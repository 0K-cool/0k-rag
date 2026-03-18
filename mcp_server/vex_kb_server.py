#!/usr/bin/env python3
"""
Vex Knowledge Base MCP Server (Plugin Version)

Provides MCP (Model Context Protocol) access to the Vex RAG system.
Enables automatic context injection into Claude Code conversations.

This is a configurable version that reads settings from .vex-rag.yml
in the project directory, making it portable across multiple projects.

Resources:
- vex://help - Get usage instructions and available capabilities
- vex://search/{query} - Search knowledge base and return top results

Tools:
- search_kb - Search the knowledge base (RECOMMENDED - always discoverable)
- index_document - Index a new document into knowledge base
- get_kb_stats - Get knowledge base statistics

100% local, zero cloud APIs, zero data exfiltration.
"""

import sys
import signal
import logging
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: FastMCP not installed. Install with: pip install fastmcp", file=sys.stderr)
    sys.exit(1)

from rag.retrieval.pipeline import RetrievalPipeline
from rag.indexing.document_loader import DocumentLoader
from rag.indexing.indexer import KnowledgeBaseIndexer, _validate_path, SecurityError
from rag.indexing.sanitizer import Sanitizer
from rag.notifications import (
    ProgressEvent,
    IndexingStage,
    NullNotifier,
    create_notifier_from_config,
)

# Load configuration
def load_config() -> Dict:
    """Load project-specific RAG configuration from .vex-rag.yml"""
    # Get config path from environment or use default
    config_path = Path(os.getenv("RAG_CONFIG", ".vex-rag.yml"))

    # If relative path, resolve from current directory
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    if not config_path.exists():
        # Fallback: try parent directory (for MCP server running from subdirectory)
        config_path = Path.cwd().parent / ".vex-rag.yml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"RAG configuration not found: {config_path}\n"
            f"Create .vex-rag.yml in your project root or set RAG_CONFIG environment variable.\n"
            f"See examples in ~/.claude/plugins/vex-rag/examples/"
        )

    with open(config_path) as f:
        return yaml.safe_load(f)

# Load configuration
try:
    config = load_config()
except FileNotFoundError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)

# Extract configuration values
PROJECT_NAME = config['project']['name']
DB_PATH = config['database']['path']
ENABLE_RERANKING = config['retrieval'].get('enable_reranking', True)
RERANKER_MODEL = config['retrieval'].get('reranker_model', 'BAAI/bge-reranker-large')
DEFAULT_TOP_K = config['retrieval'].get('default_top_k', 5)
ENABLE_SANITIZATION = config['indexing'].get('enable_sanitization', True)
LOG_LEVEL = config.get('logging', {}).get('level', 'INFO')
LOG_FILE = config.get('logging', {}).get('file', '.claude/logs/rag.log')

# Configure logging
# IMPORTANT: Only log to file, NOT stderr. Stderr output causes Claude Code
# to interpret the MCP server as "failed" even during normal shutdown.
log_file_path = Path(LOG_FILE)
log_file_path.parent.mkdir(parents=True, exist_ok=True)

# Create file handler for all logging
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure root logger with ONLY file handler (no stderr)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler]
)

logger = logging.getLogger('vex_kb_server')
logger.info(f"Vex RAG MCP Server starting for project: {PROJECT_NAME}")
logger.info(f"Configuration loaded from: {os.getenv('RAG_CONFIG', '.vex-rag.yml')}")
logger.info(f"Database path: {DB_PATH}")
logger.info(f"Reranking: {'enabled' if ENABLE_RERANKING else 'disabled'}")

# Initialize MCP server
# NOTE: FastMCP.__init__ calls configure_logging() which adds RichHandler(stderr=True)
mcp = FastMCP(f"Vex Knowledge Base ({PROJECT_NAME})")

# CRITICAL: Suppress FastMCP's stderr logging AFTER FastMCP initialization
# FastMCP.__init__ calls configure_logging() which adds RichHandler(stderr=True)
# This causes Claude Code to interpret any output as "MCP server failed"
# We MUST reconfigure these loggers AFTER FastMCP is initialized
for logger_name in ['mcp', 'mcp.server', 'mcp.server.lowlevel', 'mcp.server.fastmcp', 'FastMCP']:
    mcp_logger = logging.getLogger(logger_name)
    mcp_logger.handlers = [file_handler]  # Replace RichHandler with file-only
    mcp_logger.propagate = False  # Don't propagate to root logger

logger.info("FastMCP stderr logging suppressed (file-only logging enabled)")

# Initialize retrieval pipeline (lazy-loaded on first use)
_pipeline: Optional[RetrievalPipeline] = None
_indexer: Optional[KnowledgeBaseIndexer] = None

# Graceful shutdown handling
_shutdown_requested = False

def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully"""
    global _shutdown_requested, _pipeline, _indexer

    if _shutdown_requested:
        # Already shutting down, force exit
        logger.info("Forced shutdown")
        sys.exit(0)

    _shutdown_requested = True
    signal_name = signal.Signals(signum).name if signum else "UNKNOWN"
    logger.info(f"Received {signal_name} - initiating graceful shutdown...")

    # Cleanup resources
    try:
        if _pipeline is not None:
            logger.info("Closing retrieval pipeline...")
            # Pipeline cleanup if needed
            _pipeline = None

        if _indexer is not None:
            logger.info("Closing indexer...")
            # Indexer cleanup if needed
            _indexer = None

        logger.info("Vex Knowledge Base MCP Server shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)
# SIGHUP for terminal hangup (when Claude Code exits)
if hasattr(signal, 'SIGHUP'):
    signal.signal(signal.SIGHUP, graceful_shutdown)

logger.info("Signal handlers registered for graceful shutdown")


def get_pipeline() -> RetrievalPipeline:
    """Get or initialize retrieval pipeline"""
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing retrieval pipeline...")
        try:
            _pipeline = RetrievalPipeline(
                db_path=DB_PATH,
                enable_reranking=ENABLE_RERANKING,
                reranker_model=RERANKER_MODEL
            )
            logger.info("Retrieval pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    return _pipeline


def get_indexer() -> KnowledgeBaseIndexer:
    """Get or initialize knowledge base indexer"""
    global _indexer
    if _indexer is None:
        logger.info("Initializing knowledge base indexer...")
        try:
            _indexer = KnowledgeBaseIndexer(db_path=DB_PATH)
            _indexer.initialize()
            logger.info("Indexer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize indexer: {e}")
            raise
    return _indexer


# =============================================================================
# MCP RESOURCES
# =============================================================================

@mcp.resource("vex://help")
def get_help() -> str:
    """
    Get usage instructions for the Vex Knowledge Base.

    This resource provides onboarding information for AI agents
    discovering the plugin for the first time.
    """
    return f"""
Vex Knowledge Base - RAG System for {PROJECT_NAME}

SEARCH (use the tool - always discoverable):
  search_kb(query, top_k=5)

  Examples:
    search_kb("authentication bypass")
    search_kb("incident response workflow", top_k=3)
    search_kb("MITRE ATT&CK persistence techniques")

ALTERNATIVE (resource - may not be discoverable):
  vex://search/{{query}}

  Examples:
    vex://search/authentication bypass
    vex://search/git safety check

INDEXING:
  index_document(file_path, project=None, enable_sanitization=None)

  Example:
    index_document("/path/to/document.md")

STATS:
  get_kb_stats()

  Returns: total chunks, projects, files, and usage hints

FEATURES:
- Hybrid retrieval (vector + BM25 + RRF fusion)
- BGE reranking for high-quality results
- Native Anthropic citations support
- PII sanitization (optional)
- 100% local - zero cloud APIs

Current project: {PROJECT_NAME}
Database: {DB_PATH}
Reranking: {'enabled' if ENABLE_RERANKING else 'disabled'}
Default top_k: {DEFAULT_TOP_K}
"""


@mcp.resource("vex://search/{query}")
def search_knowledge_base(query: str) -> str:
    """
    Search the Vex knowledge base and return results with native citations.

    This resource enables automatic context injection - when Claude needs
    information about project documentation, it can automatically search
    the knowledge base.

    Results are formatted for Anthropic's native citations API, which means:
    - Cited text doesn't count toward output tokens
    - Citations have guaranteed valid character indices
    - More reliable source attribution

    NOTE: This resource may not be discoverable by AI agents.
    Prefer using the search_kb tool instead.

    Args:
        query: Search query (e.g., "git safety check workflow")

    Returns:
        JSON-formatted documents with citations enabled
    """
    logger.info(f"MCP resource search request: '{query}'")

    try:
        # Get retrieval pipeline
        pipeline = get_pipeline()

        # Execute search with full hybrid pipeline
        results = pipeline.retrieve(
            query,
            top_k=DEFAULT_TOP_K,
            enable_bm25=True,
            verbose=False
        )

        if not results:
            logger.info(f"No results found for query: '{query}'")
            return json.dumps({
                "query": query,
                "documents": [],
                "message": f"No results found for: {query}"
            }, indent=2)

        logger.info(f"Found {len(results)} results for query: '{query}'")

        # Format results with native citations
        citation_docs = pipeline.format_for_citations(
            results,
            include_context=True
        )

        # Create response with metadata
        response = {
            "query": query,
            "documents": citation_docs,
            "message": f"Retrieved {len(citation_docs)} relevant documents from {PROJECT_NAME} knowledge base"
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "query": query,
            "documents": [],
            "error": error_msg
        }, indent=2)


# =============================================================================
# MCP TOOLS (always discoverable by AI agents)
# =============================================================================

@mcp.tool()
def search_kb(query: str, top_k: int = 5) -> str:
    """
    Search the Vex knowledge base for relevant information.

    This is the PRIMARY way to query indexed documentation, skills,
    workflows, and other knowledge. Uses hybrid retrieval (vector + BM25)
    with BGE reranking for high-quality results.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default: 5, max: 20)

    Returns:
        JSON with matching documents and citations

    Examples:
        search_kb("authentication bypass")
        search_kb("git safety check workflow", top_k=3)
        search_kb("incident response procedures")
    """
    # Clamp top_k to reasonable bounds
    top_k = max(1, min(top_k, 20))

    logger.info(f"MCP search_kb tool request: '{query}' (top_k={top_k})")

    try:
        pipeline = get_pipeline()

        results = pipeline.retrieve(
            query,
            top_k=top_k,
            enable_bm25=True,
            verbose=False
        )

        if not results:
            logger.info(f"No results found for query: '{query}'")
            return json.dumps({
                "query": query,
                "top_k": top_k,
                "documents": [],
                "message": f"No results found for: {query}"
            }, indent=2)

        logger.info(f"Found {len(results)} results for query: '{query}'")

        citation_docs = pipeline.format_for_citations(
            results,
            include_context=True
        )

        response = {
            "query": query,
            "top_k": top_k,
            "documents": citation_docs,
            "message": f"Retrieved {len(citation_docs)} relevant documents from {PROJECT_NAME} knowledge base"
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "query": query,
            "documents": [],
            "error": error_msg
        }, indent=2)


class MCPProgressCollector:
    """
    Collects progress events for MCP tool response.

    Instead of displaying progress to console, collects events
    for inclusion in the tool's response message.
    """

    def __init__(self):
        self.events: List[ProgressEvent] = []
        self._file_path: Optional[str] = None
        self._start_time: Optional[float] = None

    def notify(self, event: ProgressEvent) -> None:
        """Collect progress event"""
        self.events.append(event)
        logger.debug(f"Progress: {event.stage.name} - {event.message}")

    def start(self, file_path: str, total_stages: int = 6) -> None:
        """Signal start of indexing"""
        import time
        self._file_path = file_path
        self._start_time = time.time()
        logger.info(f"MCP indexing started: {file_path}")

    def finish(self, success: bool, message: str = "") -> None:
        """Signal end of indexing"""
        import time
        duration = time.time() - self._start_time if self._start_time else 0
        logger.info(f"MCP indexing finished: success={success}, duration={duration:.1f}s, message={message}")

    def get_summary(self) -> str:
        """Get summary of progress events for response"""
        if not self.events:
            return ""

        # Get unique stages completed
        stages_completed = set()
        for event in self.events:
            if event.stage not in (IndexingStage.COMPLETE, IndexingStage.ERROR):
                stages_completed.add(event.stage.name)

        # Find last progress event for each stage
        stage_summaries = []
        for stage in [IndexingStage.LOADING, IndexingStage.SECURITY, IndexingStage.CHUNKING,
                      IndexingStage.CONTEXT, IndexingStage.EMBEDDING, IndexingStage.INDEXING]:
            stage_events = [e for e in self.events if e.stage == stage]
            if stage_events:
                last_event = stage_events[-1]
                if last_event.total > 0:
                    stage_summaries.append(f"{last_event.emoji} {last_event.stage_description}: {last_event.current}/{last_event.total}")
                else:
                    stage_summaries.append(f"{last_event.emoji} {last_event.stage_description}")

        if stage_summaries:
            return "\n".join(stage_summaries)
        return ""


@mcp.tool()
def index_document(
    file_path: str,
    project: Optional[str] = None,
    enable_sanitization: Optional[bool] = None
) -> str:
    """
    Index a new document into the Vex knowledge base.

    This tool allows manual indexing of documents during conversations.
    Useful for adding new documentation, skills, or workflows on-the-fly.

    After indexing, use search_kb(query) to search the knowledge base.

    Args:
        file_path: Absolute or relative path to document
        project: Project name (default: from config)
        enable_sanitization: Enable PII sanitization (default: from config)

    Returns:
        Status message with indexing results and progress summary

    Example:
        index_document("/path/to/document.md")
        # Then search with: search_kb("topic from document")
    """
    # Use config defaults if not specified
    if project is None:
        project = PROJECT_NAME
    if enable_sanitization is None:
        enable_sanitization = ENABLE_SANITIZATION

    logger.info(f"MCP index request: {file_path} (project={project}, sanitize={enable_sanitization})")

    # Create progress collector for MCP response
    progress_collector = MCPProgressCollector()

    # Also create webhook notifier if configured
    notifier = progress_collector  # Use collector as primary notifier

    try:
        # Resolve file path
        path = Path(file_path)
        if not path.is_absolute():
            # Assume relative to current project directory
            path = Path.cwd() / file_path

        # Security: Validate path BEFORE loading file (VUL-002 fix)
        # This prevents reading files outside allowed directories
        try:
            validated_path = _validate_path(str(path))
            logger.info(f"Path validated: {validated_path}")
        except SecurityError as e:
            error_msg = f"Path validation failed: {e}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"

        if not path.exists():
            error_msg = f"File not found: {path}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"

        # Get indexer
        indexer = get_indexer()

        # Load document
        loader = DocumentLoader()
        doc = loader.load_file(str(path), project)

        # Sanitize if enabled
        if enable_sanitization:
            sanitizer = Sanitizer(enable_ner=True)
            result = sanitizer.sanitize(doc.content, str(path))
            doc.content = result.sanitized_text

            if result.redaction_count > 0:
                logger.info(f"Sanitization: {result.redaction_count} redactions, {len(result.detected_patterns)} patterns")

        # Index document (full pipeline: chunk → context → embed → index)
        # Pass notifier for progress tracking
        chunk_count = indexer.index_document(doc, notifier=notifier)

        # CRITICAL: Invalidate the cached retrieval pipeline so the next
        # search_kb() call reopens the LanceDB table and sees the new data.
        # The pipeline and indexer maintain separate table handles — without
        # this, searches after indexing return stale results.
        global _pipeline
        if _pipeline is not None:
            logger.info("Invalidating retrieval pipeline to pick up newly indexed data")
            _pipeline = None

        success_msg = f"Successfully indexed {chunk_count} chunks from {path.name}"
        if enable_sanitization:
            success_msg += f" (sanitized)"

        # Add progress summary to response
        progress_summary = progress_collector.get_summary()
        if progress_summary:
            success_msg += f"\n\nProgress:\n{progress_summary}"

        # Add usage hint
        success_msg += f"\n\nTo search: search_kb(\"your query\")"

        logger.info(success_msg)
        return success_msg

    except Exception as e:
        error_msg = f"Indexing failed: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


@mcp.tool()
def get_kb_stats() -> Dict[str, Any]:
    """
    Get statistics about the Vex knowledge base.

    Returns total chunks, projects, files indexed, plus usage hints
    for searching the knowledge base. Also runs a live search health
    check to detect corrupted LanceDB fragments that metadata-only
    checks would miss.

    Returns:
        Dictionary with KB stats, search health, and usage examples

    Tip: Use search_kb(query) to search the knowledge base.
    If search_healthy is false, run rebuild_index() to fix.
    """
    logger.info("MCP stats request")

    try:
        pipeline = get_pipeline()
        stats = pipeline.get_stats()

        # Search health check — exercise actual search path
        search_healthy = False
        search_error = None
        try:
            test_results = pipeline.retrieve("test", top_k=1, enable_bm25=False, verbose=False)
            search_healthy = True
        except Exception as e:
            search_error = str(e)[:200]

        stats["search_healthy"] = search_healthy
        stats["search_error"] = search_error

        # Validate FTS index files actually exist (not just metadata claim)
        fts_actually_healthy = False
        try:
            indices_dir = os.path.join(pipeline.db_path, "knowledge_base.lance", "_indices")
            if os.path.isdir(indices_dir) and os.listdir(indices_dir):
                fts_actually_healthy = True
        except Exception:
            pass

        stats["fts_healthy"] = fts_actually_healthy

        # Add usage hints (ATHENA feedback - help AI agents discover search)
        stats["usage_hint"] = "Use search_kb(query, top_k) tool to search the knowledge base"
        stats["example_queries"] = [
            'search_kb("authentication bypass")',
            'search_kb("incident response workflow", top_k=3)',
            'search_kb("security best practices")'
        ]
        stats["help_resource"] = "vex://help"

        logger.info(f"Stats retrieved: {stats['total_chunks']} total chunks, search_healthy={search_healthy}, fts_healthy={fts_actually_healthy}")
        return stats

    except Exception as e:
        error_msg = f"Stats retrieval failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def rebuild_index() -> str:
    """
    Rebuild the entire knowledge base from source files.

    Drops the existing LanceDB table and re-indexes all documents
    from the paths defined in .vex-rag.yml (auto_index_paths).
    Use when search returns 0 results despite chunks being stored,
    or when the database is corrupted.

    WARNING: This is destructive — existing embeddings are regenerated.
    Source documents are NOT affected.

    Returns:
        JSON with rebuild results (files indexed, chunks created, duration)
    """
    import time
    import glob as glob_module

    logger.info("MCP rebuild_index request — dropping and recreating knowledge base")

    start_time = time.time()

    # Read auto_index_paths from config
    auto_index_paths = config.get('indexing', {}).get('auto_index_paths', [])
    auto_index_extensions = config.get('indexing', {}).get('auto_index_extensions', ['.md'])

    if not auto_index_paths:
        return json.dumps({
            "success": False,
            "error": "No auto_index_paths defined in .vex-rag.yml. Add indexing.auto_index_paths to enable rebuild."
        }, indent=2)

    try:
        # Step 1: Drop existing LanceDB table by re-initializing a fresh indexer
        global _pipeline, _indexer

        logger.info("Dropping existing knowledge base table...")

        # Force fresh indexer (will recreate table on initialize)
        _indexer = None
        _pipeline = None

        # Import lancedb to drop the table directly
        try:
            import lancedb
            db = lancedb.connect(DB_PATH)
            table_names = db.table_names()
            if "knowledge_base" in table_names:
                db.drop_table("knowledge_base")
                logger.info("Dropped existing knowledge_base table")
            else:
                logger.info("No existing knowledge_base table found — fresh start")
        except Exception as e:
            logger.warning(f"Could not drop table via lancedb (continuing): {e}")

        # Step 2: Re-initialize indexer (creates fresh table)
        indexer = get_indexer()
        loader = DocumentLoader()

        # Step 3: Collect all .md files from auto_index_paths
        files_to_index = []
        config_dir = Path(os.getenv("RAG_CONFIG", ".vex-rag.yml"))
        if not config_dir.is_absolute():
            config_dir = Path.cwd() / config_dir
        project_root = config_dir.parent

        for index_path in auto_index_paths:
            # Resolve relative paths from project root
            resolved = Path(index_path)
            if not resolved.is_absolute():
                resolved = project_root / index_path

            if resolved.is_file():
                # Direct file reference
                if any(str(resolved).endswith(ext) for ext in auto_index_extensions):
                    files_to_index.append(resolved)
            elif resolved.is_dir():
                # Directory — recurse for all matching extensions
                for ext in auto_index_extensions:
                    files_to_index.extend(resolved.rglob(f"*{ext}"))
            else:
                logger.warning(f"rebuild_index: path not found, skipping: {resolved}")

        # Deduplicate
        files_to_index = list(set(files_to_index))
        logger.info(f"rebuild_index: found {len(files_to_index)} files to index")

        if not files_to_index:
            return json.dumps({
                "success": False,
                "error": f"No files found matching {auto_index_extensions} in auto_index_paths: {auto_index_paths}"
            }, indent=2)

        # Step 4: Re-index all files
        total_chunks = 0
        indexed_files = []
        failed_files = []

        for file_path in sorted(files_to_index):
            try:
                doc = loader.load_file(str(file_path), PROJECT_NAME)
                chunk_count = indexer.index_document(doc)
                total_chunks += chunk_count
                indexed_files.append(str(file_path.name))
                logger.info(f"rebuild_index: indexed {file_path.name} ({chunk_count} chunks)")
            except Exception as e:
                failed_files.append({"file": str(file_path.name), "error": str(e)[:100]})
                logger.error(f"rebuild_index: failed to index {file_path.name}: {e}")

        # Step 5: Invalidate pipeline so next search opens fresh table
        _pipeline = None

        duration = time.time() - start_time

        result = {
            "success": True,
            "files_indexed": len(indexed_files),
            "chunks_created": total_chunks,
            "duration_seconds": round(duration, 1),
            "files": indexed_files,
        }
        if failed_files:
            result["failed_files"] = failed_files

        logger.info(f"rebuild_index complete: {len(indexed_files)} files, {total_chunks} chunks, {duration:.1f}s")
        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = f"rebuild_index failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"success": False, "error": error_msg}, indent=2)


if __name__ == "__main__":
    import atexit

    def cleanup_on_exit():
        """Cleanup handler for atexit (fallback for graceful shutdown)"""
        global _pipeline, _indexer
        if _pipeline is not None or _indexer is not None:
            logger.info("Atexit cleanup triggered")
            _pipeline = None
            _indexer = None

    atexit.register(cleanup_on_exit)

    logger.info(f"Starting Vex Knowledge Base MCP Server for {PROJECT_NAME}...")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Working directory: {Path.cwd()}")

    try:
        # Run the MCP server (stdio transport)
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt)")
        sys.exit(0)
    except SystemExit:
        # Expected during graceful shutdown, don't log as error
        raise
    except (BrokenPipeError, EOFError, ConnectionResetError, ConnectionAbortedError):
        # Expected when Claude Code closes stdin/stdout pipes during shutdown
        # These are normal termination signals, not errors
        logger.info("Server stopped (pipe closed or connection reset)")
        sys.exit(0)
    except OSError as e:
        # Handle "Bad file descriptor" and similar OS-level errors during shutdown
        if e.errno in (9, 32):  # EBADF, EPIPE
            logger.info(f"Server stopped (OS error during shutdown: {e})")
            sys.exit(0)
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Log unexpected errors but exit cleanly to avoid "failed" status
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(0)  # Exit 0 even on error to avoid "MCP server failed" message
