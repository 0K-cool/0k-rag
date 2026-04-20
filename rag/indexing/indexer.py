"""
Knowledge Base Indexer - Store contextual chunks in LanceDB

Creates and manages the LanceDB vector database for the 0K-RAG knowledge base.
Stores contextual chunks with embeddings for hybrid semantic + keyword search.

Database Location: Configured via .0k-rag.yml
Security: 100% local storage, encrypted at rest via FileVault
"""

import lancedb
import pyarrow as pa
import logging
import os
import yaml
import fcntl
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING
from datetime import datetime
from contextlib import contextmanager
import uuid
import hashlib
import time

# Type hints for notification system (avoid circular imports)
if TYPE_CHECKING:
    from rag.notifications import NotifierInterface

logger = logging.getLogger(__name__)

# Phase 2.5: Observability Framework integration
try:
    from rag.utils.observability import RAGObservability
except ImportError:
    # Graceful degradation if observability module not available
    RAGObservability = None


class SecurityError(Exception):
    """Custom exception for security violations"""
    pass


# RAG Security Scanner for anti-poisoning protection (OWASP LLM04, LLM08)
try:
    from rag.indexing.rag_security import RAGSecurityScanner
    _security_scanner = None  # Lazy-initialized
except ImportError:
    RAGSecurityScanner = None
    _security_scanner = None
    logger.warning("RAG Security Scanner not available - anti-poisoning protection disabled")


def _sanitize_sql_value(value: str) -> str:
    """
    Sanitize string values for LanceDB SQL WHERE clauses.

    Prevents SQL injection by escaping single quotes (SQL standard: ' becomes '')

    Args:
        value: User-provided string value

    Returns:
        Sanitized string safe for SQL WHERE clauses

    Security:
        - Escapes single quotes by doubling them (SQL standard)
        - Prevents SQL injection attacks (VUL-001 fix)
        - Reference: CLAUDE.md lines 277-290 (SQL Injection Prevention)
    """
    if not isinstance(value, str):
        raise TypeError(f"Expected string, got {type(value).__name__}")

    # Escape single quotes by doubling them (SQL standard)
    return value.replace("'", "''")


def _load_allowed_base_paths() -> List[Path]:
    """
    Load allowed base paths from .0k-rag.yml configuration.

    Returns:
        List of allowed base directory paths

    Security:
        - Loads security.allowed_base_paths from config
        - Falls back to current directory if config not found
        - Resolves all paths to absolute paths
    """
    try:
        config_path = Path(os.getenv("RAG_CONFIG", ".0k-rag.yml"))

        # Try to find config in current directory or parent directories
        search_path = Path.cwd()
        for _ in range(5):  # Search up to 5 parent directories
            candidate = search_path / config_path
            if candidate.exists():
                with open(candidate) as f:
                    config = yaml.safe_load(f)
                    paths = config.get("security", {}).get("allowed_base_paths", [])
                    if paths:
                        return [Path(p).expanduser().resolve() for p in paths]
                break
            search_path = search_path.parent

        # Fallback: current working directory
        logger.warning("No allowed_base_paths in config, using current directory")
        return [Path.cwd().resolve()]

    except Exception as e:
        logger.warning(f"Could not load allowed paths from config: {e}, using current directory")
        return [Path.cwd().resolve()]


def _validate_path(user_path: str, allowed_bases: Optional[List[Path]] = None) -> Path:
    """
    Validate that a path is within allowed base directories.

    Prevents path traversal attacks by ensuring resolved paths
    are within allowed directories.

    Args:
        user_path: User-provided path (may contain ../)
        allowed_bases: List of allowed base directories (loads from config if None)

    Returns:
        Validated absolute path

    Raises:
        SecurityError: If path is outside allowed directories

    Security:
        - Resolves symlinks and .. components (Path.resolve())
        - Checks if path is under allowed base directories
        - Prevents path traversal attacks (VUL-002 fix)
        - Reference: CLAUDE.md lines 155-170 (Path Traversal Prevention)

    Examples:
        >>> _validate_path("docs/readme.md", [Path("/home/user/project")])
        PosixPath('/home/user/project/docs/readme.md')

        >>> _validate_path("../../etc/passwd", [Path("/home/user/project")])
        SecurityError: Path traversal attempt detected
    """
    if not isinstance(user_path, str):
        raise TypeError(f"Expected string path, got {type(user_path).__name__}")

    # Load allowed base paths if not provided
    if allowed_bases is None:
        allowed_bases = _load_allowed_base_paths()

    # Resolve the user-provided path (expands ~, resolves .., follows symlinks)
    try:
        resolved_path = Path(user_path).expanduser().resolve()
    except (ValueError, OSError) as e:
        raise SecurityError(f"Invalid path: {user_path} ({e})")

    # Check if resolved path is within any allowed base directory
    for base in allowed_bases:
        try:
            resolved_path.relative_to(base)
            # Path is within this base - valid!
            return resolved_path
        except ValueError:
            # Not within this base, try next one
            continue

    # Path is not within any allowed base directory
    raise SecurityError(
        f"Path traversal attempt: {user_path} resolves to {resolved_path}, "
        f"which is outside allowed directories: {[str(b) for b in allowed_bases]}"
    )


class KnowledgeBaseIndexer:
    """Manage LanceDB knowledge base"""

    # --- dedup + vacuum tunables ---
    # LanceDB has no DISTINCT, so we load rows and dedupe in memory.
    # HASH_LOOKUP_LIMIT caps the per-call hash-first query (documents with
    # more chunks across historical paths than this fall back to
    # path-based dedup rather than acting on a partial set).
    # VACUUM_SCAN_ROW_LIMIT is the hard cap for vacuum_orphans(); past
    # this, vacuum aborts so we don't report false orphans.
    # VACUUM_SCAN_WARN_THRESHOLD tells operators to paginate / split by
    # project before they hit the hard cap.
    HASH_LOOKUP_LIMIT: int = 10_000
    VACUUM_SCAN_ROW_LIMIT: int = 1_000_000
    VACUUM_SCAN_WARN_THRESHOLD: int = 100_000

    def __init__(self, db_path: str = "lance_vex_kb"):  # NOTE: lance_vex_kb is the legacy default path — preserved for existing installations
        """
        Initialize indexer

        Args:
            db_path: Path to LanceDB database (relative or absolute)
        """
        # Security: Validate db_path to prevent path traversal (VUL-002 fix)
        # Note: For database paths, we validate but allow relative paths within project
        try:
            validated_db_path = _validate_path(db_path)
            self.db_path = validated_db_path
        except SecurityError as e:
            logger.warning(f"Database path validation failed: {e}")
            # For backward compatibility, fall back to expanduser() but log warning
            self.db_path = Path(db_path).expanduser()

        self.db = None
        self.table = None
        self.table_name = "knowledge_base"
        self.indexed_count = 0
        self._lock_file = None
        self._lock_path = self.db_path / ".write.lock"

        # Create database directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _write_lock(self, timeout: float = 30.0):
        """
        Acquire exclusive file lock for LanceDB writes.
        LanceDB does not handle concurrent writers — this prevents
        fragment corruption when MCP server and CLI indexer run simultaneously.
        """
        lock_path = self._lock_path
        lock_fd = None
        try:
            lock_fd = open(lock_path, 'w')
            start = time.monotonic()
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_fd.write(f"{os.getpid()}\n")
                    lock_fd.flush()
                    logger.debug(f"Write lock acquired by PID {os.getpid()}")
                    break
                except (IOError, OSError):
                    elapsed = time.monotonic() - start
                    if elapsed >= timeout:
                        raise TimeoutError(
                            f"Could not acquire write lock on {lock_path} after {timeout}s. "
                            f"Another process may be writing to this LanceDB database."
                        )
                    time.sleep(0.5)
            yield
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                    logger.debug(f"Write lock released by PID {os.getpid()}")
                except Exception:
                    pass

    def initialize(self):
        """Initialize or connect to LanceDB database"""
        try:
            # Connect to database
            self.db = lancedb.connect(str(self.db_path))
            logger.info(f"Connected to LanceDB at {self.db_path}")

            # Check if table exists
            table_names = self.db.table_names()

            if self.table_name in table_names:
                self.table = self.db.open_table(self.table_name)
                count = len(self.table)
                logger.info(f"Opened existing table '{self.table_name}' ({count} chunks)")
            else:
                logger.info(f"Table '{self.table_name}' will be created on first index")

        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            raise

    def _create_schema(self) -> pa.Schema:
        """
        Create PyArrow schema for knowledge base

        Schema matches Anthropic's contextual retrieval requirements
        """
        return pa.schema([
            # Identifiers
            pa.field("chunk_id", pa.string()),
            pa.field("chunk_index", pa.int32()),

            # Content
            pa.field("original_chunk", pa.string()),
            pa.field("contextual_chunk", pa.string()),
            pa.field("generated_context", pa.string()),

            # Embedding
            pa.field("vector", pa.list_(pa.float32(), 768)),  # nomic-embed-text dimensions

            # Metadata
            pa.field("source_file", pa.string()),
            pa.field("source_project", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("content_hash", pa.string()),  # SHA-256 hash for content-based deduplication

            # Timestamps
            pa.field("indexed_at", pa.string()),
            pa.field("last_updated", pa.string()),

            # Token counts
            pa.field("token_count", pa.int32()),

            # Provenance & Security (OWASP LLM04, LLM08)
            pa.field("trust_level", pa.string()),      # TRUSTED, VERIFIED, UNTRUSTED
            pa.field("trust_score", pa.float32()),     # 0.0 - 1.0
            pa.field("security_risk", pa.string()),    # CLEAN, LOW, MEDIUM, HIGH, CRITICAL
        ])

    def index_chunks(
        self,
        contextual_chunks: List,
        embeddings: List,
        document_path: str,
        project: str,
        file_type: str,
        content_hash: str,
        provenance_metadata: Optional[Dict] = None
    ) -> int:
        """
        Index contextual chunks with embeddings into LanceDB

        Args:
            contextual_chunks: List of ContextualChunk objects
            embeddings: List of embedding vectors (768-dim each)
            document_path: Path to source document
            project: Project name (PAI, IR-Platform, etc.)
            file_type: File extension (.md, .py, etc.)

        Returns:
            Number of chunks successfully indexed
        """
        if len(contextual_chunks) != len(embeddings):
            logger.error(f"Mismatch: {len(contextual_chunks)} chunks vs {len(embeddings)} embeddings")
            return 0

        # Prepare data for insertion
        data = []
        now = datetime.now().isoformat()

        for ctx_chunk, embedding in zip(contextual_chunks, embeddings):
            if embedding is None:
                logger.warning(f"Skipping chunk {ctx_chunk.chunk_index} (no embedding)")
                continue

            # Create chunk record
            record = {
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": ctx_chunk.chunk_index,
                "original_chunk": ctx_chunk.original_chunk,
                "contextual_chunk": ctx_chunk.contextual_chunk,
                "generated_context": ctx_chunk.generated_context,
                "vector": embedding,
                "source_file": Path(document_path).name,
                "source_project": project,
                "file_path": document_path,
                "file_type": file_type,
                "content_hash": content_hash,  # SHA-256 for content-based deduplication
                "indexed_at": now,
                "last_updated": now,
                "token_count": len(ctx_chunk.original_chunk) // 4,  # Estimate
                # Provenance & Security (OWASP LLM04, LLM08)
                "trust_level": provenance_metadata.get('trust_level', 'VERIFIED') if provenance_metadata else 'VERIFIED',
                "trust_score": provenance_metadata.get('trust_score', 0.75) if provenance_metadata else 0.75,
                "security_risk": provenance_metadata.get('security_risk', 'CLEAN') if provenance_metadata else 'CLEAN',
            }
            data.append(record)

        if not data:
            logger.error(f"No valid chunks to index")
            return 0

        try:
            with self._write_lock():
                # Create or append to table
                if self.table is None:
                    # Create new table
                    self.table = self.db.create_table(
                        self.table_name,
                        data=data,
                        schema=self._create_schema()
                    )
                    logger.info(f"Created table '{self.table_name}' with {len(data)} chunks")
                else:
                    # Append to existing table
                    self.table.add(data)
                    logger.info(f"Added {len(data)} chunks to '{self.table_name}'")

                self.indexed_count += len(data)
                return len(data)

        except TimeoutError as e:
            logger.error(f"Write lock timeout: {e}")
            return 0
        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
            return 0

    def index_document(
        self,
        document,
        enable_security_scan: bool = True,
        notifier: Optional["NotifierInterface"] = None
    ) -> int:
        """
        Index a complete document through the full RAG pipeline

        This is the high-level method that orchestrates:
        0. Security scan for injection patterns (OWASP LLM04, LLM08)
        1. Chunking the document
        2. Generating context for each chunk
        3. Embedding contextual chunks
        4. Indexing into LanceDB

        Args:
            document: Document object with content, file_path, and project
            enable_security_scan: Enable RAG anti-poisoning scan (default: True)
            notifier: Optional progress notifier for UI updates (default: None)

        Returns:
            Number of chunks successfully indexed
        """
        # Phase 2.5: Start observability tracking
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        obs = RAGObservability() if RAGObservability else None

        # Import notification models (lazy import to avoid circular dependency)
        from rag.notifications import ProgressEvent, IndexingStage, NullNotifier

        # Use null notifier if none provided (backward compatibility)
        if notifier is None:
            notifier = NullNotifier()

        from .chunker import SmartChunker
        from .context_generator import ContextGenerator
        from .embedder import Embedder

        try:
            # Signal start of indexing
            notifier.start(document.file_path, total_stages=6)

            # Stage 1: Loading
            notifier.notify(ProgressEvent(
                stage=IndexingStage.LOADING,
                message="Loading document",
                current=1,
                total=1,
                file_path=document.file_path
            ))

            # Security: Validate file_path to prevent path traversal (VUL-002 fix)
            validated_path = _validate_path(document.file_path)
            logger.info(f"Indexing document: {document.file_path} (validated: {validated_path})")

            # Stage 2: Security Scan (OWASP LLM04, LLM08 - Anti-poisoning)
            provenance_metadata = {}
            if enable_security_scan and RAGSecurityScanner:
                notifier.notify(ProgressEvent(
                    stage=IndexingStage.SECURITY,
                    message="Running security scan",
                    current=1,
                    total=1,
                    file_path=document.file_path
                ))

                global _security_scanner
                if _security_scanner is None:
                    _security_scanner = RAGSecurityScanner(
                        strict_mode=False,  # Sanitize but don't block (default)
                        indexer_id="0k-rag",
                        audit_log_path=str(Path.home() / ".0k-rag/logs/rag-security-audit.jsonl")
                    )

                is_safe, sanitized_content, provenance = _security_scanner.scan_document(
                    content=document.content,
                    source_path=document.file_path,
                    source_type="FILE",
                    metadata={"project": document.project}
                )

                if not is_safe:
                    # In strict mode, blocked documents raise an error
                    error_msg = (
                        f"Document blocked by RAG security scan: {document.file_path} "
                        f"(risk: {provenance.security_scan_result.get('risk_level', 'UNKNOWN')})"
                    )
                    notifier.notify(ProgressEvent(
                        stage=IndexingStage.ERROR,
                        message=error_msg,
                        error=error_msg,
                        file_path=document.file_path
                    ))
                    notifier.finish(success=False, message=error_msg)
                    raise SecurityError(error_msg)

                # Use sanitized content for indexing
                document.content = sanitized_content

                # Store provenance for metadata
                provenance_metadata = {
                    'trust_level': provenance.trust_level,
                    'trust_score': provenance.trust_score,
                    'security_risk': provenance.security_scan_result.get('risk_level', 'CLEAN'),
                    'pattern_count': provenance.security_scan_result.get('pattern_count', 0),
                }

                logger.info(
                    f"RAG Security: {document.file_path} - "
                    f"Trust: {provenance.trust_score:.2f} ({provenance.trust_level}), "
                    f"Risk: {provenance.security_scan_result.get('risk_level', 'CLEAN')}"
                )

            # Compute content hash for deduplication
            content_hash = hashlib.sha256(document.content.encode('utf-8')).hexdigest()
            logger.info(f"Document content hash: {content_hash[:16]}...")

            # Smart dedup — hash-first, then path-based
            #
            # Flow:
            #   1. Look up rows by content_hash (catches moves/renames).
            #      a. If hash matches at the same file_path → truly unchanged, skip.
            #      b. If hash matches at a *different* file_path → move/rename
            #         detected; update the file_path pointer on existing chunks
            #         instead of re-embedding.
            #   2. Otherwise look up by file_path (catches content changes at the
            #      same path); delete the stale chunks, then fall through to
            #      chunking + embedding.
            #
            # Rationale for hash-first: the original logic (path-first) treated
            # any moved file as a brand-new document. When the file moved from
            # path A to path B, chunks at A became orphans and B was re-embedded
            # at full cost. Hash-first avoids both the wasted embedding and the
            # orphan pollution of the search index.
            #
            # Write lock protects the read/update/delete sequence against TOCTOU.
            if self.table is not None:
                try:
                    with self._write_lock():
                        # SQL-escape is ONLY for interpolation into LanceDB WHERE
                        # clauses. In-memory comparisons (membership tests) and
                        # parameterized `update(values=...)` must use the raw
                        # path — otherwise apostrophes in filenames cause silent
                        # mismatches or get doubled in the stored data.
                        raw_path = document.file_path
                        safe_path = _sanitize_sql_value(raw_path)  # for WHERE only

                        # Step 1: hash-first lookup (catches moves and true no-ops).
                        # Bounded by HASH_LOOKUP_LIMIT — a document with that
                        # many chunks at historical paths is an extreme outlier.
                        # If we hit the cap we warn and fall through to
                        # path-based dedup rather than acting on a partial set.
                        # Note: `>=` is intentionally conservative — a document
                        # with exactly HASH_LOOKUP_LIMIT chunks is treated as
                        # truncated. Acceptable trade-off vs. the risk of
                        # losing a would-be orphan on the boundary.
                        hash_matches = (
                            self.table.search()
                            .where(f"content_hash = '{content_hash}'")
                            .limit(self.HASH_LOOKUP_LIMIT)
                            .to_list()
                        )
                        if len(hash_matches) >= self.HASH_LOOKUP_LIMIT:
                            # Lazy %-style logging: skips format eval when the
                            # warning level is disabled (ruff G004 / pylint W1203).
                            logger.warning(
                                "hash-first dedup: content_hash %s... has "
                                "%d+ chunks across historical paths — results "
                                "truncated. Skipping move-detection and "
                                "falling through to path-based dedup.",
                                content_hash[:16],
                                self.HASH_LOOKUP_LIMIT,
                            )
                            hash_matches = []

                        if hash_matches:
                            existing_paths = {row["file_path"] for row in hash_matches}

                            if raw_path in existing_paths:
                                # Case 1a — same path + same hash → unchanged, skip.
                                count_result = self.table.count_rows(
                                    f"file_path = '{safe_path}'"
                                )
                                logger.info(
                                    f"Document unchanged (path+hash match) — skipping "
                                    f"{count_result} existing chunks"
                                )
                                notifier.finish(
                                    success=True,
                                    message=f"Skipped (unchanged): {count_result} existing chunks",
                                )
                                return count_result

                            # Case 1b — move/rename. Retarget the pointer rather
                            # than re-embed. We only retarget chunks that share
                            # BOTH the old path and the content hash so we never
                            # clobber a legitimate different-content doc sitting
                            # at the same old path.
                            old_paths = sorted(existing_paths)
                            logger.info(
                                f"Move detected — content at {old_paths} now at "
                                f"{raw_path}. Updating file_path pointer."
                            )
                            new_last_updated = datetime.now().isoformat()
                            for old_path in old_paths:
                                safe_old = _sanitize_sql_value(old_path)
                                self.table.update(
                                    where=(
                                        f"content_hash = '{content_hash}' "
                                        f"AND file_path = '{safe_old}'"
                                    ),
                                    # values={} is parameterized by LanceDB — pass
                                    # raw strings, not SQL-escaped ones.
                                    values={
                                        "file_path": raw_path,
                                        "last_updated": new_last_updated,
                                    },
                                )
                            count_result = self.table.count_rows(
                                f"file_path = '{safe_path}'"
                            )
                            logger.info(
                                f"Moved {count_result} chunks to {raw_path} "
                                f"(no re-embedding)"
                            )
                            notifier.finish(
                                success=True,
                                message=f"Moved: pointer updated for {count_result} chunks",
                            )
                            return count_result

                        # Step 2: path-based lookup (catches changed content at
                        # the same path — hash didn't match above, so either the
                        # path is new or the content differs).
                        existing = (
                            self.table.search()
                            .where(f"file_path = '{safe_path}'")
                            .limit(1)
                            .to_list()
                        )
                        if existing:
                            count_result = self.table.count_rows(
                                f"file_path = '{safe_path}'"
                            )
                            logger.info(
                                f"Document content changed — removing {count_result} "
                                f"existing chunks before re-indexing"
                            )
                            self.table.delete(f"file_path = '{safe_path}'")
                            logger.info(
                                f"Deleted {count_result} existing chunks for "
                                f"{document.file_path}"
                            )
                except TimeoutError as e:
                    logger.error(f"Write lock timeout during dedup check: {e}")
                    notifier.finish(success=False, message=f"Write lock timeout: {e}")
                    return 0
                except Exception as e:
                    logger.warning(f"Could not check for existing chunks: {e}")

            # Stage 3: Chunking
            notifier.notify(ProgressEvent(
                stage=IndexingStage.CHUNKING,
                message="Chunking document",
                current=1,
                total=1,
                file_path=document.file_path
            ))

            chunker = SmartChunker(chunk_size=384, overlap_percentage=0.15)
            chunks = chunker.chunk_document(document.content, Path(document.file_path).suffix)
            logger.info(f"Chunked into {len(chunks)} chunks")

            if not chunks:
                logger.warning(f"No chunks generated from document")
                notifier.finish(success=True, message="No content to index")
                return 0

            notifier.notify(ProgressEvent(
                stage=IndexingStage.CHUNKING,
                message=f"Created {len(chunks)} chunks",
                current=1,
                total=1,
                file_path=document.file_path
            ))

            # Stage 4: Context Generation (PARALLEL + SELECTIVE + FASTER MODEL)
            # Using llama3.2:1b for 3-5x speedup vs llama3.1:8b (smaller, faster model)
            context_gen = ContextGenerator(model="llama3.2:1b")
            contextual_chunks = context_gen.generate_contexts_parallel(
                chunks=chunks,
                full_document=document.content,
                file_path=document.file_path,
                project=document.project,
                max_workers=4,  # Safe limit for 16GB+ RAM (adjust based on system)
                notifier=notifier  # Pass notifier for per-chunk progress
            )

            # Stage 5: Embedding
            embedder = Embedder(model="nomic-embed-text")
            contextual_texts = [cc.contextual_chunk for cc in contextual_chunks]
            embeddings = embedder.embed_batch(
                contextual_texts,
                show_progress=True,
                notifier=notifier  # Pass notifier for progress
            )
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Stage 6: Indexing into LanceDB
            notifier.notify(ProgressEvent(
                stage=IndexingStage.INDEXING,
                message="Writing to database",
                current=1,
                total=1,
                file_path=document.file_path
            ))

            chunk_count = self.index_chunks(
                contextual_chunks,
                embeddings,
                document.file_path,
                document.project,
                Path(document.file_path).suffix,
                content_hash,
                provenance_metadata
            )

            logger.info(f"Successfully indexed {chunk_count} chunks from {document.file_path}")

            # Phase 2.5: Log indexing operation
            if obs and obs.enabled:
                try:
                    latency_ms = int((time.time() - start_time) * 1000)
                    obs.log_index_operation(
                        file_path=document.file_path,
                        num_chunks=chunk_count,
                        latency_ms=latency_ms,
                        trace_id=trace_id
                    )
                except Exception as log_err:
                    # Graceful degradation - don't fail indexing due to logging
                    logger.debug(f"Observability logging failed: {log_err}")

            # Signal completion
            notifier.finish(success=True, message=f"Indexed {chunk_count} chunks")
            return chunk_count

        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            notifier.notify(ProgressEvent(
                stage=IndexingStage.ERROR,
                message=str(e),
                error=str(e),
                file_path=document.file_path
            ))
            notifier.finish(success=False, message=str(e))
            raise

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """
        Search knowledge base using vector similarity

        Args:
            query_embedding: Query embedding vector (768-dim)
            limit: Number of results to return

        Returns:
            List of matching chunks with metadata
        """
        if self.table is None:
            logger.error(f"No table initialized")
            return []

        try:
            results = (
                self.table
                .search(query_embedding)
                .limit(limit)
                .to_list()
            )
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def delete_by_file(self, file_path: str) -> int:
        """
        Delete all chunks from a specific file

        Args:
            file_path: Path to file

        Returns:
            Number of chunks deleted
        """
        if self.table is None:
            return 0

        try:
            with self._write_lock():
                # Security: Sanitize file_path to prevent SQL injection (VUL-001 fix)
                safe_path = _sanitize_sql_value(file_path)
                self.table.delete(f"file_path = '{safe_path}'")
                logger.info(f"Deleted chunks from {file_path}")
                return 1  # LanceDB doesn't return count

        except TimeoutError as e:
            logger.error(f"Write lock timeout: {e}")
            return 0
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return 0

    def delete_by_project(self, project: str) -> int:
        """
        Delete all chunks from a specific project

        Args:
            project: Project name

        Returns:
            Number of chunks deleted
        """
        if self.table is None:
            return 0

        try:
            with self._write_lock():
                # Security: Sanitize project to prevent SQL injection (VUL-001 fix)
                safe_project = _sanitize_sql_value(project)
                self.table.delete(f"source_project = '{safe_project}'")
                logger.info(f"Deleted chunks from project {project}")
                return 1

        except TimeoutError as e:
            logger.error(f"Write lock timeout: {e}")
            return 0
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return 0

    def vacuum_orphans(
        self,
        dry_run: bool = True,
        match: Optional[str] = None,
    ) -> Dict:
        """
        Find (and optionally delete) chunks whose file_path no longer exists
        on disk.

        Orphans accumulate when a source file is deleted or moved without
        going through index_document (pre-hash-first-dedup releases did not
        catch renames, so moves produced orphan chunks at the old path).
        This method walks every distinct file_path in the KB, stats each,
        and reports which paths are gone.

        Safety discipline: orphan detection is NOT implicit permission to
        delete. RAG knowledge may be valuable independent of whether the
        source file still exists. Deletion requires explicit dry_run=False
        PLUS an optional `match` filter that restricts pruning to a subset
        of the orphan paths — this makes it harder to accidentally wipe a
        category the caller didn't mean to touch.

        Args:
            dry_run: when True (default), list orphans without deleting.
                     Re-run with dry_run=False to actually prune.
            match:   substring filter applied to file_path before deletion.
                     Only orphan paths containing this substring are deleted
                     (the full orphan list is still reported). When None
                     and dry_run=False, every orphan is deleted — use with
                     care and only after reviewing the dry-run report.

        Returns:
            dict with:
              - 'orphan_paths': sorted list of file_paths no longer on disk
                 (unfiltered — the caller always sees the full set)
              - 'orphan_chunk_count': total chunks under those paths
              - 'deleted_paths': paths actually deleted (honors `match`)
              - 'deleted_chunk_count': 0 when dry_run, else chunks removed
              - 'scanned_paths': total distinct file_paths checked
              - 'match_filter': the filter used (or None)
        """
        result: Dict = {
            "orphan_paths": [],
            "orphan_chunk_count": 0,
            "deleted_paths": [],
            "deleted_chunk_count": 0,
            "scanned_paths": 0,
            "match_filter": match,
            "error": None,  # non-None on partial failure or early exit
        }

        if self.table is None:
            logger.warning("vacuum_orphans called before table initialized")
            result["error"] = "table_not_initialized"
            return result

        # Pulling every file_path in the KB. LanceDB doesn't expose DISTINCT,
        # so we load the column and dedupe in memory. At ~150 docs this is
        # trivial; near the hard cap we warn operators that pagination may
        # be needed to avoid memory pressure.
        try:
            all_rows = (
                self.table.search()
                .select(["file_path"])
                .limit(self.VACUUM_SCAN_ROW_LIMIT)
                .to_list()
            )
            if len(all_rows) >= self.VACUUM_SCAN_ROW_LIMIT:
                # Hard cap hit → abort early; operator needs to split the
                # sweep (e.g., per-project) before we can reason about
                # orphans. Don't also emit the warning — that would
                # double-log the same condition confusingly.
                # Lazy %-style args: skip format eval when level disabled.
                logger.error(
                    "vacuum: hit %d-row scan cap — orphan detection is "
                    "INCOMPLETE. Aborting to avoid false orphan claims.",
                    self.VACUUM_SCAN_ROW_LIMIT,
                )
                result["error"] = "scan_row_limit_reached"
                return result
            if len(all_rows) >= self.VACUUM_SCAN_WARN_THRESHOLD:
                logger.warning(
                    "vacuum: scanning %d rows — approaching the %d hard "
                    "cap. Consider paginating or running per-project "
                    "sweeps to avoid truncation.",
                    len(all_rows),
                    self.VACUUM_SCAN_ROW_LIMIT,
                )
            unique_paths = sorted({row["file_path"] for row in all_rows})
            result["scanned_paths"] = len(unique_paths)

            orphan_paths: List[str] = []
            orphan_chunk_count = 0
            for path in unique_paths:
                # Path.exists() is symlink-following; orphan detection is
                # about "can we still reach the file" not "is it canonical".
                if not Path(path).exists():
                    chunk_count = self.table.count_rows(
                        f"file_path = '{_sanitize_sql_value(path)}'"
                    )
                    orphan_paths.append(path)
                    orphan_chunk_count += chunk_count

            result["orphan_paths"] = orphan_paths
            result["orphan_chunk_count"] = orphan_chunk_count

            if not orphan_paths:
                logger.info("vacuum: no orphan chunks found")
                return result

            logger.info(
                f"vacuum: found {len(orphan_paths)} orphan paths "
                f"totalling {orphan_chunk_count} chunks (dry_run={dry_run})"
            )

            if dry_run:
                return result

            # Apply match filter BEFORE acquiring write lock — we want to
            # know exactly what's being deleted and log it before touching
            # the table. Safety rule: never delete without explicit intent.
            if match is not None:
                targeted = [p for p in orphan_paths if match in p]
                logger.info(
                    f"vacuum: --match={match!r} narrowed "
                    f"{len(orphan_paths)} orphans → {len(targeted)} targeted"
                )
            else:
                targeted = list(orphan_paths)

            if not targeted:
                logger.info("vacuum: match filter selected zero paths; nothing to delete")
                return result

            # Delete. Write lock wraps the whole sweep so concurrent
            # index_document calls don't race with our deletes.
            deleted_paths: List[str] = []
            deleted_count = 0
            with self._write_lock():
                for path in targeted:
                    safe = _sanitize_sql_value(path)
                    count = self.table.count_rows(f"file_path = '{safe}'")
                    self.table.delete(f"file_path = '{safe}'")
                    deleted_paths.append(path)
                    deleted_count += count
                    logger.info(f"vacuum: deleted {count} chunks for {path}")
            result["deleted_paths"] = deleted_paths
            result["deleted_chunk_count"] = deleted_count
            return result

        except TimeoutError as e:
            # Full exception detail stays in the logger call (local log
            # file only). The structured `error` key on the returned
            # dict carries ONLY the error class name — callers sometimes
            # serialize this to log aggregators / API responses, and
            # exception messages can embed paths, SQL fragments, or
            # connection strings we shouldn't leak there.
            logger.error("vacuum: write lock timeout: %s", e)
            result["error"] = "write_lock_timeout"
            return result
        except Exception as e:
            logger.error("vacuum failed: %s: %s", type(e).__name__, e)
            result["error"] = f"exception: {type(e).__name__}"
            return result

    def get_stats(self) -> Dict:
        """Get indexer statistics"""
        stats = {
            'db_path': str(self.db_path),
            'total_indexed': self.indexed_count
        }

        if self.table:
            try:
                stats['total_chunks'] = len(self.table)
            except:
                stats['total_chunks'] = 'unknown'

        return stats

    def create_fts_index(self):
        """
        Create full-text search index for BM25 (Phase 2)
        This will be implemented in Phase 2
        """
        if self.table is None:
            logger.error(f"No table to index")
            return

        try:
            with self._write_lock():
                self.table.create_fts_index("contextual_chunk")
                logger.info(f"Created full-text search index")
        except TimeoutError as e:
            logger.error(f"Write lock timeout during FTS creation: {e}")
        except Exception as e:
            logger.error(f"FTS index creation failed: {e}")
