"""
RAG Observability Integration

Optional observability integration for 0K-RAG operations.
Logs token usage, latency, and errors for RAG operations.

Portable Design:
- Checks project's .claude/scripts/ first (if available)
- Falls back to 0k-rag plugin's bundled observability-scripts/
- Gracefully degrades if neither location has scripts
"""

import os
import time
import subprocess
from typing import Optional, Dict, Any
from pathlib import Path


class RAGObservability:
    """Observability integration for RAG operations"""

    def __init__(self, project_dir: str = None):
        """
        Initialize observability integration

        Checks multiple locations for observability scripts:
        1. Project's .claude/scripts/ directory (if installed locally)
        2. 0k-rag plugin's bundled observability-scripts/ (fallback)

        Args:
            project_dir: Project root directory (where .claude/ exists)
        """
        self.project_dir = project_dir or os.getcwd()

        # Location 1: Project's .claude/scripts/ directory
        project_token = Path(self.project_dir) / ".claude/scripts/log-token-usage.sh"
        project_latency = Path(self.project_dir) / ".claude/scripts/log-latency-trace.sh"
        project_error = Path(self.project_dir) / ".claude/scripts/0k-rag-log-error.sh"

        # Location 2: 0k-rag plugin's bundled scripts
        # Get plugin root directory (go up from rag/utils/observability.py)
        plugin_root = Path(__file__).parent.parent.parent
        plugin_token = plugin_root / "observability-scripts/log-token-usage.sh"
        plugin_latency = plugin_root / "observability-scripts/log-latency-trace.sh"
        plugin_error = plugin_root / "observability-scripts/0k-rag-log-error.sh"

        # Check project scripts first, then fall back to plugin scripts
        if all([project_token.exists(), project_latency.exists(), project_error.exists()]):
            # Use project scripts (installed locally)
            self.token_logger = project_token
            self.latency_logger = project_latency
            self.error_logger = project_error
            self.enabled = True
            self.location = "project"
        elif all([plugin_token.exists(), plugin_latency.exists(), plugin_error.exists()]):
            # Use plugin bundled scripts (fallback)
            self.token_logger = plugin_token
            self.latency_logger = plugin_latency
            self.error_logger = plugin_error
            self.enabled = True
            self.location = "plugin"
        else:
            # No observability scripts available
            self.token_logger = None
            self.latency_logger = None
            self.error_logger = None
            self.enabled = False
            self.location = "none"

    def log_search_operation(
        self,
        query: str,
        num_chunks: int,
        latency_ms: int,
        tokens_retrieved: int,
        conversation_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ):
        """Log RAG search operation"""
        if not self.enabled:
            return

        conv_id = conversation_id or os.getenv("CONVERSATION_ID", "unknown")

        # Log token usage
        try:
            subprocess.run([
                str(self.token_logger),
                "--conversation-id", conv_id,
                "--operation-type", "rag_search",
                "--operation-name", f"search:{query[:30]}...",
                "--input-tokens", "0",  # Query tokens (negligible)
                "--output-tokens", str(tokens_retrieved),
                "--model", "nomic-embed-text",
                "--estimated"
            ], check=False, capture_output=True)
        except Exception:
            pass  # Graceful degradation

        # Log latency trace
        if trace_id:
            try:
                end_ns = time.time_ns()
                start_ns = end_ns - (latency_ms * 1_000_000)
                subprocess.run([
                    str(self.latency_logger),
                    "--conversation-id", conv_id,
                    "--trace-id", trace_id,
                    "--operation-type", "rag_search",
                    "--operation-name", f"search:{num_chunks}_chunks",
                    "--start-time", str(start_ns),
                    "--end-time", str(end_ns),
                    "--metadata", f'{{"num_chunks": {num_chunks}, "tokens": {tokens_retrieved}}}'
                ], check=False, capture_output=True)
            except Exception:
                pass

    def log_index_operation(
        self,
        file_path: str,
        num_chunks: int,
        latency_ms: int,
        conversation_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ):
        """Log RAG index operation"""
        if not self.enabled:
            return

        conv_id = conversation_id or os.getenv("CONVERSATION_ID", "unknown")

        # Log latency trace
        if trace_id:
            try:
                end_ns = time.time_ns()
                start_ns = end_ns - (latency_ms * 1_000_000)
                subprocess.run([
                    str(self.latency_logger),
                    "--conversation-id", conv_id,
                    "--trace-id", trace_id,
                    "--operation-type", "rag_index",
                    "--operation-name", Path(file_path).name,
                    "--start-time", str(start_ns),
                    "--end-time", str(end_ns),
                    "--metadata", f'{{"num_chunks": {num_chunks}, "file": "{file_path}"}}'
                ], check=False, capture_output=True)
            except Exception:
                pass
