"""
Unit tests for KnowledgeBaseIndexer.vacuum_orphans().

Strategy: populate the LanceDB table directly with synthetic rows
(skipping the chunker/embedder/context-generator pipeline, which have
external dependencies on sentence-transformers + Ollama). This lets us
test vacuum behavior deterministically without a heavy pipeline stub.

The hash-first dedup change inside index_document() is exercised by
live usage and an integration smoke test elsewhere — not covered here,
because index_document requires the real embedding/context pipeline.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path


def _synthetic_chunk_row(file_path: str, content_hash: str, chunk_idx: int = 0) -> dict:
    """Produce a row matching the indexer schema (see indexer._create_schema)."""
    now = datetime.now().isoformat()
    return {
        "chunk_id": str(uuid.uuid4()),
        "chunk_index": chunk_idx,
        "original_chunk": f"test content {chunk_idx}",
        "contextual_chunk": f"test content {chunk_idx} (contextual)",
        "generated_context": "",
        "vector": [0.0] * 768,
        "source_file": Path(file_path).name,
        "source_project": "test-project",
        "file_path": file_path,
        "file_type": Path(file_path).suffix or ".md",
        "content_hash": content_hash,
        "indexed_at": now,
        "last_updated": now,
        "token_count": 3,
        "trust_level": "TRUSTED",
        "trust_score": 1.0,
        "security_risk": "CLEAN",
        "pattern_count": 0,
    }


class VacuumOrphansTests(unittest.TestCase):
    """Directly populate the LanceDB table, then test vacuum behavior."""

    def setUp(self) -> None:
        # Temp dir inside the repo tree so the indexer's path-traversal
        # guard (defaults allowed_base_paths to cwd) accepts paths we
        # create here.
        repo_root = Path(__file__).resolve().parent.parent
        scratch_root = repo_root / "tests" / ".scratch"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.mkdtemp(prefix="vacuum-test-", dir=str(scratch_root))
        self.kb_path = os.path.join(self.tmp, "kb")

        from rag.indexing.indexer import KnowledgeBaseIndexer  # noqa: E402

        self.indexer = KnowledgeBaseIndexer(db_path=self.kb_path)
        self.indexer.initialize()

        # initialize() does NOT create the table — that happens lazily on
        # first index_chunks call (see indexer.py line 368-374). For these
        # tests we bypass the chunker/embedder pipeline entirely and write
        # synthetic rows directly, so we create an empty table up front
        # using the same schema the indexer would use.
        if self.indexer.table is None:
            self.indexer.table = self.indexer.db.create_table(
                self.indexer.table_name,
                schema=self.indexer._create_schema(),
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------- helpers ----------

    def _write_real_file(self, rel_path: str) -> str:
        """Create an actual file on disk at a path under self.tmp."""
        full = os.path.join(self.tmp, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        Path(full).write_text("real content", encoding="utf-8")
        return full

    def _add_rows(self, rows: list) -> None:
        """Append synthetic rows to the KB table."""
        import pyarrow as pa

        schema = self.indexer.table.schema
        tbl = pa.Table.from_pylist(rows, schema=schema)
        self.indexer.table.add(tbl)

    # ---------- tests ----------

    def test_clean_kb_reports_no_orphans(self) -> None:
        """When every indexed file exists on disk, nothing is orphan."""
        alive = self._write_real_file("alive.md")
        self._add_rows([_synthetic_chunk_row(alive, "hash-alive", 0)])

        report = self.indexer.vacuum_orphans(dry_run=True)

        self.assertEqual(report["scanned_paths"], 1)
        self.assertEqual(report["orphan_paths"], [])
        self.assertEqual(report["orphan_chunk_count"], 0)
        self.assertEqual(report["deleted_chunk_count"], 0)

    def test_missing_file_reported_as_orphan_dry_run(self) -> None:
        """A file_path whose file has been deleted shows up in orphans."""
        alive = self._write_real_file("alive.md")
        orphan_path = os.path.join(self.tmp, "gone.md")
        self._add_rows(
            [
                _synthetic_chunk_row(alive, "hash-alive", 0),
                _synthetic_chunk_row(orphan_path, "hash-gone", 0),
                _synthetic_chunk_row(orphan_path, "hash-gone", 1),
            ]
        )

        report = self.indexer.vacuum_orphans(dry_run=True)

        self.assertEqual(report["orphan_paths"], [orphan_path])
        self.assertEqual(report["orphan_chunk_count"], 2)
        self.assertEqual(report["deleted_chunk_count"], 0, "dry-run never deletes")

        remaining = self.indexer.table.count_rows(f"file_path = '{orphan_path}'")
        self.assertEqual(remaining, 2)

    def test_delete_without_match_removes_every_orphan(self) -> None:
        """dry_run=False with no match prunes all orphan chunks."""
        alive = self._write_real_file("alive.md")
        orphan_a = os.path.join(self.tmp, "orphan-a.md")
        orphan_b = os.path.join(self.tmp, "dir", "orphan-b.md")
        self._add_rows(
            [
                _synthetic_chunk_row(alive, "hash-alive", 0),
                _synthetic_chunk_row(orphan_a, "hash-a", 0),
                _synthetic_chunk_row(orphan_b, "hash-b", 0),
            ]
        )

        report = self.indexer.vacuum_orphans(dry_run=False)

        self.assertEqual(set(report["deleted_paths"]), {orphan_a, orphan_b})
        self.assertEqual(report["deleted_chunk_count"], 2)
        self.assertEqual(self.indexer.table.count_rows(f"file_path = '{orphan_a}'"), 0)
        self.assertEqual(self.indexer.table.count_rows(f"file_path = '{orphan_b}'"), 0)
        self.assertEqual(self.indexer.table.count_rows(f"file_path = '{alive}'"), 1)

    def test_match_filter_only_deletes_matching_paths(self) -> None:
        """With --match substring, non-matching orphans must be preserved."""
        alive = self._write_real_file("alive.md")
        keep_orphan = os.path.join(self.tmp, "research", "irreplaceable-note.md")
        prune_orphan = os.path.join(self.tmp, "scan", "proximity_scan_abc.md")
        self._add_rows(
            [
                _synthetic_chunk_row(alive, "hash-alive", 0),
                _synthetic_chunk_row(keep_orphan, "hash-keep", 0),
                _synthetic_chunk_row(prune_orphan, "hash-prune", 0),
            ]
        )

        report = self.indexer.vacuum_orphans(dry_run=False, match="proximity_scan")

        self.assertEqual(set(report["orphan_paths"]), {keep_orphan, prune_orphan})
        self.assertEqual(report["deleted_paths"], [prune_orphan])
        self.assertEqual(report["match_filter"], "proximity_scan")
        self.assertEqual(
            self.indexer.table.count_rows(f"file_path = '{prune_orphan}'"), 0
        )
        self.assertEqual(
            self.indexer.table.count_rows(f"file_path = '{keep_orphan}'"),
            1,
            "non-matching orphan MUST survive",
        )

    def test_match_filter_matches_nothing_is_a_noop(self) -> None:
        """A --match that excludes every orphan deletes zero chunks."""
        alive = self._write_real_file("alive.md")
        orphan = os.path.join(self.tmp, "orphan.md")
        self._add_rows(
            [
                _synthetic_chunk_row(alive, "hash-alive", 0),
                _synthetic_chunk_row(orphan, "hash-orphan", 0),
            ]
        )

        report = self.indexer.vacuum_orphans(dry_run=False, match="no-such-substring")

        self.assertEqual(report["orphan_paths"], [orphan])
        self.assertEqual(report["deleted_paths"], [])
        self.assertEqual(report["deleted_chunk_count"], 0)
        self.assertEqual(self.indexer.table.count_rows(f"file_path = '{orphan}'"), 1)

    def test_vacuum_on_empty_table_is_safe(self) -> None:
        """No rows in the table means no orphans, no errors."""
        report = self.indexer.vacuum_orphans(dry_run=False)

        self.assertEqual(report["scanned_paths"], 0)
        self.assertEqual(report["orphan_paths"], [])
        self.assertEqual(report["deleted_chunk_count"], 0)
        self.assertIsNone(
            report["error"], "clean sweep must not set an error"
        )

    def test_clean_sweep_reports_no_error(self) -> None:
        """Successful vacuum (orphan or not) leaves error=None."""
        alive = self._write_real_file("alive.md")
        orphan = os.path.join(self.tmp, "gone.md")
        self._add_rows([
            _synthetic_chunk_row(alive, "hash-alive", 0),
            _synthetic_chunk_row(orphan, "hash-orphan", 0),
        ])
        report = self.indexer.vacuum_orphans(dry_run=False)
        self.assertIsNone(report["error"])
        self.assertEqual(report["deleted_chunk_count"], 1)

    def test_error_key_set_when_table_uninitialized(self) -> None:
        """A raw indexer with no table set surfaces table_not_initialized."""
        from rag.indexing.indexer import KnowledgeBaseIndexer

        raw_indexer = KnowledgeBaseIndexer(
            db_path=os.path.join(self.tmp, "kb_not_init")
        )
        # Deliberately do NOT call initialize() — table stays None.
        report = raw_indexer.vacuum_orphans(dry_run=True)
        self.assertEqual(report["error"], "table_not_initialized")
        self.assertEqual(report["scanned_paths"], 0)

    def test_error_key_set_when_scan_row_limit_reached(self) -> None:
        """Hitting the hard scan cap aborts with scan_row_limit_reached.

        Uses the class-level tunable so we don't need to synthesize 1M rows.
        """
        alive = self._write_real_file("alive.md")
        self._add_rows([_synthetic_chunk_row(alive, "h1", 0)])

        # Temporarily shrink the cap so a 1-row table exhausts it. This
        # is the exact observable contract operators would see at the
        # real 1M-row cap, without paying the synthesis cost.
        original_cap = type(self.indexer).VACUUM_SCAN_ROW_LIMIT
        original_warn = type(self.indexer).VACUUM_SCAN_WARN_THRESHOLD
        try:
            type(self.indexer).VACUUM_SCAN_ROW_LIMIT = 1
            type(self.indexer).VACUUM_SCAN_WARN_THRESHOLD = 1
            report = self.indexer.vacuum_orphans(dry_run=True)
        finally:
            type(self.indexer).VACUUM_SCAN_ROW_LIMIT = original_cap
            type(self.indexer).VACUUM_SCAN_WARN_THRESHOLD = original_warn

        self.assertEqual(report["error"], "scan_row_limit_reached")
        # Abort before scanning unique paths — scanned_paths stays 0.
        self.assertEqual(report["scanned_paths"], 0)
        self.assertEqual(report["deleted_chunk_count"], 0)


if __name__ == "__main__":
    unittest.main()
