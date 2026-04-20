"""
Unit tests for the hash-first dedup branches inside
KnowledgeBaseIndexer.index_document().

The hash-first block (index_document() Step 1) returns BEFORE any
chunker / embedder / context-generator is instantiated in two cases:
  1. Content hash matches at the SAME file_path → unchanged, skip.
  2. Content hash matches at a DIFFERENT file_path → move/rename,
     retarget pointer on existing chunks.

Because these paths never touch the heavy pipeline dependencies
(sentence-transformers, Ollama), we can exercise them directly by
writing synthetic rows to the LanceDB table — identical strategy to
the vacuum_orphans tests.

Regression targets for PR #2 review findings:
  - SQL sanitize consistency: paths with apostrophes must route through
    the correct pipe (raw in-memory, SQL-escaped only in WHERE).
  - limit cap on hash lookup: documented in code; not directly
    exercised here because the cap is 10k (synthetic-row cost prohibitive).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib


@dataclass
class _StubDocument:
    """Minimal shape of rag.indexing.document_loader.Document for tests."""
    content: str
    file_path: str
    project: str
    metadata: dict


def _synthetic_chunk_row(
    file_path: str,
    content_hash: str,
    chunk_idx: int = 0,
    content: str = "stable content",
) -> dict:
    now = datetime.now().isoformat()
    return {
        "chunk_id": str(uuid.uuid4()),
        "chunk_index": chunk_idx,
        "original_chunk": content,
        "contextual_chunk": f"{content} (contextual)",
        "generated_context": "",
        "vector": [0.0] * 768,
        "source_file": Path(file_path).name,
        "source_project": "test-project",
        "file_path": file_path,
        "file_type": Path(file_path).suffix or ".md",
        "content_hash": content_hash,
        "indexed_at": now,
        "last_updated": now,
        "token_count": len(content.split()),
        "trust_level": "TRUSTED",
        "trust_score": 1.0,
        "security_risk": "CLEAN",
        "pattern_count": 0,
    }


class HashFirstDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        scratch_root = repo_root / "tests" / ".scratch"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.mkdtemp(prefix="hash-first-", dir=str(scratch_root))
        self.kb_path = os.path.join(self.tmp, "kb")

        from rag.indexing.indexer import KnowledgeBaseIndexer  # noqa: E402

        self.indexer = KnowledgeBaseIndexer(db_path=self.kb_path)
        self.indexer.initialize()

        # Pre-create the table with the indexer's schema so we can append
        # synthetic rows without running the heavy pipeline.
        if self.indexer.table is None:
            self.indexer.table = self.indexer.db.create_table(
                self.indexer.table_name,
                schema=self.indexer._create_schema(),
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------- helpers ----------

    def _write_real_file(self, rel_path: str, content: str) -> str:
        full = os.path.join(self.tmp, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        Path(full).write_text(content, encoding="utf-8")
        return full

    def _add_rows(self, rows: list) -> None:
        import pyarrow as pa
        schema = self.indexer.table.schema
        tbl = pa.Table.from_pylist(rows, schema=schema)
        self.indexer.table.add(tbl)

    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ---------- tests ----------

    def test_same_path_same_hash_is_noop(self) -> None:
        """Case 1a — re-indexing an unchanged document skips everything."""
        content = "identical content, unchanged"
        chash = self._hash(content)
        alive = self._write_real_file("same.md", content)
        self._add_rows([
            _synthetic_chunk_row(alive, chash, 0, content),
            _synthetic_chunk_row(alive, chash, 1, content),
        ])

        doc = _StubDocument(
            content=content, file_path=alive, project="test-project", metadata={},
        )
        returned = self.indexer.index_document(doc, enable_security_scan=False)

        self.assertEqual(returned, 2, "should report 2 existing chunks unchanged")
        # Row count at alive path is still 2 — no re-embedding happened.
        count = self.indexer.table.count_rows(f"file_path = '{alive}'")
        self.assertEqual(count, 2)

    def test_move_retargets_pointer_without_reembedding(self) -> None:
        """Case 1b — same content at a new path updates pointer in place."""
        content = "content that was moved"
        chash = self._hash(content)
        old_path = os.path.join(self.tmp, "subdir", "original.md")
        new_path = self._write_real_file("renamed/new-home.md", content)

        self._add_rows([
            _synthetic_chunk_row(old_path, chash, 0, content),
            _synthetic_chunk_row(old_path, chash, 1, content),
            _synthetic_chunk_row(old_path, chash, 2, content),
        ])

        doc = _StubDocument(
            content=content, file_path=new_path,
            project="test-project", metadata={},
        )
        returned = self.indexer.index_document(doc, enable_security_scan=False)

        self.assertEqual(returned, 3, "all 3 chunks retargeted")
        # Old path now has 0 chunks, new path has 3.
        self.assertEqual(
            self.indexer.table.count_rows(f"file_path = '{old_path}'"),
            0,
            "old path chunks must be retargeted away",
        )
        self.assertEqual(
            self.indexer.table.count_rows(f"file_path = '{new_path}'"),
            3,
            "new path now owns the chunks",
        )
        # Content hash preserved — same doc, just at a new path.
        all_rows = self.indexer.table.search().to_list()
        for row in all_rows:
            self.assertEqual(row["content_hash"], chash)

    def test_move_preserves_raw_path_with_apostrophe(self) -> None:
        """Path with a single quote survives the sanitize/membership pipeline.

        Regression for PR #2 review finding:
          - `safe_path in existing_paths` previously compared SQL-escaped
            to raw — any apostrophe would cause the membership test to
            miss and wrongly fall through to the path-based branch.
          - `values={"file_path": safe_path}` previously stored the
            double-escaped form on disk.
        """
        content = "kelvin's research notes"
        chash = self._hash(content)
        # Note the apostrophe in the FILENAME.
        old_path = os.path.join(self.tmp, "authors", "kelvin's old note.md")
        new_path_dir = os.path.join(self.tmp, "kelvin's renamed")
        os.makedirs(new_path_dir, exist_ok=True)
        new_path = os.path.join(new_path_dir, "note.md")
        Path(new_path).write_text(content, encoding="utf-8")

        self._add_rows([
            _synthetic_chunk_row(old_path, chash, 0, content),
            _synthetic_chunk_row(old_path, chash, 1, content),
        ])

        doc = _StubDocument(
            content=content, file_path=new_path,
            project="test-project", metadata={},
        )
        returned = self.indexer.index_document(doc, enable_security_scan=False)

        self.assertEqual(returned, 2, "apostrophe in path must not break move detection")
        # Confirm the raw path (not SQL-escaped) is what's stored on disk.
        new_rows = self.indexer.table.search().to_list()
        stored_paths = {r["file_path"] for r in new_rows}
        self.assertIn(new_path, stored_paths)
        # And specifically not a doubled-apostrophe form.
        doubled = new_path.replace("'", "''")
        if "'" in new_path:
            self.assertNotIn(
                doubled,
                stored_paths,
                "stored path must be the raw value, not SQL-escaped",
            )

    def test_same_content_same_apostrophe_path_is_noop(self) -> None:
        """Case 1a regression with an apostrophe — must still skip."""
        content = "unchanged with apostrophe"
        chash = self._hash(content)
        # File must be at an allowed_base_path (inside self.tmp).
        alive_dir = os.path.join(self.tmp, "kelvin's dir")
        os.makedirs(alive_dir, exist_ok=True)
        alive = os.path.join(alive_dir, "file.md")
        Path(alive).write_text(content, encoding="utf-8")

        self._add_rows([_synthetic_chunk_row(alive, chash, 0, content)])

        doc = _StubDocument(
            content=content, file_path=alive,
            project="test-project", metadata={},
        )
        returned = self.indexer.index_document(doc, enable_security_scan=False)

        self.assertEqual(returned, 1, "apostrophe in path must not break same-path skip")
        # Path count unchanged.
        self.assertEqual(
            self.indexer.table.count_rows(f"file_path = '{alive.replace(chr(39), chr(39)*2)}'"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
