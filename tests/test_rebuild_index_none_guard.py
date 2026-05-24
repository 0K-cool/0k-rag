"""
Regression tests for rebuild_index None-handling.

Background: when DocumentLoader.load_file returns None (file not found,
unsupported extension, parse error, empty content), the previous
rebuild_index loop called indexer.index_document(None) unconditionally,
which raised the misleading "'NoneType' object has no attribute
'file_path'" error from inside the indexer.

The fix factors the per-file load+index into a helper that handles None
explicitly and surfaces an actionable error message (most common cause
being a missing optional dependency like PyMuPDF for PDFs).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_helper():
    """Lazy import — mcp_server.ok_rag_server reads config at import time
    and sys.exits(1) without RAG_CONFIG. Scope the env var inside the
    test so it does not leak into other test modules that rely on the
    default lookup path."""
    config_path = (
        Path(__file__).parent.parent / "examples" / "config.pai.yml"
    )
    with patch.dict(os.environ, {"RAG_CONFIG": str(config_path)}):
        from mcp_server.ok_rag_server import _index_one_file
    return _index_one_file


class TestIndexOneFileNoneGuard(unittest.TestCase):
    def test_returns_failure_record_when_loader_returns_none(self):
        _index_one_file = _load_helper()
        loader = MagicMock()
        loader.load_file.return_value = None
        indexer = MagicMock()
        file_path = Path("/fake/missing.pdf")

        chunks, failure = _index_one_file(loader, indexer, file_path, "PAI")

        self.assertEqual(chunks, 0)
        self.assertIsNotNone(failure)
        self.assertEqual(failure["file"], "missing.pdf")
        self.assertIn("load_file returned None", failure["error"])
        indexer.index_document.assert_not_called()

    def test_returns_chunk_count_on_success(self):
        _index_one_file = _load_helper()
        loader = MagicMock()
        doc = MagicMock()
        loader.load_file.return_value = doc
        indexer = MagicMock()
        indexer.index_document.return_value = 7
        file_path = Path("/fake/good.md")

        chunks, failure = _index_one_file(loader, indexer, file_path, "PAI")

        self.assertEqual(chunks, 7)
        self.assertIsNone(failure)
        indexer.index_document.assert_called_once_with(doc)

    def test_returns_failure_record_when_loader_raises(self):
        _index_one_file = _load_helper()
        loader = MagicMock()
        loader.load_file.side_effect = RuntimeError("loader exploded")
        indexer = MagicMock()
        file_path = Path("/fake/explodes.pdf")

        chunks, failure = _index_one_file(loader, indexer, file_path, "PAI")

        self.assertEqual(chunks, 0)
        self.assertIsNotNone(failure)
        self.assertEqual(failure["file"], "explodes.pdf")
        self.assertIn("loader exploded", failure["error"])
        indexer.index_document.assert_not_called()

    def test_returns_failure_record_when_indexer_raises(self):
        _index_one_file = _load_helper()
        loader = MagicMock()
        loader.load_file.return_value = MagicMock()
        indexer = MagicMock()
        indexer.index_document.side_effect = RuntimeError("boom")
        file_path = Path("/fake/broken.md")

        chunks, failure = _index_one_file(loader, indexer, file_path, "PAI")

        self.assertEqual(chunks, 0)
        self.assertIsNotNone(failure)
        self.assertEqual(failure["file"], "broken.md")
        self.assertIn("boom", failure["error"])


if __name__ == "__main__":
    unittest.main()
