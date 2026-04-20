"""
Unit tests for the `0k-vacuum` CLI wrapper (rag.cli.vacuum).

Scope:
  - Exit code 0 on clean sweeps.
  - Exit code 2 when the indexer reports an error in `report["error"]`.
  - --delete / --dry-run mutual exclusion via argparse.

Uses monkeypatch to stub KnowledgeBaseIndexer so the CLI tests don't
need a live LanceDB or the embedding pipeline.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch


class _FakeIndexer:
    """Stand-in for KnowledgeBaseIndexer with a scripted vacuum report."""

    def __init__(self, *args, **kwargs):
        self._report_to_return: dict = {}

    def initialize(self) -> None:
        pass

    def vacuum_orphans(self, dry_run: bool = True, match=None) -> dict:
        # Return whatever was preloaded on the class; match the full
        # contract (every key present) so _format_human doesn't KeyError.
        base = {
            "orphan_paths": [],
            "orphan_chunk_count": 0,
            "deleted_paths": [],
            "deleted_chunk_count": 0,
            "scanned_paths": 0,
            "match_filter": match,
            "error": None,
        }
        base.update(self._report_to_return)
        return base


class VacuumCLIExitCodeTests(unittest.TestCase):
    def setUp(self) -> None:
        # Preserve argv; tests mutate it.
        self._orig_argv = sys.argv[:]

    def tearDown(self) -> None:
        sys.argv = self._orig_argv

    def _run(self, report: dict, argv: list) -> int:
        """Invoke main() with a stubbed indexer returning `report`."""
        sys.argv = ["0k-vacuum"] + argv
        fake = _FakeIndexer()
        fake._report_to_return = report
        with patch(
            "rag.cli.vacuum.KnowledgeBaseIndexer",
            MagicMock(return_value=fake),
        ):
            from rag.cli.vacuum import main
            return main()

    def test_clean_sweep_returns_zero(self) -> None:
        code = self._run({"error": None}, [])
        self.assertEqual(code, 0)

    def test_scan_row_limit_error_returns_two(self) -> None:
        code = self._run(
            {"error": "scan_row_limit_reached"}, ["--delete"],
        )
        self.assertEqual(code, 2)

    def test_timeout_error_returns_two(self) -> None:
        code = self._run(
            {"error": "write_lock_timeout: waited 30s"}, ["--delete"],
        )
        self.assertEqual(code, 2)

    def test_uninitialized_table_error_returns_two(self) -> None:
        code = self._run(
            {"error": "table_not_initialized"}, [],
        )
        self.assertEqual(code, 2)


class VacuumCLIMutualExclusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_argv = sys.argv[:]

    def tearDown(self) -> None:
        sys.argv = self._orig_argv

    def test_delete_and_dry_run_together_fails_via_argparse(self) -> None:
        """argparse's mutually-exclusive group raises SystemExit(2)."""
        sys.argv = ["0k-vacuum", "--delete", "--dry-run"]
        from rag.cli.vacuum import main

        with self.assertRaises(SystemExit) as ctx:
            main()
        # argparse exits with code 2 on argument errors.
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
