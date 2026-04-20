"""
0K-RAG Vacuum — delete chunks whose source file no longer exists on disk

Orphan chunks accumulate in the KB whenever a source file is renamed,
moved, or removed without going through `index_document` — pre-v0.x releases
of the indexer keyed deduplication on `file_path` alone, so moves produced
orphan chunks at the old path in addition to fresh chunks at the new path.

`0k-vacuum --dry-run` inspects every distinct `file_path` in the KB, stats
each one, and reports which paths are gone. Re-run without `--dry-run` to
actually delete those chunks.

Usage:
    0k-vacuum                                 # dry-run preview (default)
    0k-vacuum --delete                        # delete ALL detected orphans
    0k-vacuum --delete --match PATTERN        # delete only orphans whose
                                              # file_path contains PATTERN
                                              # (safer — recommended default)
    0k-vacuum --json                          # machine-readable report

Safety: `vacuum_orphans` detects that a file_path is no longer on disk,
but that is NOT implicit permission to delete. Review the dry-run report
first and prefer `--match` to target a specific subset. See
feedback_never_delete_from_rag_without_approval.md for the full policy.

Exit codes:
    0  success (even when orphans exist in dry-run mode)
    1  indexer initialization failed
    2  vacuum encountered an error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rag.indexing.indexer import KnowledgeBaseIndexer

logger = logging.getLogger("0k-vacuum")


def _format_human(report: dict) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("🧹 0K-RAG Vacuum Report")
    lines.append("=" * 60)
    lines.append(f"  Paths scanned   : {report['scanned_paths']}")
    lines.append(f"  Orphan paths    : {len(report['orphan_paths'])}")
    lines.append(f"  Orphan chunks   : {report['orphan_chunk_count']}")
    lines.append(f"  Match filter    : {report.get('match_filter') or '(none)'}")
    lines.append(f"  Paths deleted   : {len(report.get('deleted_paths', []))}")
    lines.append(f"  Chunks deleted  : {report['deleted_chunk_count']}")
    lines.append("")
    if report["orphan_paths"]:
        lines.append("Orphan file paths (source no longer on disk):")
        for p in report["orphan_paths"]:
            marker = "✓" if p in report.get("deleted_paths", []) else " "
            lines.append(f"  {marker} {p}")
        lines.append("")
        if report["deleted_chunk_count"] == 0:
            if report.get("match_filter"):
                lines.append(
                    f"Dry-run / no match — re-run with "
                    f"`--delete --match {report['match_filter']!r}` to prune."
                )
            else:
                lines.append(
                    "Dry-run only — re-run with `--delete` to prune all, or "
                    "`--delete --match PATTERN` to target a subset (safer)."
                )
        else:
            lines.append(
                f"✓ Deleted {report['deleted_chunk_count']} chunks across "
                f"{len(report['deleted_paths'])} paths."
            )
    else:
        lines.append("✓ No orphans — knowledge base is clean.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune orphan chunks from the 0K-RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  0k-vacuum                 # dry-run preview (default)\n"
            "  0k-vacuum --delete        # actually delete orphans\n"
            "  0k-vacuum --delete --json # machine-readable output\n"
        ),
    )
    # --delete and --dry-run are mutually exclusive by construction. argparse
    # emits a clean `--help` grouping + error message when both are passed,
    # instead of requiring a manual post-parse check.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--delete",
        action="store_true",
        help="Delete orphan chunks (default: dry-run preview only)",
    )
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Explicit dry-run flag (this is the default when --delete is "
            "absent; included for script readability)"
        ),
    )
    parser.add_argument(
        "--match",
        default=None,
        metavar="PATTERN",
        help=(
            "Substring filter applied to orphan file_paths before deletion. "
            "Only paths containing PATTERN are deleted. Recommended whenever "
            "using --delete — makes it impossible to accidentally wipe a "
            "category you did not review. Example: "
            "--match proximity_scan deletes only the MCP scan artifacts."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of a human-readable table",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help=(
            "Override the KB path (default: indexer picks it up from the "
            "0k-rag config file)"
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Instantiate the indexer. It reads the KB path from the config file
    # unless explicitly overridden.
    try:
        if args.db_path:
            indexer = KnowledgeBaseIndexer(db_path=args.db_path)
        else:
            indexer = KnowledgeBaseIndexer()
        indexer.initialize()
    except Exception as exc:
        print(f"ERROR: failed to initialize indexer: {exc}", file=sys.stderr)
        return 1

    # Run the sweep.
    try:
        report = indexer.vacuum_orphans(
            dry_run=not args.delete,
            match=args.match,
        )
    except Exception as exc:
        print(f"ERROR: vacuum failed: {exc}", file=sys.stderr)
        return 2

    # Emit before deciding the exit code — operators must always see the
    # report (including partial results), not just an error message.
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_human(report))

    # vacuum_orphans records a non-None "error" on partial/failed sweeps
    # (timeout, row-cap exhaustion, unexpected exception). Surface that as
    # a non-zero exit code so scripts / automation can distinguish a clean
    # sweep from one that stopped mid-flight with partial deletions.
    if report.get("error"):
        print(
            f"WARNING: vacuum reported an error: {report['error']}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
