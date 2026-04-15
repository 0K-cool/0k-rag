"""
0K-RAG Index - Index documents into the RAG knowledge base from command line

Usage:
    0k-index document.md
    0k-index docs/playbooks/*.pdf --project PAI
    0k-index --pattern 'docs/**/*.md'
"""

import sys
import argparse
from pathlib import Path

from rag.notifications import ConsoleNotifier, create_notifier_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Index documents into the 0K-RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  0k-index document.md
  0k-index docs/playbooks/nist-ir.pdf --project MyProject
  0k-index --pattern 'docs/**/*.pdf'
  0k-index file.md --no-sanitize
  0k-index --batch 'docs/rag/*.md'
        """
    )

    parser.add_argument(
        "file_path",
        nargs="?",
        help="Path to file to index"
    )
    parser.add_argument(
        "--pattern",
        help="Glob pattern for batch indexing (e.g., 'docs/**/*.md')"
    )
    parser.add_argument(
        "--batch",
        help="Batch index multiple files matching pattern"
    )
    parser.add_argument(
        "--project",
        help="Project name (default: from config)"
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Skip PII sanitization (not recommended)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without actually doing it"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if file was already indexed"
    )
    parser.add_argument(
        "--config",
        default=".0k-rag.yml",
        help="Path to configuration file (default: .0k-rag.yml)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress output"
    )

    args = parser.parse_args()

    if not args.file_path and not args.pattern and not args.batch:
        parser.print_help()
        return 1

    try:
        # Load configuration
        import yaml
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
            print(f"   Create .0k-rag.yml in your project root", file=sys.stderr)
            print(f"   See examples in 0k-rag/examples/", file=sys.stderr)
            return 1

        with open(config_path) as f:
            config = yaml.safe_load(f)

        db_path = config['database']['path']
        project_name = args.project or config['project']['name']
        enable_sanitization = not args.no_sanitize and config['indexing'].get('enable_sanitization', True)

        # Import RAG modules
        from rag.indexing.indexer import KnowledgeBaseIndexer
        from rag.indexing.document_loader import DocumentLoader
        from rag.indexing.sanitizer import Sanitizer

        # Initialize indexer
        print(f"Initializing 0K-RAG indexer for {project_name}...", file=sys.stderr)
        indexer = KnowledgeBaseIndexer(db_path=db_path)
        indexer.initialize()

        # Initialize document loader and sanitizer
        loader = DocumentLoader()
        sanitizer = Sanitizer() if enable_sanitization else None

        # Determine files to index
        files_to_index = []

        if args.file_path:
            file_path = Path(args.file_path)
            if not file_path.exists():
                print(f"Error: File not found: {args.file_path}", file=sys.stderr)
                return 1
            files_to_index.append(file_path)

        elif args.pattern or args.batch:
            import glob
            pattern = args.pattern or args.batch
            matched_files = glob.glob(pattern, recursive=True)
            if not matched_files:
                print(f"Error: No files matched pattern: {pattern}", file=sys.stderr)
                return 1
            files_to_index = [Path(f) for f in matched_files if Path(f).is_file()]
            print(f"Found {len(files_to_index)} file(s) matching pattern", file=sys.stderr)

        if args.dry_run:
            print("\nDRY RUN - Would index the following files:", file=sys.stderr)
            for file_path in files_to_index:
                print(f"   - {file_path}", file=sys.stderr)
            return 0

        # Create notifier from config (falls back to console if no config)
        notifier = create_notifier_from_config(config)

        # Index files
        total_chunks = 0
        for i, file_path in enumerate(files_to_index, 1):
            print(f"\n[{i}/{len(files_to_index)}] Indexing: {file_path}", file=sys.stderr)

            try:
                # Load document
                document = loader.load_file(str(file_path), project_name)
                if not document:
                    print(f"   Failed to load file", file=sys.stderr)
                    continue

                # Apply sanitization if enabled
                if sanitizer:
                    sanitization_result = sanitizer.sanitize(document.content)
                    document.content = sanitization_result.sanitized_text

                # Index document with progress notifications
                chunk_count = indexer.index_document(document, notifier=notifier)
                total_chunks += chunk_count

            except Exception as e:
                print(f"   Error indexing {file_path}: {e}", file=sys.stderr)
                continue

        print(f"\nIndexing complete: {total_chunks} total chunks from {len(files_to_index)} file(s)", file=sys.stderr)
        return 0

    except ImportError as e:
        print(f"Error: Could not import RAG indexing module", file=sys.stderr)
        print(f"   Make sure 0k-rag is properly installed: pip install -e .", file=sys.stderr)
        print(f"   Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during indexing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
