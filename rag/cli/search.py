"""
0K-RAG Search - Search the RAG knowledge base from command line

Usage:
    0k-search "your query here"
    0k-search "threat intelligence" --top-k 10
    0k-search "git workflow" --hybrid
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Search the 0K-RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  0k-search "threat intelligence"
  0k-search "git safety check" --top-k 10
  0k-search "backup procedure" --hybrid
  0k-search "memory server" --rerank
        """
    )

    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid search (vector + BM25)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use BGE reranking (slower, more accurate)"
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Don't show context, just return raw results"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--config",
        default=".0k-rag.yml",
        help="Path to configuration file (default: .0k-rag.yml)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        import yaml
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Error: Configuration file not found: {config_path}", file=sys.stderr)
            print(f"   Create .0k-rag.yml in your project root", file=sys.stderr)
            print(f"   See examples in 0k-rag/examples/", file=sys.stderr)
            return 1

        with open(config_path) as f:
            config = yaml.safe_load(f)

        db_path = config['database']['path']
        enable_reranking = args.rerank or config['retrieval'].get('enable_reranking', True)

        # Initialize retrieval pipeline
        from rag.retrieval.pipeline import RetrievalPipeline

        print(f"🔍 Searching {config['project']['name']} knowledge base...", file=sys.stderr)
        pipeline = RetrievalPipeline(
            db_path=db_path,
            enable_reranking=enable_reranking
        )

        # Perform search
        results = pipeline.retrieve(
            query=args.query,
            top_k=args.top_k,
            enable_bm25=(args.hybrid or args.rerank),
            verbose=False
        )

        # Output results
        if args.json:
            import json
            output = []
            for i, result in enumerate(results, 1):
                output.append({
                    "rank": i,
                    "chunk_id": result.get("chunk_id", ""),
                    "file_path": result.get("file_path", ""),
                    "score": float(result.get("score", 0.0)),
                    "content": result.get("original_chunk", ""),
                    "context": result.get("generated_context", ""),
                    "project": result.get("source_project", "")
                })
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            print(f"\n✅ Found {len(results)} results for: '{args.query}'\n", file=sys.stderr)

            for i, result in enumerate(results, 1):
                print(f"{'='*80}")
                print(f"Result {i}/{len(results)} - Score: {result.get('score', 0.0):.4f}")
                print(f"File: {result.get('file_path', 'unknown')}")
                print(f"Project: {result.get('source_project', 'unknown')}")
                print(f"Chunk ID: {result.get('chunk_id', 'unknown')}")
                print(f"{'-'*80}")

                # Show generated context if available and not --no-context
                if not args.no_context and result.get("generated_context"):
                    print(f"Context: {result.get('generated_context')}\n")

                # Show original chunk content
                print(result.get("original_chunk", ""))
                print()

        return 0

    except ImportError as e:
        print(f"❌ Error: Could not import RAG retrieval module", file=sys.stderr)
        print(f"   Make sure 0k-rag is properly installed: pip install -e .", file=sys.stderr)
        print(f"   Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error during search: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
