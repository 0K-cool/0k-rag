"""
Retrieval Pipeline - Complete hybrid search + reranking workflow

Implements Anthropic's recommended retrieval architecture:
1. Vector search (semantic similarity)
2. BM25 search (keyword matching)
3. Reciprocal Rank Fusion (combine rankings)
4. Local reranking (BGE-reranker-large for final ranking)

100% local, zero cloud APIs, zero data exfiltration.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
import time
import uuid

logger = logging.getLogger(__name__)

# Phase 2.5: Observability Framework integration
try:
    from rag.utils.observability import RAGObservability
except ImportError:
    # Graceful degradation if observability module not available
    RAGObservability = None

from rag.indexing.indexer import KnowledgeBaseIndexer
from rag.indexing.embedder import Embedder
from rag.retrieval.vector_search import VectorSearch
from rag.retrieval.bm25_search import BM25Search
from rag.retrieval.fusion import reciprocal_rank_fusion, get_fusion_stats
from rag.retrieval.enhancers import apply_all_enhancers
from rag.retrieval.reranker import LocalReranker


class RetrievalPipeline:
    """Complete retrieval pipeline with hybrid search and reranking"""

    def __init__(
        self,
        db_path: str = "lance_vex_kb",  # NOTE: lance_vex_kb is the legacy default path — preserved for existing installations
        enable_reranking: bool = True,
        reranker_model: str = "BAAI/bge-reranker-large"
    ):
        """
        Initialize retrieval pipeline

        Args:
            db_path: Path to LanceDB database
            enable_reranking: Whether to use BGE reranker (default: True)
            reranker_model: Reranker model to use
        """
        self.db_path = db_path
        self.enable_reranking = enable_reranking

        # Initialize indexer and get table
        self.indexer = KnowledgeBaseIndexer(db_path=db_path)
        self.indexer.initialize()
        self.table = self.indexer.table

        # Initialize embedder
        self.embedder = Embedder(model="nomic-embed-text")

        # Initialize search components
        self.vector_search = VectorSearch(self.table, self.embedder)
        self.bm25_search = BM25Search(self.table)

        # Create FTS index for BM25 search (idempotent — skips if already exists)
        self.bm25_search.create_index()

        # Initialize reranker
        self.reranker = None
        if enable_reranking:
            self.reranker = LocalReranker(model_name=reranker_model)
            # Pre-load model to avoid cold start delay on first search
            self.reranker.load_model()
            logger.info("Reranker pre-loaded and ready")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_limit: int = 20,
        bm25_limit: int = 20,
        fusion_limit: int = 10,
        enable_bm25: bool = True,
        enable_enhancers: bool = True,
        filters: Optional[Dict] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using hybrid search + reranking

        Args:
            query: Search query
            top_k: Final number of results to return after reranking
            vector_limit: Number of results to fetch from vector search
            bm25_limit: Number of results to fetch from BM25 search
            fusion_limit: Number of results to keep after fusion
            enable_bm25: Whether to use BM25 search (default: True)
            filters: Optional filters (e.g., {"source_project": "PAI"})
            verbose: Print detailed search progress

        Returns:
            List of top-k most relevant chunks with scores and metadata
        """
        # Phase 2.5: Start observability tracking
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        obs = RAGObservability() if RAGObservability else None

        if verbose:
            logger.info(f"Retrieving: '{query}'")
            print()

        # Step 1: Vector search
        if verbose:
            logger.info(f"1⃣  Vector search (limit={vector_limit})...")

        vector_results = self.vector_search.search(
            query,
            limit=vector_limit,
            filters=filters
        )

        if verbose:
            logger.info(f"Found {len(vector_results)} vector results")

        # Step 2: BM25 search (optional)
        bm25_results = []
        if enable_bm25:
            if verbose:
                logger.info(f"2⃣  BM25 keyword search (limit={bm25_limit})...")

            bm25_results = self.bm25_search.search(
                query,
                limit=bm25_limit,
                filters=filters
            )

            if verbose:
                logger.info(f"Found {len(bm25_results)} BM25 results")
        else:
            if verbose:
                logger.info(f"2⃣  BM25 search: Disabled")

        # Step 3: Fusion (if we have both results)
        if enable_bm25 and bm25_results:
            if verbose:
                logger.info(f"3⃣  Reciprocal Rank Fusion...")

            fused_results = reciprocal_rank_fusion(
                vector_results,
                bm25_results,
                top_k=fusion_limit
            )

            if verbose:
                stats = get_fusion_stats(fused_results)
                logger.info(f"Fused to {len(fused_results)} results")
                print(f"   📊 Vector only: {stats['vector_only']}, "
                      f"BM25 only: {stats['bm25_only']}, "
                      f"Both: {stats['both_methods']}")
        else:
            # Use only vector results if BM25 is disabled
            fused_results = vector_results[:fusion_limit]
            if verbose:
                logger.info(f"3⃣  Fusion: Skipped (BM25 disabled)")

        # Step 4: Reranking
        if self.enable_reranking and self.reranker:
            if verbose:
                logger.info(f"4⃣  Local reranking (BGE-reranker-large)...")

            final_results = self.reranker.rerank(
                query,
                fused_results,
                top_k=top_k
            )

            if verbose:
                logger.info(f"Reranked to top {len(final_results)} results")
        else:
            # Return top-k from fusion
            final_results = fused_results[:top_k]
            if verbose:
                logger.info(f"4⃣  Reranking: Disabled")

        # Step 5: Retrieval Enhancers (post-rerank boosting)
        # Runs AFTER reranker so boosts modify rerank_score, not rrf_score
        if enable_enhancers:
            if verbose:
                logger.info(f"5⃣  Applying retrieval enhancers...")

            # Switch score key: enhancers boost whichever score is active
            score_key = 'rerank_score' if self.enable_reranking else 'rrf_score'
            for r in final_results:
                if score_key not in r:
                    r[score_key] = r.get('rrf_score', 0)

            final_results = apply_all_enhancers(
                final_results,
                query,
                score_key=score_key,
                verbose=verbose
            )

            # Re-sort by the boosted score
            final_results.sort(key=lambda x: x.get(score_key, 0), reverse=True)

            if verbose:
                logger.info(f"Enhancers applied to {len(final_results)} results")

        if verbose:
            print()
            logger.info(f"Retrieved {len(final_results)} final results")
            print()

        # Phase 2.5: Calculate metrics and log operation
        if obs and obs.enabled:
            try:
                latency_ms = int((time.time() - start_time) * 1000)
                num_chunks = len(final_results)

                # Estimate tokens in retrieved context (rough: words / 0.75)
                tokens_retrieved = 0
                for result in final_results:
                    text = result.get('original_chunk', '')
                    word_count = len(text.split())
                    tokens_retrieved += int(word_count / 0.75)

                # Log search operation
                obs.log_search_operation(
                    query=query,
                    num_chunks=num_chunks,
                    latency_ms=latency_ms,
                    tokens_retrieved=tokens_retrieved,
                    trace_id=trace_id
                )
            except Exception as e:
                # Graceful degradation - don't fail retrieval due to logging
                logger.debug(f"Observability logging failed: {e}")

        return final_results

    def retrieve_by_project(
        self,
        query: str,
        project: str,
        top_k: int = 5,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Retrieve chunks from a specific project

        Args:
            query: Search query
            project: Project name (PAI, IR-Platform, etc.)
            top_k: Number of results to return
            verbose: Print detailed progress

        Returns:
            List of relevant chunks from the specified project
        """
        return self.retrieve(
            query,
            top_k=top_k,
            filters={"source_project": project},
            verbose=verbose
        )

    def format_results(
        self,
        results: List[Dict],
        include_context: bool = True,
        include_scores: bool = True
    ) -> str:
        """
        Format retrieval results for display

        Args:
            results: Retrieval results
            include_context: Include generated context in output
            include_scores: Include relevance scores in output

        Returns:
            Formatted string representation of results
        """
        if not results:
            return "No results found."

        formatted = []
        for idx, result in enumerate(results, 1):
            lines = [f"\n{'='*80}"]
            lines.append(f"Result #{idx}")
            lines.append(f"{'='*80}")

            # Source information
            lines.append(f"📄 Source: {result.get('source_file', 'Unknown')}")
            lines.append(f"📂 Project: {result.get('source_project', 'Unknown')}")
            lines.append(f"📝 Type: {result.get('file_type', 'Unknown')}")

            # Scores (if available and requested)
            if include_scores:
                lines.append("")
                lines.append("📊 Scores:")
                if 'rerank_score' in result:
                    lines.append(f"   Rerank: {result['rerank_score']:.4f}")
                if 'rrf_score' in result:
                    lines.append(f"   RRF: {result['rrf_score']:.4f}")
                if 'vector_rank' in result:
                    lines.append(f"   Vector rank: {result['vector_rank']}")
                if 'bm25_rank' in result:
                    lines.append(f"   BM25 rank: {result['bm25_rank']}")

            # Context (if requested)
            if include_context and 'generated_context' in result:
                lines.append("")
                lines.append("🧠 Context:")
                lines.append(f"   {result['generated_context']}")

            # Original chunk content
            lines.append("")
            lines.append("📖 Content:")
            content = result.get('original_chunk', '')
            # Truncate if too long
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"   {content}")

            formatted.append("\n".join(lines))

        return "\n".join(formatted)

    def format_for_citations(
        self,
        results: List[Dict],
        include_context: bool = True
    ) -> List[Dict]:
        """
        Format retrieval results for Anthropic's native citations API

        Converts chunks to document format with citations enabled.
        This allows Claude to cite specific passages without those citations
        counting toward output tokens.

        Args:
            results: Retrieval results from pipeline
            include_context: Include generated context as document metadata

        Returns:
            List of documents in citation format:
            [
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": <original chunk content>
                    },
                    "title": <source file name>,
                    "context": <generated context>,
                    "citations": {"enabled": True}
                },
                ...
            ]
        """
        documents = []

        for idx, chunk in enumerate(results):
            # Build document title from source info
            source_file = chunk.get('source_file', 'Unknown')
            project = chunk.get('source_project', 'Unknown')
            title = f"{source_file} ({project})"

            # Create document in citation format
            doc = {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    # Use original chunk (not contextual) for citations
                    "data": chunk.get('original_chunk', '')
                },
                "title": title,
                "citations": {"enabled": True}
            }

            # Add context as metadata (not cited from, but provides situating info)
            if include_context and 'generated_context' in chunk:
                doc["context"] = chunk['generated_context']

            documents.append(doc)

        return documents

    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        stats = {
            'db_path': str(self.indexer.db_path),
            'total_chunks': len(self.table) if self.table else 0,
            'vector_search': self.vector_search.get_stats(),
            'bm25_search': self.bm25_search.get_stats(),
            'reranking_enabled': self.enable_reranking
        }

        if self.reranker:
            stats['reranker'] = self.reranker.get_stats()

        return stats
