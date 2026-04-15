---
name: rag-search
description: Search the RAG knowledge base with hybrid retrieval
---

# RAG Search Command

Search the knowledge base using vector search, BM25, RRF fusion, and BGE reranking.

## Usage

When the user requests to search for information in the knowledge base:
1. Use the `0k-rag-knowledge-base` MCP server's search resources
2. Retrieve top-k results (default: 5)
3. Present results with sources and scores

## Examples

- "Search for git safety check workflow"
- "Find documentation on backup procedures"
- "What does the rag-maintenance agent do?"
- "Search the knowledge base for threat intelligence workflows"
- "Find all documentation related to MCP servers"

## Implementation

Use the MCP resource:
```
0k-rag://search/{query}
```

This resource automatically:
- Performs hybrid search (vector + BM25 + RRF + BGE reranking)
- Returns top-k most relevant chunks
- Includes native citations for accurate source attribution
- Handles result formatting and scoring

Or use the CLI tool directly:
```bash
0k-search "your query here" --top-k 10
```

## Response Format

Results include:
- **Source file** and project name
- **Relevance score** (from reranker)
- **Generated context** (summary of chunk)
- **Original content** (actual chunk text)
- **Citations** (file path and line numbers)

## Configuration

Controlled by `.0k-rag.yml`:
- `retrieval.default_top_k` - Number of results to return
- `retrieval.enable_reranking` - Use BGE reranker for final ranking
- `retrieval.reranker_model` - Reranker model to use

## Performance

- First search: ~6 seconds (includes BGE model load)
- Subsequent searches: ~2 seconds
- 100% local processing (zero cloud APIs)
