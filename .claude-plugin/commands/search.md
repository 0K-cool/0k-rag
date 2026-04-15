---
name: rag-search
description: Search 0k-rag knowledge base manually
---

# RAG Search

Manually search the local knowledge base for relevant documentation.

## Usage

```
/rag-search "your query here"
```

## Examples

```
/rag-search "git safety check workflow"
/rag-search "what are the mandatory protocols"
/rag-search "how to use 0k-rag"
```

## How It Works

This command calls the `0k-search` CLI tool to query your local knowledge base:
- Hybrid search (vector + BM25 + RRF)
- BGE reranking for relevance
- Top 5 results with citations
- 100% local processing

## When to Use

**Most of the time: Don't use this command**
- The MCP server automatically retrieves relevant context when you ask questions
- Automatic retrieval is faster and more seamless

**Use this command when:**
- You want to manually verify what's in the knowledge base
- You need to see all results for a specific query
- You're debugging indexing or retrieval issues
- You want explicit control over search queries

## Configuration

Search uses your project's `.vex-rag.yml` configuration:
- Database location
- Reranking settings
- Number of results
- Project isolation

## Notes

- Requires 0k-rag plugin to be installed (`pip install -e .`)
- Requires Ollama models: `nomic-embed-text`, `llama3.1:8b`
- First search loads BGE reranker (~6 seconds), subsequent searches are fast (~2 seconds)
