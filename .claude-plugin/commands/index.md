---
name: rag-index
description: Index a document into 0k-rag knowledge base
---

# RAG Index

Manually index a document into the local knowledge base.

## Usage

```
/rag-index path/to/document.md
/rag-index path/to/document.pdf --no-sanitize
```

## Examples

```
/rag-index docs/new-feature.md
/rag-index output/research/threat-analysis.md
/rag-index ~/Downloads/security-report.pdf
```

## How It Works

This command calls the `0k-index` CLI tool to process and index documents:
1. **Load document** - Supports MD, TXT, PDF, DOCX, PPTX
2. **Sanitize PII** - Multi-layer sanitization (configurable)
3. **Chunk intelligently** - Boundary-aware, 384 tokens, 15% overlap
4. **Generate context** - Llama 3.1 8B adds situating context
5. **Embed** - nomic-embed-text creates 768-dim vectors
6. **Store** - LanceDB local vector database

## Options

**`--no-sanitize`** - Skip PII sanitization (use for public documents only)
**`--project NAME`** - Index to specific project (default: current project)

## When to Use

**Most of the time: Use git post-commit hook**
- Automatically indexes `.md` files after commits
- Zero manual intervention
- Configured per-project in `.git/hooks/post-commit`

**Use this command when:**
- Indexing a document outside your git repo
- Immediately need a document indexed (can't wait for next commit)
- Testing indexing with specific settings
- Indexing non-markdown files (PDF, DOCX, etc.)

## Path Validation

For security, only files within allowed directories can be indexed:
- Your project directory
- Configured allowed paths in `.vex-rag.yml`

Attempts to index files outside allowed directories will be rejected (prevents path traversal).

## Configuration

Indexing uses your project's `.vex-rag.yml` configuration:
- Database location
- PII sanitization settings
- Chunking parameters
- Allowed base paths

## Notes

- Requires 0k-rag plugin to be installed (`pip install -e .`)
- Requires Ollama models: `nomic-embed-text`, `llama3.1:8b`
- Indexing time: ~10-30 seconds per document (depends on size)
- Duplicate detection: Same content hash skips re-indexing
