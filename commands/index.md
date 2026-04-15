---
name: rag-index
description: Index new documents into the knowledge base
---

# RAG Index Command

Index new documents into the knowledge base with contextual chunking and sanitization.

## Usage

When the user requests to index a document:
1. Use the `index_document` MCP tool
2. Provide file path and project name
3. Enable/disable sanitization based on document type

## Examples

- "Index this new playbook into the knowledge base"
- "Add docs/new-feature.md to the RAG system"
- "Index all markdown files in docs/ directory"
- "Add this skill documentation to the knowledge base"
- "Index the README file without sanitization"

## Implementation

Use the MCP tool:
```python
index_document(
    file_path="docs/new-doc.md",
    project="PAI",  # Optional, defaults to config
    enable_sanitization=True  # Optional, defaults to config
)
```

Or use the CLI tool directly:
```bash
# Index single file
0k-index docs/new-doc.md

# Batch index directory
0k-index --pattern 'docs/**/*.md'

# Index without sanitization
0k-index file.md --no-sanitize

# Dry run to see what would be indexed
0k-index --pattern 'docs/**/*.pdf' --dry-run
```

## Indexing Pipeline

Each document goes through:
1. **Document Loading** - Parse file (MD, PDF, DOCX, PPTX, TXT)
2. **PII Sanitization** - Multi-layer redaction (if enabled)
3. **Contextual Chunking** - Boundary-aware segmentation (384 tokens)
4. **Context Generation** - Llama 3.1 8B generates chunk summaries
5. **Embedding** - nomic-embed-text creates 768-dim vectors
6. **LanceDB Indexing** - Store chunks with vectors and metadata

## Configuration

Controlled by `.vex-rag.yml`:
- `indexing.chunk_size` - Chunk size in tokens (default: 384)
- `indexing.chunk_overlap` - Overlap percentage (default: 0.15)
- `indexing.context_model` - Ollama model for context generation
- `indexing.embedding_model` - Embedding model to use
- `indexing.enable_sanitization` - PII sanitization toggle
- `indexing.auto_index_extensions` - File types to auto-index
- `indexing.auto_index_paths` - Directories to watch

## Performance

- ~2-5 chunks per document (depends on length)
- ~1-2 seconds per chunk (context generation + embedding)
- 100% local processing (zero cloud APIs)
- Auto-indexed files appear immediately in searches

## Supported Formats

- Markdown (`.md`)
- PDF (`.pdf`)
- Word Documents (`.docx`)
- PowerPoint (`.pptx`)
- Plain Text (`.txt`)

## Auto-Indexing

When git post-commit hook is installed, modified files are automatically indexed after commits (configurable via `auto_index_extensions` and `auto_index_paths`).
