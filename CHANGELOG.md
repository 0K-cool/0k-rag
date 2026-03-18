# Changelog

All notable changes to the Vex RAG Plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-03-18

### Added
- **Search Health Check** in `get_kb_stats()` — exercises actual vector search path and reports `search_healthy: true/false`. Detects corrupted LanceDB fragments that metadata-only checks miss.
- **FTS Index Validation** — verifies FTS/BM25 index files exist on disk before reporting `fts_enabled: true`
- **`rebuild_index` MCP tool** — drops and recreates the knowledge base from source files. Use when search returns 0 results or database is corrupted. Source documents are never affected.

### Fixed
- **BM25/FTS index not created at startup** (v1.3.2) — `create_index()` now called eagerly at pipeline init
- **Corrupted LanceDB detection** — `get_kb_stats()` previously reported healthy status even when vector search was broken due to missing fragment files

## [1.3.2] - 2026-03-18

### Fixed
- **BM25/FTS index not created at startup** — `RetrievalPipeline.__init__()` never called `bm25_search.create_index()`. The FTS index relied on lazy creation during first search, but if that failed silently (LanceDB version issue, table lock), BM25 was permanently disabled. Hybrid search fell back to vector-only, missing keyword matches for CVE IDs, tool names, and exact command syntax. Fix: eagerly create FTS index at init (idempotent — skips if already exists).

## [1.3.1] - 2026-02-02

### Added
- **AI Agent Discoverability** (ATHENA feedback)
  - `search_kb(query, top_k)` MCP tool - always discoverable by AI agents
  - `vex://help` MCP resource - onboarding documentation for agents
  - Enhanced `get_kb_stats()` with `usage_hint` and `example_queries`

- **Progress Notification System**
  - Pluggable notification architecture with `NotifierInterface` protocol
  - `ConsoleNotifier` - terminal progress with progress bars and ANSI colors
  - `WebhookNotifier` - Discord, Slack, Teams webhook templates
  - `CompositeNotifier` - combine multiple notifiers
  - `NullNotifier` - no-op for backward compatibility
  - `MCPProgressCollector` - collects progress for MCP tool responses
  - Pipeline integration (indexer, context generator, embedder)

### Fixed
- **MCP Resource Discoverability** - Templated MCP resources (`vex://search/{query}`) aren't enumerable by AI agents. Added `search_kb` tool as always-discoverable alternative.

### Changed
- `index_document()` now includes progress summary in response
- `get_kb_stats()` returns usage hints and example queries

## [1.0.1] - 2026-01-02

### Fixed
- **CLI Shebang Portability** - Replaced hardcoded Python interpreter paths with proper package entry points
  - Created `rag/cli/search.py` and `rag/cli/index.py` as standard Python modules
  - CLI commands (`vex-search`, `vex-index`) now install correctly via `pip install -e .`
  - Removed system-specific shebang lines that prevented installation on other machines

- **Ollama Model Verification Warning** - Fixed cosmetic warning during indexing
  - Improved error handling in `embedder.py` and `context_generator.py`
  - Now handles both 'name' and 'model' keys from Ollama API response (API variation support)
  - Clearer error messages when model verification fails

- **Package Configuration** - Removed non-existent `rag.mcp_server` from setuptools packages list
  - MCP server lives in PAI project (`.claude/mcp_servers/`), not in plugin
  - Package now installs without errors

### Changed
- CLI tools now use standard Python entry points instead of standalone scripts
- Improved installation documentation (recommend `pip install -e .` after clone)

### Security
- **Privacy Improvements for Public Repository**
  - Changed author email from personal to public-facing email (`0k.8csjy@8shield.net`)
  - Replaced internal project names with generic equivalents for privacy:
    - `VERSANT-IR` → `IR-Platform`
    - `VERSANT-ATHENA` → `Pentest-Platform`
    - `lance_athena_kb` → `lance_pentest_kb`
  - Renamed `examples/config.athena.yml` → `examples/config.pentest.yml`
  - Updated all example configurations, documentation, and Python docstrings with generic names
  - Ensures repository is suitable for public consumption without exposing internal project details

## [1.0.0] - 2026-01-02

### Added
- **Initial plugin extraction from PAI monolithic system**
- Configuration-driven architecture via `.vex-rag.yml`
- Project-portable MCP server with automatic context injection
- Slash commands: `/rag-search`, `/rag-index`
- RAG maintenance subagent for system operations
- Auto-indexing git hook for post-commit document indexing
- CLI tools: `vex-search`, `vex-index`
- Example configurations for PAI, IR platform, Pentest platform
- Comprehensive README with installation and usage guide
- Python package structure with proper imports (rag.*)
- Multi-project support via configuration files
- Configurable logging and backup locations

### Features
- **Core RAG Pipeline:**
  - Contextual chunking (Llama 3.1 8B, boundary-aware, 384 tokens)
  - Vector search (nomic-embed-text, 768-dim)
  - BM25 keyword search (LanceDB FTS)
  - Reciprocal Rank Fusion (RRF)
  - BGE reranking (local, Apple Silicon GPU)
  - Native citations support (Anthropic format)

- **Infrastructure:**
  - 100% local processing (zero cloud APIs)
  - MCP server integration (automatic context injection)
  - Multi-project support (configurable via YAML)
  - PII sanitization (multi-layer, configurable)
  - Auto-indexing (git post-commit hooks)
  - Configurable backup/restore

- **Developer Experience:**
  - CLI tools (vex-search, vex-index)
  - Slash commands (/rag-search, /rag-index)
  - RAG maintenance subagent
  - Comprehensive documentation
  - Example configurations

### Performance
- First search: ~6 seconds (BGE model load)
- Subsequent searches: ~2 seconds
- Indexing: ~1-2 seconds per chunk
- Storage efficiency: ~4KB per chunk

### Security
- 100% local processing (zero data exfiltration)
- PII sanitization enabled by default
- Configurable per-project sanitization settings
- File permissions: 600/700 (owner only)

### Documentation
- Comprehensive README.md with installation guide
- Plugin design document (vex-rag-plugin-design.md)
- Example configurations for 3 project types
- Slash command documentation
- RAG maintenance agent guide
- Troubleshooting section

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security Notes
- PII sanitization enabled by default
- All processing done locally (zero cloud APIs)
- Configurable per-project sanitization settings

---

## [Unreleased]

### Planned for v1.0.1
- End-to-end test suite
- Bug fixes from user feedback
- Performance optimizations
- Enhanced error messages

### Planned for v1.1.0
- Multi-modal support (images, OCR, audio, video)
- Image indexing for screenshots and diagrams
- OCR for scanned documents
- Audio transcription support

### Planned for v1.2.0
- Advanced query features
- Query expansion/rewriting
- Agentic retrieval (iterative multi-step)
- Parent-child chunking
- Time-based filtering

### Planned for v1.3.0
- Testing & observability
- Comprehensive test suite
- Performance benchmarking
- Metrics dashboard
- Retry logic for failures

### Planned for v2.0.0
- Enterprise features
- Multi-user support
- Access control per project
- Audit logging
- Cost tracking

---

## Version Naming Convention

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR version** (v2.0.0) - Incompatible API changes
- **MINOR version** (v1.1.0) - New features, backwards compatible
- **PATCH version** (v1.0.1) - Bug fixes, backwards compatible

---

## Release Notes

### v1.0.0 - Plugin Extraction (January 2, 2026)

**Summary:** Initial extraction of Vex RAG system from Personal AI Infrastructure (PAI) monolithic codebase into a standalone, reusable Claude Code Plugin.

**Key Changes:**
- Extracted core RAG modules (indexing, retrieval) into `rag/` package
- Created configurable MCP server that reads from `.vex-rag.yml`
- Built portable CLI tools (vex-search, vex-index)
- Designed slash commands for Claude Code integration
- Implemented auto-indexing via git hook
- Created RAG maintenance subagent for operations

**Migration Path:**
- PAI (monolithic) → v1.0.0 (plugin) requires:
  1. Create `.vex-rag.yml` in project root
  2. Install plugin: `claude plugin install ~/tools/vex-rag`
  3. Copy existing LanceDB: `lance_vex_kb/` (no changes needed)
  4. Update MCP config to use new server location

**Breaking Changes:**
- None (initial release)

**Known Issues:**
- First search always slow (~6s) due to BGE model load (expected behavior)
- Auto-indexing hook requires manual installation per project
- Backup scripts are project-specific (not yet in plugin)

**Upgrade Notes:**
- None (initial release)

---

## Contributors

**v1.0.0:**
- Kelvin Lomboy (@kelvinlomboy) - Initial plugin extraction and design

---

## Links

- **Repository:** https://github.com/0K-cool/vex-rag
- **Issues:** https://github.com/0K-cool/vex-rag/issues
- **Documentation:** https://github.com/0K-cool/vex-rag/blob/main/README.md
- **Changelog:** https://github.com/0K-cool/vex-rag/blob/main/CHANGELOG.md
