#!/bin/bash
# 0K-RAG Auto-Indexing Hook
# Automatically index modified files after git commit
#
# Installation:
#   cp hooks/post-commit.sh .git/hooks/post-commit
#   chmod +x .git/hooks/post-commit
#
# Configuration:
#   Edit .0k-rag.yml to configure auto-index settings:
#   - indexing.auto_index_extensions (default: [.md, .pdf, .txt])
#   - indexing.auto_index_paths (directories to watch)

# Exit on error
set -e

# Load configuration
CONFIG_FILE=".0k-rag.yml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚠️  RAG config not found: $CONFIG_FILE" >&2
    echo "   Auto-indexing disabled. Create .0k-rag.yml to enable." >&2
    exit 0
fi

# Check if yq is available for YAML parsing
if ! command -v yq &> /dev/null; then
    echo "⚠️  yq not installed. Auto-indexing disabled." >&2
    echo "   Install yq: brew install yq" >&2
    exit 0
fi

# Check if 0k-index is available
if ! command -v 0k-index &> /dev/null; then
    echo "⚠️  0k-index not found in PATH. Auto-indexing disabled." >&2
    echo "   Ensure 0k-rag plugin is installed and CLI tools are in PATH." >&2
    exit 0
fi

# Extract auto-index settings from YAML
AUTO_INDEX_EXTENSIONS=$(yq -r '.indexing.auto_index_extensions[]' "$CONFIG_FILE" 2>/dev/null | tr '\n' '|' | sed 's/|$//')
AUTO_INDEX_PATHS=$(yq -r '.indexing.auto_index_paths[]' "$CONFIG_FILE" 2>/dev/null)

# Default to .md if not configured
if [ -z "$AUTO_INDEX_EXTENSIONS" ]; then
    AUTO_INDEX_EXTENSIONS=".md"
fi

# Get list of modified files in last commit
MODIFIED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD)

# Filter for auto-indexable files
FILES_TO_INDEX=""
for file in $MODIFIED_FILES; do
    # Check if file exists (not deleted)
    if [ ! -f "$file" ]; then
        continue
    fi

    # Check if file matches extensions
    FILE_EXT="${file##*.}"
    if ! echo "$AUTO_INDEX_EXTENSIONS" | grep -q "\.$FILE_EXT"; then
        continue
    fi

    # Check if file is in auto-index paths
    if [ -n "$AUTO_INDEX_PATHS" ]; then
        MATCH_FOUND=false
        for path in $AUTO_INDEX_PATHS; do
            # Remove trailing slash from path
            path="${path%/}"
            # Check if file starts with path
            if [[ "$file" == "$path"* ]] || [[ "$file" == "$path" ]]; then
                MATCH_FOUND=true
                break
            fi
        done

        if [ "$MATCH_FOUND" = false ]; then
            continue
        fi
    fi

    FILES_TO_INDEX="$FILES_TO_INDEX $file"
done

# Index files if any found
if [ -n "$FILES_TO_INDEX" ]; then
    echo "📚 Auto-indexing modified files..." >&2
    INDEXED_COUNT=0
    FAILED_COUNT=0

    for file in $FILES_TO_INDEX; do
        echo "   Indexing: $file" >&2
        if 0k-index "$file" 2>&1 | grep -q "✅"; then
            ((INDEXED_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
    done

    echo "   ✅ Indexed $INDEXED_COUNT file(s)" >&2
    if [ $FAILED_COUNT -gt 0 ]; then
        echo "   ⚠️  Failed to index $FAILED_COUNT file(s)" >&2
    fi
else
    # No files to index - this is normal, not an error
    exit 0
fi
