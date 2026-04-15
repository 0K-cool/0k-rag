#!/bin/bash
# install-observability.sh - Install vex-rag observability to a project
# Part of vex-rag plugin v1.0.0
#
# Usage: ./install-observability.sh [target_directory]
#
# Installs observability infrastructure (scripts and logs) to a project directory.
# This enables RAG performance tracking, cost monitoring, and health dashboards.

set -euo pipefail

# Configuration
TARGET_DIR="${1:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   vex-rag Observability Installation Helper 📊            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Validate target directory
if [[ ! -d "$TARGET_DIR" ]]; then
    echo -e "${YELLOW}⚠️  Target directory does not exist: $TARGET_DIR${NC}"
    echo "Creating directory..."
    mkdir -p "$TARGET_DIR"
fi

echo -e "📂 Target directory: ${GREEN}$TARGET_DIR${NC}"
echo -e "📦 Source scripts: ${GREEN}$SCRIPT_DIR${NC}"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p "$TARGET_DIR/.claude/scripts"
mkdir -p "$TARGET_DIR/.claude/logs/traces"
mkdir -p "$TARGET_DIR/.claude/logs/errors-archive"

# Copy observability scripts
echo "Copying observability scripts..."
cp "$SCRIPT_DIR/log-token-usage.sh" "$TARGET_DIR/.claude/scripts/"
cp "$SCRIPT_DIR/log-latency-trace.sh" "$TARGET_DIR/.claude/scripts/"
cp "$SCRIPT_DIR/vex-log-error.sh" "$TARGET_DIR/.claude/scripts/"
cp "$SCRIPT_DIR/vex-health-check.sh" "$TARGET_DIR/.claude/scripts/"
cp "$SCRIPT_DIR/vex-token-report.sh" "$TARGET_DIR/.claude/scripts/"
cp "$SCRIPT_DIR/vex-latency-report.sh" "$TARGET_DIR/.claude/scripts/"
cp "$SCRIPT_DIR/vex-error-report.sh" "$TARGET_DIR/.claude/scripts/"

# Set executable permissions
echo "Setting executable permissions..."
chmod +x "$TARGET_DIR/.claude/scripts/"*.sh

# Create empty log files
echo "Creating log files..."
touch "$TARGET_DIR/.claude/logs/token-usage.jsonl"
touch "$TARGET_DIR/.claude/logs/errors.jsonl"

# Set restrictive permissions on logs (owner read/write only)
chmod 600 "$TARGET_DIR/.claude/logs/token-usage.jsonl"
chmod 600 "$TARGET_DIR/.claude/logs/errors.jsonl"
chmod 700 "$TARGET_DIR/.claude/logs/traces"

echo ""
echo -e "${GREEN}✅ Installation complete!${NC}"
echo ""
echo "Installed:"
echo "  - 7 observability scripts → .claude/scripts/"
echo "  - 2 log files → .claude/logs/"
echo "  - 2 directories → .claude/logs/traces/ and errors-archive/"
echo ""
echo -e "${BLUE}Available Commands:${NC}"
echo "  ${GREEN}./.claude/scripts/vex-health-check.sh --period=24h${NC}"
echo "    └─ System health dashboard (24h/7d/30d)"
echo ""
echo "  ${GREEN}./.claude/scripts/vex-token-report.sh 2026-01${NC}"
echo "    └─ Monthly token usage and cost report"
echo ""
echo "  ${GREEN}./.claude/scripts/vex-latency-report.sh 2026-01${NC}"
echo "    └─ Monthly latency performance report"
echo ""
echo "  ${GREEN}./.claude/scripts/vex-error-report.sh 2026-01${NC}"
echo "    └─ Monthly error analysis report"
echo ""
echo -e "${BLUE}Log Files:${NC}"
echo "  ${GREEN}$TARGET_DIR/.claude/logs/token-usage.jsonl${NC}"
echo "    └─ Token usage and cost tracking"
echo ""
echo "  ${GREEN}$TARGET_DIR/.claude/logs/errors.jsonl${NC}"
echo "    └─ Error and warning logs"
echo ""
echo "  ${GREEN}$TARGET_DIR/.claude/logs/traces/*.jsonl${NC}"
echo "    └─ Operation latency traces (per-conversation)"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. RAG operations will now automatically log metrics"
echo "  2. Run 0k-search or 0k-index to test observability"
echo "  3. Check health dashboard: ./.claude/scripts/vex-health-check.sh"
echo ""
echo -e "${YELLOW}📝 Note: Logs are stored locally and never sent to cloud services.${NC}"
echo ""
