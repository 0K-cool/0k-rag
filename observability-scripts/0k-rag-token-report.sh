#!/usr/bin/env bash

# 0k-rag-token-report.sh - Generate monthly token usage report
# Part of Priority 2: Observability Framework (Phase 1)
# Usage: 0k-rag-token-report.sh [YYYY-MM]
#        If no month specified, generates report for current month
# Compatible with Bash 3.2+ (macOS default)

set -eo pipefail

# Phase 3: Error Tracking Integration
ERROR_LOG_SCRIPT="${OKRAG_LOG_DIR:-${HOME}/.0k-rag/logs}/../scripts/0k-rag-log-error.sh"

# Error handler
error_handler() {
  local exit_code=$?
  local line_number=$1
  local command="$BASH_COMMAND"

  # Log error if 0k-rag-log-error.sh is available
  if [[ -x "$ERROR_LOG_SCRIPT" ]]; then
    "$ERROR_LOG_SCRIPT" \
      --severity=error \
      --source="0k-rag-token-report.sh" \
      --error-type="execution_failure" \
      --message="Token report generation failed: $command" \
      --exit-code="$exit_code" \
      --context="{\"line\": $line_number, \"command\": \"$command\", \"month\": \"${MONTH:-unknown}\"}" \
      --resolution="Check log file existence and jq availability" || true
  fi

  exit $exit_code
}

# Set trap for errors
trap 'error_handler ${LINENO}' ERR

# Configuration
if [[ -n "${PAI_DIR:-}" ]]; then
    PROJECT_ROOT="$(dirname "$PAI_DIR")"
else
    PROJECT_ROOT="."
fi
LOG_DIR="${PROJECT_ROOT}/.claude/logs"
LOG_FILE="${LOG_DIR}/token-usage.jsonl"
REPORT_DIR="${LOG_DIR}/reports"

# Parse month argument (default to current month)
if [[ $# -eq 0 ]]; then
    MONTH=$(date +"%Y-%m")
else
    MONTH="$1"
fi

YEAR=$(echo "$MONTH" | cut -d'-' -f1)
MONTH_NUM=$(echo "$MONTH" | cut -d'-' -f2)
MONTH_NAME=$(date -j -f "%Y-%m" "$MONTH" +"%B %Y" 2>/dev/null || echo "$MONTH")

# Check if log file exists
if [[ ! -f "$LOG_FILE" ]]; then
    echo "Error: Token log file not found at $LOG_FILE" >&2
    exit 1
fi

# Create reports directory if it doesn't exist
mkdir -p "$REPORT_DIR"
chmod 700 "$REPORT_DIR"

# Output file
REPORT_FILE="${REPORT_DIR}/token-usage-${MONTH}.md"

# Filter logs for the specified month
MONTH_LOGS=$(grep "\"timestamp\":\"${YEAR}-${MONTH_NUM}" "$LOG_FILE" || true)

if [[ -z "$MONTH_LOGS" ]]; then
    echo "No token usage data found for $MONTH_NAME" >&2
    exit 0
fi

# Calculate totals using jq
TOTAL_TOKENS=$(echo "$MONTH_LOGS" | jq -s 'map(.tokens.total) | add')
TOTAL_COST=$(echo "$MONTH_LOGS" | jq -s 'map(.cost_usd) | add')
ENTRY_COUNT=$(echo "$MONTH_LOGS" | wc -l | tr -d ' ')
ESTIMATED_COUNT=$(echo "$MONTH_LOGS" | jq -s 'map(select(.estimated == true)) | length')

# Calculate daily average
DAYS_IN_MONTH=$(cal "$MONTH_NUM" "$YEAR" 2>/dev/null | awk 'NF {DAYS = $NF}; END {print DAYS}' || echo "30")
DAILY_AVG_TOKENS=$((TOTAL_TOKENS / DAYS_IN_MONTH))
DAILY_AVG_COST=$(echo "scale=2; $TOTAL_COST / $DAYS_IN_MONTH" | bc -l)

# Format costs
TOTAL_COST_FMT=$(printf "%.2f" "$TOTAL_COST")
DAILY_AVG_COST_FMT=$(printf "%.2f" "$DAILY_AVG_COST")

# Aggregate by operation type
OP_TYPE_AGG=$(echo "$MONTH_LOGS" | jq -r '[.operation_type, .tokens.total, .cost_usd] | @tsv' | \
    awk '{type[$1]+=$2; cost[$1]+=$3} END {for (t in type) print t"\t"type[t]"\t"cost[t]}' | \
    sort -t$'\t' -k3 -rn)

# Aggregate by operation name (top 10)
OP_NAME_AGG=$(echo "$MONTH_LOGS" | jq -r '[.operation_name, .tokens.total, .cost_usd] | @tsv' | \
    awk '{name[$1]+=$2; cost[$1]+=$3} END {for (n in name) print n"\t"name[n]"\t"cost[n]}' | \
    sort -t$'\t' -k3 -rn | head -10)

# Generate report
cat > "$REPORT_FILE" <<EOF
# 0K-RAG Token Usage Report - $MONTH_NAME

**Generated:** $(date +"%Y-%m-%d %H:%M:%S %Z")
**Data Source:** \`.claude/logs/token-usage.jsonl\`

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Tokens** | $(printf "%'d" "$TOTAL_TOKENS") |
| **Total Cost** | \$$TOTAL_COST_FMT |
| **Daily Average** | $(printf "%'d" "$DAILY_AVG_TOKENS") tokens/day (\$$DAILY_AVG_COST_FMT/day) |
| **Days in Month** | $DAYS_IN_MONTH |

---

## By Operation Type

| Operation Type | Tokens | Cost |
|----------------|--------|------|
EOF

# Add operation type rows
echo "$OP_TYPE_AGG" | while IFS=$'\t' read -r op_type tokens cost; do
    tokens_fmt=$(printf "%'d" "$tokens")
    cost_fmt=$(printf "%.2f" "$cost")
    echo "| $op_type | $tokens_fmt | \$$cost_fmt |" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" <<EOF

---

## Top 10 Most Expensive Operations

| Rank | Operation Name | Tokens | Cost |
|------|----------------|--------|------|
EOF

# Add operation name rows
rank=1
echo "$OP_NAME_AGG" | while IFS=$'\t' read -r op_name tokens cost; do
    tokens_fmt=$(printf "%'d" "$tokens")
    cost_fmt=$(printf "%.2f" "$cost")
    echo "| $rank | $op_name | $tokens_fmt | \$$cost_fmt |" >> "$REPORT_FILE"
    ((rank++))
done

cat >> "$REPORT_FILE" <<EOF

---

## Cost Breakdown

\`\`\`
Total Cost:         \$$TOTAL_COST_FMT
Daily Average:      \$$DAILY_AVG_COST_FMT/day
Monthly Projection: \$$TOTAL_COST_FMT (actual)
\`\`\`

**Note:** All token estimates are based on the formula: ~4 characters = 1 token (±15% accuracy).

---

## Data Quality

| Metric | Value |
|--------|-------|
| Total Log Entries | $ENTRY_COUNT |
| Estimated Tokens | $ESTIMATED_COUNT |
| Exact Tokens | $((ENTRY_COUNT - ESTIMATED_COUNT)) |

---

**Report saved to:** \`$REPORT_FILE\`
EOF

# Set restrictive permissions
chmod 600 "$REPORT_FILE"

# Print report to stdout
cat "$REPORT_FILE"

echo ""
echo "📊 Report saved to: $REPORT_FILE"
