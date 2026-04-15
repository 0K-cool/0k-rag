#!/usr/bin/env bash

# 0k-rag-latency-report.sh - Generate monthly latency report
# Part of Priority 2: Observability Framework (Phase 2)
# Usage: 0k-rag-latency-report.sh [YYYY-MM]
#        If no month specified, generates report for current month

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
      --source="0k-rag-latency-report.sh" \
      --error-type="execution_failure" \
      --message="Latency report generation failed: $command" \
      --exit-code="$exit_code" \
      --context="{\"line\": $line_number, \"command\": \"$command\", \"month\": \"${MONTH:-unknown}\"}" \
      --resolution="Check trace files and jq availability" || true
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
TRACE_DIR="${PROJECT_ROOT}/.claude/logs/traces"
REPORT_DIR="${PROJECT_ROOT}/.claude/logs/reports"

# Parse month argument (default to current month)
if [[ $# -eq 0 ]]; then
    MONTH=$(date +"%Y-%m")
else
    MONTH="$1"
fi

YEAR=$(echo "$MONTH" | cut -d'-' -f1)
MONTH_NUM=$(echo "$MONTH" | cut -d'-' -f2)
MONTH_NAME=$(date -j -f "%Y-%m" "$MONTH" +"%B %Y" 2>/dev/null || echo "$MONTH")

# Create reports directory if it doesn't exist
mkdir -p "$REPORT_DIR"
chmod 700 "$REPORT_DIR"

# Output file
REPORT_FILE="${REPORT_DIR}/latency-${MONTH}.md"

# Aggregate all trace data for the month
ALL_TRACES=""
for trace_file in "$TRACE_DIR"/*.jsonl; do
    if [[ -f "$trace_file" ]]; then
        # Filter by month in timestamp
        month_data=$(grep "\"start_time\":\"${YEAR}-${MONTH_NUM}" "$trace_file" 2>/dev/null || true)
        if [[ -n "$month_data" ]]; then
            ALL_TRACES="${ALL_TRACES}${month_data}"$'\n'
        fi
    fi
done

# Check if we have data
if [[ -z "$ALL_TRACES" ]]; then
    echo "No latency data found for $MONTH_NAME" >&2
    exit 0
fi

# Calculate statistics
TOTAL_OPERATIONS=$(echo "$ALL_TRACES" | grep -c . || echo "0")
TOTAL_DURATION=$(echo "$ALL_TRACES" | jq -s 'map(.duration_ms) | add')
AVG_DURATION=$(echo "scale=2; $TOTAL_DURATION / $TOTAL_OPERATIONS" | bc -l)
SLOW_OPS=$(echo "$ALL_TRACES" | jq -s 'map(select(.duration_ms > 10000)) | length')

# Calculate percentiles
P50=$(echo "$ALL_TRACES" | jq -s 'map(.duration_ms) | sort | .[length / 2 | floor]')
P95=$(echo "$ALL_TRACES" | jq -s 'map(.duration_ms) | sort | .[length * 0.95 | floor]')
P99=$(echo "$ALL_TRACES" | jq -s 'map(.duration_ms) | sort | .[length * 0.99 | floor]')

# Find slowest operations
SLOWEST_OPS=$(echo "$ALL_TRACES" | jq -s 'sort_by(-.duration_ms) | .[0:10] | .[] | "\(.operation_name): \(.duration_ms)ms"')

# Aggregate by operation type
OP_TYPE_STATS=$(echo "$ALL_TRACES" | jq -r '[.operation_type, .duration_ms] | @tsv' | \
    awk '{type[$1]++; sum[$1]+=$2; if($2>max[$1])max[$1]=$2} END {for(t in type) print t"\t"type[t]"\t"sum[t]"\t"max[t]}' | \
    sort -t$'\t' -k3 -rn)

# Generate report
cat > "$REPORT_FILE" <<EOF
# 0K-RAG Latency Report - $MONTH_NAME

**Generated:** $(date +"%Y-%m-%d %H:%M:%S %Z")
**Data Source:** \`.claude/logs/traces/*.jsonl\`

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Operations** | $(printf "%'d" "$TOTAL_OPERATIONS") |
| **Total Duration** | $(echo "scale=2; $TOTAL_DURATION / 1000" | bc -l)s |
| **Average Duration** | ${AVG_DURATION}ms |
| **Slow Operations (>10s)** | $SLOW_OPS |

---

## Performance Percentiles

| Percentile | Duration |
|------------|----------|
| **P50 (Median)** | ${P50}ms |
| **P95** | ${P95}ms |
| **P99** | ${P99}ms |

---

## By Operation Type

| Operation Type | Count | Total Duration | Max Duration |
|----------------|-------|----------------|--------------|
EOF

# Add operation type rows
echo "$OP_TYPE_STATS" | while IFS=$'\t' read -r op_type count total_ms max_ms; do
    total_sec=$(echo "scale=2; $total_ms / 1000" | bc -l)
    echo "| $op_type | $(printf "%'d" "$count") | ${total_sec}s | ${max_ms}ms |" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" <<EOF

---

## Top 10 Slowest Operations

| Rank | Operation | Duration |
|------|-----------|----------|
EOF

# Add slowest operations
rank=1
echo "$SLOWEST_OPS" | while IFS= read -r line; do
    echo "| $rank | $line |" >> "$REPORT_FILE"
    ((rank++))
done

# Calculate daily average
DAYS_IN_MONTH=$(cal "$MONTH_NUM" "$YEAR" 2>/dev/null | awk 'NF {DAYS = $NF}; END {print DAYS}' || echo "30")
DAILY_AVG_OPS=$((TOTAL_OPERATIONS / DAYS_IN_MONTH))

cat >> "$REPORT_FILE" <<EOF

---

## Performance Health

\`\`\`
Total Operations:   $(printf "%'d" "$TOTAL_OPERATIONS")
Daily Average:      $(printf "%'d" "$DAILY_AVG_OPS") operations/day
Slow Operations:    $SLOW_OPS ($(echo "scale=1; $SLOW_OPS * 100 / $TOTAL_OPERATIONS" | bc -l)%)
Average Duration:   ${AVG_DURATION}ms
P95 Duration:       ${P95}ms
\`\`\`

**Performance Grade:**
EOF

# Calculate performance grade
SLOW_PERCENTAGE=$(echo "scale=1; $SLOW_OPS * 100 / $TOTAL_OPERATIONS" | bc -l | cut -d. -f1)
if [[ $SLOW_PERCENTAGE -lt 5 ]]; then
    echo "- ✅ **EXCELLENT** - Less than 5% slow operations" >> "$REPORT_FILE"
elif [[ $SLOW_PERCENTAGE -lt 10 ]]; then
    echo "- ✅ **GOOD** - Less than 10% slow operations" >> "$REPORT_FILE"
elif [[ $SLOW_PERCENTAGE -lt 20 ]]; then
    echo "- ⚠️  **FAIR** - 10-20% slow operations (consider optimization)" >> "$REPORT_FILE"
else
    echo "- 🚨 **POOR** - More than 20% slow operations (optimization needed)" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" <<EOF

---

**Report saved to:** \`$REPORT_FILE\`
EOF

# Set restrictive permissions
chmod 600 "$REPORT_FILE"

# Print report to stdout
cat "$REPORT_FILE"

echo ""
echo "📊 Report saved to: $REPORT_FILE"
