#!/bin/bash
# 0k-rag-health-check.sh - 0K-RAG system health dashboard
# Part of 0K-RAG Observability Framework - Phase 3: Error Tracking
# Usage: 0k-rag-health-check.sh [--period=24h|7d|30d] [--json]

set -euo pipefail

# Configuration
ERROR_LOG="${OKRAG_LOG_DIR:-${HOME}/.0k-rag/logs}/errors.jsonl"
TOKEN_LOG="${OKRAG_LOG_DIR:-${HOME}/.0k-rag/logs}/token-usage.jsonl"
TRACES_DIR="${OKRAG_LOG_DIR:-${HOME}/.0k-rag/logs}/traces"
PERIOD="24h"
OUTPUT_JSON=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --period=*)
      PERIOD="${1#*=}"
      shift
      ;;
    --json)
      OUTPUT_JSON=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Calculate time window
case $PERIOD in
  24h)
    HOURS=24
    DISPLAY_PERIOD="Last 24 hours"
    ;;
  7d)
    HOURS=$((7 * 24))
    DISPLAY_PERIOD="Last 7 days"
    ;;
  30d)
    HOURS=$((30 * 24))
    DISPLAY_PERIOD="Last 30 days"
    ;;
  *)
    echo "Invalid period: $PERIOD. Use 24h, 7d, or 30d" >&2
    exit 1
    ;;
esac

# Get cutoff timestamp (macOS compatible)
CUTOFF=$(python3 -c "from datetime import datetime, timedelta; print((datetime.now() - timedelta(hours=$HOURS)).strftime('%Y-%m-%dT%H:%M:%S'))")

# Count errors by severity (handle missing file gracefully)
if [[ -f "$ERROR_LOG" ]]; then
  ERROR_COUNT=$(jq -r --arg cutoff "$CUTOFF" 'select(.timestamp >= $cutoff and .severity == "error")' "$ERROR_LOG" 2>/dev/null | wc -l | tr -d ' ')
  WARNING_COUNT=$(jq -r --arg cutoff "$CUTOFF" 'select(.timestamp >= $cutoff and .severity == "warning")' "$ERROR_LOG" 2>/dev/null | wc -l | tr -d ' ')
  INFO_COUNT=$(jq -r --arg cutoff "$CUTOFF" 'select(.timestamp >= $cutoff and .severity == "info")' "$ERROR_LOG" 2>/dev/null | wc -l | tr -d ' ')
else
  ERROR_COUNT=0
  WARNING_COUNT=0
  INFO_COUNT=0
fi

# Count total operations (token log entries)
if [[ -f "$TOKEN_LOG" ]]; then
  TOTAL_OPS=$(jq -r --arg cutoff "$CUTOFF" 'select(.timestamp >= $cutoff)' "$TOKEN_LOG" 2>/dev/null | wc -l | tr -d ' ')
else
  TOTAL_OPS=0
fi

# Calculate error rate
if [[ $TOTAL_OPS -gt 0 ]]; then
  ERROR_RATE=$(python3 -c "print(f'{($ERROR_COUNT / $TOTAL_OPS) * 100:.2f}')")
else
  ERROR_RATE="0.00"
fi

# Determine health status
HEALTH_STATUS="🟢 EXCELLENT"
HEALTH_GRADE="A+"

if (( $(echo "$ERROR_RATE >= 0.5" | bc -l) )); then
  HEALTH_STATUS="🟢 HEALTHY"
  HEALTH_GRADE="A"
fi

if (( $(echo "$ERROR_RATE >= 1.0" | bc -l) )); then
  HEALTH_STATUS="🟡 FAIR"
  HEALTH_GRADE="B"
fi

if (( $(echo "$ERROR_RATE >= 2.0" | bc -l) )); then
  HEALTH_STATUS="🔴 NEEDS ATTENTION"
  HEALTH_GRADE="C"
fi

# Get recent errors (top 5)
if [[ -f "$ERROR_LOG" && $ERROR_COUNT -gt 0 ]]; then
  RECENT_ERRORS=$(jq -r --arg cutoff "$CUTOFF" 'select(.timestamp >= $cutoff and .severity == "error") | "\(.timestamp)|\(.source)|\(.error_message)"' "$ERROR_LOG" 2>/dev/null | tail -5)
else
  RECENT_ERRORS=""
fi

# Output format
if [[ "$OUTPUT_JSON" == "true" ]]; then
  # JSON output
  jq -n \
    --arg period "$DISPLAY_PERIOD" \
    --arg errors "$ERROR_COUNT" \
    --arg warnings "$WARNING_COUNT" \
    --arg info "$INFO_COUNT" \
    --arg total "$TOTAL_OPS" \
    --arg rate "$ERROR_RATE" \
    --arg status "$HEALTH_STATUS" \
    --arg grade "$HEALTH_GRADE" \
    '{
      period: $period,
      error_count: ($errors | tonumber),
      warning_count: ($warnings | tonumber),
      info_count: ($info | tonumber),
      total_operations: ($total | tonumber),
      error_rate_percent: ($rate | tonumber),
      health_status: $status,
      health_grade: $grade
    }'
else
  # Human-readable output
  cat <<EOF
╔════════════════════════════════════════════════════════════╗
║           0K-RAG System Health Dashboard                    ║
╚════════════════════════════════════════════════════════════╝

📅 Period: $DISPLAY_PERIOD
🏥 Health Status: $HEALTH_STATUS (Grade: $HEALTH_GRADE)

📊 Error Statistics:
────────────────────────────────────────────────────────────
   Errors:     $ERROR_COUNT
   Warnings:   $WARNING_COUNT
   Info:       $INFO_COUNT
   ─────────────────────
   Total Ops:  $TOTAL_OPS
   Error Rate: ${ERROR_RATE}%

EOF

  if [[ -n "$RECENT_ERRORS" ]]; then
    echo "🚨 Recent Errors (Top 5):"
    echo "────────────────────────────────────────────────────────────"
    echo "$RECENT_ERRORS" | while IFS='|' read -r timestamp source message; do
      # Truncate message if too long
      if [[ ${#message} -gt 60 ]]; then
        message="${message:0:57}..."
      fi
      echo "   • [${timestamp:0:19}] $source"
      echo "     └─ $message"
    done
    echo ""
  else
    echo "✅ No errors in this period!"
    echo ""
  fi

  echo "═══════════════════════════════════════════════════════════"
fi

exit 0
