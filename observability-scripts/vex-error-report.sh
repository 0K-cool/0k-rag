#!/bin/bash
# vex-error-report.sh - Generate monthly error analysis report
# Part of Vex Observability Framework - Phase 3: Error Tracking
# Usage: vex-error-report.sh [YYYY-MM] [--output=/path/to/report.md]

set -euo pipefail

# Configuration
ERROR_LOG="${OKRAG_LOG_DIR:-${HOME}/.0k-rag/logs}/errors.jsonl"
YEAR_MONTH="${1:-$(date +%Y-%m)}"
OUTPUT_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output=*)
      OUTPUT_FILE="${1#*=}"
      shift
      ;;
    [0-9][0-9][0-9][0-9]-[0-9][0-9])
      YEAR_MONTH="$1"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Validate year-month format
if [[ ! "$YEAR_MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
  echo "Error: Invalid format. Use YYYY-MM (e.g., 2026-01)" >&2
  exit 1
fi

# Check if error log exists
if [[ ! -f "$ERROR_LOG" ]]; then
  echo "Error: Error log not found at $ERROR_LOG" >&2
  exit 1
fi

# Generate report header
REPORT=$(cat <<EOF
# 0K-RAG Error Report - $YEAR_MONTH

**Generated:** $(date +"%Y-%m-%d %H:%M:%S %Z")
**Period:** $YEAR_MONTH

---

## Summary

EOF
)

# Count errors by severity
ERROR_COUNT=$(jq -r --arg ym "$YEAR_MONTH" 'select(.timestamp | startswith($ym)) | select(.severity == "error")' "$ERROR_LOG" 2>/dev/null | wc -l | tr -d ' ')
WARNING_COUNT=$(jq -r --arg ym "$YEAR_MONTH" 'select(.timestamp | startswith($ym)) | select(.severity == "warning")' "$ERROR_LOG" 2>/dev/null | wc -l | tr -d ' ')
INFO_COUNT=$(jq -r --arg ym "$YEAR_MONTH" 'select(.timestamp | startswith($ym)) | select(.severity == "info")' "$ERROR_LOG" 2>/dev/null | wc -l | tr -d ' ')
TOTAL_EVENTS=$((ERROR_COUNT + WARNING_COUNT + INFO_COUNT))

REPORT+=$(cat <<EOF
- **Errors:** $ERROR_COUNT
- **Warnings:** $WARNING_COUNT
- **Info:** $INFO_COUNT
- **Total Events:** $TOTAL_EVENTS

EOF
)

# Determine health grade
if [[ $ERROR_COUNT -eq 0 ]]; then
  HEALTH_GRADE="🟢 EXCELLENT (No errors)"
elif [[ $ERROR_COUNT -lt 10 ]]; then
  HEALTH_GRADE="🟢 HEALTHY ($ERROR_COUNT errors)"
elif [[ $ERROR_COUNT -lt 50 ]]; then
  HEALTH_GRADE="🟡 FAIR ($ERROR_COUNT errors)"
else
  HEALTH_GRADE="🔴 NEEDS ATTENTION ($ERROR_COUNT errors)"
fi

REPORT+=$(cat <<EOF
- **Health Grade:** $HEALTH_GRADE

---

## Top Error Sources

EOF
)

# Top 5 error sources
if [[ $ERROR_COUNT -gt 0 ]]; then
  TOP_SOURCES=$(jq -r --arg ym "$YEAR_MONTH" 'select(.timestamp | startswith($ym)) | select(.severity == "error") | .source' "$ERROR_LOG" 2>/dev/null | sort | uniq -c | sort -rn | head -5)

  REPORT+=$(cat <<EOF
\`\`\`
$TOP_SOURCES
\`\`\`

EOF
  )
else
  REPORT+="No errors in this period.

"
fi

REPORT+=$(cat <<EOF
---

## Top Error Types

EOF
)

# Top 5 error types
if [[ $ERROR_COUNT -gt 0 ]]; then
  TOP_TYPES=$(jq -r --arg ym "$YEAR_MONTH" 'select(.timestamp | startswith($ym)) | select(.severity == "error") | .error_type' "$ERROR_LOG" 2>/dev/null | sort | uniq -c | sort -rn | head -5)

  REPORT+=$(cat <<EOF
\`\`\`
$TOP_TYPES
\`\`\`

EOF
  )
else
  REPORT+="No errors in this period.

"
fi

REPORT+=$(cat <<EOF
---

## Recent Critical Errors (Last 10)

EOF
)

# Last 10 critical errors
if [[ $ERROR_COUNT -gt 0 ]]; then
  RECENT=$(jq -r --arg ym "$YEAR_MONTH" 'select(.timestamp | startswith($ym)) | select(.severity == "error") | "- **[\(.timestamp)]** \(.source): \(.error_message)"' "$ERROR_LOG" 2>/dev/null | tail -10)

  REPORT+=$(cat <<EOF
$RECENT

EOF
  )
else
  REPORT+="No errors in this period.

"
fi

REPORT+=$(cat <<EOF
---

## Recommendations

EOF
)

# Generate recommendations based on patterns
if [[ $ERROR_COUNT -gt 50 ]]; then
  REPORT+="- ⚠️ **High error count ($ERROR_COUNT)** - Review instrumentation and error handling
"
fi

if [[ $WARNING_COUNT -gt 100 ]]; then
  REPORT+="- ⚠️ **High warning count ($WARNING_COUNT)** - Investigate degraded performance
"
fi

if [[ $ERROR_COUNT -eq 0 ]]; then
  REPORT+="- ✅ **Zero errors** - Excellent system health!
"
fi

if [[ $TOTAL_EVENTS -eq 0 ]]; then
  REPORT+="- ℹ️  **No events logged** - System may not be fully instrumented
"
fi

REPORT+=$(cat <<EOF

---

## Error Details

EOF
)

# Error breakdown by source
if [[ $ERROR_COUNT -gt 0 ]]; then
  SOURCES=$(jq -r --arg ym "$YEAR_MONTH" 'select(.timestamp | startswith($ym)) | select(.severity == "error") | .source' "$ERROR_LOG" 2>/dev/null | sort -u)

  for source in $SOURCES; do
    SOURCE_COUNT=$(jq -r --arg ym "$YEAR_MONTH" --arg src "$source" 'select(.timestamp | startswith($ym)) | select(.severity == "error") | select(.source == $src)' "$ERROR_LOG" 2>/dev/null | wc -l | tr -d ' ')

    REPORT+="### $source ($SOURCE_COUNT errors)

"

    SOURCE_ERRORS=$(jq -r --arg ym "$YEAR_MONTH" --arg src "$source" 'select(.timestamp | startswith($ym)) | select(.severity == "error") | select(.source == $src) | "- **[\(.timestamp)]** \(.error_type): \(.error_message)"' "$ERROR_LOG" 2>/dev/null | head -5)

    REPORT+="$SOURCE_ERRORS

"
  done
fi

REPORT+=$(cat <<EOF
---

**End of Report**
EOF
)

# Output
if [[ -n "$OUTPUT_FILE" ]]; then
  echo "$REPORT" > "$OUTPUT_FILE"
  echo "✅ Report saved to: $OUTPUT_FILE"
else
  echo "$REPORT"
fi

exit 0
