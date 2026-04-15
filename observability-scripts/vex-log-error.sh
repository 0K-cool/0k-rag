#!/bin/bash
# vex-log-error.sh - Centralized error logging utility
# Part of Vex Observability Framework - Phase 3: Error Tracking
# Usage: vex-log-error.sh --severity=error --source="script.sh" --message="Error occurred" [options]

set -euo pipefail

# Configuration
ERROR_LOG="${OKRAG_LOG_DIR:-${HOME}/.0k-rag/logs}/errors.jsonl"
CONVERSATION_ID="${CONVERSATION_ID:-unknown}"

# Parse arguments
SEVERITY=""
SOURCE=""
MESSAGE=""
ERROR_TYPE=""
EXIT_CODE="0"
CONTEXT="{}"
STACK_TRACE=""
RESOLUTION_HINT=""
RECOVERED="false"

while [[ $# -gt 0 ]]; do
  case $1 in
    --severity=*)
      SEVERITY="${1#*=}"
      shift
      ;;
    --source=*)
      SOURCE="${1#*=}"
      shift
      ;;
    --message=*)
      MESSAGE="${1#*=}"
      shift
      ;;
    --error-type=*)
      ERROR_TYPE="${1#*=}"
      shift
      ;;
    --exit-code=*)
      EXIT_CODE="${1#*=}"
      shift
      ;;
    --context=*)
      CONTEXT="${1#*=}"
      shift
      ;;
    --stack-trace=*)
      STACK_TRACE="${1#*=}"
      shift
      ;;
    --resolution=*)
      RESOLUTION_HINT="${1#*=}"
      shift
      ;;
    --recovered)
      RECOVERED="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# Validate required fields
if [[ -z "$SEVERITY" || -z "$SOURCE" || -z "$MESSAGE" ]]; then
  echo "Error: --severity, --source, and --message are required" >&2
  echo "Usage: vex-log-error.sh --severity=error --source=script.sh --message=\"Error message\" [options]" >&2
  exit 1
fi

# Validate severity level
if [[ ! "$SEVERITY" =~ ^(error|warning|info)$ ]]; then
  echo "Error: severity must be 'error', 'warning', or 'info'" >&2
  exit 1
fi

# Create log directory if not exists
mkdir -p "$(dirname "$ERROR_LOG")"

# Generate timestamp with timezone (AST) - macOS compatible
# Use Python for microseconds since macOS date doesn't support %N
TIMESTAMP=$(python3 -c "from datetime import datetime; print(datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f%z'))")

# Build JSON log entry (compact for JSONL format)
LOG_ENTRY=$(jq -nc \
  --arg ts "$TIMESTAMP" \
  --arg conv "$CONVERSATION_ID" \
  --arg sev "$SEVERITY" \
  --arg src "$SOURCE" \
  --arg type "$ERROR_TYPE" \
  --arg msg "$MESSAGE" \
  --arg code "$EXIT_CODE" \
  --argjson ctx "$CONTEXT" \
  --arg trace "$STACK_TRACE" \
  --arg hint "$RESOLUTION_HINT" \
  --argjson rec "$RECOVERED" \
  '{
    timestamp: $ts,
    conversation_id: $conv,
    severity: $sev,
    source: $src,
    error_type: $type,
    error_message: $msg,
    exit_code: ($code | tonumber),
    context: $ctx,
    stack_trace: $trace,
    resolution_hint: $hint,
    recovered: $rec
  }')

# Append to log file
echo "$LOG_ENTRY" >> "$ERROR_LOG"

# Output to stderr for immediate visibility (only if not recovered)
if [[ "$RECOVERED" == "false" ]]; then
  if [[ "$SEVERITY" == "error" ]]; then
    echo "❌ ERROR logged: $MESSAGE" >&2
  elif [[ "$SEVERITY" == "warning" ]]; then
    echo "⚠️  WARNING logged: $MESSAGE" >&2
  else
    echo "ℹ️  INFO logged: $MESSAGE" >&2
  fi
fi

exit 0
