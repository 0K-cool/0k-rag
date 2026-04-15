#!/usr/bin/env bash

# log-latency-trace.sh - Append latency trace to observability log
# Part of Priority 2: Observability Framework (Phase 2)
# Usage: log-latency-trace.sh --conversation-id ID --trace-id ID --operation-type TYPE --operation-name NAME --start-time NS --end-time NS

set -eo pipefail

# Phase 3: Error Tracking Integration
ERROR_LOG_SCRIPT="${OKRAG_LOG_DIR:-${HOME}/.0k-rag/logs}/../scripts/vex-log-error.sh"

# Error handler
error_handler() {
  local exit_code=$?
  local line_number=$1
  local command="$BASH_COMMAND"

  # Log error if vex-log-error.sh is available
  if [[ -x "$ERROR_LOG_SCRIPT" ]]; then
    "$ERROR_LOG_SCRIPT" \
      --severity=error \
      --source="log-latency-trace.sh" \
      --error-type="execution_failure" \
      --message="Latency trace logging failed: $command" \
      --exit-code="$exit_code" \
      --context="{\"line\": $line_number, \"command\": \"$command\", \"operation\": \"${OPERATION_TYPE:-unknown}/${OPERATION_NAME:-unknown}\"}" \
      --resolution="Check log directory permissions and Python availability" || true
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
LOG_DIR="${PROJECT_ROOT}/.claude/logs/traces"

# Parse arguments
CONVERSATION_ID=""
TRACE_ID=""
PARENT_TRACE_ID=""
OPERATION_TYPE=""
OPERATION_NAME=""
START_TIME_NS=""
END_TIME_NS=""
METADATA="{}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --conversation-id)
            CONVERSATION_ID="$2"
            shift 2
            ;;
        --trace-id)
            TRACE_ID="$2"
            shift 2
            ;;
        --parent-trace-id)
            PARENT_TRACE_ID="$2"
            shift 2
            ;;
        --operation-type)
            OPERATION_TYPE="$2"
            shift 2
            ;;
        --operation-name)
            OPERATION_NAME="$2"
            shift 2
            ;;
        --start-time)
            START_TIME_NS="$2"
            shift 2
            ;;
        --end-time)
            END_TIME_NS="$2"
            shift 2
            ;;
        --metadata)
            METADATA="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Validate required fields
if [[ -z "$CONVERSATION_ID" ]]; then
    echo "Error: --conversation-id is required" >&2
    exit 1
fi
if [[ -z "$TRACE_ID" ]]; then
    echo "Error: --trace-id is required" >&2
    exit 1
fi
if [[ -z "$OPERATION_TYPE" ]]; then
    echo "Error: --operation-type is required" >&2
    exit 1
fi
if [[ -z "$OPERATION_NAME" ]]; then
    echo "Error: --operation-name is required" >&2
    exit 1
fi
if [[ -z "$START_TIME_NS" ]]; then
    echo "Error: --start-time is required" >&2
    exit 1
fi
if [[ -z "$END_TIME_NS" ]]; then
    echo "Error: --end-time is required" >&2
    exit 1
fi

# Calculate duration in milliseconds
DURATION_NS=$((END_TIME_NS - START_TIME_NS))
DURATION_MS=$((DURATION_NS / 1000000))

# Convert nanosecond timestamps to ISO 8601 format
# Using Python for reliable timestamp conversion
START_TIME_ISO=$(python3 -c "import datetime; print(datetime.datetime.fromtimestamp($START_TIME_NS / 1000000000.0, tz=datetime.timezone.utc).isoformat())")
END_TIME_ISO=$(python3 -c "import datetime; print(datetime.datetime.fromtimestamp($END_TIME_NS / 1000000000.0, tz=datetime.timezone.utc).isoformat())")

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"
chmod 700 "$LOG_DIR"

# Log file for this conversation
LOG_FILE="${LOG_DIR}/${CONVERSATION_ID}.jsonl"

# Build parent_trace_id field (null if not provided)
if [[ -z "$PARENT_TRACE_ID" ]]; then
    PARENT_FIELD="null"
else
    PARENT_FIELD="\"$PARENT_TRACE_ID\""
fi

# Create JSONL entry
LOG_ENTRY=$(cat <<EOF
{"conversation_id":"$CONVERSATION_ID","trace_id":"$TRACE_ID","parent_trace_id":$PARENT_FIELD,"operation_type":"$OPERATION_TYPE","operation_name":"$OPERATION_NAME","start_time":"$START_TIME_ISO","end_time":"$END_TIME_ISO","duration_ms":$DURATION_MS,"metadata":$METADATA}
EOF
)

# Atomic append to log file
echo "$LOG_ENTRY" >> "$LOG_FILE"

# Set restrictive permissions
chmod 600 "$LOG_FILE"

# Output duration for caller (useful for scripts)
echo "$DURATION_MS"
