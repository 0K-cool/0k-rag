#!/usr/bin/env bash

# log-token-usage.sh - Append token usage to observability log
# Part of Priority 2: Observability Framework (Phase 1)
# Usage: log-token-usage.sh --conversation-id ID --operation-type TYPE --operation-name NAME --input-tokens N --output-tokens N --model MODEL

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
      --source="log-token-usage.sh" \
      --error-type="execution_failure" \
      --message="Token logging failed: $command" \
      --exit-code="$exit_code" \
      --context="{\"line\": $line_number, \"command\": \"$command\", \"operation\": \"${OPERATION_TYPE:-unknown}/${OPERATION_NAME:-unknown}\"}" \
      --resolution="Check log permissions and disk space" || true
  fi

  exit $exit_code
}

# Set trap for errors
trap 'error_handler ${LINENO}' ERR

# Configuration
# PAI_DIR points to .claude directory, so we need to go up one level for the project root
if [[ -n "${PAI_DIR:-}" ]]; then
    PROJECT_ROOT="$(dirname "$PAI_DIR")"
else
    PROJECT_ROOT="."
fi
LOG_DIR="${PROJECT_ROOT}/.claude/logs"
LOG_FILE="${LOG_DIR}/token-usage.jsonl"

# Anthropic pricing (as of January 2026)
# Sonnet 4.5: $3/MTok input, $15/MTok output
# Haiku: $0.80/MTok input, $4/MTok output
# Opus 4.5: $15/MTok input, $75/MTok output
get_input_price() {
    case "$1" in
        "claude-sonnet-4-5-20250929")
            echo "0.000003"
            ;;
        "claude-haiku-3-5-20241022")
            echo "0.0000008"
            ;;
        "claude-opus-4-5-20251101")
            echo "0.000015"
            ;;
        *)
            echo "0.000003"  # Default to Sonnet pricing
            ;;
    esac
}

get_output_price() {
    case "$1" in
        "claude-sonnet-4-5-20250929")
            echo "0.000015"
            ;;
        "claude-haiku-3-5-20241022")
            echo "0.000004"
            ;;
        "claude-opus-4-5-20251101")
            echo "0.000075"
            ;;
        *)
            echo "0.000015"  # Default to Sonnet pricing
            ;;
    esac
}

# Parse arguments
CONVERSATION_ID=""
OPERATION_TYPE=""
OPERATION_NAME=""
INPUT_TOKENS=0
OUTPUT_TOKENS=0
MODEL="claude-sonnet-4-5-20250929"
ESTIMATED=false
METADATA="{}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --conversation-id)
            CONVERSATION_ID="$2"
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
        --input-tokens)
            INPUT_TOKENS="$2"
            shift 2
            ;;
        --output-tokens)
            OUTPUT_TOKENS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --estimated)
            ESTIMATED=true
            shift
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
if [[ -z "$OPERATION_TYPE" ]]; then
    echo "Error: --operation-type is required" >&2
    exit 1
fi
if [[ -z "$OPERATION_NAME" ]]; then
    echo "Error: --operation-name is required" >&2
    exit 1
fi

# Calculate total tokens
TOTAL_TOKENS=$((INPUT_TOKENS + OUTPUT_TOKENS))

# Calculate cost
INPUT_PRICE_VAL=$(get_input_price "$MODEL")
OUTPUT_PRICE_VAL=$(get_output_price "$MODEL")
INPUT_COST=$(echo "$INPUT_TOKENS * $INPUT_PRICE_VAL" | bc -l)
OUTPUT_COST=$(echo "$OUTPUT_TOKENS * $OUTPUT_PRICE_VAL" | bc -l)
TOTAL_COST=$(echo "$INPUT_COST + $OUTPUT_COST" | bc -l)

# Format cost to 6 decimal places
TOTAL_COST=$(printf "%.6f" "$TOTAL_COST")

# Get timestamp in ISO 8601 format (macOS compatible)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"
chmod 700 "$LOG_DIR"

# Create JSONL entry
LOG_ENTRY=$(cat <<EOF
{"timestamp":"$TIMESTAMP","conversation_id":"$CONVERSATION_ID","operation_type":"$OPERATION_TYPE","operation_name":"$OPERATION_NAME","tokens":{"input":$INPUT_TOKENS,"output":$OUTPUT_TOKENS,"total":$TOTAL_TOKENS},"model":"$MODEL","cost_usd":$TOTAL_COST,"estimated":$ESTIMATED,"metadata":$METADATA}
EOF
)

# Atomic append to log file
echo "$LOG_ENTRY" >> "$LOG_FILE"

# Set restrictive permissions
chmod 600 "$LOG_FILE"

# Optional: Echo to stderr for debugging (commented out for production)
# echo "[TOKEN LOG] $OPERATION_TYPE/$OPERATION_NAME: ${TOTAL_TOKENS} tokens (\$${TOTAL_COST})" >&2
