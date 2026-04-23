#!/usr/bin/env bash
# Readable wrapper around the LangGraph dev server.
# Usage: ./chat.sh "your question"
#   LANGGRAPH_URL overrides the base URL (default http://localhost:2024)

set -euo pipefail

URL="${LANGGRAPH_URL:-http://localhost:2024}"
MSG="${1:-}"

if [[ -z "$MSG" ]]; then
  echo "usage: $0 \"your message\"" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (brew install jq)" >&2
  exit 1
fi

payload=$(jq -n --arg m "$MSG" '{
  assistant_id: "agent",
  input: { messages: [{ role: "user", content: $m }] }
}')

curl -sS --fail-with-body -X POST "$URL/runs/wait" \
  -H "content-type: application/json" \
  -d "$payload" \
| jq -r '
    def as_text:
      if type == "string" then .
      elif type == "array" then
        [.[] | if type == "object" then (.text // "") else tostring end] | join("")
      elif . == null then ""
      else tostring end;

    def truncate($n):
      if (. | length) > $n then (.[0:$n] + "…") else . end;

    (.messages // [])[] |
      if .type == "human" then
        "\n[36mYou:[0m " + (.content | as_text)
      elif .type == "ai" then
        (if ((.tool_calls // []) | length) > 0 then
           "[33m→ tool call:[0m " +
           ((.tool_calls // []) | map("\(.name)(\(.args | tostring))") | join(", "))
         else
           "[32mAssistant:[0m " + (.content | as_text)
         end)
      elif .type == "tool" then
        "[90m  ↳ [\(.name // "tool")]:[0m " + ((.content | as_text) | truncate(400))
      else empty end
  '
