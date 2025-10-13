#!/bin/bash
# ======================================================
# 🧠 Bear MCP 回归测试脚本（含自动验证 + 错误日志）
# ======================================================

BASE_URL="http://localhost:8000"
LOG_FILE="mcp_test_fail.log"
TMP_FILE=$(mktemp)
PASS_COUNT=0
FAIL_COUNT=0

# 清空旧日志
> "$LOG_FILE"

# 打印标题
echo ""
echo "🚀 Running Bear MCP regression tests..."
echo "Server: $BASE_URL"
echo "--------------------------------------------"

# 定义测试函数
run_test() {
  local name="$1"
  local path="$2"
  local payload="$3"
  local key="$4"

  echo "▶️  Testing $name ($path)"
  response=$(curl -s -X POST "$BASE_URL/$path" -H "Content-Type: application/json" -d "$payload")
  echo "$response" > "$TMP_FILE"

  # 检查返回码与关键字段
  if echo "$response" | jq -e ".${key}" >/dev/null 2>&1; then
    echo "✅ PASSED — found key: ${key}"
    ((PASS_COUNT++))
  else
    echo "❌ FAILED — missing key: ${key}"
    ((FAIL_COUNT++))
    {
      echo "--------------------------------------------"
      echo "[❌ FAIL] $name ($path)"
      echo "Payload: $payload"
      echo "Timestamp: $(date)"
      echo "Response:"
      cat "$TMP_FILE"
      echo ""
    } >> "$LOG_FILE"
  fi

  echo "--------------------------------------------"
}

# ==========================================
# 🔍 逐项测试
# ==========================================

run_test "🧭 Search Notes" \
  "search_notes" \
  '{"args":{"query":"每日复盘"}}' \
  "notes"

run_test "🧠 Retrieve for RAG" \
  "retrieve_for_rag" \
  '{"args":{"query":"自由由生长"}}' \
  "context"

run_test "🔮 Daily Insight Context" \
  "daily_insight_context" \
  '{"args":{"date":"2025-10-13"}}' \
  "system_prompt"

run_test "🏷️  Get Tags" \
  "get_tags" \
  '{"args":{}}' \
  "tags"

run_test "📄 Get Note" \
  "get_note" \
  '{"args":{"id":"537FB3B5-9086-443C-8E8D-E040CD82C2A8"}}' \
  "note"

# ==========================================
# 🧾 测试汇总
# ==========================================
echo ""
echo "--------------------------------------------"
echo "✅ Passed: $PASS_COUNT | ❌ Failed: $FAIL_COUNT"
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "📄 Failure log saved to: $LOG_FILE"
else
  echo "🎉 All tests passed successfully!"
fi
echo "--------------------------------------------"

# 清理临时文件
rm -f "$TMP_FILE"
