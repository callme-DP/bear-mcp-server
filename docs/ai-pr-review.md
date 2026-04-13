# Bear MCP PR 自动审查（Codex Agent）

## 1. 已配置内容
- 工作流：`.github/workflows/pr-review.yml`
- 审查脚本：`scripts/analyze-pr.mjs`
- 触发时机：PR `opened / synchronize / reopened`
- 行为：拉取 PR diff -> 调用 OpenAI -> 在 PR 评论区创建或更新一条审查评论

## 2. GitHub 仓库配置（必须）
在仓库 `Settings -> Secrets and variables -> Actions` 里配置：

- `Secrets`
  - `OPENAI_API_KEY`：你的 OpenAI API Key
- `Variables`（可选）
  - `REVIEW_MODEL`：默认 `gpt-4.1`，可改成你要的模型

说明：
- 工作流默认使用 `${{ github.token }}` 发表评论，不需要额外配置 `GITHUB_TOKEN` Secret。
- 对 fork PR 默认不运行（避免泄露 Secret）。

## 3. Codex 侧建议配置
如果你同时在用 Codex Connector（PR 行内建议）和本仓库 Action（Summary 评论），建议分工：

- Codex Connector：行内审查建议（inline）
- GitHub Action：结构化总评（summary）

建议你在 Codex/Connector 里保持固定审查模板：
- Correctness
- Security
- Performance
- Maintainability
- Final verdict

这样两条审查链路不会互相重复。

可直接放到 Codex 审查提示词里的模板：

```md
You are reviewing a Bear MCP Server pull request.
Focus on:
1) Correctness
2) Security
3) Performance
4) Maintainability

Output must include:
- Severity (High/Medium/Low)
- File path
- Why it is a problem
- Concrete fix suggestion
- Final verdict: Approve or Request changes
```

## 4. 验证步骤
本地 dry-run（不访问 GitHub/OpenAI）：

```bash
cat > /tmp/mock-pr-event.json <<'JSON'
{"pull_request":{"number":123}}
JSON

cat > /tmp/mock-pr.diff <<'DIFF'
diff --git a/src/example.js b/src/example.js
index 1111111..2222222 100644
--- a/src/example.js
+++ b/src/example.js
@@ -1,3 +1,4 @@
 const a = 1;
+const b = 2;
 export { a };
DIFF

GITHUB_TOKEN=dummy \
GITHUB_REPOSITORY=owner/repo \
GITHUB_EVENT_PATH=/tmp/mock-pr-event.json \
PR_DIFF_PATH=/tmp/mock-pr.diff \
DRY_RUN=1 \
MOCK_REVIEW_TEXT='## 1. 变更摘要\n- mock\n\n## 2. 主要问题（按严重级别排序）\n- 未发现\n\n## 3. 风险与回归点\n- 无\n\n## 4. 建议修改\n- 无\n\n## 5. 审查结论\n- Approve' \
node scripts/analyze-pr.mjs
```

线上验证（真实 PR）：
1. 新建一个测试分支，制造一个小改动并发起 PR。
2. 确认 PR 页面出现工作流 `PR Review (Codex Agent)`。
3. 查看日志中 `Run Codex PR review` 步骤：
   - 能看到 `开始审查 owner/repo#<number>`
   - 能看到 `评论已创建` 或 `评论已更新`
4. 在同一 PR 再 push 一次，确认旧评论被更新而不是重复新增。

## 5. 常见问题
- 报 `OPENAI_API_KEY 未配置`：检查仓库 Secret 名称是否正确。
- 报 GitHub API 403：确认仓库 Actions 权限未被限制，且 PR 不是 fork。
- 评论内容为空：检查 OpenAI Key 额度、模型名、以及 API 调用日志。
