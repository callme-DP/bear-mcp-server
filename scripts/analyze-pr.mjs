import fs from "node:fs";

const REQUIRED_ENV = ["GITHUB_TOKEN", "GITHUB_REPOSITORY", "GITHUB_EVENT_PATH"];
const REVIEW_MARKER = "<!-- codex-pr-review -->";
const REVIEW_HEADER = "## AI 自动代码审查（Codex Agent）";

function readJson(path) {
  return JSON.parse(fs.readFileSync(path, "utf8"));
}

function assertEnv() {
  const missing = REQUIRED_ENV.filter((key) => !process.env[key]);
  if (missing.length > 0) {
    throw new Error(`缺少环境变量: ${missing.join(", ")}`);
  }
}

async function githubRequest(path, { method = "GET", body, accept } = {}) {
  const res = await fetch(`https://api.github.com${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${process.env.GITHUB_TOKEN}`,
      "Content-Type": "application/json",
      Accept: accept || "application/vnd.github+json",
      "User-Agent": "bear-mcp-pr-review-bot",
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`GitHub API ${method} ${path} 失败: ${res.status} ${text}`);
  }

  if (res.status === 204) return null;
  return res.json();
}

async function fetchPrDiff(repo, prNumber) {
  if (process.env.PR_DIFF_PATH) {
    return fs.readFileSync(process.env.PR_DIFF_PATH, "utf8");
  }

  const res = await fetch(`https://api.github.com/repos/${repo}/pulls/${prNumber}`, {
    headers: {
      Authorization: `Bearer ${process.env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github.v3.diff",
      "User-Agent": "bear-mcp-pr-review-bot",
    },
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`获取 PR diff 失败: ${res.status} ${text}`);
  }

  return res.text();
}

function buildPrompt(diff, repo, prNumber) {
  return `你是 Bear MCP Server 项目的资深 Reviewer。\n\n请基于下面 PR diff 做审查，并按固定结构输出。\n\n输出格式（必须保持下面 5 个标题）：\n\n## 1. 变更摘要\n- 列出本次改动涉及的模块\n\n## 2. 主要问题（按严重级别排序）\n- 每条问题给出：严重级别（High/Medium/Low）、文件路径、原因\n\n## 3. 风险与回归点\n- 列出可能引发回归的问题\n\n## 4. 建议修改\n- 给出可执行的改法\n\n## 5. 审查结论\n- 只能二选一：Approve / Request changes\n\n审查边界：\n- 重点看正确性、稳定性、安全性、性能与可维护性\n- 避免空泛建议；如果没有明显问题，明确写“未发现阻塞问题”\n\n仓库：${repo}\nPR：#${prNumber}\n\n以下是 diff（截断至 18000 字符）：\n\n${diff.slice(0, 18000)}`;
}

async function callOpenAI(prompt) {
  if (process.env.MOCK_REVIEW_TEXT) {
    return process.env.MOCK_REVIEW_TEXT;
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.log("OPENAI_API_KEY 未配置，跳过 AI 审查。");
    return null;
  }

  const model = process.env.REVIEW_MODEL || "gpt-4.1";
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      temperature: 0.1,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OpenAI API 调用失败: ${response.status} ${text}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error("OpenAI 返回内容为空");
  }

  return content;
}

async function upsertComment(repo, prNumber, body) {
  const comments = await githubRequest(`/repos/${repo}/issues/${prNumber}/comments?per_page=100`);
  const existing = comments.find((item) => item.body?.includes(REVIEW_MARKER));

  if (existing) {
    await githubRequest(`/repos/${repo}/issues/comments/${existing.id}`, {
      method: "PATCH",
      body: { body },
    });
    return { mode: "updated", id: existing.id };
  }

  const created = await githubRequest(`/repos/${repo}/issues/${prNumber}/comments`, {
    method: "POST",
    body: { body },
  });
  return { mode: "created", id: created.id };
}

async function main() {
  assertEnv();

  const event = readJson(process.env.GITHUB_EVENT_PATH);
  const prNumber = event?.pull_request?.number;
  if (!prNumber) {
    throw new Error("当前事件不是 pull_request，无法定位 PR number");
  }

  const repo = process.env.GITHUB_REPOSITORY;

  console.log(`开始审查 ${repo}#${prNumber}`);
  const diff = await fetchPrDiff(repo, prNumber);
  if (!diff.trim()) {
    console.log("PR diff 为空，跳过。");
    return;
  }

  const prompt = buildPrompt(diff, repo, prNumber);
  const review = await callOpenAI(prompt);
  if (!review) {
    return;
  }

  const body = [
    REVIEW_MARKER,
    REVIEW_HEADER,
    "",
    review,
    "",
    "---",
    `模型：\`${process.env.REVIEW_MODEL || "gpt-4.1"}\``,
    "由 GitHub Actions 自动生成。",
  ].join("\n");

  if (process.env.DRY_RUN === "1") {
    console.log("DRY_RUN=1，跳过 GitHub 评论更新。");
    console.log(body.slice(0, 600));
    return;
  }

  const result = await upsertComment(repo, prNumber, body);
  console.log(`评论${result.mode === "updated" ? "已更新" : "已创建"}: ${result.id}`);
}

main().catch((err) => {
  console.error(`执行失败: ${err.message}`);
  process.exit(1);
});
