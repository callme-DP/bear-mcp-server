import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { Planner } from "./planner.js";
import { Executor } from "./executor.js";
import { BearTools } from "./tools.js";
import { Memory } from "./memory.js";
import { Validator } from "./validator.js";

function extractTag(text) {
  if (!text) return null;
  const regexList = [
    /包含[“"'‘]?([\w\-\u4e00-\u9fa5]+)[”"'’]?标签/,
    /标签[是为]?\s*[“"'‘]?([\w\-\u4e00-\u9fa5]+)[”"'’]?/,
    /tag\s*[:=]\s*([\w\-]+)/i,
  ];
  for (const reg of regexList) {
    const match = text.match(reg);
    if (match?.[1]) return match[1];
  }
  return null;
}

function parseControl(message) {
  const text = (message || "").trim();
  if (!text) return { cmd: "continue" };
  if (text === "继续") return { cmd: "continue" };
  if (text === "结束") return { cmd: "stop" };

  if (text.startsWith("调整")) {
    const payload = text.replace(/^调整[:：]?\s*/, "").trim();
    return { cmd: "adjust", payload };
  }

  return { cmd: "adjust", payload: text };
}

async function callOpenAI(messages, { model = process.env.REVIEW_MODEL || "gpt-4.1" } = {}) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return null;

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      messages,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`OpenAI 调用失败: ${res.status} ${text}`);
  }

  const data = await res.json();
  return data?.choices?.[0]?.message?.content || null;
}

class BossAgentSession {
  constructor(goal) {
    this.goal = goal;
    this.memory = new Memory();
    this.planner = new Planner();
    this.tools = new BearTools();
    this.validator = new Validator();
    this.executor = new Executor(this.tools, this.memory, { maxRetries: 1 });
    this.round = 0;
    this.maxRounds = 5;
    this.state = {
      task: goal,
      currentTag: extractTag(goal) || "AI",
      noteLimit: 5,
      notes: [],
      noteDetails: [],
      finalOutput: "",
    };
    this.chatHistory = [{ role: "boss", content: goal }];
  }

  async init() {
    await this.tools.init();
  }

  applyBossAdjustment(message) {
    this.chatHistory.push({ role: "boss", content: message });

    const foundTag = extractTag(message);
    if (foundTag) {
      this.state.currentTag = foundTag;
    }

    this.state.task = `${this.goal}\n补充要求：${message}`;
  }

  async runRound() {
    this.round += 1;
    const loop = this.round;
    console.log(`\n[boss-agent] round=${loop} think -> act -> observe`);

    this.memory.add({ phase: "think", loop, goal: this.state.task });
    const plan = this.planner.createPlan(this.state.task, this.state);
    this.memory.add({ phase: "think", loop, plan });

    try {
      await this.executor.executePlan(plan, this.state);
    } catch (err) {
      this.memory.add({ phase: "observe", loop, status: "loop_error", error: err.message });
      console.log(`[boss-agent] round=${loop} error=${err.message}`);
    }

    const check = this.validator.validate(this.state);
    this.memory.add({ phase: "observe", loop, validation: check });

    const report = await this.buildReport(plan, check);
    this.chatHistory.push({ role: "agent", content: report });

    return {
      done: check.ok,
      check,
      report,
      output: this.state.finalOutput,
      round: this.round,
      maxRounds: this.maxRounds,
    };
  }

  async buildReport(plan, check) {
    const fallback = [
      "【Agent 本轮汇报】",
      `- 目标：${this.goal}`,
      `- 当前标签：${this.state.currentTag}`,
      `- 执行轮次：${this.round}/${this.maxRounds}`,
      `- 进度状态：${check.ok ? "已达成可交付结果" : "未达成，建议继续循环"}`,
      `- 本轮计划步数：${plan.length}`,
      `- 结果预览：${(this.state.finalOutput || "(空)").slice(0, 240)}`,
      `- 建议下一步：${check.ok ? "请老板确认是否结束" : "请回复“继续”或“调整 + 新要求”"}`,
    ].join("\n");

    try {
      const aiText = await callOpenAI([
        {
          role: "system",
          content:
            "你是项目执行型员工，向老板汇报。请用中文，少讲实现细节，重点汇报目标、进度、风险、下一步。",
        },
        {
          role: "user",
          content: JSON.stringify(
            {
              goal: this.goal,
              currentTag: this.state.currentTag,
              round: this.round,
              maxRounds: this.maxRounds,
              done: check.ok,
              checkReason: check.reason,
              outputPreview: (this.state.finalOutput || "").slice(0, 600),
            },
            null,
            2
          ),
        },
      ]);

      return aiText || fallback;
    } catch (err) {
      return `${fallback}\n- GPT 汇报降级：${err.message}`;
    }
  }
}

async function main() {
  const rl = readline.createInterface({ input, output });

  try {
    const initialGoal = (process.argv.slice(2).join(" ") || "").trim();
    const goal = initialGoal || (await rl.question("老板目标：")).trim();

    if (!goal) {
      console.log("未提供目标，结束。");
      return;
    }

    console.log("\n[boss-agent] 已接收目标，开始执行。可用指令：继续 / 调整 <要求> / 结束\n");

    const session = new BossAgentSession(goal);
    await session.init();

    let stop = false;

    while (!stop && session.round < session.maxRounds) {
      const result = await session.runRound();
      console.log(`\n${result.report}\n`);

      if (result.done) {
        console.log("[boss-agent] 当前结果已通过校验。请输入“结束”完成，或“继续/调整”迭代。\n");
      }

      let cmdText = "";
      try {
        cmdText = (await rl.question("老板指令：")).trim();
      } catch (err) {
        if ((err?.message || "").includes("readline was closed")) {
          console.log("[boss-agent] 输入流已关闭，自动结束会话。");
          stop = true;
          break;
        }
        throw err;
      }
      const parsed = parseControl(cmdText);

      if (parsed.cmd === "stop") {
        stop = true;
        break;
      }

      if (parsed.cmd === "adjust" && parsed.payload) {
        session.applyBossAdjustment(parsed.payload);
        console.log(`[boss-agent] 已接收调整：${parsed.payload}`);
      }
    }

    console.log("\n[boss-agent] 会话结束");
    console.log("\n最终结果：");
    console.log(session.state.finalOutput || "(空)");
  } finally {
    rl.close();
  }
}

main().catch((err) => {
  console.error(`[boss-agent] fatal: ${err.message}`);
  process.exitCode = 1;
});
