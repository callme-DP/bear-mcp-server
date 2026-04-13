import { randomUUID } from "node:crypto";
import { Planner } from "./planner.js";
import { Executor } from "./executor.js";
import { BearTools } from "./tools.js";
import { Memory } from "./memory.js";
import { Validator } from "./validator.js";

function extractTag(text) {
  if (!text) return null;
  const regexList = [
    /包含[“"'‘]?([\w\-\u4e00-\u9fa5]+)[”"'’]?标签/,
    /标签(?:改成|改为|设为|是|为)?\s*[“"'‘]?([\w\-\u4e00-\u9fa5]+)[”"'’]?/,
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
  if (!text || text === "继续") return { cmd: "continue" };
  if (text === "结束") return { cmd: "stop" };
  if (text.startsWith("调整")) {
    return { cmd: "adjust", payload: text.replace(/^调整[:：]?\s*/, "").trim() };
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

export class BossAgentService {
  constructor(options = {}) {
    this.sessions = new Map();
    this.maxRounds = options.maxRounds ?? 6;
    this.maxRetries = options.maxRetries ?? 1;
    this.planner = new Planner();
    this.validator = new Validator();
    this.tools = new BearTools();
    this._toolsReady = false;
    this._toolsInitPromise = null;
  }

  async ensureToolsReady() {
    if (this._toolsReady) return;
    if (!this._toolsInitPromise) {
      this._toolsInitPromise = this.tools.init().then(() => {
        this._toolsReady = true;
      });
    }
    await this._toolsInitPromise;
  }

  async startSession(goal) {
    const trimmedGoal = (goal || "").trim();
    if (!trimmedGoal) {
      throw new Error("goal 不能为空");
    }

    await this.ensureToolsReady();

    const sessionId = randomUUID();
    const memory = new Memory();
    const state = {
      task: trimmedGoal,
      currentTag: extractTag(trimmedGoal) || "AI",
      noteLimit: 5,
      notes: [],
      noteDetails: [],
      finalOutput: "",
    };

    const session = {
      id: sessionId,
      goal: trimmedGoal,
      status: "active",
      round: 0,
      maxRounds: this.maxRounds,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      chatHistory: [{ role: "boss", content: trimmedGoal }],
      memory,
      state,
      lastValidation: null,
      lastReport: "",
    };

    this.sessions.set(sessionId, session);
    console.log(`[boss-agent-service] 新会话已创建 sessionId=${sessionId}`);

    const roundResult = await this._runRound(session, "start");
    return {
      sessionId,
      ...roundResult,
      session: this._serializeSession(session, { withMemory: false }),
    };
  }

  async messageSession(sessionId, message) {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error("session 不存在");
    }

    const control = parseControl(message);
    console.log(
      `[boss-agent-service] 收到指令 sessionId=${sessionId} cmd=${control.cmd} msg=${(message || "").trim()}`
    );

    if (control.cmd === "stop") {
      session.status = "ended";
      session.updatedAt = new Date().toISOString();
      const report = [
        "【Agent 会话已结束】",
        `- 目标：${session.goal}`,
        `- 轮次：${session.round}/${session.maxRounds}`,
        `- 最终输出：${(session.state.finalOutput || "(空)").slice(0, 600)}`,
      ].join("\n");
      session.lastReport = report;
      session.chatHistory.push({ role: "boss", content: message || "结束" });
      session.chatHistory.push({ role: "agent", content: report });
      return {
        done: true,
        status: session.status,
        round: session.round,
        report,
        output: session.state.finalOutput,
        validation: session.lastValidation,
        session: this._serializeSession(session, { withMemory: false }),
      };
    }

    if (session.round >= session.maxRounds) {
      session.status = "max_rounds_reached";
      session.updatedAt = new Date().toISOString();
      return {
        done: !!session.lastValidation?.ok,
        status: session.status,
        round: session.round,
        report: "已达到最大轮次，请使用“调整 ...”聚焦目标后重试，或“结束”终止会话。",
        output: session.state.finalOutput,
        validation: session.lastValidation,
        session: this._serializeSession(session, { withMemory: false }),
      };
    }

    if (control.cmd === "adjust" && control.payload) {
      this._applyAdjustment(session, control.payload);
    } else {
      session.chatHistory.push({ role: "boss", content: message || "继续" });
    }

    const roundResult = await this._runRound(session, control.cmd);
    return {
      ...roundResult,
      session: this._serializeSession(session, { withMemory: false }),
    };
  }

  getSession(sessionId) {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error("session 不存在");
    }
    return this._serializeSession(session, { withMemory: true });
  }

  async _runRound(session, trigger) {
    session.round += 1;
    session.updatedAt = new Date().toISOString();
    const loop = session.round;
    const executor = new Executor(this.tools, session.memory, {
      maxRetries: this.maxRetries,
    });

    console.log(
      `[boss-agent-service] round=${loop} trigger=${trigger} think -> act -> observe sessionId=${session.id}`
    );

    session.memory.add({ phase: "think", loop, trigger, goal: session.state.task });
    const plan = this.planner.createPlan(session.state.task, session.state);
    session.memory.add({ phase: "think", loop, plan });

    try {
      await executor.executePlan(plan, session.state);
    } catch (err) {
      session.memory.add({ phase: "observe", loop, status: "loop_error", error: err.message });
      console.log(`[boss-agent-service] round=${loop} 执行失败: ${err.message}`);
    }

    const validation = this.validator.validate(session.state);
    session.lastValidation = validation;
    if (validation.ok) {
      session.status = "done";
    } else if (session.round >= session.maxRounds) {
      session.status = "max_rounds_reached";
    } else {
      session.status = "active";
    }
    session.memory.add({ phase: "observe", loop, validation });

    const report = await this._buildReport(session, plan, validation);
    session.lastReport = report;
    session.chatHistory.push({ role: "agent", content: report });

    return {
      done: validation.ok,
      status: session.status,
      round: session.round,
      maxRounds: session.maxRounds,
      report,
      output: session.state.finalOutput,
      validation,
    };
  }

  _applyAdjustment(session, payload) {
    const msg = payload.trim();
    session.chatHistory.push({ role: "boss", content: `调整 ${msg}` });
    const foundTag = extractTag(msg);
    if (foundTag) {
      session.state.currentTag = foundTag;
    }
    session.state.task = `${session.goal}\n补充要求：${msg}`;
    console.log(
      `[boss-agent-service] 已应用调整 sessionId=${session.id} currentTag=${session.state.currentTag}`
    );
  }

  async _buildReport(session, plan, validation) {
    const fallback = [
      "【Agent 本轮汇报】",
      `- 目标：${session.goal}`,
      `- 当前标签：${session.state.currentTag}`,
      `- 执行轮次：${session.round}/${session.maxRounds}`,
      `- 进度状态：${validation.ok ? "已达成可交付结果" : "未达成，建议继续循环"}`,
      `- 本轮计划步数：${plan.length}`,
      `- 工具来源：${session.state.searchSource || (this.tools.mockMode ? "mock" : "unknown")}`,
      `- 结果预览：${(session.state.finalOutput || "(空)").slice(0, 240)}`,
      `- 建议下一步：${validation.ok ? "请回复“结束”确认收尾，或“调整 ...”继续优化" : "请回复“继续”或“调整 ...”"}`,
    ].join("\n");

    try {
      const aiText = await callOpenAI([
        {
          role: "system",
          content:
            "你是执行型员工，向老板做中文周报式汇报。简洁、结论先行，包含：结果、风险、下一步。",
        },
        {
          role: "user",
          content: JSON.stringify(
            {
              goal: session.goal,
              round: session.round,
              maxRounds: session.maxRounds,
              validation,
              currentTag: session.state.currentTag,
              outputPreview: (session.state.finalOutput || "").slice(0, 900),
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

  _serializeSession(session, { withMemory = false } = {}) {
    return {
      id: session.id,
      goal: session.goal,
      status: session.status,
      round: session.round,
      maxRounds: session.maxRounds,
      createdAt: session.createdAt,
      updatedAt: session.updatedAt,
      currentTag: session.state.currentTag,
      noteCount: Array.isArray(session.state.notes) ? session.state.notes.length : 0,
      detailCount: Array.isArray(session.state.noteDetails) ? session.state.noteDetails.length : 0,
      output: session.state.finalOutput || "",
      lastValidation: session.lastValidation,
      lastReport: session.lastReport,
      chatHistory: session.chatHistory,
      memory: withMemory ? session.memory.all() : undefined,
    };
  }
}

export const bossAgentService = new BossAgentService();
