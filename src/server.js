import express from 'express';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';
import { handleTool, initContext } from './handle.js';
import { bossAgentService } from '../harness/bossAgentService.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();

function formatLocalTimestamp() {
  // Use local time with offset for readable logs, e.g., 2025-11-28 08:19:22 GMT+8
  return new Date().toLocaleString('sv-SE', {
    hour12: false,
    timeZoneName: 'short'
  });
}

// JSON 请求解析
app.use(express.json());

// ✅ 静态文件路由：公开 .well-known 文件夹
// 标准插件发现路径：/.well-known/ai-plugin.json
app.use('/.well-known', express.static(path.join(__dirname, "../.well-known")));
// 向后兼容旧路径（如 /openapi.yaml）
app.use(express.static(path.join(__dirname, "../.well-known")));

// 健康检查（非必须）
app.get('/', (req, res) => {
  res.send('Bear MCP server is running');
});

app.get('/legal', (req, res) => {
  res.type('text/plain; charset=utf-8');
  res.send('Bear MCP Plugin Legal: local dev bridge for personal notes. Use at your own risk.');
});

// 笔记趋势缓存读取（由 Inner-Orchard 每日离线脚本生成）
app.get('/notes-trend', async (req, res) => {
  const defaultTrendPath = path.resolve(
    __dirname,
    '../../Inner-Orchard/artifacts/notes/notes-trend.json'
  );
  const trendPath = process.env.NOTES_TREND_PATH || defaultTrendPath;
  try {
    const raw = await fs.readFile(trendPath, 'utf8');
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.send(raw);
  } catch (err) {
    res.status(404).json({
      error: 'notes trend cache not found',
      pathTried: trendPath,
    });
  }
});

// Boss-Agent: 启动会话（自动执行首轮）
app.post('/agent/session/start', async (req, res) => {
  try {
    const goal = typeof req.body?.goal === 'string' ? req.body.goal : '';
    const result = await bossAgentService.startSession(goal);
    res.json(result);
  } catch (err) {
    const msg = err?.message || 'start session failed';
    const status = msg.includes('不能为空') ? 400 : 500;
    console.error(`[agent/session/start] ${msg}`);
    res.status(status).json({ error: msg });
  }
});

// Boss-Agent: 发送指令（继续/调整.../结束）
app.post('/agent/session/message', async (req, res) => {
  try {
    const sessionId = typeof req.body?.sessionId === 'string' ? req.body.sessionId : '';
    const message = typeof req.body?.message === 'string' ? req.body.message : '';
    if (!sessionId) {
      return res.status(400).json({ error: 'sessionId 不能为空' });
    }
    const result = await bossAgentService.messageSession(sessionId, message);
    res.json(result);
  } catch (err) {
    const msg = err?.message || 'message session failed';
    const status = msg.includes('session 不存在') ? 404 : 500;
    console.error(`[agent/session/message] ${msg}`);
    res.status(status).json({ error: msg });
  }
});

// Boss-Agent: 查询会话状态
app.get('/agent/session/:id', async (req, res) => {
  try {
    const result = bossAgentService.getSession(req.params.id);
    res.json(result);
  } catch (err) {
    const msg = err?.message || 'get session failed';
    const status = msg.includes('session 不存在') ? 404 : 500;
    console.error(`[agent/session/:id] ${msg}`);
    res.status(status).json({ error: msg });
  }
});

// 初始化模型
await initContext();


app.post("/:tool", async (req, res) => {
  const { tool } = req.params;
  const args = req.body.args || {};
  // Add local timestamp so calls can be correlated in logs
  console.log(`[${formatLocalTimestamp()}] [Router] tool=${tool}`, args);
  // Debug helper（打印请求参数：排查用）: reconstruct curl for quick repro (remove if too noisy)
  const bodyString = JSON.stringify(req.body || {});
  console.log(`curl -X POST http://localhost:8000/${tool} -H "Content-Type: application/json" -d '${bodyString}'`);
  try {
    const result = await handleTool(tool, args);
    // Debug helper（打印返回体：排查用）: print response for inspection (remove if too noisy)
    // console.log(`[${formatLocalTimestamp()}] [Router][response] tool=${tool}`, result);
    res.json(result);
  } catch (err) {
    console.error(`[ERROR:${tool}]`, err);
    res.status(500).json({ error: err.message });
  }
});


// 启动服务
const PORT = 8000;
app.listen(PORT, () => {
  console.log(`✅ Bear MCP server listening at http://localhost:${PORT}`);
});
