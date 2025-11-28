import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { handleTool, initContext } from './handle.js';

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
app.use(express.static(path.join(__dirname, "../.well-known")));

// 健康检查（非必须）
app.get('/', (req, res) => {
  res.send('Bear MCP server is running');
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
