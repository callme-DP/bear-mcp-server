import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { handleTool, initContext } from './handle.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();

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
  console.log(`[Router] tool=${tool}`, args);
  try {
    const result = await handleTool(tool, args);
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
