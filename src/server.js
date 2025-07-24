import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { handleQuery, initContext } from './handle.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();

// JSON 请求解析
app.use(express.json());

// 静态目录：暴露 openapi.yaml + ai-plugin.json
app.use('/.well-known', express.static(path.join(__dirname, '../.well-known'), {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.yaml')) {
      res.setHeader('Content-Type', 'application/x-yaml');
    } else if (filePath.endsWith('.json')) {
      res.setHeader('Content-Type', 'application/json');
    }
  }
}));

// 健康检查（非必须）
app.get('/', (req, res) => {
  res.send('Bear MCP server is running');
});

// 初始化模型
await initContext();

// 主接口：query
app.post('/query', async (req, res) => {
  try {
    const result = await handleQuery(req.body);
    res.json(result);
  } catch (err) {
    console.error('❌ /query error:', err);
    res.status(500).json({ error: err.message });
  }
});

// 启动服务
const PORT = 8000;
app.listen(PORT, () => {
  console.log(`✅ Bear MCP server listening at http://localhost:${PORT}`);
});
