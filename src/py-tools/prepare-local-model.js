/**
 * 一键下载本地模型到 node_modules/@xenova/transformers/models/
 * 用于后续完全离线推理。
 */

import fs from "fs";
import path from "path";
import { pipeline, env } from "@xenova/transformers";

const __dirname = path.resolve();

async function main() {
  // 目标目录：@xenova/transformers/models/distilbart-cnn-6-6
  const MODEL_DIR = path.resolve(
    __dirname,
    "node_modules/@xenova/transformers/models/distilbart-cnn-6-6"
  );

  console.log("📦 模型将下载到：", MODEL_DIR);

  // 若目录不存在则创建
  fs.mkdirSync(MODEL_DIR, { recursive: true });

  // 环境配置
  env.allowLocalModels = true;
  env.allowRemoteModels = true; // 允许联网下载
  env.cacheDir = path.resolve(__dirname, "node_modules/@xenova/transformers/models");

  console.log("🌐 启动下载：Xenova/distilbart-cnn-6-6 ...");

  // 自动下载 HuggingFace 上的 ONNX 模型
  const summarizer = await pipeline("summarization", "Xenova/distilbart-cnn-6-6", {
    quantized: true,
    device: "cpu",
  });

  console.log("✅ 模型下载完成！");
  console.log("📂 已缓存路径：", env.cacheDir);

  // 简单验证一次
  const summary = await summarizer("This is a test to verify the downloaded model.");
  console.log("🧠 摘要测试输出：", summary);
}

main().catch((err) => {
  console.error("❌ 下载失败：", err);
  process.exit(1);
});
