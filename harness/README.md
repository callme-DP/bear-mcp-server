# Boss-Agent 对话模式

## 目标
以“老板给目标、Agent循环执行并汇报”的方式推进任务，减少你对实现细节的关注。

## 启动

```bash
cd /Users/yangdongpeng/Projects/bear-mcp-server
npm run harness:boss
```

或直接传入目标：

```bash
npm run harness:boss -- "查找包含AI标签的笔记并做摘要"
```

## 对话指令

- `继续`：进入下一轮 loop
- `调整 <要求>`：更新目标约束并继续执行
- `结束`：结束会话

## 运行机制

每轮自动执行：

- think：生成计划
- act：调用工具执行（含重试）
- observe：验证结果是否为空
- report：向老板输出中文汇报（有 `OPENAI_API_KEY` 时由 GPT 生成更自然汇报）

## 环境变量

- `OPENAI_API_KEY`：可选；配置后用于生成更自然的“员工式汇报”
- `REVIEW_MODEL`：可选；默认 `gpt-4.1`
