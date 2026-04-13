import { Planner } from "./planner.js";
import { Executor } from "./executor.js";
import { BearTools } from "./tools.js";
import { Memory } from "./memory.js";
import { Validator } from "./validator.js";

async function runHarness(taskInput) {
  const task = taskInput || "查找包含‘AI’标签的笔记并做摘要";

  const memory = new Memory();
  const planner = new Planner();
  const tools = new BearTools();
  const validator = new Validator();
  const executor = new Executor(tools, memory, { maxRetries: 1 });

  const state = {
    task,
    currentTag: "AI",
    noteLimit: 5,
    notes: [],
    noteDetails: [],
    finalOutput: "",
  };

  const maxLoops = 3;
  let loop = 0;
  let done = false;

  console.log(`[harness] task=${task}`);
  await tools.init();

  while (!done && loop < maxLoops) {
    loop += 1;
    console.log(`\n[harness] loop=${loop} think -> act -> observe`);

    memory.add({ phase: "think", loop, message: "planning steps" });
    const plan = planner.createPlan(task, state);
    memory.add({ phase: "think", loop, plan });
    console.log(`[think] plan size=${plan.length}`);

    try {
      await executor.executePlan(plan, state);
    } catch (err) {
      memory.add({ phase: "observe", loop, status: "loop_error", error: err.message });
      console.log(`[observe] loop execution error: ${err.message}`);
    }

    const check = validator.validate(state);
    memory.add({ phase: "observe", loop, validation: check });
    console.log(`[observe] validator ok=${check.ok} reason=${check.reason}`);

    if (check.ok) {
      done = true;
      break;
    }

    if (loop < maxLoops) {
      console.log("[harness] validator failed, adjust and continue next round");
      state.noteLimit += 3;
    }
  }

  console.log("\n[harness] final result");
  if (state.finalOutput) {
    console.log(state.finalOutput);
  } else {
    console.log("(empty)");
  }

  console.log("\n[harness] memory steps");
  console.log(JSON.stringify(memory.all(), null, 2));

  return {
    done,
    loops: loop,
    output: state.finalOutput,
    memory: memory.all(),
  };
}

const cliTask = process.argv.slice(2).join(" ").trim();
runHarness(cliTask).catch((err) => {
  console.error(`[harness] fatal error: ${err.message}`);
  process.exitCode = 1;
});
