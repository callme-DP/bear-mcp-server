export class Executor {
  constructor(tools, memory, options = {}) {
    this.tools = tools;
    this.memory = memory;
    this.maxRetries = options.maxRetries ?? 1;
  }

  async executePlan(plan, state) {
    for (const step of plan) {
      await this._executeStepWithRetry(step, state);
    }
    return state;
  }

  async _executeStepWithRetry(step, state) {
    let attempt = 0;
    let lastError = null;

    while (attempt <= this.maxRetries) {
      try {
        attempt += 1;
        this.memory.add({ phase: "act", stepId: step.id, attempt, status: "start", step });
        console.log(`[act] ${step.id} (${step.type}) attempt=${attempt}`);

        await this._executeStep(step, state);

        this.memory.add({ phase: "observe", stepId: step.id, attempt, status: "success" });
        console.log(`[observe] ${step.id} success`);
        return;
      } catch (err) {
        lastError = err;
        this.memory.add({
          phase: "observe",
          stepId: step.id,
          attempt,
          status: "error",
          error: err.message,
        });
        console.log(`[observe] ${step.id} failed: ${err.message}`);
      }
    }

    throw lastError;
  }

  async _executeStep(step, state) {
    if (step.type === "search_by_tag") {
      const { tag, limit } = step.params;
      const result = await this.tools.searchNotesByTag(tag, limit);
      state.searchSource = result.source;
      state.notes = Array.isArray(result.notes) ? result.notes : [];
      console.log(`[tools] search_by_tag source=${result.source} found=${state.notes.length}`);
      return;
    }

    if (step.type === "fetch_note_details") {
      const maxDetails = step.params.maxDetails ?? 3;
      const noteIds = (state.notes || []).slice(0, maxDetails).map((n) => n.id);
      const details = [];

      for (const noteId of noteIds) {
        const one = await this.tools.getNote(noteId);
        if (one.note) details.push(one.note);
      }

      state.noteDetails = details;
      console.log(`[tools] fetched note details count=${details.length}`);
      return;
    }

    if (step.type === "summarize_notes") {
      const maxSummaryNotes = step.params.maxSummaryNotes ?? 3;
      const maxChars = step.params.maxChars ?? 180;
      const notes = (state.noteDetails || state.notes || []).slice(0, maxSummaryNotes);

      const lines = notes.map((n, idx) => {
        const raw = `${n.title || "Untitled"}: ${(n.content || "").replace(/\s+/g, " ").trim()}`;
        const trimmed = raw.length > maxChars ? `${raw.slice(0, maxChars)}...` : raw;
        return `${idx + 1}. ${trimmed}`;
      });

      state.finalOutput = lines.join("\n");
      console.log(`[summary] built lines=${lines.length}`);
      return;
    }

    if (step.type === "validate_output") {
      const value = state.finalOutput || "";
      state.stepValidation = value.trim().length > 0;
      console.log(`[validate-step] non-empty=${state.stepValidation}`);
      return;
    }

    throw new Error(`Unknown step type: ${step.type}`);
  }
}
