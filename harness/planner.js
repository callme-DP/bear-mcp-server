export class Planner {
  createPlan(task, state) {
    const tag = state.currentTag || "AI";
    const noteLimit = state.noteLimit || 5;

    return [
      {
        id: "step_search",
        type: "search_by_tag",
        params: { tag, limit: noteLimit },
        reason: `Find notes with tag ${tag}`,
      },
      {
        id: "step_fetch",
        type: "fetch_note_details",
        params: { maxDetails: 3 },
        reason: "Load note details for summary",
      },
      {
        id: "step_summarize",
        type: "summarize_notes",
        params: { maxSummaryNotes: 3, maxChars: 180 },
        reason: "Build final summary text",
      },
      {
        id: "step_validate",
        type: "validate_output",
        params: {},
        reason: "Check output is not empty",
      },
    ];
  }
}
