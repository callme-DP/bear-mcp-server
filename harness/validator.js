export class Validator {
  validate(state) {
    const text = state.finalOutput;
    const ok = typeof text === "string" && text.trim().length > 0;
    return {
      ok,
      reason: ok ? "output is non-empty" : "output is empty",
    };
  }
}
