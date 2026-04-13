export class Memory {
  constructor() {
    this.steps = [];
  }

  add(entry) {
    const record = {
      at: new Date().toISOString(),
      ...entry,
    };
    this.steps.push(record);
    return record;
  }

  all() {
    return this.steps;
  }
}
