/**
 * Stats display: read-only counters shown in the UI.
 */

export class Stats {
  constructor() {
    this.elStep = document.getElementById('stat-step');
    this.elNodes = document.getElementById('stat-nodes');
    this.elEdges = document.getElementById('stat-edges');
    this.elOutputNorm = document.getElementById('stat-output-norm');
  }

  update(t, nodeCount, edgeCount, outputNorm) {
    this.elStep.textContent = t;
    this.elNodes.textContent = nodeCount;
    this.elEdges.textContent = edgeCount;
    if (this.elOutputNorm != null && outputNorm != null) {
      // Show small values more clearly; avoid everything looking like 0.
      if (Math.abs(outputNorm) < 1e-6) {
        this.elOutputNorm.textContent = '0';
      } else if (Math.abs(outputNorm) < 1e-2) {
        this.elOutputNorm.textContent = outputNorm.toExponential(2);
      } else {
        this.elOutputNorm.textContent = outputNorm.toFixed(4);
      }
    }
  }
}
