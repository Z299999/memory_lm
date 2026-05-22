/**
 * Chart.js line chart for the output norm ‖y(t)‖₂ over time.
 */

/* global Chart */

export class OutputView {
  constructor(canvasId, windowSelectId) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    this.windowSelect = document.getElementById(windowSelectId);

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: '‖y(t)‖₂',
          data: [],
          borderColor: 'rgba(74, 144, 217, 1)',
          backgroundColor: 'rgba(74, 144, 217, 0.15)',
          borderWidth: 1,
          pointRadius: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: {
            title: { display: false },
            ticks: { display: false },
            grid: { display: false }
          },
          y: {
            title: { display: false },
            ticks: {
              color: '#777',
              maxTicksLimit: 4
            },
            grid: { color: 'rgba(0,0,0,0.05)' },
            beginAtZero: true
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              title: items => `t = ${items[0].label}`
            }
          }
        }
      }
    });

    // Force a redraw when the window selection changes.
    if (this.windowSelect) {
      this.windowSelect.addEventListener('change', () => {
        if (this._lastHistory) {
          this.update(this._lastHistory);
        }
      });
    }

    this._lastHistory = null;
  }

  /**
   * Update the chart from a full history array of the form
   * [{ t, norm }, ...]. Windowing and light down-sampling are
   * applied for performance.
   */
  update(history) {
    this._lastHistory = history;
    if (!history || history.length === 0) {
      this.chart.data.labels = [];
      this.chart.data.datasets[0].data = [];
      this.chart.update('none');
      return;
    }

    const windowValue = this.windowSelect ? this.windowSelect.value : '100';
    let slice = history;
    if (windowValue !== 'all') {
      const n = parseInt(windowValue, 10) || 100;
      slice = history.slice(-n);
    }

    // Down-sample if there are too many points.
    const maxPoints = 1000;
    const len = slice.length;
    let step = 1;
    if (len > maxPoints) {
      step = Math.ceil(len / maxPoints);
    }

    const labels = [];
    const data = [];
    for (let i = 0; i < len; i += step) {
      const p = slice[i];
      labels.push(p.t.toString());
      data.push(p.norm);
    }
    // Ensure we include the latest point.
    const last = slice[len - 1];
    if (labels.length === 0 || labels[labels.length - 1] !== last.t.toString()) {
      labels.push(last.t.toString());
      data.push(last.norm);
    }

    this.chart.data.labels = labels;
    this.chart.data.datasets[0].data = data;
    this.chart.update('none');
  }
}

