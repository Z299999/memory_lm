const NODE_COLORS = {
  input: '#4a90d9',
  hidden: '#7f848f',
  output: '#d94a4a',
  dopamine: '#2d936c',
};

const EDGE_COLORS = {
  default: '#6a6f7a',
  highlight: '#0b63ce',
  selected: '#d95f0e',
  dimmed: '#c8d0db',
};

const FORM_FIELDS = [
  { key: 'run_name', label: 'Run name', type: 'text', wide: true },
  { key: 'resume_latest', label: 'Resume latest model.pt', type: 'checkbox', wide: true },
  { key: 'task_name', label: 'Task name', type: 'select' },
  { key: 'seed', label: 'Seed', type: 'number', step: '1' },
  { key: 'epochs', label: 'Continue epochs', type: 'number', step: '1' },
  { key: 'lambda', label: 'Lambda', type: 'number', step: '0.01' },
  { key: 'input_dim', label: 'Input dim', type: 'number', step: '1' },
  { key: 'output_dim', label: 'Output dim', type: 'number', step: '1' },
  { key: 'trunk_dims', label: 'Trunk dims', type: 'text', wide: true, placeholder: '16,16' },
  { key: 'coverage_c', label: 'Coverage c', type: 'number', step: '1' },
  { key: 'dopamine_m_override', label: 'Dopamine m override', type: 'number', step: '1', optional: true },
  { key: 'batch_size', label: 'Batch size', type: 'number', step: '1' },
  { key: 'lr_bp', label: 'Learning rate', type: 'number', step: '0.0001' },
  { key: 'eta_int', label: 'Internal eta', type: 'number', step: '0.00001' },
  { key: 'gamma', label: 'Gamma', type: 'number', step: '0.1' },
  { key: 'num_train', label: 'Train samples', type: 'number', step: '1' },
  { key: 'num_val', label: 'Val samples', type: 'number', step: '1' },
  { key: 'num_plot', label: 'Plot samples', type: 'number', step: '1' },
  { key: 'x_min', label: 'x min', type: 'number', step: '0.1' },
  { key: 'x_max', label: 'x max', type: 'number', step: '0.1' },
  { key: 'resume_from', label: 'Resume from', type: 'text', wide: true },
  { key: 'enable_diagnostics', label: 'Enable diagnostics', type: 'checkbox', wide: true },
];

const FORM_SECTIONS = [
  {
    key: 'run',
    title: 'Run',
    fields: ['run_name', 'resume_latest', 'resume_from', 'seed'],
  },
  {
    key: 'task',
    title: 'Task & Architecture',
    fields: ['task_name', 'input_dim', 'output_dim', 'trunk_dims'],
  },
  {
    key: 'dopamine',
    title: 'Dopamine Coverage',
    fields: ['coverage_c', 'dopamine_m_override'],
    summaryId: 'dopamine-preview-summary',
  },
  {
    key: 'training',
    title: 'Training',
    fields: ['epochs', 'lambda', 'batch_size', 'lr_bp', 'eta_int', 'gamma'],
  },
  {
    key: 'data',
    title: 'Data',
    fields: ['num_train', 'num_val', 'num_plot', 'x_min', 'x_max', 'enable_diagnostics'],
  },
];

const FIELD_MAP = new Map(FORM_FIELDS.map((field) => [field.key, field]));

const POLL_INTERVAL_MS = 1000;

const state = {
  defaultConfig: null,
  taskNames: [],
  defaultGraphPayload: null,
  defaultPreviewState: null,
  defaultPreviewSummary: null,
  previewGraphPayload: null,
  previewState: null,
  previewSummary: null,
  confirmedPayload: null,
  activeRun: null,
  cy: null,
  graphPayload: null,
  graphKey: null,
  selectedDopamineId: null,
  selectedNodeId: null,
  selectedEdgeId: null,
  pollHandle: null,
  isDirty: true,
};

function graphPayloadKey(payload) {
  return JSON.stringify(payload);
}

function trunkDimsToText(trunkDims) {
  if (!Array.isArray(trunkDims)) {
    return '';
  }
  return trunkDims.join(', ');
}

function parseTrunkDims(text) {
  return String(text)
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number(part));
}

function getLayerLabelMap(payload) {
  const entries = (payload?.layerMetadata || []).map((item) => [item.key, item.label]);
  return new Map(entries);
}

function getArchitecture(payload) {
  return payload?.architecture || state.activeRun?.architecture || null;
}

function isRunActive(runState) {
  return Boolean(runState && ['running', 'starting', 'stopping'].includes(runState.status));
}

function buildLiveSnapshot(runState) {
  return {
    node_activation_snapshot: runState?.node_activation_snapshot || {},
    edge_weight_snapshot: runState?.edge_weight_snapshot || {},
  };
}

function getCurrentGraphContext() {
  if (isRunActive(state.activeRun) && state.activeRun?.graph_payload) {
    return {
      payload: state.activeRun.graph_payload,
      previewState: buildLiveSnapshot(state.activeRun),
    };
  }

  if (state.previewGraphPayload) {
    const previewKey = graphPayloadKey(state.previewGraphPayload);
    const activeKey = state.activeRun?.graph_payload ? graphPayloadKey(state.activeRun.graph_payload) : null;
    if (activeKey && activeKey === previewKey) {
      return {
        payload: state.previewGraphPayload,
        previewState: buildLiveSnapshot(state.activeRun),
      };
    }
    return {
      payload: state.previewGraphPayload,
      previewState: state.previewState,
    };
  }

  if (state.activeRun?.graph_payload) {
    return {
      payload: state.activeRun.graph_payload,
      previewState: buildLiveSnapshot(state.activeRun),
    };
  }

  return {
    payload: state.defaultGraphPayload,
    previewState: state.defaultPreviewState,
  };
}

function createNodeElement(node, payload) {
  const columnCount = Math.max(...payload.nodes.map((item) => item.column)) + 1;
  const left = -520;
  const right = 520;
  const x = columnCount <= 1
    ? 0
    : left + (node.column / (columnCount - 1)) * (right - left);
  const spacing = node.isDopamine ? 52 : 46;
  const offset = (node.rowCount - 1) / 2;
  const y = (node.row - offset) * spacing;
  return {
    group: 'nodes',
    data: {
      id: node.id,
      label: node.label,
      kind: node.kind,
      isDopamine: node.isDopamine,
      color: node.isDopamine ? NODE_COLORS.dopamine : NODE_COLORS[node.kind],
    },
    position: { x, y },
    classes: node.isDopamine ? 'dopamine-node' : '',
  };
}

function createEdgeElement(edge) {
  return {
    group: 'edges',
    data: {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      layerKey: edge.layerKey,
      layerLabel: edge.layerLabel,
      controllingDopamineIds: edge.controllingDopamineIds,
      color: EDGE_COLORS.default,
      thickness: 1.6,
      opacity: 0.68,
    },
  };
}

function createCy(payload) {
  return cytoscape({
    container: document.getElementById('cy'),
    elements: [
      ...payload.nodes.map((node) => createNodeElement(node, payload)),
      ...payload.edges.map(createEdgeElement),
    ],
    layout: { name: 'preset', fit: true, padding: 24 },
    minZoom: 0.25,
    maxZoom: 3,
    wheelSensitivity: 0.25,
    style: [
      {
        selector: 'node',
        style: {
          label: 'data(label)',
          'font-size': 9,
          'text-valign': 'center',
          'text-halign': 'center',
          color: '#ffffff',
          'text-outline-color': '#334155',
          'text-outline-width': 1.2,
          'background-color': 'data(color)',
          width: 26,
          height: 26,
          'border-width': 0,
        },
      },
      {
        selector: 'node.dopamine-node',
        style: {
          width: 30,
          height: 30,
        },
      },
      {
        selector: 'node.selected-dopamine',
        style: {
          'border-width': 4,
          'border-color': EDGE_COLORS.highlight,
        },
      },
      {
        selector: 'edge',
        style: {
          width: 'data(thickness)',
          'line-color': 'data(color)',
          'target-arrow-color': 'data(color)',
          'target-arrow-shape': 'triangle',
          'arrow-scale': 0.75,
          'curve-style': 'bezier',
          opacity: 'data(opacity)',
        },
      },
    ],
  });
}

function ensureGraph(payload) {
  if (!payload) {
    return;
  }
  const key = graphPayloadKey(payload);
  if (state.cy && state.graphKey === key) {
    return;
  }
  if (state.cy) {
    state.cy.destroy();
  }
  state.graphPayload = payload;
  state.graphKey = key;
  state.cy = createCy(payload);
  bindGraphInteractions(state.cy);
  applyGraphSelection();
}

function renderStatusPanel() {
  const panel = document.getElementById('status-panel');
  const activeRun = state.activeRun;

  if (!activeRun) {
    panel.innerHTML = `
      <div class="detail-card">
        <p class="detail-kicker">Active run</p>
        <h2 class="detail-title">Idle</h2>
        <p class="muted">
          Confirm a preview on the left, then start or resume a single active run from the toolbar.
        </p>
      </div>
    `;
    return;
  }

  const statusClass = `status-${activeRun.status || 'idle'}`;
  const localDisplay = `${activeRun.local_epoch ?? activeRun.epoch ?? 0} / ${activeRun.local_epochs_total ?? activeRun.epochs_total ?? '-'}`;
  const globalDisplay = `${activeRun.global_epoch ?? 0}`;
  const globalRange = `${activeRun.global_epoch_start ?? '?'}-${activeRun.global_epoch_end ?? '?'}`;

  panel.innerHTML = `
    <div class="detail-card">
      <p class="detail-kicker">Active run</p>
      <h2 class="detail-title">${activeRun.run_name || 'unnamed run'}</h2>
      <div class="stats-grid">
        <div class="stat-box ${statusClass}">
          <p class="stat-label">Status</p>
          <p class="stat-value">${activeRun.status || 'idle'}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Local epoch</p>
          <p class="stat-value">${localDisplay}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Global epoch</p>
          <p class="stat-value">${globalDisplay}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Global range</p>
          <p class="stat-value">${globalRange}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Train loss</p>
          <p class="stat-value">${formatMetric(activeRun.train_loss)}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Val loss</p>
          <p class="stat-value">${formatMetric(activeRun.val_loss)}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Best val</p>
          <p class="stat-value">${formatMetric(activeRun.best_val_loss)}</p>
        </div>
      </div>
      <div class="meta-strip">
        <span class="meta-chip">task ${activeRun.task_name || 'n/a'}</span>
        <span class="meta-chip">lambda ${formatMetric(activeRun.lambda, 2)}</span>
        <span class="meta-chip">c ${activeRun.coverage_c ?? 'n/a'}</span>
        <span class="meta-chip">m ${activeRun.dopamine_m ?? 'n/a'}</span>
      </div>
      <p class="run-dir-line" title="${escapeHtml(activeRun.run_dir || '')}">
        Run dir: ${formatShortPath(activeRun.run_dir)}
      </p>
      ${activeRun.error ? `<p class="muted">Error: ${escapeHtml(activeRun.error)}</p>` : ''}
    </div>
  `;
}

function renderLossPanel() {
  const panel = document.getElementById('loss-panel');
  const activeRun = state.activeRun;
  const localHistory = activeRun?.local_loss_history || [];
  const globalHistory = activeRun?.global_loss_history || [];

  if (!activeRun) {
    panel.innerHTML = `
      <div class="detail-card">
        <p class="detail-kicker">Loss</p>
        <h2 class="detail-title">Waiting for run</h2>
        <p class="muted">
          Once training starts, this panel will redraw train and validation MSE in real time
          using the current run's live JSON history.
        </p>
      </div>
    `;
    return;
  }

  panel.innerHTML = `
    <div class="detail-card">
      <p class="detail-kicker">Loss</p>
      <div class="loss-section">
        <p class="loss-section-label">Local</p>
        ${renderLossChart(localHistory, 'local')}
      </div>
      <div class="loss-section">
        <p class="loss-section-label">Global</p>
        ${renderLossChart(globalHistory, 'global')}
      </div>
    </div>
  `;
}

function renderLossChart(history, epochMode) {
  if (!history || history.length === 0) {
    return `<p class="muted">No epochs completed yet. The curve will appear after the first validation pass.</p>`;
  }

  const width = 320;
  const height = 180;
  const padLeft = 42;
  const padRight = 18;
  const padTop = 16;
  const padBottom = 28;
  const xKey = epochMode === 'local' ? 'local_epoch' : 'global_epoch';
  const xs = history.map((row) => Number(row[xKey]));
  const train = history.map((row) => Math.log10(Math.max(Number(row.train_loss), 1e-12)));
  const val = history.map((row) => Math.log10(Math.max(Number(row.val_loss), 1e-12)));
  const allY = [...train, ...val];
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...allY);
  const maxY = Math.max(...allY);
  const xSpan = Math.max(maxX - minX, 1);
  const ySpan = Math.max(maxY - minY, 1e-6);

  const scaleX = (value) => padLeft + ((value - minX) / xSpan) * (width - padLeft - padRight);
  const scaleY = (value) => height - padBottom - ((value - minY) / ySpan) * (height - padTop - padBottom);
  const toPolyline = (series) => series.map((value, idx) => `${scaleX(xs[idx]).toFixed(1)},${scaleY(value).toFixed(1)}`).join(' ');

  const gridValues = [0, 0.5, 1].map((t) => minY + t * ySpan);
  const gridSvg = gridValues.map((value) => {
    const y = scaleY(value).toFixed(1);
    const label = `1e${value.toFixed(1)}`;
    return `
      <g>
        <line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" stroke="#e2e8f0" stroke-width="1" />
        <text x="${padLeft - 8}" y="${Number(y) + 4}" text-anchor="end" font-size="10" fill="#64748b">${label}</text>
      </g>
    `;
  }).join('');

  return `
    <div class="loss-chart-shell">
      <svg class="loss-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Live train and validation loss curves">
        ${gridSvg}
        <line x1="${padLeft}" y1="${height - padBottom}" x2="${width - padRight}" y2="${height - padBottom}" stroke="#94a3b8" stroke-width="1.2" />
        <line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${height - padBottom}" stroke="#94a3b8" stroke-width="1.2" />
        <polyline fill="none" stroke="${EDGE_COLORS.highlight}" stroke-width="2.2" points="${toPolyline(train)}" />
        <polyline fill="none" stroke="${NODE_COLORS.output}" stroke-width="2.2" points="${toPolyline(val)}" />
        <text x="${padLeft}" y="${height - 6}" font-size="10" fill="#64748b">${epochMode} ${minX}</text>
        <text x="${width - padRight}" y="${height - 6}" text-anchor="end" font-size="10" fill="#64748b">${epochMode} ${maxX}</text>
      </svg>
      <div class="loss-legend">
        <span><span class="legend-swatch legend-train"></span>train</span>
        <span><span class="legend-swatch legend-val"></span>val</span>
      </div>
    </div>
  `;
}

function renderDefaultDetail(panel, payload) {
  panel.innerHTML = `
    <div class="detail-card">
      <p class="detail-kicker">Overview</p>
      <h2 class="detail-title">Structure summary</h2>
      <p class="muted">
        Click a dopamine hidden node to inspect which forward edges it controls,
        or click any forward edge to inspect its source, target, and controlling dopamine nodes.
      </p>
    </div>
    <div class="detail-card">
      <p class="detail-kicker">Network scale</p>
      <div class="stats-grid">
        <div class="stat-box">
          <p class="stat-label">Layers</p>
          <p class="stat-value">${payload.totalLayers}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Nodes</p>
          <p class="stat-value">${payload.totalNodes}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Controllable edges</p>
          <p class="stat-value">${payload.totalEdges}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Dopamine nodes</p>
          <p class="stat-value">${payload.dopamineM}</p>
        </div>
      </div>
    </div>
  `;
}

function buildLayerBreakdownHtml(layerCounts, payload) {
  const totals = payload.edges.reduce((acc, edge) => {
    acc[edge.layerKey] = (acc[edge.layerKey] || 0) + 1;
    return acc;
  }, {});
  const layerMetadata = payload.layerMetadata || [];
  return layerMetadata
    .map((layer) => {
      const controlled = layerCounts[layer.key] ?? 0;
      const layerTotal = totals[layer.key] ?? 0;
      const widthPercent = layerTotal === 0 ? 0 : (controlled / layerTotal) * 100;
      return `
        <div class="layer-row">
          <div>
            <p class="layer-name">${layer.label}</p>
            <div class="bar-track">
              <div class="bar-fill" style="width: ${widthPercent}%"></div>
            </div>
          </div>
          <div class="layer-metric">${controlled}/${layerTotal}</div>
        </div>
      `;
    })
    .join('');
}

function renderDopamineDetail(panel, nodeId, payload) {
  const dopamineStats = new Map((payload.dopamineStats || []).map((item) => [item.node_id, item]));
  const selectedStat = dopamineStats.get(nodeId);
  const controlledEdges = payload.edges.filter((edge) => (edge.controllingDopamineIds || []).includes(nodeId));
  const layerCounts = controlledEdges.reduce((acc, edge) => {
    acc[edge.layerKey] = (acc[edge.layerKey] || 0) + 1;
    return acc;
  }, {});

  const previewState = getCurrentGraphContext().previewState;
  const activationValue = previewState?.node_activation_snapshot?.[nodeId];
  panel.innerHTML = `
    <div class="detail-card">
      <p class="detail-kicker">Selected dopamine node</p>
      <h2 class="detail-title">${nodeId}</h2>
      <p class="muted">
        Highlighting every forward edge currently assigned to this hidden dopamine neuron.
      </p>
    </div>
    <div class="detail-card">
      <p class="detail-kicker">Coverage summary</p>
      <div class="stats-grid">
        <div class="stat-box">
          <p class="stat-label">Activation</p>
          <p class="stat-value">${formatMaybeMetric(activationValue)}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Controlled edges</p>
          <p class="stat-value">${selectedStat?.edge_count ?? controlledEdges.length}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Coverage ratio</p>
          <p class="stat-value">${formatPercent(selectedStat?.coverage_ratio)}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Rank</p>
          <p class="stat-value">${selectedStat?.rank ?? 'n/a'}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">c_r</p>
          <p class="stat-value">${formatPercent(selectedStat?.c_r)}</p>
        </div>
      </div>
      <div class="layer-breakdown">
        ${buildLayerBreakdownHtml(layerCounts, payload)}
      </div>
    </div>
  `;
}

function renderNodeDetail(panel, nodeId, payload) {
  const previewState = getCurrentGraphContext().previewState;
  const activationValue = previewState?.node_activation_snapshot?.[nodeId];
  const node = payload.nodes.find((item) => item.id === nodeId);
  panel.innerHTML = `
    <div class="detail-card">
      <p class="detail-kicker">Selected node</p>
      <h2 class="detail-title">${nodeId}</h2>
      <table class="detail-table">
        <tr><th>Kind</th><td>${node?.kind ?? 'n/a'}</td></tr>
        <tr><th>Activation mean</th><td>${formatMaybeMetric(activationValue)}</td></tr>
      </table>
      <p class="muted">Activation values appear after the latest validation pass. Preview-only states leave this as not run yet.</p>
    </div>
  `;
}

function renderEdgeDetail(panel, edge, payload) {
  const pills = (edge.controllingDopamineIds || [])
    .map((nodeId) => `<span class="pill">${nodeId}</span>`)
    .join('');
  const layerLabel = edge.layerLabel || getLayerLabelMap(payload).get(edge.layerKey) || edge.layerKey;
  const previewState = getCurrentGraphContext().previewState;
  const weightValue = previewState?.edge_weight_snapshot?.[edge.id];

  panel.innerHTML = `
    <div class="detail-card">
      <p class="detail-kicker">Selected edge</p>
      <h2 class="detail-title">${edge.id}</h2>
      <table class="detail-table">
        <tr><th>Source</th><td><code>${edge.source}</code></td></tr>
        <tr><th>Target</th><td><code>${edge.target}</code></td></tr>
        <tr><th>Layer</th><td>${layerLabel}</td></tr>
        <tr><th>Order index</th><td>${edge.orderIndex}</td></tr>
        <tr><th>Weight</th><td>${formatMaybeMetric(weightValue)}</td></tr>
      </table>
    </div>
    <div class="detail-card">
      <p class="detail-kicker">Controlling dopamine nodes</p>
      <div class="pill-list">${pills}</div>
    </div>
  `;
}

function renderDetailPanel() {
  const panel = document.getElementById('detail-panel');
  const payload = state.graphPayload;
  if (!payload) {
    panel.innerHTML = '';
    return;
  }
  if (state.selectedDopamineId) {
    renderDopamineDetail(panel, state.selectedDopamineId, payload);
    return;
  }
  if (state.selectedNodeId) {
    renderNodeDetail(panel, state.selectedNodeId, payload);
    return;
  }
  if (state.selectedEdgeId) {
    const edge = payload.edges.find((item) => item.id === state.selectedEdgeId);
    if (edge) {
      renderEdgeDetail(panel, edge, payload);
      return;
    }
  }
  renderDefaultDetail(panel, payload);
}

function clearEdgeStyling(cy) {
  cy.edges().forEach((edgeEle) => {
    edgeEle.data('color', EDGE_COLORS.default);
    edgeEle.data('thickness', 1.6);
    edgeEle.data('opacity', 0.68);
  });
  cy.nodes('.selected-dopamine').removeClass('selected-dopamine');
}

function highlightEdgesForDopamine(cy, nodeId) {
  clearEdgeStyling(cy);
  cy.getElementById(nodeId).addClass('selected-dopamine');
  cy.edges().forEach((edgeEle) => {
    const controlling = edgeEle.data('controllingDopamineIds') || [];
    if (controlling.includes(nodeId)) {
      edgeEle.data('color', EDGE_COLORS.highlight);
      edgeEle.data('thickness', 3.2);
      edgeEle.data('opacity', 0.95);
    } else {
      edgeEle.data('color', EDGE_COLORS.dimmed);
      edgeEle.data('thickness', 1.1);
      edgeEle.data('opacity', 0.15);
    }
  });
}

function highlightSelectedEdge(cy, edgeId) {
  clearEdgeStyling(cy);
  cy.edges().forEach((edgeEle) => {
    if (edgeEle.id() === edgeId) {
      edgeEle.data('color', EDGE_COLORS.selected);
      edgeEle.data('thickness', 3.6);
      edgeEle.data('opacity', 1);
    } else {
      edgeEle.data('color', EDGE_COLORS.dimmed);
      edgeEle.data('thickness', 1.1);
      edgeEle.data('opacity', 0.15);
    }
  });
}

function applyGraphSelection() {
  if (!state.cy) {
    return;
  }
  if (state.selectedDopamineId) {
    highlightEdgesForDopamine(state.cy, state.selectedDopamineId);
  } else if (state.selectedEdgeId) {
    highlightSelectedEdge(state.cy, state.selectedEdgeId);
  } else {
    clearEdgeStyling(state.cy);
  }
}

function bindGraphInteractions(cy) {
  cy.on('tap', 'node', (event) => {
    const node = event.target;
    state.selectedDopamineId = node.data('isDopamine') ? node.id() : null;
    state.selectedNodeId = node.id();
    state.selectedEdgeId = null;
    applyGraphSelection();
    renderDetailPanel();
  });

  cy.on('tap', 'edge', (event) => {
    state.selectedEdgeId = event.target.id();
    state.selectedDopamineId = null;
    state.selectedNodeId = null;
    applyGraphSelection();
    renderDetailPanel();
  });

  cy.on('tap', (event) => {
    if (event.target === cy) {
      state.selectedDopamineId = null;
      state.selectedNodeId = null;
      state.selectedEdgeId = null;
      applyGraphSelection();
      renderDetailPanel();
    }
  });

  window.addEventListener('resize', () => {
    cy.resize();
    cy.fit(undefined, 24);
  });
}

function createFieldMarkup(field) {
  if (field.type === 'checkbox') {
    return `
      <div class="form-field wide">
        <div class="checkbox-row">
          <input id="field-${field.key}" name="${field.key}" type="checkbox">
          <label for="field-${field.key}">${field.label}</label>
        </div>
      </div>
    `;
  }

  if (field.type === 'select') {
    const options = state.taskNames
      .map((taskName) => `<option value="${taskName}">${taskName}</option>`)
      .join('');
    return `
      <div class="form-field ${field.wide ? 'wide' : ''}">
        <label for="field-${field.key}">${field.label}</label>
        <select id="field-${field.key}" name="${field.key}">
          ${options}
        </select>
      </div>
    `;
  }

  const stepAttr = field.step ? `step="${field.step}"` : '';
  const placeholder = field.placeholder
    ? `placeholder="${field.placeholder}"`
    : (field.optional ? 'placeholder="auto"' : '');
  return `
    <div class="form-field ${field.wide ? 'wide' : ''}">
      <label for="field-${field.key}">${field.label}</label>
      <input
        id="field-${field.key}"
        name="${field.key}"
        type="${field.type}"
        ${stepAttr}
        ${placeholder}
      >
    </div>
  `;
}

function createSectionMarkup(section) {
  const fields = section.fields
    .map((fieldKey) => createFieldMarkup(FIELD_MAP.get(fieldKey)))
    .join('');
  const summaryMarkup = section.summaryId
    ? `<div id="${section.summaryId}" class="derived-summary"></div>`
    : '';
  return `
    <fieldset class="config-card">
      <legend>${section.title}</legend>
      <div class="form-grid section-grid">
        ${fields}
      </div>
      ${summaryMarkup}
    </fieldset>
  `;
}

function renderConfigForm() {
  const container = document.getElementById('config-sections');
  container.innerHTML = FORM_SECTIONS.map(createSectionMarkup).join('');
  const resumeLatest = document.getElementById('field-resume_latest');
  if (resumeLatest) {
    resumeLatest.addEventListener('change', syncResumeModeUi);
  }
}

function fillForm(config) {
  FORM_FIELDS.forEach((field) => {
    const element = document.getElementById(`field-${field.key}`);
    if (!element) {
      return;
    }
    if (field.type === 'checkbox') {
      element.checked = Boolean(config[field.key]);
    } else if (field.key === 'trunk_dims') {
      element.value = trunkDimsToText(config[field.key]);
    } else if (field.optional && (config[field.key] === null || config[field.key] === undefined)) {
      element.value = '';
    } else {
      element.value = config[field.key] ?? '';
    }
  });
  syncResumeModeUi();
}

function readFormPayload() {
  const payload = {};
  FORM_FIELDS.forEach((field) => {
    const element = document.getElementById(`field-${field.key}`);
    if (!element) {
      return;
    }
    if (field.type === 'checkbox') {
      payload[field.key] = element.checked;
      return;
    }
    if (field.key === 'trunk_dims') {
      payload[field.key] = parseTrunkDims(element.value);
      return;
    }
    if (field.type === 'number') {
      if (field.optional && element.value.trim() === '') {
        payload[field.key] = null;
      } else {
        payload[field.key] = Number(element.value);
      }
      return;
    }
    payload[field.key] = element.value;
  });
  return payload;
}

function syncResumeModeUi() {
  const resumeLatest = document.getElementById('field-resume_latest');
  const resumeFrom = document.getElementById('field-resume_from');
  if (!resumeLatest || !resumeFrom) {
    return;
  }

  const useLatest = resumeLatest.checked;
  resumeFrom.disabled = useLatest;
  resumeFrom.readOnly = useLatest;
  resumeFrom.placeholder = useLatest ? 'auto: latest runs/*/model.pt' : '';
  if (useLatest) {
    resumeFrom.classList.add('is-disabled');
    resumeFrom.title = 'Using the newest runs/*/model.pt automatically.';
  } else {
    resumeFrom.classList.remove('is-disabled');
    resumeFrom.title = '';
  }
}

function renderDopaminePreviewSummary(summary) {
  const container = document.getElementById('dopamine-preview-summary');
  if (!container) {
    return;
  }
  if (!summary) {
    container.innerHTML = '';
    return;
  }
  container.innerHTML = `
    <div class="derived-summary-grid">
      <div class="derived-item">
        <span class="derived-label">recommended m</span>
        <span class="derived-value">${summary.recommended_dopamine_m ?? 'n/a'}</span>
      </div>
      <div class="derived-item">
        <span class="derived-label">effective m</span>
        <span class="derived-value">${summary.effective_dopamine_m ?? 'n/a'}</span>
      </div>
      <div class="derived-item">
        <span class="derived-label">hidden pool</span>
        <span class="derived-value">${summary.hidden_pool_size ?? 'n/a'}</span>
      </div>
      <div class="derived-item">
        <span class="derived-label">avg edges / dopamine</span>
        <span class="derived-value">${formatMetric(summary.average_edges_per_dopamine, 2)}</span>
      </div>
    </div>
    <p class="derived-footnote">This summary updates on Confirm and defines the previewed dopamine assignment.</p>
  `;
}

function setFlashMessage(message, kind = 'info') {
  const box = document.getElementById('flash-message');
  if (!message) {
    box.className = 'flash-message hidden';
    box.textContent = '';
    return;
  }
  box.className = `flash-message ${kind}`;
  box.textContent = message;
}

function updateButtons() {
  const confirmBtn = document.getElementById('btn-confirm');
  const runBtn = document.getElementById('btn-run');
  const stopBtn = document.getElementById('btn-stop');
  const running = isRunActive(state.activeRun);
  const canRun = Boolean(state.confirmedPayload) && !state.isDirty && !running;
  confirmBtn.disabled = running;
  runBtn.disabled = !canRun;
  stopBtn.disabled = !running;
}

function formatMetric(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 'n/a';
  }
  return Number(value).toFixed(digits);
}

function formatMaybeMetric(value, digits = 4) {
  if (value === null || value === undefined) {
    return 'not run yet';
  }
  return formatMetric(value, digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 'n/a';
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatShortPath(path) {
  if (!path) {
    return 'n/a';
  }
  const parts = String(path).split('/').filter(Boolean);
  if (parts.length <= 3) {
    return String(path);
  }
  return `.../${parts.slice(-3).join('/')}`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function updateToolbarStatus() {
  const previewEl = document.getElementById('toolbar-preview-state');
  const statusEl = document.getElementById('toolbar-run-status');
  const epochEl = document.getElementById('toolbar-global-epoch');
  const previewStatus = state.confirmedPayload && !state.isDirty ? 'confirmed' : 'dirty';
  previewEl.textContent = previewStatus;
  statusEl.textContent = state.activeRun?.status || 'idle';
  epochEl.textContent = String(state.activeRun?.global_epoch ?? 0);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed: ${response.status}`);
    error.payload = payload;
    throw error;
  }
  return payload;
}

function syncGraphFromState() {
  const graphContext = getCurrentGraphContext();
  ensureGraph(graphContext.payload);
  renderDopaminePreviewSummary(state.previewSummary || state.defaultPreviewSummary);
  renderDetailPanel();
  renderStatusPanel();
  renderLossPanel();
  updateToolbarStatus();
}

async function loadDefaults() {
  const payload = await fetchJson('/api/default-config');
  state.defaultConfig = payload.config;
  state.taskNames = payload.task_names || [];
  state.defaultGraphPayload = payload.graph_payload || null;
  state.defaultPreviewState = payload.preview_state || null;
  state.defaultPreviewSummary = payload.preview_summary || null;
  state.previewGraphPayload = state.defaultGraphPayload;
  state.previewState = state.defaultPreviewState;
  state.previewSummary = state.defaultPreviewSummary;
  state.confirmedPayload = null;
  state.isDirty = true;
  renderConfigForm();
  fillForm(state.defaultConfig);
  syncGraphFromState();
  updateButtons();
}

async function refreshState(showMessage = false) {
  const payload = await fetchJson('/api/state');
  state.activeRun = payload.active_run;
  syncGraphFromState();
  updateButtons();
  if (showMessage) {
    setFlashMessage('Dashboard state refreshed.', 'info');
  }
}

async function handleConfirm() {
  try {
    const payload = readFormPayload();
    const result = await fetchJson('/api/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.previewGraphPayload = result.graph_payload;
    state.previewState = result.preview_state;
    state.previewSummary = result.preview_summary || null;
    state.confirmedPayload = result.config;
    state.isDirty = false;
    fillForm(result.config);
    state.selectedDopamineId = null;
    state.selectedNodeId = null;
    state.selectedEdgeId = null;
    syncGraphFromState();
    updateButtons();
    setFlashMessage('Preview confirmed. Graph and dopamine assignment updated; Run is now enabled.', 'success');
  } catch (error) {
    setFlashMessage(error.message, 'error');
  }
}

async function handleRun() {
  if (!state.confirmedPayload || state.isDirty) {
    setFlashMessage('Confirm the current configuration before starting a run.', 'error');
    return;
  }
  try {
    const result = await fetchJson('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.confirmedPayload),
    });
    state.activeRun = result.active_run;
    syncGraphFromState();
    updateButtons();
    setFlashMessage('Training run started.', 'success');
  } catch (error) {
    const payload = error.payload || {};
    if (payload.active_run !== undefined) {
      state.activeRun = payload.active_run;
      syncGraphFromState();
      updateButtons();
    }
    setFlashMessage(error.message, 'error');
  }
}

async function handleStop() {
  try {
    const result = await fetchJson('/api/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    state.activeRun = result.active_run;
    syncGraphFromState();
    updateButtons();
    setFlashMessage('Stop requested. Training will finish the current epoch and then write artifacts.', 'info');
  } catch (error) {
    setFlashMessage(error.message, 'error');
  }
}

function bindDashboardControls() {
  document.getElementById('btn-confirm').addEventListener('click', handleConfirm);
  document.getElementById('btn-run').addEventListener('click', handleRun);
  document.getElementById('btn-stop').addEventListener('click', handleStop);
  document.getElementById('btn-refresh').addEventListener('click', () => refreshState(true).catch((error) => {
    setFlashMessage(error.message, 'error');
  }));
  document.getElementById('btn-reset-form').addEventListener('click', () => {
    if (state.defaultConfig) {
      fillForm(state.defaultConfig);
      state.previewGraphPayload = state.defaultGraphPayload;
      state.previewState = state.defaultPreviewState;
      state.previewSummary = state.defaultPreviewSummary;
      state.confirmedPayload = null;
      state.isDirty = true;
      state.selectedDopamineId = null;
      state.selectedNodeId = null;
      state.selectedEdgeId = null;
      syncGraphFromState();
      updateButtons();
      setFlashMessage('Form reset to default values. Confirm again before running.', 'info');
    }
  });

  document.getElementById('config-form').addEventListener('input', () => {
    state.isDirty = true;
    updateButtons();
    updateToolbarStatus();
  });
  document.getElementById('config-form').addEventListener('change', () => {
    state.isDirty = true;
    updateButtons();
    updateToolbarStatus();
  });
}

function startPolling() {
  if (state.pollHandle) {
    clearInterval(state.pollHandle);
  }
  state.pollHandle = window.setInterval(() => {
    refreshState(false).catch(() => {
      // Keep polling silently; the user can still press Refresh for details.
    });
  }, POLL_INTERVAL_MS);
}

async function initializeDashboard() {
  await loadDefaults();
  bindDashboardControls();
  await refreshState(false);
  startPolling();
}

initializeDashboard().catch((error) => {
  setFlashMessage(`Failed to initialize dashboard: ${error.message}`, 'error');
});
