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

const LAYER_LABELS = {
  'input->trunk1': 'Input -> Hidden L1',
  'trunk1->trunk2': 'Hidden L1 -> Hidden L2',
  'trunk2->y': 'Hidden L2 -> Output',
};

const FORM_FIELDS = [
  { key: 'run_name', label: 'Run name', type: 'text', wide: true },
  { key: 'task_name', label: 'Task name', type: 'select' },
  { key: 'seed', label: 'Seed', type: 'number', step: '1' },
  { key: 'epochs', label: 'Continue epochs', type: 'number', step: '1' },
  { key: 'lambda', label: 'Lambda', type: 'number', step: '0.01' },
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

const POLL_INTERVAL_MS = 1000;

const state = {
  defaultConfig: null,
  taskNames: [],
  defaultGraphPayload: null,
  activeRun: null,
  cy: null,
  graphPayload: null,
  graphKey: null,
  selectedDopamineId: null,
  selectedEdgeId: null,
  pollHandle: null,
};

function graphPayloadKey(payload) {
  return JSON.stringify(payload);
}

function createNodeElement(node) {
  const xPositions = [-520, -180, 160, 500];
  const x = xPositions[node.column] ?? 0;
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
      ...payload.nodes.map(createNodeElement),
      ...payload.edges.map(createEdgeElement),
    ],
    layout: { name: 'preset', fit: true, padding: 24 },
    minZoom: 0.35,
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
        <p class="detail-kicker">Dashboard status</p>
        <h2 class="detail-title">Idle</h2>
        <p class="muted">
          Edit the config on the left, then start a single active run. The graph previews the
          current hidden-dopamine assignment for the default configuration.
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
        <div class="stat-box">
          <p class="stat-label">Lambda</p>
          <p class="stat-value">${formatMetric(activeRun.lambda, 2)}</p>
        </div>
      </div>
      <table class="detail-table">
        <tr><th>Task</th><td>${activeRun.task_name || 'n/a'}</td></tr>
        <tr><th>Seed</th><td>${activeRun.seed ?? 'n/a'}</td></tr>
        <tr><th>Coverage c</th><td>${activeRun.coverage_c ?? 'n/a'}</td></tr>
        <tr><th>Dopamine m</th><td>${activeRun.dopamine_m ?? 'n/a'} (rec ${activeRun.recommended_dopamine_m ?? 'n/a'})</td></tr>
        <tr><th>Run dir</th><td class="subtle-path">${activeRun.run_dir || 'n/a'}</td></tr>
        <tr><th>Updated</th><td>${activeRun.updated_at || 'n/a'}</td></tr>
      </table>
      ${activeRun.error ? `<p class="muted">Error: ${escapeHtml(activeRun.error)}</p>` : ''}
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
  const orderedLayerKeys = ['input->trunk1', 'trunk1->trunk2', 'trunk2->y'];
  return orderedLayerKeys
    .map((layerKey) => {
      const controlled = layerCounts[layerKey] ?? 0;
      const layerTotal = totals[layerKey] ?? 0;
      const widthPercent = layerTotal === 0 ? 0 : (controlled / layerTotal) * 100;
      return `
        <div class="layer-row">
          <div>
            <p class="layer-name">${LAYER_LABELS[layerKey]}</p>
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

function renderEdgeDetail(panel, edge) {
  const pills = (edge.controllingDopamineIds || [])
    .map((nodeId) => `<span class="pill">${nodeId}</span>`)
    .join('');

  panel.innerHTML = `
    <div class="detail-card">
      <p class="detail-kicker">Selected edge</p>
      <h2 class="detail-title">${edge.id}</h2>
      <table class="detail-table">
        <tr><th>Source</th><td><code>${edge.source}</code></td></tr>
        <tr><th>Target</th><td><code>${edge.target}</code></td></tr>
        <tr><th>Layer</th><td>${LAYER_LABELS[edge.layerKey]}</td></tr>
        <tr><th>Order index</th><td>${edge.orderIndex}</td></tr>
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
  if (state.selectedEdgeId) {
    const edge = payload.edges.find((item) => item.id === state.selectedEdgeId);
    if (edge) {
      renderEdgeDetail(panel, edge);
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
    if (!node.data('isDopamine')) {
      return;
    }
    state.selectedDopamineId = node.id();
    state.selectedEdgeId = null;
    applyGraphSelection();
    renderDetailPanel();
  });

  cy.on('tap', 'edge', (event) => {
    state.selectedEdgeId = event.target.id();
    state.selectedDopamineId = null;
    applyGraphSelection();
    renderDetailPanel();
  });

  cy.on('tap', (event) => {
    if (event.target === cy) {
      state.selectedDopamineId = null;
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
  const placeholder = field.optional ? 'placeholder="auto"' : '';
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

function renderConfigForm() {
  const container = document.getElementById('config-fields');
  container.innerHTML = FORM_FIELDS.map(createFieldMarkup).join('');
}

function fillForm(config) {
  FORM_FIELDS.forEach((field) => {
    const element = document.getElementById(`field-${field.key}`);
    if (!element) {
      return;
    }
    if (field.type === 'checkbox') {
      element.checked = Boolean(config[field.key]);
    } else if (field.optional && (config[field.key] === null || config[field.key] === undefined)) {
      element.value = '';
    } else {
      element.value = config[field.key] ?? '';
    }
  });
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
  const runBtn = document.getElementById('btn-run');
  const stopBtn = document.getElementById('btn-stop');
  const running = Boolean(state.activeRun && ['running', 'starting', 'stopping'].includes(state.activeRun.status));
  runBtn.disabled = running;
  stopBtn.disabled = !running;
}

function formatMetric(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 'n/a';
  }
  return Number(value).toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 'n/a';
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
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
  const payload = state.activeRun?.graph_payload || state.defaultGraphPayload;
  ensureGraph(payload);
  renderDetailPanel();
  renderStatusPanel();
}

async function loadDefaults() {
  const payload = await fetchJson('/api/default-config');
  state.defaultConfig = payload.config;
  state.taskNames = payload.task_names || [];
  state.defaultGraphPayload = payload.graph_payload || null;
  renderConfigForm();
  fillForm(state.defaultConfig);
  syncGraphFromState();
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

async function handleRun() {
  try {
    const payload = readFormPayload();
    const result = await fetchJson('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
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
  document.getElementById('btn-run').addEventListener('click', handleRun);
  document.getElementById('btn-stop').addEventListener('click', handleStop);
  document.getElementById('btn-refresh').addEventListener('click', () => refreshState(true).catch((error) => {
    setFlashMessage(error.message, 'error');
  }));
  document.getElementById('btn-reset-form').addEventListener('click', () => {
    if (state.defaultConfig) {
      fillForm(state.defaultConfig);
      if (!state.activeRun) {
        state.selectedDopamineId = null;
        state.selectedEdgeId = null;
        syncGraphFromState();
      }
      setFlashMessage('Form reset to default configuration.', 'info');
    }
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
