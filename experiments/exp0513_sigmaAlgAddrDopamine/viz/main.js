const ARCH = {
  inputDim: 1,
  trunk1Dim: 16,
  trunk2Dim: 16,
  yDim: 1,
  qDim: 9,
};

const NODE_COLORS = {
  input: '#4a90d9',
  hidden: '#7f848f',
  y: '#d94a4a',
  q: '#2d936c',
};

const EDGE_COLORS = {
  default: '#6a6f7a',
  highlight: '#0b63ce',
  selected: '#d95f0e',
  dimmed: '#c8d0db',
};

const LAYER_LABELS = {
  'input->trunk1': 'Input -> Trunk L1',
  'trunk1->trunk2': 'Trunk L1 -> Trunk L2',
  'trunk2->y': 'Trunk L2 -> y head',
  'trunk2->q': 'Trunk L2 -> q head',
};

const FORM_FIELDS = [
  { key: 'run_name', label: 'Run name', type: 'text', wide: true },
  { key: 'task_name', label: 'Task name', type: 'select' },
  { key: 'seed', label: 'Seed', type: 'number', step: '1' },
  { key: 'epochs', label: 'Epochs', type: 'number', step: '1' },
  { key: 'lambda', label: 'Lambda', type: 'number', step: '0.01' },
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
  activeRun: null,
  cy: null,
  data: null,
  selectedQId: null,
  selectedEdgeId: null,
  pollHandle: null,
};

function createNode(id, label, kind, column, row, rowCount) {
  const xPositions = [-520, -240, 40, 320, 540];
  const x = xPositions[column] ?? 0;
  const spacing = kind === 'q' ? 54 : 46;
  const offset = (rowCount - 1) / 2;
  const y = (row - offset) * spacing;

  return {
    group: 'nodes',
    data: {
      id,
      label,
      kind,
      color: NODE_COLORS[kind],
    },
    position: { x, y },
    classes: kind === 'q' ? 'q-node' : '',
  };
}

function buildNodes() {
  const nodes = [];
  nodes.push(createNode('x0', 'x0', 'input', 0, 0, 1));

  for (let i = 0; i < ARCH.trunk1Dim; i += 1) {
    nodes.push(createNode(`h1_${i}`, `h1_${i}`, 'hidden', 1, i, ARCH.trunk1Dim));
  }
  for (let i = 0; i < ARCH.trunk2Dim; i += 1) {
    nodes.push(createNode(`h2_${i}`, `h2_${i}`, 'hidden', 2, i, ARCH.trunk2Dim));
  }

  nodes.push(createNode('y0', 'y0', 'y', 3, 0, 1));

  for (let i = 0; i < ARCH.qDim; i += 1) {
    nodes.push(createNode(`q${i}`, `q${i}`, 'q', 4, i, ARCH.qDim));
  }

  return nodes;
}

function buildForwardEdges() {
  const edges = [];

  for (let outIdx = 0; outIdx < ARCH.trunk1Dim; outIdx += 1) {
    edges.push({
      id: `trunk1[${outIdx},0]`,
      source: 'x0',
      target: `h1_${outIdx}`,
      layerKey: 'input->trunk1',
    });
  }

  for (let outIdx = 0; outIdx < ARCH.trunk2Dim; outIdx += 1) {
    for (let inIdx = 0; inIdx < ARCH.trunk1Dim; inIdx += 1) {
      edges.push({
        id: `trunk2[${outIdx},${inIdx}]`,
        source: `h1_${inIdx}`,
        target: `h2_${outIdx}`,
        layerKey: 'trunk1->trunk2',
      });
    }
  }

  for (let inIdx = 0; inIdx < ARCH.trunk2Dim; inIdx += 1) {
    edges.push({
      id: `y_head[0,${inIdx}]`,
      source: `h2_${inIdx}`,
      target: 'y0',
      layerKey: 'trunk2->y',
    });
  }

  for (let outIdx = 0; outIdx < ARCH.qDim; outIdx += 1) {
    for (let inIdx = 0; inIdx < ARCH.trunk2Dim; inIdx += 1) {
      edges.push({
        id: `q_head[${outIdx},${inIdx}]`,
        source: `h2_${inIdx}`,
        target: `q${outIdx}`,
        layerKey: 'trunk2->q',
      });
    }
  }

  return edges.map((edge, index) => ({
    ...edge,
    orderIndex: index,
    controllingQIds: [],
  }));
}

function generateNonzeroBinaryCodes(length) {
  const codes = [];
  const total = 2 ** length;
  for (let value = 1; value < total; value += 1) {
    const bits = value
      .toString(2)
      .padStart(length, '0')
      .split('')
      .map((digit) => Number(digit));
    codes.push(bits);
  }
  return codes;
}

function compareCodes(a, b, midpoint) {
  const weightA = a.reduce((sum, bit) => sum + bit, 0);
  const weightB = b.reduce((sum, bit) => sum + bit, 0);
  const balanceDiff = Math.abs(weightA - midpoint) - Math.abs(weightB - midpoint);
  if (balanceDiff !== 0) {
    return balanceDiff;
  }
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return a[i] - b[i];
    }
  }
  return 0;
}

function buildStaticAssignment(edgeCount, m) {
  const midpoint = m / 2;
  const codes = generateNonzeroBinaryCodes(m);
  codes.sort((a, b) => compareCodes(a, b, midpoint));
  return codes.slice(0, edgeCount);
}

function enrichEdgesWithControl(edges, assignmentRows) {
  const qToEdges = new Map();
  for (let qIndex = 0; qIndex < ARCH.qDim; qIndex += 1) {
    qToEdges.set(`q${qIndex}`, []);
  }

  edges.forEach((edge, edgeIndex) => {
    const code = assignmentRows[edgeIndex];
    const controlling = [];
    code.forEach((bit, qIndex) => {
      if (bit === 1) {
        const qId = `q${qIndex}`;
        controlling.push(qId);
        qToEdges.get(qId).push(edge.id);
      }
    });
    edge.controllingQIds = controlling;
  });

  return qToEdges;
}

function buildVizData() {
  const nodes = buildNodes();
  const edges = buildForwardEdges();
  const assignmentRows = buildStaticAssignment(edges.length, ARCH.qDim);
  const qToEdges = enrichEdgesWithControl(edges, assignmentRows);

  return {
    nodes,
    edges,
    qToEdges,
    totalNodes: nodes.length,
    totalEdges: edges.length,
    totalLayers: 5,
  };
}

function createCytoscapeElements(data) {
  const edgeElements = data.edges.map((edge) => ({
    group: 'edges',
    data: {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      layerKey: edge.layerKey,
      controllingQIds: edge.controllingQIds,
      color: EDGE_COLORS.default,
      thickness: 1.6,
      opacity: 0.68,
    },
  }));

  return [...data.nodes, ...edgeElements];
}

function createCy(data) {
  return cytoscape({
    container: document.getElementById('cy'),
    elements: createCytoscapeElements(data),
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
        selector: 'node.q-node',
        style: {
          width: 28,
          height: 28,
        },
      },
      {
        selector: 'node.selected-q',
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

function renderStatusPanel() {
  const panel = document.getElementById('status-panel');
  const activeRun = state.activeRun;

  if (!activeRun) {
    panel.innerHTML = `
      <div class="detail-card">
        <p class="detail-kicker">Dashboard status</p>
        <h2 class="detail-title">Idle</h2>
        <p class="muted">
          Load the default configuration, edit any parameter in the left panel, and click <code>Run</code>
          to start a single active training session.
        </p>
      </div>
    `;
    return;
  }

  const statusClass = `status-${activeRun.status || 'idle'}`;
  const epochDisplay = `${activeRun.epoch ?? 0} / ${activeRun.epochs_total ?? '-'}`;
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
          <p class="stat-label">Epoch</p>
          <p class="stat-value">${epochDisplay}</p>
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
        <tr><th>Run dir</th><td class="subtle-path">${activeRun.run_dir || 'n/a'}</td></tr>
        <tr><th>Updated</th><td>${activeRun.updated_at || 'n/a'}</td></tr>
      </table>
      ${activeRun.error ? `<p class="muted">Error: ${escapeHtml(activeRun.error)}</p>` : ''}
    </div>
  `;
}

function renderDefaultDetail(panel, data) {
  panel.innerHTML = `
    <div class="detail-card">
      <p class="detail-kicker">Overview</p>
      <h2 class="detail-title">Structure summary</h2>
      <p class="muted">
        Click a <code>q_i</code> node to inspect which forward edges it controls,
        or click any forward edge to inspect its source, target, and controlling heads.
      </p>
    </div>
    <div class="detail-card">
      <p class="detail-kicker">Network scale</p>
      <div class="stats-grid">
        <div class="stat-box">
          <p class="stat-label">Layers</p>
          <p class="stat-value">${data.totalLayers}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Nodes</p>
          <p class="stat-value">${data.totalNodes}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Controllable edges</p>
          <p class="stat-value">${data.totalEdges}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">q heads</p>
          <p class="stat-value">${ARCH.qDim}</p>
        </div>
      </div>
    </div>
  `;
}

function buildLayerBreakdownHtml(layerCounts) {
  const orderedLayerKeys = ['input->trunk1', 'trunk1->trunk2', 'trunk2->y', 'trunk2->q'];
  return orderedLayerKeys
    .map((layerKey) => {
      const controlled = layerCounts[layerKey] ?? 0;
      const layerTotal = {
        'input->trunk1': 16,
        'trunk1->trunk2': 256,
        'trunk2->y': 16,
        'trunk2->q': 144,
      }[layerKey];
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

function renderQDetail(panel, qId, data) {
  const controlledEdgeIds = data.qToEdges.get(qId) ?? [];
  const controlledEdges = data.edges.filter((edge) => controlledEdgeIds.includes(edge.id));
  const layerCounts = controlledEdges.reduce((acc, edge) => {
    acc[edge.layerKey] = (acc[edge.layerKey] ?? 0) + 1;
    return acc;
  }, {});
  const coverage = ((controlledEdges.length / data.totalEdges) * 100).toFixed(1);

  panel.innerHTML = `
    <div class="detail-card">
      <p class="detail-kicker">Selected q head</p>
      <h2 class="detail-title">${qId}</h2>
      <p class="muted">
        Highlighting every forward edge whose assignment code includes <code>${qId}</code>.
      </p>
    </div>
    <div class="detail-card">
      <p class="detail-kicker">Control summary</p>
      <div class="stats-grid">
        <div class="stat-box">
          <p class="stat-label">Controlled edges</p>
          <p class="stat-value">${controlledEdges.length}</p>
        </div>
        <div class="stat-box">
          <p class="stat-label">Coverage</p>
          <p class="stat-value">${coverage}%</p>
        </div>
      </div>
      <div class="layer-breakdown">
        ${buildLayerBreakdownHtml(layerCounts)}
      </div>
    </div>
  `;
}

function renderEdgeDetail(panel, edge) {
  const qPills = edge.controllingQIds
    .map((qId) => `<span class="pill">${qId}</span>`)
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
      <p class="detail-kicker">Controlling q heads</p>
      <div class="pill-list">${qPills}</div>
    </div>
  `;
}

function renderDetailPanel() {
  const panel = document.getElementById('detail-panel');
  if (state.selectedQId) {
    renderQDetail(panel, state.selectedQId, state.data);
    return;
  }
  if (state.selectedEdgeId) {
    const edge = state.data.edges.find((item) => item.id === state.selectedEdgeId);
    if (edge) {
      renderEdgeDetail(panel, edge);
      return;
    }
  }
  renderDefaultDetail(panel, state.data);
}

function clearEdgeStyling(cy) {
  cy.edges().forEach((edgeEle) => {
    edgeEle.data('color', EDGE_COLORS.default);
    edgeEle.data('thickness', 1.6);
    edgeEle.data('opacity', 0.68);
  });
  cy.nodes('.selected-q').removeClass('selected-q');
}

function highlightEdgesForQ(cy, qId) {
  clearEdgeStyling(cy);
  cy.getElementById(qId).addClass('selected-q');

  cy.edges().forEach((edgeEle) => {
    const controlling = edgeEle.data('controllingQIds') || [];
    if (controlling.includes(qId)) {
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
  if (state.selectedQId) {
    highlightEdgesForQ(state.cy, state.selectedQId);
  } else if (state.selectedEdgeId) {
    highlightSelectedEdge(state.cy, state.selectedEdgeId);
  } else {
    clearEdgeStyling(state.cy);
  }
}

function bindGraphInteractions(cy, data) {
  cy.on('tap', 'node', (event) => {
    const node = event.target;
    if (node.data('kind') !== 'q') {
      return;
    }
    state.selectedQId = node.id();
    state.selectedEdgeId = null;
    applyGraphSelection();
    renderDetailPanel();
  });

  cy.on('tap', 'edge', (event) => {
    state.selectedEdgeId = event.target.id();
    state.selectedQId = null;
    applyGraphSelection();
    renderDetailPanel();
  });

  cy.on('tap', (event) => {
    if (event.target === cy) {
      state.selectedQId = null;
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
  return `
    <div class="form-field ${field.wide ? 'wide' : ''}">
      <label for="field-${field.key}">${field.label}</label>
      <input
        id="field-${field.key}"
        name="${field.key}"
        type="${field.type}"
        ${stepAttr}
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
      payload[field.key] = Number(element.value);
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

async function loadDefaults() {
  const payload = await fetchJson('/api/default-config');
  state.defaultConfig = payload.config;
  state.taskNames = payload.task_names || [];
  renderConfigForm();
  fillForm(state.defaultConfig);
}

async function refreshState(showMessage = false) {
  const payload = await fetchJson('/api/state');
  state.activeRun = payload.active_run;
  renderStatusPanel();
  renderDetailPanel();
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
    renderStatusPanel();
    updateButtons();
    setFlashMessage('Training run started.', 'success');
  } catch (error) {
    const payload = error.payload || {};
    if (payload.active_run !== undefined) {
      state.activeRun = payload.active_run;
      renderStatusPanel();
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
    renderStatusPanel();
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
  state.data = buildVizData();
  state.cy = createCy(state.data);
  bindGraphInteractions(state.cy, state.data);

  await loadDefaults();
  bindDashboardControls();
  renderStatusPanel();
  renderDetailPanel();
  await refreshState(false);
  startPolling();
}

initializeDashboard().catch((error) => {
  setFlashMessage(`Failed to initialize dashboard: ${error.message}`, 'error');
});
