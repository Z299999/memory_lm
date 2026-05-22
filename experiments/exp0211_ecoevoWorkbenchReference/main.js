/**
 * Main orchestrator for the eco-evo simulation demo.
 * Wires up simulation logic, UI controls, graph view, and chart.
 */

import { Graph } from './sim/graph.js';
import { simulationStep } from './sim/step.js';
import { Controls } from './ui/controls.js';
import { GraphView } from './ui/graph-view.js';
import { ChartView, ActivationView, WeightView } from './ui/distributions.js';
import { OutputView } from './ui/output-view.js';
import { Stats } from './ui/stats.js';

// --- State ---
let graph = null;
let t = 0;
let playing = false;
let lastBridged = new Set();
let genesisM = 3; // track m used at last reset
let activationKind = 'tanh';
let weightControlKind = 'vanilla';
let mode = 'evolve'; // 'evolve' | 'test'
let testStepIndex = 0;
let currentTest = null; // { inputIndex, amplitude, steps }
let testRunning = false; // within test mode: running vs idle

// --- UI components (initialized after DOM ready) ---
let controls, graphView, chartView, activationView, weightView, outputView, stats;

// --- Timing ---
let lastFrameTime = 0;
let accumulator = 0;

/** Compute L2 norm of output activations ‖y(t)‖₂. */
function computeOutputNorm(graph) {
  if (!graph) return 0;
  let sumSq = 0;
  for (const [, node] of graph.nodes) {
    if (node.type === 'output') {
      const a = node.activation || 0;
      sumSq += a * a;
    }
  }
  return Math.sqrt(sumSq);
}

// History of output norms for plotting: [{ t, norm }, ...]
const outputHistory = [];
let recordOutput = true; // controls whether we record and plot ‖y(t)‖₂

function updateModeUI() {
  const runtimeFs = document.getElementById('fieldset-runtime');
  const testFs = document.getElementById('fieldset-test');
  const btnTest = document.getElementById('btn-test-toggle');
  if (!runtimeFs || !testFs || !btnTest) return;

  if (mode === 'test') {
    runtimeFs.style.display = 'none';
    testFs.style.display = '';
    btnTest.textContent = 'End test';
  } else {
    runtimeFs.style.display = '';
    testFs.style.display = 'none';
    btnTest.textContent = 'Start test';
  }

  updateTopButtons();
  updateTestButtons();
}

function updateTopButtons() {
  const btnPlay = document.getElementById('btn-play');
  const btnStep = document.getElementById('btn-step');
  const btnReset = document.getElementById('btn-reset');
  if (!btnPlay || !btnStep || !btnReset) return;

  if (mode === 'test') {
    btnPlay.disabled = true;
    btnStep.disabled = true;
    // In test mode we still allow a full reset when not playing.
    btnReset.disabled = playing;
  } else {
    btnPlay.disabled = false;
    if (playing) {
      btnStep.disabled = true;
      btnReset.disabled = true;
    } else {
      btnStep.disabled = false;
      btnReset.disabled = false;
    }
  }
}

function updateTestButtons() {
  const btnInject = document.getElementById('btn-test-inject');
  const btnStop = document.getElementById('btn-test-stop-recording');
  if (!btnInject || !btnStop) return;

  if (mode !== 'test') {
    btnInject.disabled = true;
    btnStop.disabled = true;
    return;
  }

  if (testRunning) {
    btnInject.disabled = true;
    btnStop.disabled = false;
  } else {
    btnInject.disabled = false;
    btnStop.disabled = true;
  }
}

let constructionMode = 'bridge'; // 'bridge' | 'random'

/** Initialize or re-initialize the simulation. */
function reset() {
  const { m, n, kInternal, arch, activation, weightControl, construction } = controls.getStartParams();
  genesisM = Math.max(1, m); // guard against m=0
  const safeN = Math.max(1, n);
  activationKind =
    activation === 'relu' || activation === 'relu-thresh' || activation === 'identity' || activation === 'maxabs'
      ? activation
      : 'tanh';
  weightControlKind =
    weightControl === 'tanh' ||
    weightControl === 'ou' ||
    weightControl === 'hebbian' ||
    weightControl === 'hebbian-tanh' ||
    weightControl === 'hebb-ou'
      ? weightControl
      : 'vanilla';
  constructionMode = construction === 'random' ? 'random' : 'bridge';
  const safeK = Math.max(1, kInternal | 0);
  if (arch === 'mlp2') {
    graph = Graph.genesisMLP2(genesisM, safeN);
  } else {
    graph = Graph.genesis(genesisM, safeN, safeK);
  }
  recordOutput = true;
  t = 0;
  lastBridged = new Set();
  outputHistory.length = 0;
  mode = 'evolve';
  testStepIndex = 0;
  currentTest = null;
  updateModeUI();

  graphView.rebuild(graph, lastBridged);
  chartView.update(graph.degreeHistogram());
  if (weightView) {
    weightView.update(computeWeightHistogram(graph));
  }
  const outputNorm = computeOutputNorm(graph);
  outputHistory.push({ t, norm: outputNorm });
  outputView.update(outputHistory);
  stats.update(t, graph.nodeCount, graph.edgeCount, outputNorm);
  if (activationView) {
    activationView.update(computeActivationHistogram(graph));
  }
}

/** Run a single eco-evo evolution step and update the UI. */
function evolveStep() {
  if (!graph) return;

  // Use the m from genesis, not the current slider value
  const runParams = {
    ...controls.getRunParams(),
    m: genesisM,
    activation: activationKind,
    construction: constructionMode,
    weightTanh: weightControlKind === 'tanh' || weightControlKind === 'hebbian-tanh',
    useOU: weightControlKind === 'ou',
    useHebbian: weightControlKind === 'hebbian' || weightControlKind === 'hebbian-tanh',
    useHebbOU: weightControlKind === 'hebb-ou'
  };

  const events = simulationStep(graph, t, runParams);
  t++;

  // Track bridged nodes for highlighting (clear after a few steps)
  lastBridged = new Set(events.bridged);

  // Incremental graph update
  graphView.update(graph, lastBridged);

  // Update stats every step
  const outputNorm = computeOutputNorm(graph);
  if (recordOutput) {
    outputHistory.push({ t, norm: outputNorm });
    outputView.update(outputHistory);
  }
  stats.update(t, graph.nodeCount, graph.edgeCount, outputNorm);
  if (activationView) {
    activationView.update(computeActivationHistogram(graph));
  }
  if (weightView) {
    weightView.update(computeWeightHistogram(graph));
  }
  // Update degree chart every 10 steps
  if (t % 10 === 0) {
    chartView.update(graph.degreeHistogram());
  }
}

/** One impulse-response test step with frozen graph (no evolution). */
function impulseTestStep() {
  if (!graph || !currentTest) return;

  const { inputIndex, amplitude, steps, signalType } = currentTest;
  if (testStepIndex >= steps) {
    // Auto-stop test playback but stay in test mode until user ends it.
    playing = false;
    testRunning = false;
    document.getElementById('btn-play').textContent = 'Play';
    updateTestButtons();
    return;
  }

  const useImpulse = !signalType || signalType === 'impulse';

  // 1) Set input activations:
  //    - impulse: amplitude only at k=0,
  //    - constant: amplitude at all steps.
  for (let i = 0; i < genesisM; i++) {
    const node = graph.nodes.get(`x${i}`);
    if (!node) continue;
    const isActiveStep =
      useImpulse ? testStepIndex === 0 : true;
    if (i === inputIndex && isActiveStep) {
      node.activation = amplitude;
    } else {
      node.activation = 0;
    }
  }

  // 2) Forward pass for non-input nodes (same order as simulationStep).
  const { theta } = controls.getRunParams();
  const thetaVal = Number.isFinite(theta) ? theta : 0;
  const useTanhConstraint =
    weightControlKind === 'tanh' || weightControlKind === 'hebbian-tanh';
  let actFn;
  if (activationKind === 'relu') {
    actFn = x => (x > 0 ? x : 0);
  } else if (activationKind === 'relu-thresh') {
    actFn = x => {
      const s = x - thetaVal;
      return s > 0 ? s : 0;
    };
  } else if (activationKind === 'identity') {
    actFn = x => x;
  } else if (activationKind === 'maxabs') {
    actFn = null;
  } else {
    actFn = x => Math.tanh(x);
  }
  const order = graph.getForwardOrder();

  // Use previous activations to advance one "hop" per test step.
  const prevActivation = new Map();
  for (const [id, node] of graph.nodes) {
    prevActivation.set(id, node.activation || 0);
  }

  for (const nodeId of order) {
    const node = graph.nodes.get(nodeId);
    if (!node || node.type === 'input') continue;

    const inEdges = graph.adjIn.get(nodeId);
    if (activationKind === 'maxabs') {
      let best = 0;
      let hasInput = false;
      if (inEdges) {
        for (const eid of inEdges) {
          const edge = graph.edges.get(eid);
          if (!edge) continue;
          const x = prevActivation.get(edge.src) || 0;
          let contrib;
          if (useTanhConstraint) {
            contrib = Math.tanh(edge.w * x);
          } else {
            contrib = edge.w * x;
          }
          if (!hasInput || Math.abs(contrib) > Math.abs(best)) {
            best = contrib;
            hasInput = true;
          }
        }
      }
      node.activation = hasInput ? best : 0;
    } else {
      let z = 0;
      if (inEdges) {
        for (const eid of inEdges) {
          const edge = graph.edges.get(eid);
          if (!edge) continue;
          const x = prevActivation.get(edge.src) || 0;
          if (useTanhConstraint) {
            z += Math.tanh(edge.w * x);
          } else {
            z += edge.w * x;
          }
        }
      }
      node.activation = actFn(z);
    }
  }

  // No bridging / weight updates / cleanup: graph structure is frozen.
  lastBridged = new Set();

  // Update view and statistics. Use local testStepIndex on the time axis.
  graphView.update(graph, lastBridged);
  if (activationView) {
    activationView.update(computeActivationHistogram(graph));
  }

  const outputNorm = computeOutputNorm(graph);
  if (recordOutput) {
    outputHistory.push({ t: testStepIndex, norm: outputNorm });
    outputView.update(outputHistory);
  }
  stats.update(t, graph.nodeCount, graph.edgeCount, outputNorm);
  if (weightView) {
    weightView.update(computeWeightHistogram(graph));
  }

  testStepIndex++;
}

/** Run a full impulse-response experiment in test mode in one shot. */
function runImpulseOnce() {
  if (!graph) return;
  if (mode !== 'test') return;

  const params = controls.getTestParams();
  const { inputIndex, amplitude, steps, signalType } = params;
  currentTest = params;
  testStepIndex = 0;

  // Reset activations and history
  for (const [, node] of graph.nodes) {
    node.activation = 0;
  }

  // Determine starting time index so that multiple impulse
  // tests within the same session can be concatenated on
  // the output plot instead of overwriting previous runs.
  const startT =
    outputHistory.length
      ? outputHistory[outputHistory.length - 1].t + 1
      : 0;

  const useTanhConstraint =
    weightControlKind === 'tanh' || weightControlKind === 'hebbian-tanh';
  const { theta: thetaRun } = controls.getRunParams();
  const thetaRunVal = Number.isFinite(thetaRun) ? thetaRun : 0;
  let actFn;
  if (activationKind === 'relu') {
    actFn = x => (x > 0 ? x : 0);
  } else if (activationKind === 'relu-thresh') {
    actFn = x => {
      const s = x - thetaRunVal;
      return s > 0 ? s : 0;
    };
  } else if (activationKind === 'identity') {
    actFn = x => x;
  } else if (activationKind === 'maxabs') {
    actFn = null;
  } else {
    actFn = x => Math.tanh(x);
  }

  const useImpulse = !signalType || signalType === 'impulse';

  for (let k = 0; k < steps; k++) {
    // Inputs:
    //   - impulse: amplitude only at k=0 on selected channel,
    //   - constant: amplitude for all k on selected channel,
    //   - other channels: 0.
    for (let i = 0; i < genesisM; i++) {
      const node = graph.nodes.get(`x${i}`);
      if (!node) continue;
      const isActiveStep =
        useImpulse ? k === 0 : true;
      if (i === inputIndex && isActiveStep) {
        node.activation = amplitude;
      } else {
        node.activation = 0;
      }
    }

    // Forward pass for non-input nodes using previous activations so that
    // each test step advances the impulse by one hop.
    const prevActivation = new Map();
    for (const [id, node] of graph.nodes) {
      prevActivation.set(id, node.activation || 0);
    }

    // Forward pass for non-input nodes.
    const order = graph.getForwardOrder();
    for (const nodeId of order) {
      const node = graph.nodes.get(nodeId);
      if (!node || node.type === 'input') continue;

      const inEdges = graph.adjIn.get(nodeId);
      if (activationKind === 'maxabs') {
        let best = 0;
        let hasInput = false;
        if (inEdges) {
          for (const eid of inEdges) {
            const edge = graph.edges.get(eid);
            if (!edge) continue;
            const x = prevActivation.get(edge.src) || 0;
            let contrib;
            if (useTanhConstraint) {
              contrib = Math.tanh(edge.w * x);
            } else {
              contrib = edge.w * x;
            }
            if (!hasInput || Math.abs(contrib) > Math.abs(best)) {
              best = contrib;
              hasInput = true;
            }
          }
        }
        node.activation = hasInput ? best : 0;
      } else {
        let z = 0;
        if (inEdges) {
          for (const eid of inEdges) {
            const edge = graph.edges.get(eid);
            if (!edge) continue;
            const x = prevActivation.get(edge.src) || 0;
            if (useTanhConstraint) {
              z += Math.tanh(edge.w * x);
            } else {
              z += edge.w * x;
            }
          }
        }
        node.activation = actFn(z);
      }
    }

    const outputNorm = computeOutputNorm(graph);
    outputHistory.push({ t: startT + k, norm: outputNorm });
  }

  // Final view/state corresponds to the last step.
  lastBridged = new Set();
  graphView.update(graph, lastBridged);
  if (activationView) {
    activationView.update(computeActivationHistogram(graph));
  }
  if (weightView) {
    weightView.update(computeWeightHistogram(graph));
  }
  const finalNorm =
    outputHistory.length
      ? outputHistory[outputHistory.length - 1].norm
      : computeOutputNorm(graph);
  outputView.update(outputHistory);
  stats.update(t, graph.nodeCount, graph.edgeCount, finalNorm);
}

/** Dispatch a single step depending on mode. */
function step() {
  if (mode === 'test') {
    impulseTestStep();
  } else {
    evolveStep();
  }
}

/** Animation loop using requestAnimationFrame + accumulator. */
function loop(timestamp) {
  if (!playing) return;

  if (lastFrameTime === 0) lastFrameTime = timestamp;
  const dt = (timestamp - lastFrameTime) / 1000; // seconds
  lastFrameTime = timestamp;

  const speed = controls.getSpeed(); // steps per second
  const stepInterval = 1 / speed;

  accumulator += dt;

  // Cap to avoid spiral of death
  const maxStepsPerFrame = Math.min(speed, 50);
  let stepsThisFrame = 0;

  while (accumulator >= stepInterval && stepsThisFrame < maxStepsPerFrame) {
    step();
    accumulator -= stepInterval;
    stepsThisFrame++;
  }

  // Clamp leftover accumulator to avoid runaway catch-up
  if (accumulator > stepInterval) {
    accumulator = stepInterval;
  }

  requestAnimationFrame(loop);
}

function play() {
  if (playing) return;
  playing = true;
  if (mode === 'test') {
    testRunning = true;
    updateTestButtons();
  }
  updateTopButtons();
  lastFrameTime = 0;
  accumulator = 0;
  document.getElementById('btn-play').textContent = 'Pause';
  requestAnimationFrame(loop);
}

function pause() {
  playing = false;
  document.getElementById('btn-play').textContent = 'Play';
  if (mode === 'test') {
    testRunning = false;
    updateTestButtons();
  }
   updateTopButtons();
}

function togglePlay() {
  if (playing) pause();
  else play();
}

function computeActivationHistogram(graph) {
  const activations = [];
  for (const [, node] of graph.nodes) {
    if (node.activation == null || Number.isNaN(node.activation)) continue;
    activations.push(node.activation);
  }
  if (activations.length === 0) {
    return { centers: [], counts: [] };
  }

  let maxAbs = 0;
  for (const a of activations) {
    const v = Math.abs(a);
    if (v > maxAbs) maxAbs = v;
  }
  // Window tightly around current activations; only clip extreme outliers.
  // This avoids everything collapsing into a single central bin when
  // activations are very small.
  let limit;
  if (maxAbs < 1e-6) {
    limit = 1e-3;
  } else {
    limit = Math.min(maxAbs * 1.2, 10);
  }
  const minVal = -limit;
  const maxVal = limit;
  const bins = 21;
  const width = (maxVal - minVal) / bins;
  const counts = new Array(bins).fill(0);

  for (const a of activations) {
    let idx = Math.floor((Math.max(Math.min(a, maxVal - 1e-9), minVal) - minVal) / width);
    if (idx < 0) idx = 0;
    if (idx >= bins) idx = bins - 1;
    counts[idx]++;
  }

  const centers = [];
  for (let i = 0; i < bins; i++) {
    centers.push(minVal + (i + 0.5) * width);
  }
  return { centers, counts };
}

function computeWeightHistogram(graph) {
  const weights = [];
  for (const [, edge] of graph.edges) {
    if (edge.w == null || Number.isNaN(edge.w)) continue;
    weights.push(edge.w);
  }
  if (weights.length === 0) {
    return { centers: [], counts: [] };
  }

  let maxAbs = 0;
  for (const w of weights) {
    const v = Math.abs(w);
    if (v > maxAbs) maxAbs = v;
  }
  let limit;
  if (maxAbs < 1e-6) {
    limit = 1e-3;
  } else {
    limit = Math.min(maxAbs * 1.2, 10);
  }
  const minVal = -limit;
  const maxVal = limit;
  const bins = 21;
  const width = (maxVal - minVal) / bins;
  const counts = new Array(bins).fill(0);

  for (const w of weights) {
    let clamped = Math.max(Math.min(w, maxVal - 1e-9), minVal);
    let idx = Math.floor((clamped - minVal) / width);
    if (idx < 0) idx = 0;
    if (idx >= bins) idx = bins - 1;
    counts[idx]++;
  }

  const centers = [];
  for (let i = 0; i < bins; i++) {
    centers.push(minVal + (i + 0.5) * width);
  }
  return { centers, counts };
}

function startImpulseTest() {
  if (!graph) return;
  pause();
  // Reset activations to zero
  for (const [, node] of graph.nodes) {
    node.activation = 0;
  }
  currentTest = controls.getTestParams();
  testStepIndex = 0;
  mode = 'test';
  testRunning = false;
  updateModeUI();
}

function updateWeightDynamicsUI() {
  const sel = document.getElementById('param-weight-control');
  const muSlider = document.getElementById('param-mu');
  const ouSlider = document.getElementById('param-ou-mean');
  const etaHebbSlider = document.getElementById('param-eta-hebb');
  const hebbThreshSlider = document.getElementById('param-hebb-thresh');
  if (!sel || !muSlider || !ouSlider) return;
  const muLabel = document.getElementById('label-mu');
  const ouLabel = document.getElementById('label-ou-mean');
  const etaHebbLabel = document.getElementById('label-eta-hebb');
  const hebbThreshLabel = document.getElementById('label-hebb-thresh');
  const modeVal = sel.value;

  const setSliderState = (slider, label, enabled) => {
    if (!slider || !label) return;
    slider.disabled = !enabled;
    if (enabled) {
      label.classList.remove('disabled-slider');
    } else {
      label.classList.add('disabled-slider');
    }
  };

  if (modeVal === 'ou') {
    setSliderState(muSlider, muLabel, false);
    setSliderState(ouSlider, ouLabel, true);
    if (etaHebbSlider && etaHebbLabel) {
      setSliderState(etaHebbSlider, etaHebbLabel, false);
    }
    if (hebbThreshSlider && hebbThreshLabel) {
      setSliderState(hebbThreshSlider, hebbThreshLabel, false);
    }
  } else if (modeVal === 'hebbian' || modeVal === 'hebbian-tanh' || modeVal === 'hebb-ou') {
    setSliderState(muSlider, muLabel, false);
    setSliderState(ouSlider, ouLabel, false);
    if (etaHebbSlider && etaHebbLabel) {
      setSliderState(etaHebbSlider, etaHebbLabel, true);
    }
    if (hebbThreshSlider && hebbThreshLabel) {
      setSliderState(hebbThreshSlider, hebbThreshLabel, true);
    }
  } else {
    setSliderState(muSlider, muLabel, true);
    setSliderState(ouSlider, ouLabel, false);
    if (etaHebbSlider && etaHebbLabel) {
      setSliderState(etaHebbSlider, etaHebbLabel, false);
    }
    if (hebbThreshSlider && hebbThreshLabel) {
      setSliderState(hebbThreshSlider, hebbThreshLabel, false);
    }
  }
}

function endImpulseTest() {
  pause();
  mode = 'evolve';
  currentTest = null;
  testStepIndex = 0;
  testRunning = false;
  if (graph) {
    for (const [, node] of graph.nodes) {
      node.activation = 0;
    }
  }
  // Clear output history so the main mode starts a fresh trace
  // after leaving test mode.
  outputHistory.length = 0;
  outputView.update(outputHistory);
  const outputNorm = computeOutputNorm(graph);
  stats.update(t, graph.nodeCount, graph.edgeCount, outputNorm);
  if (activationView) {
    activationView.update(computeActivationHistogram(graph));
  }
  if (weightView) {
    weightView.update(computeWeightHistogram(graph));
  }
  updateModeUI();
}

function updateConstructionUI() {
  const constructionSelect = document.getElementById('param-construction');
  const tBridgeSlider = document.getElementById('param-tbridge');
  const omegaSlider = document.getElementById('param-omega');
  const kSlider = document.getElementById('param-K');
  const pAddEdgeSlider = document.getElementById('param-p-add-edge');
  const pAddNodeSlider = document.getElementById('param-p-add-node');
  const randAlphaSlider = document.getElementById('param-rand-alpha');
  const randDMaxSlider = document.getElementById('param-rand-dmax');
  if (!constructionSelect || !tBridgeSlider || !omegaSlider || !kSlider) return;

  const tBridgeLabel = tBridgeSlider.closest('label');
  const omegaLabel = omegaSlider.closest('label');
  const kLabel = kSlider.closest('label');
  const pAddEdgeLabel = pAddEdgeSlider ? pAddEdgeSlider.closest('label') : null;
  const pAddNodeLabel = pAddNodeSlider ? pAddNodeSlider.closest('label') : null;
  const randAlphaLabel = randAlphaSlider ? randAlphaSlider.closest('label') : null;
  const randDMaxLabel = randDMaxSlider ? randDMaxSlider.closest('label') : null;

  const isRandom = constructionSelect.value === 'random';
  tBridgeSlider.disabled = isRandom;
  omegaSlider.disabled = isRandom;
  kSlider.disabled = isRandom;
  if (pAddEdgeSlider) pAddEdgeSlider.disabled = !isRandom;
  if (pAddNodeSlider) pAddNodeSlider.disabled = !isRandom;
  if (randAlphaSlider) randAlphaSlider.disabled = !isRandom;
  if (randDMaxSlider) randDMaxSlider.disabled = !isRandom;

  const applyClass = (label, disabled) => {
    if (!label) return;
    if (disabled) label.classList.add('disabled-slider');
    else label.classList.remove('disabled-slider');
  };

  applyClass(tBridgeLabel, isRandom);
  applyClass(omegaLabel, isRandom);
  applyClass(kLabel, isRandom);
  applyClass(pAddEdgeLabel, !isRandom);
  applyClass(pAddNodeLabel, !isRandom);
  applyClass(randAlphaLabel, !isRandom);
  applyClass(randDMaxLabel, !isRandom);
}

// --- Initialize ---
// ES modules are deferred, so the DOM is ready when this runs.
function init() {
  controls = new Controls();
  graphView = new GraphView('cy');
  chartView = new ChartView('degree-chart');
  activationView = new ActivationView('activation-chart');
  weightView = new WeightView('weight-chart');
  outputView = new OutputView('output-chart', 'output-window');
  stats = new Stats();

  // Weight dynamics UI: disable/enable μ vs OU mean based on Genesis selection
  const weightCtrlSelect = document.getElementById('param-weight-control');
  if (weightCtrlSelect) {
    weightCtrlSelect.addEventListener('change', updateWeightDynamicsUI);
    // Initialize once based on current selection
    updateWeightDynamicsUI();
  }

  // Construction UI: disable T_bridge, omega, K when using random construction.
  const constructionSelect = document.getElementById('param-construction');
  if (constructionSelect) {
    constructionSelect.addEventListener('change', updateConstructionUI);
    updateConstructionUI();
  }

  // Activation UI: enable θ slider only for thresholded ReLU.
  const activationSelect = document.getElementById('param-activation');
  const thetaSlider = document.getElementById('param-theta');
  if (activationSelect && thetaSlider) {
    const thetaLabel = thetaSlider.closest('label');
    const updateThetaUI = () => {
      const isThresh = activationSelect.value === 'relu-thresh';
      thetaSlider.disabled = !isThresh;
      if (thetaLabel) {
        if (isThresh) thetaLabel.classList.remove('disabled-slider');
        else thetaLabel.classList.add('disabled-slider');
      }
    };
    activationSelect.addEventListener('change', updateThetaUI);
    updateThetaUI();
  }

  // Button bindings
  document.getElementById('btn-play').addEventListener('click', togglePlay);
  document.getElementById('btn-step').addEventListener('click', () => {
    if (!playing) step();
  });
  document.getElementById('btn-reset').addEventListener('click', () => {
    pause();
    reset();
  });

  // Help button: show tutorial overlay
  const helpBtn = document.getElementById('btn-help');
  const helpOverlay = document.getElementById('help-overlay');
  const helpClose = document.getElementById('btn-help-close');
  if (helpBtn && helpOverlay && helpClose) {
    helpBtn.addEventListener('click', () => {
      helpOverlay.style.display = 'flex';
    });
    helpClose.addEventListener('click', () => {
      helpOverlay.style.display = 'none';
    });
    helpOverlay.addEventListener('click', evt => {
      if (evt.target === helpOverlay) {
        helpOverlay.style.display = 'none';
      }
    });
  }

  // Impulse test toggle button
  document.getElementById('btn-test-toggle').addEventListener('click', () => {
    if (mode === 'test') {
      endImpulseTest();
    } else {
      startImpulseTest();
    }
  });

  // Inject impulse once in test mode
  document.getElementById('btn-test-inject').addEventListener('click', () => {
    if (!graph || mode !== 'test' || testRunning) return;
    recordOutput = true;
    // Reset activations and history for a fresh run
    for (const [, node] of graph.nodes) {
      node.activation = 0;
    }
    outputHistory.length = 0;
    outputView.update(outputHistory);
    currentTest = controls.getTestParams();
    testStepIndex = 0;
    testRunning = true;
    updateTestButtons();
    play();
  });

  // Stop current impulse test run
  document.getElementById('btn-test-stop-recording').addEventListener('click', () => {
    if (mode !== 'test') return;
    pause();
  });

  // Handle window resize
  window.addEventListener('resize', () => graphView.resize());

  // Initial reset
  reset();
  updateModeUI();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
