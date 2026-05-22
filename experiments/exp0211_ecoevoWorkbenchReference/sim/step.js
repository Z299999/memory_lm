/**
 * Simulation step logic for the eco-evo model.
 * Implements the exact step order specified in the model.
 */

import { getInputValue } from './input.js';

// Fixed constants that are not currently exposed in the UI
const EPS_DEFAULT = 1e-3;   // default near-zero threshold

// Bridge-specific constant: sqrt(2) / 2, used for the canonical 2-cycle
// and the xi -> z1 fan-in weight.
const BRIDGE_BASE = Math.SQRT1_2;

/** Standard normal via Box-Muller transform. */
function randn() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 || 1e-30)) * Math.cos(2 * Math.PI * u2);
}

// --- Random growth helpers (used when construction === 'random') ---
/**
 * Sample an integer distance d ∈ {1, ..., dMax} from a discrete power-law
 * distribution P(d) ∝ 1 / d^alpha.
 */
function sampleDistance(dMax, alpha) {
  const weights = [];
  let total = 0;
  for (let d = 1; d <= dMax; d++) {
    const w = 1 / Math.pow(d, alpha);
    weights.push({ d, w });
    total += w;
  }
  let u = Math.random() * total;
  for (const { d, w } of weights) {
    if (u <= w) return d;
    u -= w;
  }
  return dMax;
}

/**
 * Randomly add a new weak edge from an internal or input node to another node
 * that is "nearby" in graph distance (power-law over distances).
 * - Sources: internal or input nodes.
 * - Targets: if source is input, only internal nodes;
 *            if source is internal, internal or output nodes.
 * We avoid duplicate edges.
 */
function randomEdgeGrowth(graph, pAddEdge, alpha, dMax) {
  if (pAddEdge <= 0) return null;
  if (Math.random() >= pAddEdge) return null;

  const candidateSrcIds = [];
  for (const [id, node] of graph.nodes) {
    if (node.type === 'internal' || node.type === 'input') {
      candidateSrcIds.push(id);
    }
  }
  if (candidateSrcIds.length === 0) return null;

  const srcId = candidateSrcIds[Math.floor(Math.random() * candidateSrcIds.length)];
  const srcNode = graph.nodes.get(srcId);
  const srcType = srcNode ? srcNode.type : 'internal';

  const targetDistance = sampleDistance(dMax, alpha);

  // BFS in the underlying undirected graph to find nodes at the target distance.
  const queue = [srcId];
  const dist = new Map([[srcId, 0]]);
  const candidates = [];

  while (queue.length > 0) {
    const current = queue.shift();
    const d = dist.get(current);
    if (d >= targetDistance) continue;

    const neighbors = new Set();
    const outEdges = graph.adjOut.get(current);
    if (outEdges) {
      for (const eid of outEdges) {
        const e = graph.edges.get(eid);
        if (e) neighbors.add(e.dst);
      }
    }
    const inEdges = graph.adjIn.get(current);
    if (inEdges) {
      for (const eid of inEdges) {
        const e = graph.edges.get(eid);
        if (e) neighbors.add(e.src);
      }
    }

    for (const nb of neighbors) {
      if (dist.has(nb)) continue;
      const nd = d + 1;
      dist.set(nb, nd);
      if (nd < targetDistance) {
        queue.push(nb);
      } else if (nd === targetDistance) {
        const node = graph.nodes.get(nb);
        if (!node) continue;
        if (nb === srcId) continue;
        // If the source is an input node, only connect to internal nodes.
        if (srcType === 'input') {
          if (node.type === 'internal') {
            candidates.push(nb);
          }
        } else if (node.type === 'internal' || node.type === 'output') {
          candidates.push(nb);
        }
      }
    }
  }

  if (candidates.length === 0) return null;

  const dstId = candidates[Math.floor(Math.random() * candidates.length)];

  // Avoid duplicate edges from srcId to dstId.
  const outEdgesFromSrc = graph.adjOut.get(srcId);
  if (outEdgesFromSrc) {
    for (const eid of outEdgesFromSrc) {
      const e = graph.edges.get(eid);
      if (e && e.dst === dstId) {
        return null;
      }
    }
  }

  const wNew = 0.02 * randn(); // small initial weight
  graph.addEdge(srcId, dstId, wNew);
  return srcId;
}

/**
 * Randomly insert a new internal node along an existing edge, without removing
 * the original edge. The original edge remains unchanged; the new node gets
 * two weak incident edges so that the structural perturbation is small.
 */
function randomNodeGrowth(graph, pAddNode) {
  if (pAddNode <= 0) return null;
  if (Math.random() >= pAddNode) return null;

  const candidateEdges = [];
  for (const [eid, edge] of graph.edges) {
    const src = graph.nodes.get(edge.src);
    const dst = graph.nodes.get(edge.dst);
    if (!src || !dst) continue;
    const isInternalInternal =
      src.type === 'internal' && dst.type === 'internal';
    const isInternalOutput =
      src.type === 'internal' && dst.type === 'output';
    const isInputInternal =
      src.type === 'input' && dst.type === 'internal';
    if (isInternalInternal || isInternalOutput || isInputInternal) {
      candidateEdges.push(edge);
    }
  }
  if (candidateEdges.length === 0) return null;

  const edge = candidateEdges[Math.floor(Math.random() * candidateEdges.length)];
  const srcId = edge.src;
  const dstId = edge.dst;

  const newId = graph.newInternalId();
  graph.addNode(newId, 'internal');

  const base = 0.02;
  const wIn = base * randn();
  const wOut = base * randn();
  graph.addEdge(srcId, newId, wIn);
  graph.addEdge(newId, dstId, wOut);
  return newId;
}

/**
 * Recursively prune internal sink nodes (nodes with out-degree 0).
 * Starting from `startId`, any internal node with no outgoing edges is
 * removed together with all its incoming edges. Predecessors of a
 * removed node are re-checked, propagating the pruning upstream.
 */
function pruneSinkCascade(graph, startId, events) {
  const queue = [startId];
  const visited = new Set();

  while (queue.length > 0) {
    const nodeId = queue.pop();
    if (visited.has(nodeId)) continue;
    visited.add(nodeId);

    const node = graph.nodes.get(nodeId);
    if (!node) continue;
    if (node.type === 'input' || node.type === 'output') continue;

    if (graph.outDegree(nodeId) > 0) continue;

    const inEdges = graph.adjIn.get(nodeId)
      ? Array.from(graph.adjIn.get(nodeId))
      : [];
    const predecessors = [];
    for (const eid of inEdges) {
      const e = graph.edges.get(eid);
      if (e) predecessors.push(e.src);
    }

    graph.removeNode(nodeId);
    events.removedNodes++;

    for (const srcId of predecessors) {
      queue.push(srcId);
    }
  }
}

/**
 * Apply the structural deletion rules for a near-zero internal edge
 * z1 -> z2 with weight w12:
 *  - First remove the edge.
 *  - If z2 becomes a source (deg-(z2)=0, deg+(z2)>0), rewire all
 *    outgoing edges z2 -> y to z1 -> y with
 *      w_new = epsZero * w_{z2y} * sign(w12),
 *    then delete z2.
 *  - Otherwise, treat z1 as a potential sink and prune sinks recursively.
 */
function deleteInternalEdgeWithStructure(graph, edgeId, edge, epsZero, events) {
  const z1Id = edge.src;
  const z2Id = edge.dst;
  const w12 = edge.w;

  graph.removeEdge(edgeId);
  events.removedEdges++;

  const z1 = graph.nodes.get(z1Id);
  const z2 = graph.nodes.get(z2Id);
  if (!z1 || !z2) return;

  const degIn2 = graph.inDegree(z2Id);
  const degOut2 = graph.outDegree(z2Id);

  // Case 1: z2 becomes a source (no incoming edges, but has outgoing edges)
  if (degIn2 === 0 && degOut2 > 0) {
    const outEdges = graph.adjOut.get(z2Id)
      ? Array.from(graph.adjOut.get(z2Id))
      : [];
    const signW12 = Math.sign(w12) || (Math.random() < 0.5 ? 1 : -1);
    for (const oeid of outEdges) {
      const e = graph.edges.get(oeid);
      if (!e) continue;
      const newW = epsZero * e.w * signW12;
      graph.addEdge(z1Id, e.dst, newW);
    }
    graph.removeNode(z2Id);
    events.removedNodes++;
    return;
  }

  // Case 2: z2 does not become a source — prune sinks starting from z1.
  pruneSinkCascade(graph, z1Id, events);
}

/**
 * Execute one simulation step.
 * @param {Graph} graph
 * @param {number} t - current step counter
 * @param {object} params - {
 *   mu, pFlip, tBridge, sigma, omega, epsilon, K, theta,
 *   inputSource, m, activation, construction,
 *   weightTanh, useOU, useHebbian, useHebbOU, ouMean, etaHebb, hebbThresh
 * }
 * @returns {object} events - { bridged: [], removedEdges: number, removedNodes: number }
 */
export function simulationStep(graph, t, params) {
  const {
    mu,
    pFlip,
    tBridge,
    sigma,
    omega,
    epsilon,
    K,
    theta,
    inputSource,
    m,
    activation,
    construction,
    pAddEdge,
    pAddNode,
    randAlpha,
    randDMax,
    weightTanh,
    useOU,
    useHebbian,
    useHebbOU,
    ouMean,
    etaHebb,
    hebbThresh
  } = params;

  // Defaults if UI values are missing
  const sigmaVal = Number.isFinite(sigma) ? sigma : 0.02;
  const omegaVal = Number.isFinite(omega) ? omega : 0.05;
  const epsZero = Number.isFinite(epsilon) ? epsilon : EPS_DEFAULT;
  const cooldownK = Number.isFinite(K) ? K : 10;
  const pAddEdgeVal = Number.isFinite(pAddEdge) ? Math.max(0, pAddEdge) : 0.01;
  const pAddNodeVal = Number.isFinite(pAddNode) ? Math.max(0, pAddNode) : 0.005;
  const alphaVal = Number.isFinite(randAlpha) ? Math.max(0.1, randAlpha) : 1.5;
  const dMaxVal = Number.isFinite(randDMax) ? Math.max(1, Math.min(10, Math.floor(randDMax))) : 4;
  const etaHebbVal = Number.isFinite(etaHebb) ? etaHebb : 0;
  const hebbThreshVal = Number.isFinite(hebbThresh) ? hebbThresh : 0;
  const events = { bridged: [], removedEdges: 0, removedNodes: 0 };

  // 1) Set input node activations
  for (let i = 0; i < m; i++) {
    const node = graph.nodes.get(`x${i}`);
    if (node) {
      node.activation = getInputValue(inputSource, i, t);
    }
  }

  let actFn;
  const thetaVal = Number.isFinite(theta) ? theta : 0;
  if (activation === 'relu') {
    actFn = x => (x > 0 ? x : 0);
  } else if (activation === 'relu-thresh') {
    actFn = x => {
      const s = x - thetaVal;
      return s > 0 ? s : 0;
    };
  } else if (activation === 'identity') {
    actFn = x => x;
  } else if (activation === 'maxabs') {
    // handled explicitly below; actFn unused
    actFn = null;
  } else {
    actFn = x => Math.tanh(x);
  }

  // Snapshot previous activations so that each step only moves one "hop":
  // new activations depend on the previous step, not on values updated
  // earlier in this same step.
  const prevActivation = new Map();
  for (const [id, node] of graph.nodes) {
    prevActivation.set(id, node.activation || 0);
  }

  // 2) Forward pass for non-input nodes (in creation order),
  // using prevActivation for all sources.
  const order = graph.getForwardOrder();
  for (const nodeId of order) {
    const node = graph.nodes.get(nodeId);
    if (!node || node.type === 'input') continue;

    const inEdges = graph.adjIn.get(nodeId);
    if (activation === 'maxabs') {
      let best = 0;
      let hasInput = false;
      if (inEdges) {
        for (const eid of inEdges) {
          const edge = graph.edges.get(eid);
          if (!edge) continue;
          const x = prevActivation.get(edge.src) || 0;
          let contrib;
          if (weightTanh) {
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
          if (weightTanh) {
            z += Math.tanh(edge.w * x);
          } else {
            z += edge.w * x;
          }
        }
      }
      node.activation = actFn(z);
    }
  }

  if (construction !== 'random') {
    // 3) Bridging trigger: mark nodes with |activation| > T_bridge.
    // Only internal nodes (z0 and all bridge nodes z1, z2, ...) are allowed to
    // trigger bridges; inputs/outputs are excluded.
    const triggered = [];
    for (const [nodeId, node] of graph.nodes) {
      if (node.type !== 'internal') continue;
      if (Math.abs(node.activation) > tBridge) {
        // Cooldown check
        if (t - node.lastBridge >= cooldownK) {
          triggered.push(nodeId);
        }
      }
    }

    // 4) Bridging action (new bridge mechanism):
    //
    // For each triggered node z0:
    //   - Create two new internal nodes z1, z2
    //   - Add a canonical 2-cycle: z1 -> z2 = sqrt(2)/2, z2 -> z1 = sqrt(2)/2
    //   - Halve all incoming weights into z0: 2 w_i -> w_i (implemented as w_i /= 2)
    //   - For each incoming edge xi -> z0 (now weight w_i), add xi -> z1 with
    //     weight (sqrt(2)/2) * w_i
    //   - For each outgoing edge z0 -> yj with weight v_j, add a parallel edge
    //     z2 -> yj with the same weight v_j (keeping the original z0 -> yj)
    //   - Add stabilizing feedback edges: z1 -> z0 = -epsilon, z0 -> z2 = epsilon
    for (const nodeId of triggered) {
      const z0 = graph.nodes.get(nodeId);
      if (!z0) continue;
      z0.lastBridge = t;

      // New internal bridge nodes (closer to inputs / closer to outputs)
      const z1Id = graph.newInternalId();
      const z2Id = graph.newInternalId();
      graph.addNode(z1Id, 'internal');
      graph.addNode(z2Id, 'internal');

      // Canonical 2-cycle between z1 and z2
      graph.addEdge(z1Id, z2Id, BRIDGE_BASE);
      graph.addEdge(z2Id, z1Id, BRIDGE_BASE);

      // Snapshot of incoming and outgoing edges to z0 BEFORE we modify them
      const inEdges = graph.adjIn.get(nodeId)
        ? Array.from(graph.adjIn.get(nodeId))
        : [];
      const outEdges = graph.adjOut.get(nodeId)
        ? Array.from(graph.adjOut.get(nodeId))
        : [];

      // Incoming edges: 2 w_i -> w_i, and add xi -> z1 with (sqrt(2)/2) * w_i
      for (const eid of inEdges) {
        const edge = graph.edges.get(eid);
        if (!edge || edge.dst !== nodeId) continue;
        // Halve the existing weight
        edge.w *= 0.5;
        const w_i = edge.w;
        const newWeight = BRIDGE_BASE * w_i;
        graph.addEdge(edge.src, z1Id, newWeight);
      }

      // Outgoing edges: duplicate to z2 with the same weight v_j
      for (const eid of outEdges) {
        const edge = graph.edges.get(eid);
        if (!edge || edge.src !== nodeId) continue;
        graph.addEdge(z2Id, edge.dst, edge.w);
      }

      // Stabilizing feedback edges: z1 -> z0 = -omega, z0 -> z2 = omega
      graph.addEdge(z1Id, nodeId, -omegaVal);
      graph.addEdge(nodeId, z2Id, omegaVal);

      events.bridged.push(nodeId);
    }
  } else {
    // Random construction mode: skip bridging and instead apply random growth.
    const growthNodes = [];
    const edgeSrc = randomEdgeGrowth(graph, pAddEdgeVal, alphaVal, dMaxVal);
    if (edgeSrc) growthNodes.push(edgeSrc);
    const newNode = randomNodeGrowth(graph, pAddNodeVal);
    if (newNode) growthNodes.push(newNode);
    for (const id of growthNodes) {
      events.bridged.push(id);
    }
  }

  // 5) Weight update for every edge
  if (useHebbOU) {
    // Hebbian-defined OU mean m_e(a_pre, a_post) (no separate ODE state),
    // then OU update of w_e around this instantaneous mean.
    const gamma = 0.05;
    const aOU = Math.exp(-gamma);
    const bOU = sigmaVal * Math.sqrt((1 - aOU * aOU) / (2 * gamma));

    for (const [, edge] of graph.edges) {
      const srcNode = graph.nodes.get(edge.src);
      const dstNode = graph.nodes.get(edge.dst);
      const aPre = srcNode ? srcNode.activation || 0 : 0;
      const aPost = dstNode ? dstNode.activation || 0 : 0;
      const p = aPre * aPost;

      // Require both nodes to be strongly active before Hebb-OU has any effect.
      const absPre = Math.abs(aPre);
      const absPost = Math.abs(aPost);

      // Determine role sign (excitatory/inhibitory) in a stable way.
      let roleSign = 0;
      if (edge.w !== 0) {
        roleSign = Math.sign(edge.w);
      } else if (p !== 0) {
        roleSign = Math.sign(p);
      } else {
        roleSign = Math.random() < 0.5 ? 1 : -1;
      }

      // Instantaneous Hebbian mean in [-1, 1]:
      // only when both |a_pre| and |a_post| exceed θ_hebb.
      let mInst = 0;
      if (absPre > hebbThreshVal && absPost > hebbThreshVal) {
        const hebbInput = etaHebbVal * p * roleSign;
        mInst = Math.tanh(hebbInput);
      } else {
        mInst = 0;
      }

      // OU update of the actual weight around instantaneous mean mInst
      const wOld = edge.w;
      edge.w = mInst + aOU * (wOld - mInst) + bOU * randn();
    }
  } else if (useHebbian) {
    // Brownian motion with Hebbian drift:
    // drift magnitude ∝ |a_pre * a_post|,
    // but only if |a_pre * a_post| > θ_hebb;
    // drift direction = sign(w).
    for (const [, edge] of graph.edges) {
      const srcNode = graph.nodes.get(edge.src);
      const dstNode = graph.nodes.get(edge.dst);
      const aPre = srcNode ? srcNode.activation || 0 : 0;
      const aPost = dstNode ? dstNode.activation || 0 : 0;
      const productMag = Math.abs(aPre * aPost);
      const driftMag = productMag > hebbThreshVal ? etaHebbVal * productMag : 0;
      const signW = Math.sign(edge.w) || 0;
      edge.w += driftMag * signW + sigmaVal * randn();
    }
  } else if (useOU) {
    // Ornstein–Uhlenbeck update with mean reversion.
    // We interpret ouMean as a magnitude m >= 0 and use
    // a sign-dependent mean: +m for positive edges and
    // -m for negative edges. This gives each sign its own
    // stable position at ±m.
    const gamma = 0.05; // fixed mean-reversion rate for now
    const a = Math.exp(-gamma);
    const b = sigmaVal * Math.sqrt((1 - a * a) / (2 * gamma));
    const mMag = Number.isFinite(ouMean) ? Math.abs(ouMean) : 0;
    for (const [, edge] of graph.edges) {
      const w = edge.w;
      const s = Math.sign(w) || (Math.random() < 0.5 ? 1 : -1);
      const mE = s * mMag;
      edge.w = mE + a * (w - mE) + b * randn();
    }
  } else {
    for (const [, edge] of graph.edges) {
      edge.w += sigmaVal * randn() + mu * Math.sign(edge.w);
    }
  }

  // 6) Near-zero event with structural deletion and rewiring.
  // First collect candidate edges to avoid mutating the map while iterating.
  const nearZeroEdges = [];
  for (const [eid, edge] of graph.edges) {
    if (Math.abs(edge.w) < epsZero) {
      nearZeroEdges.push(eid);
    }
  }

  for (const eid of nearZeroEdges) {
    const edge = graph.edges.get(eid);
    if (!edge) continue;

    // With probability pFlip, perform a small sign-flip and keep the edge.
    if (Math.random() < pFlip) {
      const oldSign = Math.sign(edge.w) || (Math.random() < 0.5 ? 1 : -1);
      const mag = Math.random() * epsZero;
      edge.w = -oldSign * mag;
      continue;
    }

    const srcNode = graph.nodes.get(edge.src);
    const dstNode = graph.nodes.get(edge.dst);

    // Apply the full structural rule only for internal -> internal edges.
    if (srcNode && dstNode &&
        srcNode.type === 'internal' &&
        dstNode.type === 'internal') {
      deleteInternalEdgeWithStructure(graph, eid, edge, epsZero, events);
    } else {
      // For all other edges, simply remove the edge. If the source is an
      // internal node that becomes a sink, prune sinks recursively.
      const srcId = edge.src;
      graph.removeEdge(eid);
      events.removedEdges++;
      const src = graph.nodes.get(srcId);
      if (src && src.type === 'internal') {
        pruneSinkCascade(graph, srcId, events);
      }
    }
  }

  return events;
}
