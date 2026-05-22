/**
 * Graph data structure for the eco-evo simulation.
 * Manages nodes, edges, adjacency, and degree tracking.
 */

export class Graph {
  constructor() {
    this.nodes = new Map();   // id -> { id, type, activation, lastBridge }
    this.edges = new Map();   // id -> { id, src, dst, w, m }
    this.adjOut = new Map();  // src -> Set of edge ids
    this.adjIn = new Map();   // dst -> Set of edge ids
    // Counter for internal nodes z1, z2, z3, ...
    // z0 (and any additional z's created at genesis) use explicit ids.
    this._nextZIndex = 1;
    this._nextEdgeId = 0;
  }

  /**
   * Create the genesis graph with m inputs, k internal nodes, and n outputs.
   * Internal nodes z0, z1, ..., z{k-1} form a fully connected directed
   * graph (no self-loops). Every input x_i connects to every z_j, and
   * every z_j connects to every output y_l.
   */
  static genesis(m, n, k = 1) {
    const g = new Graph();

    // Input nodes x0, ..., x{m-1}
    const inputs = [];
    for (let i = 0; i < m; i++) {
      const id = `x${i}`;
      g.addNode(id, 'input');
      inputs.push(id);
    }

    // Internal nodes z0, z1, ..., z{k-1}
    const internals = [];
    const kInt = Math.max(1, k | 0);
    for (let j = 0; j < kInt; j++) {
      const id = j === 0 ? 'z0' : g.newInternalId();
      g.addNode(id, 'internal');
      internals.push(id);
    }

    // Output nodes y0, ..., y{n-1}
    const outputs = [];
    for (let j = 0; j < n; j++) {
      const id = `y${j}`;
      g.addNode(id, 'output');
      outputs.push(id);
    }

    // Edges: fully connect internal nodes (no self-loops)
    if (internals.length > 1) {
      const wZZ = 1 / internals.length;
      for (const src of internals) {
        for (const dst of internals) {
          if (src === dst) continue;
          g.addEdge(src, dst, wZZ);
        }
      }
    }

    // Edges: xi -> z_j with weight 1/m
    const wIn = 1 / m;
    for (const inId of inputs) {
      for (const zId of internals) {
        g.addEdge(inId, zId, wIn);
      }
    }

    // Edges: z_j -> y_l with weight 1/kInt
    const wOut = 1 / kInt;
    for (const zId of internals) {
      for (const outId of outputs) {
        g.addEdge(zId, outId, wOut);
      }
    }

    return g;
  }

  /**
   * Create a 2-hidden-layer feed-forward MLP-style graph:
   *  - layer 1: m internal nodes
   *  - layer 2: m internal nodes
   *  - fully connected x -> layer1 -> layer2 -> y
   */
  static genesisMLP2(m, n) {
    const g = new Graph();

    const mSafe = Math.max(1, m | 0);
    const nSafe = Math.max(1, n | 0);

    // Inputs x0, ..., x{m-1}
    const inputs = [];
    for (let i = 0; i < mSafe; i++) {
      const id = `x${i}`;
      g.addNode(id, 'input');
      inputs.push(id);
    }

    // Hidden layer 1: z0, z1, ..., z{m-1}
    const h1 = [];
    for (let j = 0; j < mSafe; j++) {
      const id = j === 0 ? 'z0' : g.newInternalId();
      g.addNode(id, 'internal');
      h1.push(id);
    }

    // Hidden layer 2: z{m}, ..., z{2m-1}
    const h2 = [];
    for (let j = 0; j < mSafe; j++) {
      const id = g.newInternalId();
      g.addNode(id, 'internal');
      h2.push(id);
    }

    // Outputs y0, ..., y{n-1}
    const outputs = [];
    for (let j = 0; j < nSafe; j++) {
      const id = `y${j}`;
      g.addNode(id, 'output');
      outputs.push(id);
    }

    // x -> hidden1
    const wIn = 1 / mSafe;
    for (const xId of inputs) {
      for (const zId of h1) {
        g.addEdge(xId, zId, wIn);
      }
    }

    // hidden1 -> hidden2
    const wH = 1 / mSafe;
    for (const z1 of h1) {
      for (const z2 of h2) {
        g.addEdge(z1, z2, wH);
      }
    }

    // hidden2 -> outputs
    const wOut = 1 / mSafe;
    for (const z2 of h2) {
      for (const yId of outputs) {
        g.addEdge(z2, yId, wOut);
      }
    }

    return g;
  }

  addNode(id, type) {
    if (this.nodes.has(id)) return;
    this.nodes.set(id, {
      id,
      type,          // 'input' | 'internal' | 'output'
      activation: 0,
      lastBridge: -Infinity  // step when last bridging occurred
    });
    this.adjOut.set(id, new Set());
    this.adjIn.set(id, new Set());
  }

  /** Generate a unique internal node id: z1, z2, z3, ... */
  newInternalId() {
    const id = `z${this._nextZIndex++}`;
    return id;
  }

  /** Generate a unique edge id. */
  newEdgeId() {
    return `e_${this._nextEdgeId++}`;
  }

  addEdge(src, dst, w) {
    const id = this.newEdgeId();
    this.edges.set(id, { id, src, dst, w, m: w });
    this.adjOut.get(src).add(id);
    this.adjIn.get(dst).add(id);
    return id;
  }

  removeEdge(edgeId) {
    const e = this.edges.get(edgeId);
    if (!e) return;
    this.adjOut.get(e.src)?.delete(edgeId);
    this.adjIn.get(e.dst)?.delete(edgeId);
    this.edges.delete(edgeId);
  }

  removeNode(nodeId) {
    // Remove all incident edges first
    const outEdges = this.adjOut.get(nodeId);
    if (outEdges) {
      for (const eid of [...outEdges]) this.removeEdge(eid);
    }
    const inEdges = this.adjIn.get(nodeId);
    if (inEdges) {
      for (const eid of [...inEdges]) this.removeEdge(eid);
    }
    this.adjOut.delete(nodeId);
    this.adjIn.delete(nodeId);
    this.nodes.delete(nodeId);
  }

  inDegree(nodeId) {
    return this.adjIn.get(nodeId)?.size ?? 0;
  }

  outDegree(nodeId) {
    return this.adjOut.get(nodeId)?.size ?? 0;
  }

  totalDegree(nodeId) {
    return this.inDegree(nodeId) + this.outDegree(nodeId);
  }

  /** Compute degree histogram: { k: count } */
  degreeHistogram() {
    const hist = {};
    for (const [id] of this.nodes) {
      const k = this.totalDegree(id);
      hist[k] = (hist[k] || 0) + 1;
    }
    return hist;
  }

  /** Get ordered list of node ids for forward sweep (inputs first, then by creation order). */
  getForwardOrder() {
    const inputs = [];
    const rest = [];
    for (const [id, node] of this.nodes) {
      if (node.type === 'input') inputs.push(id);
      else rest.push(id);
    }
    return [...inputs, ...rest];
  }

  get nodeCount() { return this.nodes.size; }
  get edgeCount() { return this.edges.size; }
}
