/**
 * Cytoscape.js graph visualization.
 */

/* global cytoscape */

const NODE_COLORS = {
  input: '#4a90d9',
  internal: '#7c7c7c',
  output: '#d94a4a'
};

// Edge colours: positive / non-negative vs negative weights.
const EDGE_POS_COLOR = '#000000';   // black
const EDGE_NEG_COLOR = '#b71c1c';   // deep red
const BRIDGE_BORDER_COLOR = '#ffd600';

export class GraphView {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.cy = cytoscape({
      container: this.container,
      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(id)',
            'font-size': 'data(fontSize)',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': 'data(color)',
            'width': 'data(size)',
            'height': 'data(size)',
            'border-width': 0,
            'border-color': BRIDGE_BORDER_COLOR,
            'color': '#fff',
            'text-outline-color': '#333',
            'text-outline-width': 'data(textOutlineWidth)'
          }
        },
        {
          selector: 'node.bridged',
          style: {
            'border-width': 4,
            'border-color': BRIDGE_BORDER_COLOR
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 'data(thickness)',
            'line-color': 'data(color)',
            'target-arrow-color': 'data(color)',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'opacity': 0.7
          }
        }
      ],
      // We use explicit positions (preset layout) with a simple
      // square grid arrangement for all nodes.
      layout: { name: 'preset' },
      minZoom: 0.2,
      maxZoom: 5,
      wheelSensitivity: 0.3
    });

    this._initInfoPopup();
    this._bindInteractionHandlers();
  }

  /**
   * Compute deterministic positions using three non-overlapping
   * regions from left to right:
   *  - left:  vertical column grid for input nodes x*
   *  - middle: square grid for internal nodes z*
   *  - right: vertical column grid for output nodes y*
   *
   * Input/output columns keep the same shape and x-positions; the
   * central z grid adapts to the number of internal nodes.
   */
  _computePositions(graph) {
    const positions = new Map();

    const inputs = [];
    const internals = [];
    const outputs = [];

    for (const [id, node] of graph.nodes) {
      if (node.type === 'input') inputs.push(id);
      else if (node.type === 'output') outputs.push(id);
      else internals.push(id);
    }

    // Stable ordering within each group: sort by trailing index if present.
    const sortByIndex = (a, b) => {
      const parse = id => {
        const m = id.match(/(\d+)$/);
        return m ? parseInt(m[1], 10) : 0;
      };
      return parse(a) - parse(b);
    };
    inputs.sort(sortByIndex);
    outputs.sort(sortByIndex);
    internals.sort();

    // Fixed x-positions for left/middle/right regions.
    const xInput = -300;
    const xOutput = 300;
    const inputOutputSpacingY = 60;

    // Place input nodes in a vertical grid on the left.
    const nIn = inputs.length;
    if (nIn > 0) {
      const offsetIn = (nIn - 1) / 2;
      inputs.forEach((id, idx) => {
        const y = (idx - offsetIn) * inputOutputSpacingY;
        positions.set(id, { x: xInput, y });
      });
    }

    // Place output nodes in a vertical grid on the right.
    const nOut = outputs.length;
    if (nOut > 0) {
      const offsetOut = (nOut - 1) / 2;
      outputs.forEach((id, idx) => {
        const y = (idx - offsetOut) * inputOutputSpacingY;
        positions.set(id, { x: xOutput, y });
      });
    }

    // Central square region for internal nodes z*, between the two columns.
    const nInt = internals.length;
    if (nInt > 0) {
      const innerMargin = 60;
      const innerLeft = xInput + innerMargin;
      const innerRight = xOutput - innerMargin;
      const maxSide = innerRight - innerLeft;
      const minSide = 120; // keep small clusters visually compact

      const cols = Math.ceil(Math.sqrt(nInt));
      const rows = Math.ceil(nInt / cols);

      // Try to keep roughly 60px between neighbours; shrink/grow
      // the central square depending on how many z nodes we have.
      const targetSpacing = 60;
      const neededWidth = (cols - 1) * targetSpacing;
      const neededHeight = (rows - 1) * targetSpacing;
      let side = Math.max(minSide, Math.max(neededWidth, neededHeight));
      side = Math.min(side, maxSide);

      const spacing = cols > 1 ? side / (cols - 1) : 0;
      const width = (cols - 1) * spacing;
      const height = rows > 1 ? (rows - 1) * spacing : 0;

      internals.forEach((id, idx) => {
        const col = idx % cols;
        const row = Math.floor(idx / cols);
        const x = innerLeft + (maxSide - side) / 2 + col * spacing;
        const y = row * spacing - height / 2;
        positions.set(id, { x, y });
      });
    }

    return positions;
  }

  /** Full rebuild of the cytoscape graph from simulation graph. */
  rebuild(graph, bridgedSet) {
    this.cy.elements().remove();

    const positions = this._computePositions(graph);
    const elements = [];

    // Add nodes
    for (const [id, node] of graph.nodes) {
      const size = 28;
      const fontSize = 10;
      const textOutlineWidth = 1;
      elements.push({
        group: 'nodes',
        data: {
          id,
          activation: node.activation || 0,
          color: NODE_COLORS[node.type] || NODE_COLORS.internal,
          size,
          fontSize,
          textOutlineWidth
        },
        position: positions.get(id),
        classes: bridgedSet.has(id) ? 'bridged' : ''
      });
    }

    // Add edges
    for (const [eid, edge] of graph.edges) {
      elements.push({
        group: 'edges',
        data: {
          id: eid,
          source: edge.src,
          target: edge.dst,
          weight: edge.w,
          color: edge.w >= 0 ? EDGE_POS_COLOR : EDGE_NEG_COLOR,
          thickness: Math.max(0.5, Math.min(Math.abs(edge.w), 3) * 2)
        }
      });
    }

    this.cy.add(elements);
    this.cy.fit();
  }

  /** Incremental update: sync node classes and edge styles without full rebuild. */
  update(graph, bridgedSet) {
    const cyNodeIds = new Set();
    const cyEdgeIds = new Set();

    // Collect current cy element ids
    this.cy.nodes().forEach(n => cyNodeIds.add(n.id()));
    this.cy.edges().forEach(e => cyEdgeIds.add(e.id()));

    const graphNodeIds = new Set(graph.nodes.keys());
    const graphEdgeIds = new Set(graph.edges.keys());

    // Recompute desired positions on the square grid
    const positions = this._computePositions(graph);

    // Remove nodes/edges no longer in graph
    this.cy.nodes().forEach(n => {
      if (!graphNodeIds.has(n.id())) n.remove();
    });
    this.cy.edges().forEach(e => {
      if (!graphEdgeIds.has(e.id())) e.remove();
    });

    // Add new nodes
    const newElements = [];
    for (const [id, node] of graph.nodes) {
      if (!cyNodeIds.has(id)) {
        const size = 28;
        const fontSize = 10;
        const textOutlineWidth = 1;
        newElements.push({
          group: 'nodes',
          data: {
            id,
            activation: node.activation || 0,
            color: NODE_COLORS[node.type] || NODE_COLORS.internal,
            size,
            fontSize,
            textOutlineWidth
          },
          position: positions.get(id)
        });
      }
    }

    // Add new edges
    for (const [eid, edge] of graph.edges) {
      if (!cyEdgeIds.has(eid)) {
        newElements.push({
          group: 'edges',
          data: {
            id: eid,
            source: edge.src,
            target: edge.dst,
            weight: edge.w,
            color: edge.w >= 0 ? EDGE_POS_COLOR : EDGE_NEG_COLOR,
            thickness: Math.max(0.5, Math.min(Math.abs(edge.w), 3) * 2)
          }
        });
      }
    }

    if (newElements.length > 0) {
      this.cy.add(newElements);
    }

    // Update positions and sizes of all existing nodes to keep columns neat
    for (const [id, node] of graph.nodes) {
      const pos = positions.get(id);
      if (!pos) continue;
      const cyNode = this.cy.getElementById(id);
      if (cyNode.length) {
        cyNode.position(pos);
        const size = 28;
        const fontSize = 10;
        const textOutlineWidth = 1;
        cyNode.data('size', size);
        cyNode.data('fontSize', fontSize);
        cyNode.data('textOutlineWidth', textOutlineWidth);
        cyNode.data('activation', node.activation || 0);
      }
    }

    // Update existing edge styles
    for (const [eid, edge] of graph.edges) {
      const cyEdge = this.cy.getElementById(eid);
      if (cyEdge.length) {
        cyEdge.data('color', edge.w >= 0 ? EDGE_POS_COLOR : EDGE_NEG_COLOR);
        cyEdge.data('thickness', Math.max(0.5, Math.min(Math.abs(edge.w), 3) * 2));
        cyEdge.data('weight', edge.w);
      }
    }

    // Update bridged classes
    this.cy.nodes().forEach(n => {
      if (bridgedSet.has(n.id())) {
        n.addClass('bridged');
      } else {
        n.removeClass('bridged');
      }
    });

    // Keep the whole graph in view
    if (newElements.some(e => e.group === 'nodes')) {
      this.cy.fit();
    }
  }

  resize() {
    this.cy.resize();
    this.cy.fit();
  }

  _initInfoPopup() {
    const popup = document.createElement('div');
    popup.style.position = 'absolute';
    popup.style.zIndex = '10';
    popup.style.background = 'rgba(0, 0, 0, 0.8)';
    popup.style.color = '#fff';
    popup.style.padding = '4px 6px';
    popup.style.borderRadius = '4px';
    popup.style.fontSize = '11px';
    popup.style.pointerEvents = 'none';
    popup.style.display = 'none';
    this.container.style.position = this.container.style.position || 'relative';
    this.container.appendChild(popup);
    this.infoPopup = popup;
  }

  _showInfoPopup(text, renderedPosition) {
    if (!this.infoPopup) return;
    this.infoPopup.textContent = text;
    const rect = this.container.getBoundingClientRect();
    const x = renderedPosition.x - rect.left;
    const y = renderedPosition.y - rect.top;
    this.infoPopup.style.left = `${x + 8}px`;
    this.infoPopup.style.top = `${y + 8}px`;
    this.infoPopup.style.display = 'block';
  }

  _hideInfoPopup() {
    if (!this.infoPopup) return;
    this.infoPopup.style.display = 'none';
  }

  _bindInteractionHandlers() {
    // Click on node: show activation
    this.cy.on('tap', 'node', evt => {
      const node = evt.target;
      const id = node.id();
      const a = node.data('activation');
      const val =
        a == null || Number.isNaN(a) ? 'n/a' : a.toFixed(4);
      const oe = evt.originalEvent;
      const pos = oe && typeof oe.clientX === 'number'
        ? { x: oe.clientX, y: oe.clientY }
        : evt.renderedPosition;
      this._showInfoPopup(`node ${id}\nactivation = ${val}`, pos);
    });

    // Click on edge: show weight
    this.cy.on('tap', 'edge', evt => {
      const edge = evt.target;
      const id = edge.id();
      const w = edge.data('weight');
      const val =
        w == null || Number.isNaN(w) ? 'n/a' : w.toFixed(4);
      const oe = evt.originalEvent;
      const pos = oe && typeof oe.clientX === 'number'
        ? { x: oe.clientX, y: oe.clientY }
        : evt.renderedPosition;
      this._showInfoPopup(`edge ${id}\nweight = ${val}`, pos);
    });

    // Click on background: hide popup
    this.cy.on('tap', evt => {
      if (evt.target === this.cy) {
        this._hideInfoPopup();
      }
    });
  }
}
