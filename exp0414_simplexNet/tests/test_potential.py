"""Unit tests for potential function and graph properties."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lattice import generate_lattice
from potential import (
    vertex_potentials,
    node_potential,
    get_admissible_edges,
    move_mass,
    F_in, F_out, F_mid, backbone,
)
from graph import SimplexMemoryGraph


def test_vertex_potentials():
    """Test vertex potential assignment: β₀=0, β₁=2, βᵢ=1 for i≥2."""
    for n in [2, 3, 4]:
        beta = vertex_potentials(n)
        assert beta[0] == 0, f"β₀ should be 0, got {beta[0]}"
        assert beta[1] == 2, f"β₁ should be 2, got {beta[1]}"
        for i in range(2, n + 1):
            assert beta[i] == 1, f"β_{i} should be 1, got {beta[i]}"
    print("✓ test_vertex_potentials passed")


def test_potential_increases():
    """Test that H strictly increases along every admissible edge."""
    for n in [2, 3]:
        for m in [3, 4]:
            lattice = generate_lattice(n, m)
            beta = vertex_potentials(n)
            for α in lattice:
                edges = get_admissible_edges(α, beta)
                for i, j in edges:
                    α_prime = move_mass(α, i, j)
                    H_before = node_potential(α, beta)
                    H_after = node_potential(α_prime, beta)
                    assert H_after > H_before, f"Potential not increasing: {α} → {α_prime}"
    print("✓ test_potential_increases passed")


def test_f_mid_isopotential():
    """Test that F_mid has constant H = m-1."""
    for n in [2, 3]:
        for m in [2, 3, 4]:
            lattice = generate_lattice(n, m)
            beta = vertex_potentials(n)
            f_mid = F_mid(lattice)
            h_values = set(node_potential(α, beta) for α in f_mid)
            if f_mid:
                assert len(h_values) == 1, f"F_mid has multiple H values: {h_values}"
                assert h_values.pop() == m - 1, f"F_mid H should be {m-1}"
    print("✓ test_f_mid_isopotential passed")


def test_facet_intersection():
    """Test F_mid = F_in ∩ F_out."""
    for n in [2, 3]:
        for m in [2, 3, 4]:
            lattice = generate_lattice(n, m)
            f_in = set(F_in(lattice))
            f_out = set(F_out(lattice))
            f_mid = set(F_mid(lattice))
            assert f_mid == (f_in & f_out), f"F_mid ≠ F_in ∩ F_out"
    print("✓ test_facet_intersection passed")


def test_backbone_size():
    """Test |𝓑_{n,m}| = m."""
    for n in [2, 3, 4]:
        for m in [2, 3, 4, 5]:
            lattice = generate_lattice(n, m)
            bb = backbone(lattice)
            assert len(bb) == m, f"|backbone| = {len(bb)}, expected {m}"
    print("✓ test_backbone_size passed")


def test_graph_acyclicity():
    """Test that the directed graph is acyclic."""
    for n in [2, 3]:
        for m in [2, 3, 4]:
            graph = SimplexMemoryGraph(n=n, m=m, n_in=1, n_out=1)
            # Check: potential strictly increases along every edge
            beta = vertex_potentials(n)
            for src, dst in graph.edges:
                if src[0] == "core" and dst[0] == "core":
                    H_src = node_potential(src[1:], beta)
                    H_dst = node_potential(dst[1:], beta)
                    assert H_dst > H_src, f"Edge {src} → {dst} does not increase potential"
    print("✓ test_graph_acyclicity passed")


def test_graph_connectivity():
    """Test that all core + output nodes have predecessors."""
    for n in [2, 3]:
        for m in [2, 3]:
            graph = SimplexMemoryGraph(n=n, m=m, n_in=1, n_out=1)
            for node in graph.core_nodes + graph.output_nodes:
                assert graph.preds[node], f"Node {node} has no predecessors"
    print("✓ test_graph_connectivity passed")


if __name__ == "__main__":
    test_vertex_potentials()
    test_potential_increases()
    test_f_mid_isopotential()
    test_facet_intersection()
    test_backbone_size()
    test_graph_acyclicity()
    test_graph_connectivity()
    print("\nAll potential and graph tests passed!")
