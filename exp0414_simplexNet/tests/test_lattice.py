"""Unit tests for lattice generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lattice import generate_lattice, cardinality


def test_cardinality():
    """Test |V_{n,m}| = C(m+n-1, n)."""
    test_cases = [
        (2, 2, 3),    # triangle, m=2: 3 vertices
        (2, 3, 6),    # triangle, m=3: 6 points
        (2, 4, 10),   # triangle, m=4: 10 points
        (3, 2, 4),    # tetrahedron, m=2: 4 vertices
        (3, 3, 10),   # tetrahedron, m=3: 10 points
        (4, 2, 5),    # 4-simplex, m=2: 5 vertices
    ]
    for n, m, expected in test_cases:
        result = cardinality(n, m)
        assert result == expected, f"cardinality({n}, {m}) = {result}, expected {expected}"
    print("✓ test_cardinality passed")


def test_sum_constraint():
    """Test that all lattice points satisfy Σαᵢ = m-1."""
    for n in [2, 3, 4]:
        for m in [2, 3, 4, 5]:
            lattice = generate_lattice(n, m)
            for α in lattice:
                assert len(α) == n + 1, f"α has wrong length: {len(α)}, expected {n+1}"
                assert sum(α) == m - 1, f"sum(α) = {sum(α)}, expected {m-1}"
    print("✓ test_sum_constraint passed")


def test_nonnegative():
    """Test that all lattice points have nonnegative coordinates."""
    for n in [2, 3]:
        for m in [2, 3, 4]:
            lattice = generate_lattice(n, m)
            for α in lattice:
                assert all(a >= 0 for a in α), f"Found negative coordinate in {α}"
    print("✓ test_nonnegative passed")


def test_vertex_cases():
    """Test that vertices e₀, e₁, ..., eₙ are in the lattice for m=2."""
    for n in [2, 3, 4]:
        m = 2
        lattice = generate_lattice(n, m)
        # Vertices should be permutations of (1, 0, 0, ...)
        for i in range(n + 1):
            expected = tuple(1 if j == i else 0 for j in range(n + 1))
            assert expected in lattice, f"Vertex e_{i} not in lattice for n={n}"
    print("✓ test_vertex_cases passed")


if __name__ == "__main__":
    test_cardinality()
    test_sum_constraint()
    test_nonnegative()
    test_vertex_cases()
    print("\nAll lattice tests passed!")
