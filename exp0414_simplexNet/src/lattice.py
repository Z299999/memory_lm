"""Lattice generation for simplex memory networks.

The lattice V_{n,m} is the set of nonnegative integer vectors
α = (α₀, ..., αₙ) ∈ ℤ_≥0^{n+1} such that Σαᵢ = m-1.

This is implemented using the stars-and-bars enumeration.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Iterator


def generate_lattice(n: int, m: int) -> list[tuple[int, ...]]:
    """Generate V_{n,m} = {α ∈ ℤ_≥0^{n+1} : Σαᵢ = m-1}.

    Uses recursive enumeration (stars-and-bars conceptually).

    Args:
        n: Simplex dimension (number of vertices is n+1)
        m: Resolution parameter (each edge has m lattice points)

    Returns:
        List of tuples α = (α₀, ..., αₙ) with Σαᵢ = m-1.
        The cardinality is C(m+n-1, n).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if m < 2:
        raise ValueError(f"m must be >= 2, got {m}")

    s = m - 1  # Sum of coordinates
    result = []

    def helper(k: int, remaining: int, current: list[int]) -> None:
        """Enumerate (α₀, ..., αₖ) with sum = remaining."""
        if k == 0:
            current.append(remaining)
            result.append(tuple(current))
            current.pop()
            return

        for α_k in range(remaining + 1):
            current.append(α_k)
            helper(k - 1, remaining - α_k, current)
            current.pop()

    helper(n, s, [])
    return result


def cardinality(n: int, m: int) -> int:
    """Return |V_{n,m}| = C(m+n-1, n)."""
    from math import comb
    return comb(m + n - 1, n)


# Example usage / quick test
if __name__ == "__main__":
    # Test: V_{2,3} should have C(3+2-1, 2) = C(4,2) = 6 points
    n, m = 2, 3
    lattice = generate_lattice(n, m)
    print(f"V_{{{n},{m}}} has {len(lattice)} points (expected {cardinality(n, m)})")
    for α in lattice:
        print(f"  α = {α}, Σαᵢ = {sum(α)}")

    # Verify cardinality for a few more cases
    for n_test in [2, 3, 4]:
        for m_test in [2, 3, 4]:
            lat = generate_lattice(n_test, m_test)
            expected = cardinality(n_test, m_test)
            assert len(lat) == expected, f"Mismatch for n={n_test}, m={m_test}"
            assert all(sum(α) == m_test - 1 for α in lat), f"Sum check failed for n={n_test}, m={m_test}"
    print("Verified: cardinality and sum constraints")
