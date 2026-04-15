"""Potential function and edge orientation for simplex memory networks.

The potential function H: V_{n,m} → ℤ is defined by
H(α) = Σᵢ βᵢ αᵢ

where β is the vertex potential:
- β₀ = 0 (vertex a)
- β₁ = 2 (vertex b)
- βᵢ = 1 for i ≥ 2 (vertices c₁, ..., c_{n-1})

A directed edge α → α' exists iff:
- α' = α + eᵢ - eⱼ for some i ≠ j
- αⱼ ≥ 1 (can move mass from j)
- βᵢ > βⱼ (potential increases)

Admissible edge types:
- a → cₖ (0 → k+1): β increases from 0 to 1, ΔH = 1
- cₖ → b (k+1 → 1): β increases from 1 to 2, ΔH = 1
- a → b (0 → 1): β increases from 0 to 2, ΔH = 2
"""

from __future__ import annotations


def vertex_potentials(n: int) -> dict[int, int]:
    """Return vertex potentials β = {0: 0, 1: 2, 2..n: 1}.

    Args:
        n: Simplex dimension (vertices are 0, 1, ..., n)

    Returns:
        Dictionary mapping vertex index to potential.
    """
    beta = {0: 0, 1: 2}
    for k in range(2, n + 1):
        beta[k] = 1
    return beta


def node_potential(alpha: tuple[int, ...], beta: dict[int, int]) -> int:
    """Compute H(α) = Σᵢ β αᵢ.

    Args:
        alpha: Lattice point α = (α₀, ..., αₙ)
        beta: Vertex potentials

    Returns:
        H(α)
    """
    return sum(beta[i] * alpha[i] for i in range(len(alpha)))


def is_admissible_edge(
    alpha: tuple[int, ...],
    i: int,
    j: int,
    beta: dict[int, int]
) -> bool:
    """Check if α → α + eᵢ - eⱼ is an admissible directed edge.

    Args:
        alpha: Source lattice point
        i: Target vertex index (receiving mass)
        j: Source vertex index (losing mass)
        beta: Vertex potentials

    Returns:
        True iff αⱼ ≥ 1 and βᵢ > β
    """
    if alpha[j] < 1:
        return False
    return beta[i] > beta[j]


def move_mass(
    alpha: tuple[int, ...],
    i: int,
    j: int
) -> tuple[int, ...]:
    """Compute α' = α + eᵢ - eⱼ.

    Args:
        alpha: Source lattice point
        i: Target vertex index
        j: Source vertex index

    Returns:
        New lattice point α'
    """
    alpha_list = list(alpha)
    alpha_list[i] += 1
    alpha_list[j] -= 1
    return tuple(alpha_list)


def get_admissible_edges(
    alpha: tuple[int, ...],
    beta: dict[int, int]
) -> list[tuple[int, int]]:
    """Get all admissible outgoing edges from α.

    Args:
        alpha: Source lattice point
        beta: Vertex potentials

    Returns:
        List of (i, j) pairs representing edges α → α + eᵢ - eⱼ
    """
    n = len(alpha) - 1
    edges = []
    for j in range(n + 1):
        if alpha[j] < 1:
            continue
        for i in range(n + 1):
            if i == j:
                continue
            if beta[i] > beta[j]:
                edges.append((i, j))
    return edges


def edge_type(i: int, j: int, n: int) -> str:
    """Classify edge type.

    Args:
        i: Target vertex
        j: Source vertex
        n: Simplex dimension

    Returns:
        String describing edge type
    """
    if j == 0 and i == 1:
        return "a->b (skip)"
    elif j == 0 and i >= 2:
        return f"a->c_{i-1}"
    elif j >= 2 and i == 1:
        return f"c_{j-1}->b"
    else:
        return f"{j}->{i}"


def F_in(lattice: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Return input facet F_in = {α : α₁ = 0}.

    Args:
        lattice: List of lattice points

    Returns:
        Subset of lattice with α₁ = 0
    """
    return [α for α in lattice if α[1] == 0]


def F_out(lattice: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Return output facet F_out = {α : α₀ = 0}.

    Args:
        lattice: List of lattice points

    Returns:
        Subset of lattice with α₀ = 0
    """
    return [α for α in lattice if α[0] == 0]


def F_mid(lattice: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Return shared face F_mid = F_in ∩ F_out = {α : α₀ = α₁ = 0}.

    Args:
        lattice: List of lattice points

    Returns:
        Subset of lattice with α₀ = α₁ = 0
    """
    return [α for α in lattice if α[0] == 0 and α[1] == 0]


def backbone(lattice: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Return backbone 𝓑_{n,m} = {α : α₂ = ... = αₙ = 0}.

    Args:
        lattice: List of lattice points

    Returns:
        Subset of lattice with α₂ = ... = αₙ = 0
    """
    return [α for α in lattice if all(α[k] == 0 for k in range(2, len(α)))]


# Example usage / quick test
if __name__ == "__main__":
    from lattice import generate_lattice, cardinality

    n, m = 2, 3
    lattice = generate_lattice(n, m)
    beta = vertex_potentials(n)

    print(f"Vertex potentials: β = {beta}")
    print(f"\nLattice V_{{{n},{m}}}:")
    for α in lattice:
        H = node_potential(α, beta)
        edges = get_admissible_edges(α, beta)
        edge_str = ", ".join([f"α+e[{i}]-e[{j}]" for i, j in edges]) if edges else "(none)"
        print(f"  α = {α}, H(α) = {H}, edges: {edge_str}")

    # Verify F_mid properties
    f_in = F_in(lattice)
    f_out = F_out(lattice)
    f_mid = F_mid(lattice)
    print(f"\n|F_in| = {len(f_in)} (expected C(m+n-2, n-1) = {cardinality(n-1, m)})")
    print(f"|F_out| = {len(f_out)} (expected C(m+n-2, n-1) = {cardinality(n-1, m)})")
    print(f"|F_mid| = {len(f_mid)} (expected C(m+n-3, n-2) = {cardinality(n-2, m) if n >= 2 else 0})")

    # Verify H is constant on F_mid
    h_mid = [node_potential(α, beta) for α in f_mid]
    if h_mid:
        print(f"\nH values on F_mid: {set(h_mid)} (should be singleton {{m-1}} = {{{m-1}}})")

    # Verify backbone
    bb = backbone(lattice)
    print(f"\n|backbone| = {len(bb)} (expected m = {m})")
    for α in bb:
        print(f"  α = {α}, H(α) = {node_potential(α, beta)}")
