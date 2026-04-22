"""Unit tests for SMNModule.

Run from the project root:
    PYTHONPATH=src python3 tests/test_smn.py

Tests are standalone — no Config, no data.py dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from smn_fitter import SMNModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_shape(module: SMNModule, batch: int) -> None:
    x = torch.randn(batch, module.n_in)
    y = module(x)
    assert y.shape == (batch, module.n_out), (
        f"Expected ({batch}, {module.n_out}), got {y.shape}"
    )


def _check_output_range(module: SMNModule, n_samples: int = 256) -> None:
    """Output should be in (-1, 1) due to final tanh."""
    x = torch.randn(n_samples, module.n_in) * 5
    y = module(x)
    assert y.min() > -1.0 and y.max() < 1.0, (
        f"Output out of tanh range: min={y.min():.4f}, max={y.max():.4f}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_siso_shape():
    """SISO: (B, 1) → (B, 1)."""
    m = SMNModule(n=2, m=3, n_in=1, n_out=1)
    _check_shape(m, batch=32)
    print("PASS  test_siso_shape")


def test_mimo_shape():
    """MIMO: (B, 3) → (B, 2)."""
    m = SMNModule(n=3, m=3, n_in=3, n_out=2,
                  x_bounds=[(-1., 1.), (-2., 2.), (0., 5.)])
    _check_shape(m, batch=16)
    print("PASS  test_mimo_shape")


def test_1d_input_squeezed():
    """Accepts (B,) input for n_in=1."""
    m = SMNModule(n=2, m=2, n_in=1, n_out=1)
    x = torch.randn(8)          # 1-D, not (8, 1)
    y = m(x)
    assert y.shape == (8, 1), f"Expected (8, 1), got {y.shape}"
    print("PASS  test_1d_input_squeezed")


def test_output_range():
    """Outputs lie strictly in (-1, 1)."""
    m = SMNModule(n=3, m=4)
    _check_output_range(m)
    print("PASS  test_output_range")


def test_x_bounds_normalisation():
    """Two models with same weights but different x_bounds should differ."""
    # Use the same seed so both get identical random weights
    torch.manual_seed(7)
    m_narrow = SMNModule(n=2, m=3, n_in=1, n_out=1, x_bounds=[(0., 2.)])
    torch.manual_seed(7)
    m_wide   = SMNModule(n=2, m=3, n_in=1, n_out=1, x_bounds=[(0., 10.)])

    # x=1.0 normalises to 0.0 for narrow, -0.8 for wide — outputs must differ
    x = torch.tensor([[1.0]])
    y_narrow = m_narrow(x)
    y_wide   = m_wide(x)
    assert not torch.allclose(y_narrow, y_wide), (
        f"Different x_bounds should change output: narrow={y_narrow.item():.4f}, "
        f"wide={y_wide.item():.4f}"
    )
    print("PASS  test_x_bounds_normalisation")


def test_x_bounds_default_identity():
    """Default x_bounds=[(-1,1)] acts as identity normalisation."""
    m = SMNModule(n=2, m=2, n_in=1, n_out=1)   # x_bounds defaults to [(-1,1)]
    # Input already in [-1, 1]; normalised value should equal input
    x = torch.tensor([[0.5]])
    # We can't directly inspect the normalised value, but we can confirm
    # x_bounds default is [(-1,1)] via the registered buffers
    assert m._x_min.item() == -1.0
    assert m._x_max.item() ==  1.0
    print("PASS  test_x_bounds_default_identity")


def test_x_bounds_mismatch_raises():
    """x_bounds length != n_in should raise ValueError."""
    try:
        SMNModule(n=2, m=2, n_in=2, n_out=1, x_bounds=[(-1., 1.)])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS  test_x_bounds_mismatch_raises")


def test_param_count_positive():
    """param_count should be > 0."""
    m = SMNModule(n=2, m=3)
    assert m.param_count > 0
    print("PASS  test_param_count_positive")


def test_arch_str_format():
    """arch_str should contain key fields."""
    m = SMNModule(n=3, m=4, n_in=2, n_out=2)
    s = m.arch_str
    assert "n=3" in s and "m=4" in s and "2→2" in s
    print("PASS  test_arch_str_format")


def test_gradients_flow():
    """Backward pass should produce non-None, non-zero gradients."""
    m = SMNModule(n=2, m=3, n_in=1, n_out=1)
    x = torch.randn(16, 1)
    y = m(x)
    loss = y.mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed"
    assert any(g.abs().sum() > 0 for g in grads), "All gradients are zero"
    print("PASS  test_gradients_flow")


def test_various_n_m():
    """Smoke test across several (n, m) combinations."""
    for n, m in [(2, 2), (2, 5), (3, 3), (4, 2)]:
        mod = SMNModule(n=n, m=m)
        _check_shape(mod, batch=8)
    print("PASS  test_various_n_m")


def test_state_dict_roundtrip():
    """save / load state_dict preserves output exactly."""
    m = SMNModule(n=2, m=3, n_in=1, n_out=1)
    x = torch.randn(4, 1)
    y_before = m(x).detach()

    state = m.state_dict()
    m2 = SMNModule(n=2, m=3, n_in=1, n_out=1)
    m2.load_state_dict(state)
    y_after = m2(x).detach()

    assert torch.allclose(y_before, y_after), "state_dict roundtrip changed output"
    print("PASS  test_state_dict_roundtrip")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_siso_shape,
        test_mimo_shape,
        test_1d_input_squeezed,
        test_output_range,
        test_x_bounds_normalisation,
        test_x_bounds_default_identity,
        test_x_bounds_mismatch_raises,
        test_param_count_positive,
        test_arch_str_format,
        test_gradients_flow,
        test_various_n_m,
        test_state_dict_roundtrip,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed:
        sys.exit(1)
