#!/usr/bin/env python3
"""Verify SMNModule is working correctly."""

import torch
import sys
sys.path.insert(0, 'src')

from smn_module import SMNModule

def test_basic():
    """Test basic forward pass."""
    print("Testing SMNModule basic forward pass...")

    # SISO configuration
    module = SMNModule(n=2, m=3, n_in=1, n_out=1, x_bounds=[(-10, 10)])
    print(f"Created: {module}")
    print(f"Architecture: {module.arch_str}")
    print(f"Parameters: {module.param_count}")

    # Forward pass
    x = torch.randn(32, 1)
    y = module(x)
    assert y.shape == (32, 1), f"Expected (32, 1), got {y.shape}"
    print(f"Forward pass OK: input {x.shape} -> output {y.shape}")

    # Gradient flow
    loss = y.sum()
    loss.backward()

    # Check gradients exist
    for name, param in module.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    print("Gradient flow OK")

    # State dict roundtrip
    state = module.state_dict()
    module2 = SMNModule(n=2, m=3, n_in=1, n_out=1, x_bounds=[(-10, 10)])
    module2.load_state_dict(state)
    print("State dict roundtrip OK")

    print("\n" + "="*50)
    print("All SMNModule tests PASSED!")
    print("="*50)

def test_dqn_config():
    """Test configuration for DQN (1 input, 7 discrete actions)."""
    print("\nTesting DQN configuration (1→7)...")

    module = SMNModule(
        n=2, m=4,
        n_in=1, n_out=7,  # 7 discrete actions
        activation='relu',
        x_bounds=[(-10, 10)]
    )
    print(f"Created: {module}")

    x = torch.randn(64, 1)
    y = module(x)
    assert y.shape == (64, 7), f"Expected (64, 7), got {y.shape}"
    print(f"Forward pass OK: input {x.shape} -> output {y.shape}")

    print("DQN configuration PASSED!")

def test_reinforce_config():
    """Test configuration for REINFORCE (1 input, 2 outputs for continuous action)."""
    print("\nTesting REINFORCE configuration (1→2 for continuous action)...")

    module = SMNModule(
        n=2, m=4,
        n_in=1, n_out=2,  # [action_mean, action_log_std]
        activation='relu',
        x_bounds=[(-10, 10)]
    )
    print(f"Created: {module}")

    x = torch.randn(64, 1)
    y = module(x)
    assert y.shape == (64, 2), f"Expected (64, 2), got {y.shape}"
    print(f"Forward pass OK: input {x.shape} -> output {y.shape}")

    print("REINFORCE configuration PASSED!")

if __name__ == "__main__":
    test_basic()
    test_dqn_config()
    test_reinforce_config()
    print("\n✓ SMNModule verification complete!")
