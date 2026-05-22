"""
Neural network controller for agents.

MLP with ReLU activations, forward-only (no backprop).
"""

import torch
import torch.nn as nn
from typing import List


class NeuralController(nn.Module):
    """
    MLP controller that maps observations to velocity commands and birth intent.

    Supports two input modes:
    - "vector" mode: Input is [S_vec_x, S_vec_y, S_scalar, energy, age] (5D)
    - "stencil" mode: Input is [nose_0..8, energy, age] (11D total)

    Output: (vx, vy, y_birth) where:
    - vx, vy: velocity command (scaled by max_speed)
    - y_birth: birth intent signal in [-1, 1] (compared against theta_birth)

    Uses ReLU activations and tanh output for bounded outputs.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        max_speed: float = 5.0,
        input_size: int = 5,
    ):
        """
        Initialize the neural controller.

        Args:
            hidden_sizes: List of hidden layer sizes
            max_speed: Maximum speed for velocity output
            input_size: Size of input observation (5 for vector mode, 11 for stencil)
        """
        super().__init__()
        self.max_speed = max_speed
        self.input_size = input_size

        # Build MLP layers
        layers = []
        prev_size = self.input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size

        # Output layer: 3D (vx, vy, y_birth)
        layers.append(nn.Linear(prev_size, 3))
        layers.append(nn.Tanh())  # Bound output to [-1, 1]

        self.mlp = nn.Sequential(*layers)

        # Set to eval mode and disable gradients
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute velocity command and birth intent.

        Args:
            x: (N, input_size) tensor of observations

        Returns:
            (N, 3) tensor of [vx * max_speed, vy * max_speed, y_birth]
            where y_birth remains in [-1, 1] for threshold comparison
        """
        with torch.no_grad():
            out = self.mlp(x)  # (N, 3) in [-1, 1]
            # Scale velocity components in-place (faster than sliced assignment)
            out[:, 0] *= self.max_speed
            out[:, 1] *= self.max_speed
            return out

    def get_num_neurons(self) -> int:
        """
        Count total number of neurons (hidden units) in the network.

        Returns:
            Total neuron count (sum of hidden layer sizes)
        """
        count = 0
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                # Count output features of linear layer (except final output)
                count += module.out_features
        # Subtract output layer neurons (3) since we only count hidden
        count -= 3
        return count

    def get_num_edges(self) -> int:
        """
        Count total number of edges (trainable weights) in the network.

        This is the primary measure of agent "mass" for energy scaling.
        Edges = sum over all Linear layers of (in_features * out_features).
        Biases are excluded.

        Returns:
            Total edge count (number of weight parameters, excluding biases)
        """
        count = 0
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                # Each edge connects one input to one output
                count += module.in_features * module.out_features
        return count

    def get_parameters_flat(self) -> torch.Tensor:
        """
        Get all parameters as a flat tensor.

        Returns:
            1D tensor of all parameters
        """
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)

    def set_parameters_flat(self, flat_params: torch.Tensor):
        """
        Set parameters from a flat tensor.

        Args:
            flat_params: 1D tensor of all parameters
        """
        idx = 0
        for param in self.parameters():
            numel = param.numel()
            param.data.copy_(flat_params[idx : idx + numel].view(param.shape))
            idx += numel

    def clone(self) -> "NeuralController":
        """
        Create a deep copy of this controller.

        Returns:
            New NeuralController with same architecture and parameters
        """
        # Get hidden sizes from existing architecture
        hidden_sizes = []
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                if module.out_features != 3:  # Not output layer
                    hidden_sizes.append(module.out_features)

        # Create new controller with same input size
        new_controller = NeuralController(
            hidden_sizes=hidden_sizes,
            max_speed=self.max_speed,
            input_size=self.input_size,
        )
        # Copy parameters
        new_controller.set_parameters_flat(self.get_parameters_flat().clone())
        return new_controller


def create_controller(
    hidden_sizes: List[int],
    max_speed: float,
    rng: torch.Generator,
    input_size: int = 5,
) -> NeuralController:
    """
    Create a neural controller with random initialization.

    Args:
        hidden_sizes: List of hidden layer sizes
        max_speed: Maximum speed for velocity output
        rng: PyTorch random generator for reproducibility
        input_size: Size of input observation (5 for vector, 11 for stencil)

    Returns:
        Initialized NeuralController
    """
    controller = NeuralController(hidden_sizes, max_speed, input_size)

    # Reinitialize weights with the provided RNG
    for module in controller.mlp:
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization
            fan_in = module.weight.size(1)
            fan_out = module.weight.size(0)
            std = (2.0 / (fan_in + fan_out)) ** 0.5
            module.weight.data.normal_(0, std, generator=rng)
            if module.bias is not None:
                module.bias.data.zero_()

    return controller


def batch_forward(
    controllers: List[NeuralController],
    observations: torch.Tensor,
) -> torch.Tensor:
    """
    Batch forward pass for multiple controllers.

    Since each agent has different parameters, we need to iterate.
    However, we can still process efficiently by stacking.

    Args:
        controllers: List of N controllers
        observations: (N, input_size) tensor of observations

    Returns:
        (N, 3) tensor of [vx, vy, y_birth] outputs
        - vx, vy: velocity commands (scaled by max_speed)
        - y_birth: birth intent signal in [-1, 1]
    """
    if len(controllers) == 0:
        return torch.empty((0, 3))

    outputs = []
    with torch.no_grad():
        for i, ctrl in enumerate(controllers):
            obs = observations[i : i + 1]  # (1, input_size)
            out = ctrl(obs)  # (1, 3)
            outputs.append(out)

    return torch.cat(outputs, dim=0)  # (N, 3)
