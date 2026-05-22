"""
Frame rendering module using PIL for lightweight rendering.

Only imported if make_video is True.
"""

import numpy as np
from typing import Tuple, Optional


def create_frame(
    agent_positions: np.ndarray,
    agent_energies: np.ndarray,
    prey_positions: np.ndarray,
    world_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    day: int,
    population: int,
    frame_size: Tuple[int, int] = (800, 800),
) -> np.ndarray:
    """
    Create a single frame of the simulation.

    Uses PIL for lightweight rendering.

    Args:
        agent_positions: (N, 2) array of agent positions
        agent_energies: (N,) array of agent energies
        prey_positions: (M, 2) array of prey positions
        world_bounds: ((x_min, x_max), (y_min, y_max))
        day: Current day number
        population: Current population count
        frame_size: (width, height) in pixels

    Returns:
        (H, W, 3) uint8 numpy array of RGB values
    """
    # Import PIL here to keep it lazy
    from PIL import Image, ImageDraw, ImageFont

    width, height = frame_size
    (x_min, x_max), (y_min, y_max) = world_bounds

    # Create blank white image
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw border
    draw.rectangle([0, 0, width - 1, height - 1], outline=(0, 0, 0), width=2)

    # Scale factors
    scale_x = (width - 20) / (x_max - x_min)
    scale_y = (height - 40) / (y_max - y_min)  # Leave room for text

    def to_pixel(x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        px = int(10 + (x - x_min) * scale_x)
        py = int(10 + (y - y_min) * scale_y)
        return px, py

    # Draw preys as green circles
    prey_radius = 3
    for i in range(len(prey_positions)):
        px, py = to_pixel(prey_positions[i, 0], prey_positions[i, 1])
        draw.ellipse(
            [px - prey_radius, py - prey_radius, px + prey_radius, py + prey_radius],
            fill=(0, 200, 0),
            outline=(0, 150, 0),
        )

    # Draw agents as blue circles with size based on energy
    if len(agent_positions) > 0:
        # Normalize energies for visualization
        E_min = np.min(agent_energies) if len(agent_energies) > 0 else 0
        E_max = np.max(agent_energies) if len(agent_energies) > 0 else 1
        E_range = max(E_max - E_min, 1)

        for i in range(len(agent_positions)):
            px, py = to_pixel(agent_positions[i, 0], agent_positions[i, 1])

            # Size based on energy (2-6 pixels)
            energy_norm = (agent_energies[i] - E_min) / E_range
            agent_radius = int(2 + 4 * energy_norm)

            # Color: blue to red based on energy
            r = int(50 + 200 * (1 - energy_norm))
            g = 50
            b = int(50 + 200 * energy_norm)

            draw.ellipse(
                [
                    px - agent_radius,
                    py - agent_radius,
                    px + agent_radius,
                    py + agent_radius,
                ],
                fill=(r, g, b),
                outline=(0, 0, 0),
            )

    # Draw text info
    text = f"Day: {day}  Population: {population}  Prey: {len(prey_positions)}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()

    draw.text((10, height - 25), text, fill=(0, 0, 0), font=font)

    # Convert to numpy array
    return np.array(img)


class FrameBuffer:
    """
    Buffer for storing frames before video encoding.
    """

    def __init__(self):
        """Initialize empty frame buffer."""
        self.frames = []

    def add_frame(self, frame: np.ndarray):
        """
        Add a frame to the buffer.

        Args:
            frame: (H, W, 3) uint8 numpy array
        """
        self.frames.append(frame)

    def get_frames(self):
        """Get all stored frames."""
        return self.frames

    def clear(self):
        """Clear all stored frames."""
        self.frames = []

    def count(self) -> int:
        """Return number of stored frames."""
        return len(self.frames)
