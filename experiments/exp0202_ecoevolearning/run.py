#!/usr/bin/env python3
"""
Entry point for the eco-evolutionary hunting simulation.

Usage:
    python run.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from sim import Simulation


def main():
    parser = argparse.ArgumentParser(
        description="Eco-evolutionary hunting simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = Config.from_yaml(config_path)

    # Create and run simulation
    sim = Simulation(config)
    sim.run()


if __name__ == "__main__":
    main()
