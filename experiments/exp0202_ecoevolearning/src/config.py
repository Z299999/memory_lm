"""
Configuration loading and management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml


@dataclass
class Config:
    """Configuration container for the simulation."""

    # Random seed
    seed: int = 0

    # Simulation time
    T_days: int = 2000
    dt: float = 1.0
    log_every: int = 10

    # World parameters
    omega_x: Tuple[float, float] = (0.0, 100.0)
    omega_y: Tuple[float, float] = (0.0, 100.0)
    boundary_mode: str = "periodic"  # Options: "periodic", "reflect"

    # Agent parameters
    n: int = 200
    R: float = 10.0          # Nose (scent) detection radius
    r_eat: float = 1.0

    # Per-edge energy parameters (scaled by mass = num_edges at runtime)
    eps_survival: float = 0.0           # Energy per edge to survive (E0 = mass * eps_survival)
    eps_birth_cost: float = 0.2         # Energy per edge paid for reproduction
    eps_birth_energy: float = 0.2       # Initial energy per edge for offspring

    e0: float = 0.01  # Basal metabolic cost per edge per day
    A: int = 200
    delta: float = 0.01

    # Neural controller
    hidden_sizes: List[int] = field(default_factory=lambda: [16, 16])
    max_speed: float = 2.0
    motion_cost_coef: float = 0.01  # Coefficient for motion cost (coef * 0.5 * m * v^2)
    theta_birth: float = 0.0  # Birth intent threshold (agent reproduces if y_birth > theta_birth)

    # Prey parameters
    e_prey: float = 10.0
    T_prey: int = 20
    E_f: float = 1000.0
    eta_window_days: int = 100  # Rolling window for utilization metrics

    # Video settings
    make_video: bool = False
    video_fps: int = 24
    frames_per_day: int = 1
    save_last_seconds: int = 0  # Only save last N seconds (0 = save all)

    # Metabolism parameters (unified basal + aging cost)
    metabolism_enabled: bool = True
    metabolism_mode: str = "sigmoid_age"  # "sigmoid_age" or "legacy"
    metabolism_k: float = 0.06            # Sigmoid steepness
    metabolism_a0_ref: float = 250.0      # Onset age at reference mass
    metabolism_m_ref: float = 1000.0      # Reference mass (edges)
    metabolism_lifespan_exp: float = 0.25 # Quarter-power scaling
    metabolism_c_age: float = 2.0         # Max old-age metabolic multiplier
    metabolism_precompute_enabled: bool = True
    metabolism_precompute_max_age: int = 2000

    # Hunting parameters
    intake_rate_per_neuron: float = 0.2  # Max energy intake per edge per day (legacy name)

    # Sensing parameters
    scent_mode: str = "stencil"  # "vector" (5D) or "stencil" (11D: nose 9D + 2 internal)
    stencil_h: float = 2.0       # Sampling distance for nose stencil points
    stencil_geometry: str = "circle"  # Ignored (kept for backward compatibility)
    R_eye: float = 10.0          # Reserved for future eye implementation (unused)

    # Input gains (for staged learning / ablations)
    scent_gain: float = 1.0      # Global gain for nose inputs
    birth_gain: float = 1.0      # Gain on birth output channel

    # Post-processing
    plot_after_run: bool = True

    # Plot toggles
    plot_eta: bool = True
    plot_mass: bool = True
    plot_birth_death: bool = True
    plot_pop_energy: bool = True
    plot_energy_flux: bool = True
    plot_energy_budget: bool = True
    plot_age_distribution_final: bool = True
    plot_smooth_window: int = 50  # Moving average smoothing window (days)

    # Checkpoint settings
    checkpoint_enabled: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_days: int = 0  # 0 = only at end
    checkpoint_keep_last: int = 3   # 0 = keep all
    checkpoint_resume: bool = False
    checkpoint_resume_strict: bool = True
    checkpoint_resume_mode: str = "continue_population"  # or "exact_replay"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Convert lists to tuples for bounds
        if "omega_x" in data:
            data["omega_x"] = tuple(data["omega_x"])
        if "omega_y" in data:
            data["omega_y"] = tuple(data["omega_y"])

        # Flatten nested metabolism config (replaces old aging config)
        if "metabolism" in data:
            metabolism = data.pop("metabolism")
            if isinstance(metabolism, dict):
                data["metabolism_enabled"] = metabolism.get("enabled", True)
                data["metabolism_mode"] = metabolism.get("mode", "sigmoid_age")
                # Sigmoid parameters
                sigmoid = metabolism.get("sigmoid", {})
                data["metabolism_k"] = sigmoid.get("k", 0.06)
                data["metabolism_a0_ref"] = sigmoid.get("a0_ref", 250.0)
                data["metabolism_m_ref"] = sigmoid.get("m_ref", 1000.0)
                data["metabolism_lifespan_exp"] = sigmoid.get("lifespan_exp", 0.25)
                data["metabolism_c_age"] = metabolism.get("c_age", 2.0)
                # Precompute parameters
                precompute = metabolism.get("precompute", {})
                data["metabolism_precompute_enabled"] = precompute.get("enabled", True)
                data["metabolism_precompute_max_age"] = precompute.get("max_age", 2000)
        # Legacy aging config (for backward compatibility)
        elif "aging" in data:
            aging = data.pop("aging")
            if isinstance(aging, dict):
                # Map legacy aging to disabled metabolism (sim.py will handle legacy)
                data["metabolism_enabled"] = aging.get("enabled", True)
                data["metabolism_mode"] = "legacy"

        # Flatten nested hunting config
        if "hunting" in data:
            hunting = data.pop("hunting")
            if isinstance(hunting, dict):
                # Support both new name (intake_rate_per_edge) and legacy name (intake_rate_per_neuron)
                data["intake_rate_per_neuron"] = hunting.get(
                    "intake_rate_per_edge",
                    hunting.get("intake_rate_per_neuron", 0.2)
                )

        # Flatten nested sensing config
        if "sensing" in data:
            sensing = data.pop("sensing")
            if isinstance(sensing, dict):
                data["scent_mode"] = sensing.get("scent_mode", "stencil")
                data["stencil_h"] = sensing.get("stencil_h", 2.0)
                data["stencil_geometry"] = sensing.get("stencil_geometry", "circle")
                data["R_eye"] = sensing.get("R_eye", 10.0)

        # Optional gains block for input scaling
        if "gains" in data:
            gains = data.pop("gains")
            if isinstance(gains, dict):
                data["scent_gain"] = gains.get("scent_gain", 1.0)
                data["birth_gain"] = gains.get("birth_gain", 1.0)

        # Flatten nested postprocess config
        if "postprocess" in data:
            postprocess = data.pop("postprocess")
            if isinstance(postprocess, dict):
                data["plot_after_run"] = postprocess.get("plot_after_run", True)

        # Flatten nested plots config
        if "plots" in data:
            plots = data.pop("plots")
            if isinstance(plots, dict):
                data["plot_eta"] = plots.get("eta", True)
                data["plot_mass"] = plots.get("mass", True)
                data["plot_birth_death"] = plots.get("birth_death", True)
                data["plot_pop_energy"] = plots.get("pop_energy", True)
                data["plot_energy_flux"] = plots.get("energy_flux", True)
                data["plot_energy_budget"] = plots.get("energy_budget", True)
                data["plot_age_distribution_final"] = plots.get("age_distribution_final", True)
                data["plot_smooth_window"] = plots.get("smooth_window", 50)

        # Flatten nested checkpoint config
        if "checkpoint" in data:
            checkpoint = data.pop("checkpoint")
            if isinstance(checkpoint, dict):
                data["checkpoint_enabled"] = checkpoint.get("enabled", True)
                data["checkpoint_dir"] = checkpoint.get("dir", "checkpoints")
                data["checkpoint_every_days"] = checkpoint.get("every_days", 0)
                data["checkpoint_keep_last"] = checkpoint.get("keep_last", 3)
                data["checkpoint_resume"] = checkpoint.get("resume", False)
                data["checkpoint_resume_strict"] = checkpoint.get("resume_strict", True)
                data["checkpoint_resume_mode"] = checkpoint.get("resume_mode", "continue_population")

        return cls(**data)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "seed": self.seed,
            "T_days": self.T_days,
            "dt": self.dt,
            "log_every": self.log_every,
            "omega_x": list(self.omega_x),
            "omega_y": list(self.omega_y),
            "boundary_mode": self.boundary_mode,
            "n": self.n,
            "R": self.R,
            "r_eat": self.r_eat,
            "eps_survival": self.eps_survival,
            "eps_birth_cost": self.eps_birth_cost,
            "eps_birth_energy": self.eps_birth_energy,
            "e0": self.e0,
            "A": self.A,
            "delta": self.delta,
            "hidden_sizes": self.hidden_sizes,
            "max_speed": self.max_speed,
            "motion_cost_coef": self.motion_cost_coef,
            "theta_birth": self.theta_birth,
            "e_prey": self.e_prey,
            "T_prey": self.T_prey,
            "E_f": self.E_f,
            "eta_window_days": self.eta_window_days,
            "make_video": self.make_video,
            "video_fps": self.video_fps,
            "frames_per_day": self.frames_per_day,
            "save_last_seconds": self.save_last_seconds,
            "metabolism": {
                "enabled": self.metabolism_enabled,
                "mode": self.metabolism_mode,
                "sigmoid": {
                    "k": self.metabolism_k,
                    "a0_ref": self.metabolism_a0_ref,
                    "m_ref": self.metabolism_m_ref,
                    "lifespan_exp": self.metabolism_lifespan_exp,
                },
                "c_age": self.metabolism_c_age,
                "precompute": {
                    "enabled": self.metabolism_precompute_enabled,
                    "max_age": self.metabolism_precompute_max_age,
                },
            },
            "hunting": {
                "intake_rate_per_edge": self.intake_rate_per_neuron,
            },
            "sensing": {
                "scent_mode": self.scent_mode,
                "stencil_h": self.stencil_h,
                "stencil_geometry": self.stencil_geometry,
                "R_eye": self.R_eye,
            },
            "gains": {
                "scent_gain": self.scent_gain,
                "birth_gain": self.birth_gain,
            },
            "postprocess": {
                "plot_after_run": self.plot_after_run,
            },
            "plots": {
                "eta": self.plot_eta,
                "mass": self.plot_mass,
                "birth_death": self.plot_birth_death,
                "pop_energy": self.plot_pop_energy,
                "energy_flux": self.plot_energy_flux,
                "energy_budget": self.plot_energy_budget,
                "age_distribution_final": self.plot_age_distribution_final,
                "smooth_window": self.plot_smooth_window,
            },
            "checkpoint": {
                "enabled": self.checkpoint_enabled,
                "dir": self.checkpoint_dir,
                "every_days": self.checkpoint_every_days,
                "keep_last": self.checkpoint_keep_last,
                "resume": self.checkpoint_resume,
                "resume_strict": self.checkpoint_resume_strict,
                "resume_mode": self.checkpoint_resume_mode,
            },
        }

    def save_yaml(self, path: Path):
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
