"""
Main simulation module.

Orchestrates the eco-evolutionary hunting simulation.
"""

import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional
import csv

from config import Config
from world import World
from agents import AgentManager
from prey import PreyManager
from observer import compute_observations
from controller import batch_forward
from dynamics import (
    update_positions,
    compute_motion_costs,
    compute_basal_costs,
    MetabolismCalculator,
    apply_energy_costs,
    velocities_from_torch,
)
from hunting import resolve_hunting
from selector import apply_selection_with_birth_intent
from reproduction import prepare_births, execute_births, BirthQueue
from metrics import MetricsTracker, format_log_line
from console import ConsoleLogger
from checkpoint import CheckpointManager, IncompatibilityError, get_latest_checkpoint
from run_context import RunContext


class Simulation:
    """
    Main simulation class.

    Orchestrates the daily cycle of the eco-evolutionary simulation.
    """

    def __init__(self, config: Config):
        """
        Initialize the simulation.

        Args:
            config: Configuration object
        """
        self.config = config

        # Set up random number generators for reproducibility
        self.np_rng = np.random.default_rng(config.seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(config.seed)
        torch.manual_seed(config.seed)

        # Determine base directory
        self.base_dir = Path(__file__).parent.parent

        # Create RunContext for unified run management
        # If resuming, extract run_id and start_day from checkpoint
        run_id = None
        start_day = 0
        if config.checkpoint_resume and config.checkpoint_enabled:
            # Try to find the latest checkpoint and extract its info
            ckpt_info = RunContext.find_latest_checkpoint_info(self.base_dir)
            if ckpt_info is not None:
                run_id = ckpt_info["run_id"]
                start_day = ckpt_info["day"]

        # Create RunContext with segment folder (e.g., "0-1000", "1001-2000")
        self.run_context = RunContext(
            base_dir=self.base_dir,
            run_id=run_id,
            seed=config.seed,
            start_day=start_day,
            end_day=config.T_days,
        )
        self.run_id = self.run_context.run_id
        self.output_dir = self.run_context.output_dir

        # Save resolved config
        config.save_yaml(self.output_dir / "config_resolved.yaml")

        # Initialize components
        self.world = World(config.omega_x, config.omega_y)

        # Determine neural network input size based on scent mode
        # Vector mode: 5D, stencil mode: 11D (nose 9D + energy + age)
        input_size = 11 if config.scent_mode == "stencil" else 5

        self.agent_manager = AgentManager(
            n=config.n,
            eps_birth_energy=config.eps_birth_energy,
            hidden_sizes=config.hidden_sizes,
            max_speed=config.max_speed,
            np_rng=self.np_rng,
            torch_rng=self.torch_rng,
            input_size=input_size,
        )

        self.prey_manager = PreyManager(
            e_prey=config.e_prey,
            T_prey=config.T_prey,
            E_f=config.E_f,
            rng=self.np_rng,
        )

        # Initialize metabolism calculator if enabled
        self.metabolism_calculator = None
        if config.metabolism_enabled and config.metabolism_mode == "sigmoid_age":
            self.metabolism_calculator = MetabolismCalculator(
                e0=config.e0,
                k=config.metabolism_k,
                a0_ref=config.metabolism_a0_ref,
                m_ref=config.metabolism_m_ref,
                lifespan_exp=config.metabolism_lifespan_exp,
                c_age=config.metabolism_c_age,
                precompute_enabled=config.metabolism_precompute_enabled,
                precompute_max_age=config.metabolism_precompute_max_age,
            )

        self.birth_queue = BirthQueue()
        self.metrics = MetricsTracker(eta_window_days=config.eta_window_days)
        self.logger = ConsoleLogger(self.output_dir / "console.log")

        # Video components (lazy loaded)
        self._video_writer = None
        self._frame_buffer = None

        # Current simulation day
        self.day = 0

        # Checkpoint manager (uses run_context.checkpoint_dir for run-specific path)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.run_context.checkpoint_dir,
            every_days=config.checkpoint_every_days,
            keep_last=config.checkpoint_keep_last,
            logger=self.logger,
        ) if config.checkpoint_enabled else None

        # Track if we resumed from checkpoint
        self._resumed = False

    def _initialize_population(self):
        """Initialize the agent population."""
        positions = self.world.random_positions(self.config.n, self.np_rng)
        self.agent_manager.initialize_agents(self.config.n, positions)

    def _setup_video(self):
        """Set up video recording if enabled."""
        if self.config.make_video:
            from video import VideoWriter

            self._video_writer = VideoWriter(
                self.output_dir / "video.mp4",
                fps=self.config.video_fps,
                save_last_seconds=self.config.save_last_seconds,
            )
            self._video_writer.start()

    def _render_frame(self):
        """Render and save a frame if video is enabled."""
        if not self.config.make_video or self._video_writer is None:
            return

        from render import create_frame

        frame = create_frame(
            agent_positions=self.agent_manager.get_positions(),
            agent_energies=self.agent_manager.get_energies(),
            prey_positions=self.prey_manager.get_positions(),
            world_bounds=(self.config.omega_x, self.config.omega_y),
            day=self.day,
            population=self.agent_manager.count(),
        )
        self._video_writer.add_frame(frame)

    def step_day(self) -> bool:
        """
        Execute one day of simulation.

        Returns:
            True if simulation should continue, False if terminated
        """
        N_start = self.agent_manager.count()
        self.metrics.start_day(self.day, N_start)

        # Record age exposure for age-dependent rate calculation
        if N_start > 0:
            self.metrics.record_age_exposure(self.agent_manager.get_ages())

        # 1. Execute births from previous day's queue
        n_births = self.birth_queue.execute(self.agent_manager)
        self.metrics.record_births(n_births)

        # 2. Inject new preys
        self.prey_manager.inject_preys(self.config.omega_x, self.config.omega_y)
        injected_energy = int(self.config.E_f / self.config.e_prey) * self.config.e_prey
        self.metrics.record_prey_injection(injected_energy)

        # 3. Compute observations for all agents
        agent_positions = self.agent_manager.get_positions()
        agent_energies = self.agent_manager.get_energies()
        agent_ages = self.agent_manager.get_ages()
        prey_positions = self.prey_manager.get_positions()

        observations = compute_observations(
            agent_positions,
            agent_energies,
            agent_ages,
            prey_positions,
            self.config.R,
            self.world,
            self.config.boundary_mode,
            self.config.scent_mode,
            self.config.stencil_h,
            self.config.stencil_geometry,
            self.config.R_eye,
        )

        # Apply input gains for staged learning / ablations
        if self.config.scent_mode == "stencil":
            if self.config.scent_gain != 1.0:
                observations[:, 0:9] *= self.config.scent_gain   # nose_0..8
        else:
            # Vector mode: apply scent gain to all scent channels
            if self.config.scent_gain != 1.0:
                observations[:, :3] *= self.config.scent_gain

        # 4. Batched NN forward to get velocities and birth intents
        controllers = self.agent_manager.get_controllers()
        if len(controllers) > 0:
            # Controller outputs (N, 3): [vx, vy, y_birth]
            outputs_tensor = batch_forward(controllers, observations)
            # Extract velocities (first 2 columns)
            velocities_tensor = outputs_tensor[:, :2]
            velocities = velocities_from_torch(velocities_tensor)
            # Extract birth intents with gain: y_birth > theta_birth
            y_birth_raw = outputs_tensor[:, 2].numpy()
            y_birth = self.config.birth_gain * y_birth_raw
            birth_intents = y_birth > self.config.theta_birth
            self.agent_manager.set_birth_intents(birth_intents)
        else:
            velocities = np.empty((0, 2))

        # 5. Update positions with boundary handling
        if len(agent_positions) > 0:
            new_positions = update_positions(
                agent_positions,
                velocities,
                self.config.dt,
                self.world,
                self.config.boundary_mode,
            )
            self.agent_manager.set_positions(new_positions)

        # 6. Apply motion and metabolic costs
        agent_masses = self.agent_manager.get_masses()
        total_aging = 0.0
        if len(velocities) > 0:
            motion_costs = compute_motion_costs(
                velocities, agent_masses, self.config.motion_cost_coef
            )

            # Use unified metabolism model if enabled
            if self.metabolism_calculator is not None:
                agent_ages = self.agent_manager.get_ages()
                # Unified metabolic cost includes basal + aging
                metabolic_costs = self.metabolism_calculator.compute_metabolic_costs(
                    agent_ages, agent_masses
                )
                # For metrics: separate basal (young baseline) from aging component
                young_basal = compute_basal_costs(agent_masses, self.config.e0)
                total_basal = float(np.sum(young_basal))
                total_aging = float(np.sum(metabolic_costs)) - total_basal

                new_energies, total_motion, _ = apply_energy_costs(
                    self.agent_manager.get_energies(),
                    motion_costs,
                    metabolic_costs,  # Use unified metabolic cost
                )
                self.agent_manager.set_energies(new_energies)
                self.metrics.record_energy_costs(total_motion, total_basal)
            else:
                # Fallback: just use basal costs (no aging)
                basal_costs = compute_basal_costs(agent_masses, self.config.e0)
                new_energies, total_motion, total_basal = apply_energy_costs(
                    self.agent_manager.get_energies(),
                    motion_costs,
                    basal_costs,
                )
                self.agent_manager.set_energies(new_energies)
                self.metrics.record_energy_costs(total_motion, total_basal)
        else:
            self.metrics.record_energy_costs(0.0, 0.0)
        self.metrics.record_aging_cost(total_aging)

        # 7. Hunting: detect agent-prey within r_eat, resolve conflicts, transfer energy
        energy_gained, _, intake_by_age = resolve_hunting(
            self.agent_manager,
            self.prey_manager,
            self.config.r_eat,
            self.world,
            self.config.boundary_mode,
            self.config.intake_rate_per_neuron,
        )
        self.metrics.record_hunting(energy_gained)
        self.metrics.record_intake_by_age(intake_by_age)

        # 8. Prey decay: decrement shelf-life, remove rotten
        rotted_energy = self.prey_manager.decay_preys()
        self.metrics.record_rotted_energy(rotted_energy)

        # 9. Age increment
        self.agent_manager.increment_ages()

        # 10. Natural selection with self-decided reproduction
        # Birth costs are deducted here for agents with birth_intent
        # Then death check: energy < 0 or age > A
        # Survivors with birth_intent become parents
        n_deaths, parents, death_ages, dystocia_count = apply_selection_with_birth_intent(
            self.agent_manager,
            self.config.eps_birth_cost,
            self.config.eps_birth_energy,
            self.config.A,
        )
        self.metrics.record_deaths(n_deaths)
        self.metrics.record_death_ages(death_ages)

        # Calculate total birth cost from successful births
        # Each successful parent paid: mass * (eps_birth_cost + eps_birth_energy)
        total_birth_cost = sum(
            p.mass * (self.config.eps_birth_cost + self.config.eps_birth_energy)
            for p in parents
        )

        # Create offspring for surviving parents (costs already deducted in selector)
        births, n_scheduled = prepare_births(
            parents,
            self.config.eps_birth_energy,
            self.config.delta,
            self.torch_rng,
        )
        self.birth_queue.schedule(births)

        # Record birth energy cost
        self.metrics.record_birth_cost(total_birth_cost)

        # Record parent ages for age-dependent birth rate
        parent_ages = [b.parent.age for b in births]
        self.metrics.record_birth_ages(parent_ages)

        # 11. Finalize metrics
        metrics = self.metrics.end_day(
            N_end=self.agent_manager.count(),
            M=self.agent_manager.total_mass(),
            energies=self.agent_manager.get_energies(),
            prey_count=self.prey_manager.count(),
            prey_energy_total=self.prey_manager.total_energy(),
        )

        # Log every log_every days
        if self.day % self.config.log_every == 0:
            self.logger.log(format_log_line(metrics, self.config.eta_window_days))

        # Render frame if video enabled
        for _ in range(self.config.frames_per_day):
            self._render_frame()

        self.day += 1

        # Check termination conditions
        if self.agent_manager.count() == 0:
            self.logger.log(f"Population extinct at day {self.day}")
            return False

        return True

    def _try_resume(self) -> bool:
        """
        Attempt to resume from checkpoint if configured.

        Returns:
            True if successfully resumed, False otherwise
        """
        if not self.config.checkpoint_resume:
            return False

        if self.checkpoint_manager is None:
            self.logger.log("Warning: checkpoint_resume=true but checkpointing is disabled")
            return False

        self.logger.log("Attempting to resume from checkpoint...")

        success, restored_day, message = self.checkpoint_manager.load(
            config=self.config,
            agent_manager=self.agent_manager,
            prey_manager=self.prey_manager if self.config.checkpoint_resume_mode == "exact_replay" else None,
            np_rng=self.np_rng if self.config.checkpoint_resume_mode == "exact_replay" else None,
            torch_rng=self.torch_rng if self.config.checkpoint_resume_mode == "exact_replay" else None,
            strict=self.config.checkpoint_resume_strict,
            resume_mode=self.config.checkpoint_resume_mode,
        )

        if success:
            self.day = restored_day
            self._resumed = True
            self.logger.log(f"Resumed from checkpoint at day {restored_day}")
            return True
        else:
            if self.config.checkpoint_resume_strict:
                raise IncompatibilityError(message)
            else:
                self.logger.log(f"Could not resume: {message}")
                self.logger.log("Starting fresh simulation instead")
                return False

    def _save_checkpoint(self, force: bool = False):
        """
        Save checkpoint if conditions are met.

        Args:
            force: Force save regardless of schedule
        """
        if self.checkpoint_manager is None:
            return

        if force or self.checkpoint_manager.should_save(self.day):
            self.checkpoint_manager.save(
                day=self.day,
                agent_manager=self.agent_manager,
                prey_manager=self.prey_manager,
                np_rng=self.np_rng,
                torch_rng=self.torch_rng,
                config_dict=self.config.to_dict(),
                resume_mode=self.config.checkpoint_resume_mode,
            )

    def run(self):
        """Run the complete simulation."""
        self.logger.log(f"Starting simulation with seed {self.config.seed}")
        self.logger.log(f"Output directory: {self.output_dir}")
        self.logger.log(f"Boundary mode: {self.config.boundary_mode}")
        scent_info = f"Scent mode: {self.config.scent_mode}"
        if self.config.scent_mode == "stencil":
            scent_info += f" (geometry={self.config.stencil_geometry}, h={self.config.stencil_h}, input_size=11)"
        else:
            scent_info += " (input_size=5)"
        self.logger.log(scent_info)
        self.logger.log(f"Video enabled: {self.config.make_video}")
        if self.checkpoint_manager:
            self.logger.log(f"Checkpointing enabled: dir={self.config.checkpoint_dir}, every={self.config.checkpoint_every_days} days")
        self.logger.log("-" * 80)

        # Try to resume from checkpoint
        resumed = self._try_resume()

        if not resumed:
            # Initialize fresh population
            self._initialize_population()

        # Set up video recording
        self._setup_video()

        # Run simulation
        while self.day < self.config.T_days:
            if not self.step_day():
                break

            # Periodic checkpoint save
            self._save_checkpoint()

        # Finalize
        self._finalize()

    def _save_age_energy_snapshot(self):
        """
        Save final-day age-energy snapshot to CSV.

        The file age_energy_snapshot.csv contains per-age statistics:
        age,count,mean_energy
        """
        ages = self.agent_manager.get_ages()
        energies = self.agent_manager.get_energies()

        if len(ages) == 0:
            self.logger.log("Warning: No agents alive at end, skipping age-energy snapshot")
            return

        ages_arr = np.asarray(ages, dtype=int)
        energies_arr = np.asarray(energies, dtype=float)

        if ages_arr.size == 0 or energies_arr.size == 0:
            self.logger.log("Warning: Empty ages or energies, skipping age-energy snapshot")
            return

        unique_ages = np.unique(ages_arr)
        rows = []
        for age in unique_ages:
            mask = ages_arr == age
            count = int(np.sum(mask))
            if count <= 0:
                continue
            mean_energy = float(np.mean(energies_arr[mask]))
            rows.append({
                "t": int(self.day),
                "age": int(age),
                "count": count,
                "mean_energy": mean_energy,
            })

        if not rows:
            self.logger.log("Warning: No valid age-energy data, skipping age-energy snapshot")
            return

        output_path = self.output_dir / "age_energy_snapshot.csv"
        fieldnames = ["t", "age", "count", "mean_energy"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(rows, key=lambda r: r["age"]):
                writer.writerow(row)

        self.logger.log(f"Age-energy snapshot saved to: {output_path}")

    def _plot_age_distribution_final(self):
        """Generate final age distribution histogram if enabled."""
        if not self.config.plot_age_distribution_final:
            return

        ages = self.agent_manager.get_ages()
        if len(ages) == 0:
            self.logger.log("Warning: No agents alive at end, skipping age distribution plot")
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Create histogram
        max_age = max(ages) if len(ages) > 0 else 1
        bins = min(50, max_age + 1)  # Reasonable number of bins
        ax.hist(ages, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

        ax.set_xlabel('Age (days)')
        ax.set_ylabel('Count')
        ax.set_title(f'Final Age Distribution (Day {self.day}, N={len(ages)})')
        ax.set_xlim(left=0)

        plt.tight_layout()
        output_path = plots_dir / 'age_distribution_final.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.logger.log(f"Age distribution plot saved to: {output_path}")

    def _run_postprocess_plots(self):
        """Run post-processing plotting script if enabled."""
        if not self.config.plot_after_run:
            return

        import subprocess
        import sys

        script_path = Path(__file__).parent.parent / "scripts" / "plot_timeseries.py"
        if not script_path.exists():
            self.logger.log("Warning: plot_timeseries.py not found, skipping auto-plot")
            return

        self.logger.log("Running post-processing plots...")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--run_dir", str(self.output_dir),
                    "--smooth_window", str(self.config.plot_smooth_window),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                self.logger.log("Post-processing plots completed successfully")
            else:
                self.logger.log(f"Post-processing plots failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            self.logger.log("Post-processing plots timed out")
        except Exception as e:
            self.logger.log(f"Post-processing plots error: {e}")

    def _finalize(self):
        """Finalize the simulation and save outputs."""
        # Save final checkpoint
        self._save_checkpoint(force=True)

        # Save metrics to CSV
        self.metrics.save_csv(self.output_dir / "timeseries.csv")

        # Save age-at-event data for computing age-dependent rates
        self.metrics.save_age_events_csv(self.output_dir / "age_events.csv")
        self.metrics.save_age_exposure_csv(self.output_dir / "age_exposure.csv")

        # Save final-day age-energy snapshot for offline analysis
        self._save_age_energy_snapshot()

        self.logger.log("-" * 80)
        self.logger.log(f"Simulation completed at day {self.day}")
        self.logger.log(f"Final population: {self.agent_manager.count()}")
        self.logger.log(f"Metrics saved to: {self.output_dir / 'timeseries.csv'}")
        self.logger.log(f"Age events saved to: {self.output_dir / 'age_events.csv'}")

        # Close video writer
        if self._video_writer is not None:
            self._video_writer.close()
            self.logger.log(f"Video saved to: {self.output_dir / 'video.mp4'}")

        # Generate age distribution plot
        self._plot_age_distribution_final()

        # Run post-processing plots
        self._run_postprocess_plots()

        # Close logger
        self.logger.close()
