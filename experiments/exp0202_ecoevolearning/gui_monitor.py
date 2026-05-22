#!/usr/bin/env python3
"""
Simple local GUI monitor for the eco‑evo simulation.

Usage:
    cd exp2-2
    python gui_monitor.py

This script:
  - Loads config.yaml
  - Creates a Simulation instance
  - Runs the simulation in a background thread
  - Shows a small tkinter + matplotlib window with:
        * Current day
        * Population size N
        * Utilization eta_window
  - Updates the plot/labels periodically.

No HTML, no HTTP server, no browser required.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, List

import numpy as np

import matplotlib

# Use a GUI backend suitable for tkinter
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402

import tkinter as tk  # noqa: E402
from tkinter import ttk  # noqa: E402

from src.config import Config  # type: ignore
from src.sim import Simulation  # type: ignore
from src.metrics import DayMetrics  # type: ignore


class SimulationRunner:
    """
    Run Simulation in a background thread and collect metrics.
    """

    def __init__(self, config: Config):
        self.sim = Simulation(config)
        # Disable heavy outputs for interactive monitoring
        self.sim.config.plot_after_run = False
        self.sim.config.make_video = False

        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        self._lock = threading.Lock()

        # Buffers for metrics time‑series
        self.days: List[int] = []
        self.N: List[int] = []
        self.eta: List[float] = []

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._pause_flag.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def pause(self):
        self._pause_flag.set()

    def resume(self):
        self._pause_flag.clear()

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run_loop(self):
        """
        Run the simulation day by day until stopped or extinct.
        """
        while not self._stop_flag.is_set():
            # Pause handling
            while self._pause_flag.is_set() and not self._stop_flag.is_set():
                time.sleep(0.1)

            if self._stop_flag.is_set():
                break

            # One day step
            should_continue = self.sim.step_day()
            latest = self.sim.metrics.get_latest()
            if latest is not None:
                with self._lock:
                    self.days.append(int(latest.t))
                    self.N.append(int(latest.N))
                    self.eta.append(float(latest.eta_window))

            if not should_continue:
                break

    def snapshot(self):
        """
        Get a thread‑safe snapshot of the metrics series.
        """
        with self._lock:
            return (
                list(self.days),
                list(self.N),
                list(self.eta),
                self.sim.metrics.get_latest(),
            )


class MonitorGUI:
    """
    Tkinter + matplotlib GUI to display simulation progress.
    """

    def __init__(self, runner: SimulationRunner, refresh_ms: int = 500):
        self.runner = runner
        self.refresh_ms = refresh_ms

        # --- Tk root window ---
        self.root = tk.Tk()
        self.root.title("EcoEvo Simulation Monitor (exp2-2)")

        # Top info frame
        info_frame = ttk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.day_var = tk.StringVar(value="0")
        self.N_var = tk.StringVar(value="0")
        self.eta_var = tk.StringVar(value="0.00")

        ttk.Label(info_frame, text="Day:").grid(row=0, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.day_var).grid(
            row=0, column=1, sticky="w"
        )

        ttk.Label(info_frame, text="N:").grid(row=0, column=2, padx=(20, 0), sticky="w")
        ttk.Label(info_frame, textvariable=self.N_var).grid(
            row=0, column=3, sticky="w"
        )

        ttk.Label(info_frame, text="η (window):").grid(
            row=0, column=4, padx=(20, 0), sticky="w"
        )
        ttk.Label(info_frame, textvariable=self.eta_var).grid(
            row=0, column=5, sticky="w"
        )

        # Control buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.btn_start = ttk.Button(btn_frame, text="Start", command=self.on_start)
        self.btn_start.pack(side=tk.LEFT)

        self.btn_pause = ttk.Button(btn_frame, text="Pause", command=self.on_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=5)

        self.btn_resume = ttk.Button(btn_frame, text="Resume", command=self.on_resume)
        self.btn_resume.pack(side=tk.LEFT, padx=5)

        self.btn_quit = ttk.Button(btn_frame, text="Quit", command=self.on_quit)
        self.btn_quit.pack(side=tk.RIGHT)

        # Matplotlib figure inside tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.ax.set_xlabel("Day")
        self.ax.set_ylabel("Population N")
        self.ax.grid(True, alpha=0.3)

        self.line_N, = self.ax.plot([], [], label="N", color="tab:blue")
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel("η_window")
        self.line_eta, = self.ax2.plot([], [], label="eta", color="tab:orange")

        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = canvas

        # Schedule periodic updates
        self.root.after(self.refresh_ms, self._refresh)

        # Proper cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

    # --- Button callbacks ---
    def on_start(self):
        self.runner.start()

    def on_pause(self):
        self.runner.pause()

    def on_resume(self):
        self.runner.resume()

    def on_quit(self):
        self.runner.stop()
        self.root.destroy()

    # --- Periodic UI refresh ---
    def _refresh(self):
        days, N, eta, latest = self.runner.snapshot()
        if latest is not None:
            self.day_var.set(str(int(latest.t)))
            self.N_var.set(str(int(latest.N)))
            self.eta_var.set(f"{float(latest.eta_window):.2f}")

        if days:
            x = np.array(days)
            yN = np.array(N)
            yeta = np.array(eta)

            self.line_N.set_data(x, yN)
            self.line_eta.set_data(x, yeta)

            self.ax.set_xlim(float(x.min()), float(x.max()) + 1.0)
            # Avoid zero-height axis
            ymin, ymax = float(yN.min()), float(yN.max())
            if ymin == ymax:
                ymin -= 1
                ymax += 1
            self.ax.set_ylim(ymin, ymax)

            eta_min, eta_max = float(yeta.min()), float(yeta.max())
            if eta_min == eta_max:
                eta_min -= 0.1
                eta_max += 0.1
            self.ax2.set_ylim(eta_min, max(eta_max, eta_min + 0.1))

        self.canvas.draw_idle()
        # Reschedule
        self.root.after(self.refresh_ms, self._refresh)

    def run(self):
        self.root.mainloop()


def main():
    base_dir = Path(__file__).parent
    cfg_path = base_dir / "config.yaml"
    config = Config.from_yaml(cfg_path)

    # For interactive monitor: keep it lighter by default
    config.plot_after_run = False
    config.make_video = False

    runner = SimulationRunner(config)
    gui = MonitorGUI(runner)
    gui.run()


if __name__ == "__main__":
    main()

