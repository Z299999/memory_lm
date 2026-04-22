"""PySide6-based GUI for SMN Training.

This module provides an interactive GUI for training and monitoring SMN-based
RL agents with real-time visualization.

Features:
- Left panel: Environment, SMN params, algorithm selection, hyperparameters
- Right panel: Tabbed output (Training curves, Reward analysis, Network, Checkpoint)
- Background training with pause/resume support
- Overwrite protection with diff dialog
- Real-time logging

Usage::

    from tools.gui import TrainingGUI
    from simplexnet.core.SMN_RL import SMN_RL
    import gymnasium as gym

    env = gym.make('CartPole-v1')
    smn_rl = SMN_RL(env=env, algorithm='dqn', n=2, m=4, n_in=4, n_out=2)
    gui = TrainingGUI(smn_rl)
    gui.run()
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any

import numpy as np

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QDoubleSpinBox, QSpinBox,
    QGroupBox, QFormLayout, QTabWidget, QTextEdit, QFrame, QFileDialog,
    QMessageBox, QSplitter, QGridLayout, QRadioButton, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer
from PySide6.QtGui import QFont, QColor, QPalette

# matplotlib with Qt backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Local imports
try:
    from ..core.SMN_RL import SMN_RL
    from .checkpoint import CheckpointManager
except ImportError:
    import os
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.SMN_RL import SMN_RL
    from checkpoint import CheckpointManager


# =============================================================================
# Training Thread - runs training in background
# =============================================================================

class TrainingSignal(QObject):
    """Signal emitter for training thread."""
    update_signal = Signal(dict)  # episode, reward, loss, epsilon/entropy
    finished_signal = Signal()
    error_signal = Signal(str)


class TrainingThread(QThread):
    """Background thread for training."""

    def __init__(
        self,
        smn_rl: SMN_RL,
        num_episodes: int = 500,
        max_steps: int = 500,
        update_target_every: int = 100,
        checkpoint_every: int = 50,
    ):
        super().__init__()
        self.smn_rl = smn_rl
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.update_target_every = update_target_every
        self.checkpoint_every = checkpoint_every
        self.signal_emitter = TrainingSignal()
        self._paused = False
        self._stopped = False

    @property
    def update_signal(self):
        return self.signal_emitter.update_signal

    @property
    def finished_signal(self):
        return self.signal_emitter.finished_signal

    @property
    def error_signal(self):
        return self.signal_emitter.error_signal

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stopped = True

    def run(self):
        """Training loop with pause/resume support."""
        try:
            start_episode = 0
            checkpoint = self.smn_rl.checkpoint_mgr.find_latest_checkpoint()
            if checkpoint is not None:
                ckpt = self.smn_rl.checkpoint_mgr.load_latest()
                if ckpt is not None:
                    start_episode = ckpt.get('episode', 0)
                    if self.smn_rl.algorithm == 'dqn':
                        dqn_checkpoint = {
                            'q_network': ckpt['state_dict'],
                            'target_network': ckpt['state_dict'],
                            'optimizer': ckpt.get('optimizer_state'),
                            'epsilon': self.smn_rl.agent.epsilon,
                        }
                        self.smn_rl.agent.load_checkpoint_from_dict(dqn_checkpoint)
                    else:
                        reinforce_checkpoint = {
                            'policy_network': ckpt['state_dict'],
                            'optimizer': ckpt.get('optimizer_state'),
                        }
                        self.smn_rl.agent.load_checkpoint_from_dict(reinforce_checkpoint)

            if self.smn_rl._mdp is None:
                if self.smn_rl._env is not None:
                    self.smn_rl._mdp = self.smn_rl._env  # type: ignore
                else:
                    raise ValueError("No env provided")

            if self.smn_rl._collector is None:
                from ..rl.collector import TrajectoryCollector
                self.smn_rl._collector = TrajectoryCollector(self.smn_rl._mdp)  # type: ignore

            rewards_history = []
            losses_history = []

            for episode in range(start_episode, self.num_episodes):
                # Check for stop
                if self._stopped:
                    break

                # Handle pause
                while self._paused:
                    if self._stopped:
                        break
                    self.msleep(100)

                if self._stopped:
                    break

                # Collect trajectory
                trajectory = self.smn_rl._collector.collect_episode(
                    self.smn_rl.agent, max_steps=self.max_steps, training=True
                )

                # Train from trajectory
                loss = self.smn_rl.agent.train(trajectory)

                # DQN-specific: Update target network
                if self.smn_rl.algorithm == 'dqn':
                    if (episode + 1) % self.update_target_every == 0:
                        self.smn_rl.agent.update_target_network()
                    self.smn_rl.agent.decay_epsilon()

                # Record history
                episode_reward = sum(trajectory.rewards)
                avg_loss = loss if loss is not None else 0.0
                rewards_history.append(episode_reward)
                losses_history.append(avg_loss)

                self.smn_rl.training_history.append({
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'loss': avg_loss,
                })

                # Save checkpoint
                if (episode + 1) % self.checkpoint_every == 0:
                    self.smn_rl.checkpoint_mgr.save_checkpoint(
                        module=self.smn_rl.network,
                        optimizer=self.smn_rl.agent.optimizer,
                        episode=episode + 1,
                        reward=episode_reward,
                        loss=avg_loss,
                        metadata={
                            'rewards_history': rewards_history,
                            'losses_history': losses_history,
                            'algorithm': self.smn_rl.algorithm,
                        }
                    )

                # Emit update signal
                extra = {}
                if self.smn_rl.algorithm == 'dqn':
                    extra['epsilon'] = self.smn_rl.agent.epsilon
                else:
                    extra['entropy_coef'] = self.smn_rl.agent.entropy_coef

                self.update_signal.emit({
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'loss': avg_loss,
                    'rewards_history': rewards_history,
                    'losses_history': losses_history,
                    **extra,
                })

            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(str(e))


# =============================================================================
# Matplotlib Canvas Widget
# =============================================================================

class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for embedding in Qt layout."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


# =============================================================================
# Control Panel - Left Side Input
# =============================================================================

class ControlPanel(QWidget):
    """Left panel with all user input controls."""

    config_changed = Signal(dict)
    train_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, smn_rl: SMN_RL):
        super().__init__()
        self.smn_rl = smn_rl
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QFormLayout()
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            'CartPole-v1',
            'MountainCar-v0',
            'Acrobot-v1',
        ])
        env_layout.addRow("Gymnasium Env:", self.env_combo)
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)

        # SMN parameters
        smn_group = QGroupBox("SMN Parameters")
        smn_layout = QFormLayout()

        self.n_spin = QSpinBox()
        self.n_spin.setRange(2, 10)
        self.n_spin.setValue(self.smn_rl.n)
        smn_layout.addRow("n (dimension):", self.n_spin)

        self.m_spin = QSpinBox()
        self.m_spin.setRange(2, 20)
        self.m_spin.setValue(self.smn_rl.m)
        smn_layout.addRow("m (resolution):", self.m_spin)
        smn_group.setLayout(smn_layout)
        layout.addWidget(smn_group)

        # Algorithm selection
        algo_group = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout()
        self.algo_group_btn = QButtonGroup()
        self.dqn_radio = QRadioButton("DQN (value-based, discrete)")
        self.reinforce_radio = QRadioButton("REINFORCE (policy gradient)")
        self.algo_group_btn.addButton(self.dqn_radio)
        self.algo_group_btn.addButton(self.reinforce_radio)
        if self.smn_rl.algorithm == 'dqn':
            self.dqn_radio.setChecked(True)
        else:
            self.reinforce_radio.setChecked(True)
        algo_layout.addWidget(self.dqn_radio)
        algo_layout.addWidget(self.reinforce_radio)
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # Hyperparameters
        hyper_group = QGroupBox("Hyperparameters")
        hyper_layout = QFormLayout()

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setValue(1e-3)
        self.lr_spin.setDecimals(6)
        hyper_layout.addRow("Learning rate:", self.lr_spin)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0, 1.0)
        self.gamma_spin.setValue(0.99)
        self.gamma_spin.setDecimals(3)
        hyper_layout.addRow("Discount (gamma):", self.gamma_spin)

        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 1.0)
        self.epsilon_spin.setValue(1.0)
        self.epsilon_spin.setDecimals(3)
        hyper_layout.addRow("Epsilon:", self.epsilon_spin)

        self.epsilon_decay_spin = QDoubleSpinBox()
        self.epsilon_decay_spin.setRange(0.9, 1.0)
        self.epsilon_decay_spin.setValue(0.995)
        self.epsilon_decay_spin.setDecimals(4)
        hyper_layout.addRow("Epsilon decay:", self.epsilon_decay_spin)

        self.entropy_spin = QDoubleSpinBox()
        self.entropy_spin.setRange(0.0, 1.0)
        self.entropy_spin.setValue(0.0)
        self.entropy_spin.setDecimals(3)
        hyper_layout.addRow("Entropy coef:", self.entropy_spin)

        hyper_group.setLayout(hyper_layout)
        layout.addWidget(hyper_group)

        # Training settings
        train_group = QGroupBox("Training Settings")
        train_layout = QFormLayout()

        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(10, 10000)
        self.episodes_spin.setValue(500)
        train_layout.addRow("Episodes:", self.episodes_spin)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(50, 2000)
        self.max_steps_spin.setValue(500)
        train_layout.addRow("Max steps/ep:", self.max_steps_spin)

        self.checkpoint_spin = QSpinBox()
        self.checkpoint_spin.setRange(10, 500)
        self.checkpoint_spin.setValue(50)
        train_layout.addRow("Checkpoint every:", self.checkpoint_spin)

        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.train_btn = QPushButton("▶ Start Training")
        self.train_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.train_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)

        layout.addStretch()
        self.setLayout(layout)

        # Connect signals
        self.train_btn.clicked.connect(self.train_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)

    def get_config(self) -> dict:
        """Get current configuration from controls."""
        return {
            'env': self.env_combo.currentText(),
            'n': self.n_spin.value(),
            'm': self.m_spin.value(),
            'algorithm': 'dqn' if self.dqn_radio.isChecked() else 'reinforce',
            'lr': self.lr_spin.value(),
            'gamma': self.gamma_spin.value(),
            'epsilon': self.epsilon_spin.value(),
            'epsilon_decay': self.epsilon_decay_spin.value(),
            'entropy_coef': self.entropy_spin.value(),
            'num_episodes': self.episodes_spin.value(),
            'max_steps': self.max_steps_spin.value(),
            'checkpoint_every': self.checkpoint_spin.value(),
        }

    def set_training_mode(self, training: bool):
        """Enable/disable controls during training."""
        self.train_btn.setEnabled(not training)
        self.stop_btn.setEnabled(training)
        self.env_combo.setEnabled(not training)
        self.n_spin.setEnabled(not training)
        self.m_spin.setEnabled(not training)
        self.dqn_radio.setEnabled(not training)
        self.reinforce_radio.setEnabled(not training)


# =============================================================================
# Output Panel - Right Side with Tabs
# =============================================================================

class OutputPanel(QWidget):
    """Right panel with tabbed visualization output."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Tab 1: Training Curves
        self.training_curves_widget = TrainingCurvesWidget()
        self.tabs.addTab(self.training_curves_widget, "📈 Training Curves")

        # Tab 2: Reward Analysis
        self.reward_analysis_widget = RewardAnalysisWidget()
        self.tabs.addTab(self.reward_analysis_widget, "📊 Reward Analysis")

        # Tab 3: Network Info
        self.network_widget = NetworkInfoWidget()
        self.tabs.addTab(self.network_widget, "🧠 Network")

        # Tab 4: Checkpoint Manager
        self.checkpoint_widget = CheckpointManagerWidget()
        self.tabs.addTab(self.checkpoint_widget, "💾 Checkpoint")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def update_training_curves(self, data: dict):
        self.training_curves_widget.update_data(data)

    def update_reward_analysis(self, data: dict):
        self.reward_analysis_widget.update_data(data)

    def update_network_info(self, smn_rl: SMN_RL):
        self.network_widget.update_info(smn_rl)

    def update_checkpoints(self, checkpoint_dir: Path):
        self.checkpoint_widget.update_checkpoints(checkpoint_dir)


class TrainingCurvesWidget(QWidget):
    """Training curves visualization."""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.rewards = []
        self.losses = []
        self.epsilons = []

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.canvas = MplCanvas(self, width=6, height=5, dpi=100)
        self.ax1 = self.canvas.fig.add_subplot(311)
        self.ax2 = self.canvas.fig.add_subplot(312)
        self.ax3 = self.canvas.fig.add_subplot(313)
        layout.addWidget(self.canvas)

    def update_data(self, data: dict):
        """Update curves with new data."""
        self.rewards = data.get('rewards_history', self.rewards)
        self.losses = data.get('losses_history', self.losses)
        self.epsilons = data.get('epsilon', self.epsilons)
        if not isinstance(self.epsilons, list):
            # Single value, append
            if len(self.epsilons) == 0:
                self.epsilons = [self.epsilons] if not isinstance(self.epsilons, list) else self.epsilons + [data.get('epsilon', 1.0)]
            else:
                self.epsilons = list(self.epsilons) + [data.get('epsilon', 1.0)]

        # Clear and redraw
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        episodes = list(range(1, len(self.rewards) + 1))

        # Reward
        self.ax1.plot(episodes, self.rewards, 'b-', linewidth=1, label='Reward')
        if len(self.rewards) >= 20:
            ma = np.convolve(self.rewards, np.ones(20)/20, mode='valid')
            self.ax1.plot(range(20, len(self.rewards) + 1), ma, 'r-', linewidth=2, label='20-ep avg')
        self.ax1.set_ylabel('Reward')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)

        # Loss
        self.ax2.plot(episodes, self.losses, 'g-', linewidth=1, label='Loss')
        self.ax2.set_ylabel('Loss')
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)

        # Epsilon (DQN only)
        if len(self.epsilons) > 0 and self.epsilons[0] is not None:
            self.ax3.plot(range(1, len(self.epsilons) + 1), self.epsilons, 'orange', linewidth=1, label='Epsilon')
            self.ax3.set_ylabel('Epsilon')
            self.ax3.set_xlabel('Episode')
            self.ax3.legend(loc='upper right')
            self.ax3.grid(True, alpha=0.3)
        else:
            self.ax3.text(0.5, 0.5, 'REINFORCE mode', ha='center', va='center', transform=self.ax3.transAxes)

        self.canvas.fig.tight_layout()
        self.canvas.draw()


class RewardAnalysisWidget(QWidget):
    """Reward analysis visualization."""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.rewards = []

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.canvas = MplCanvas(self, width=6, height=5, dpi=100)
        self.ax1 = self.canvas.fig.add_subplot(121)
        self.ax2 = self.canvas.fig.add_subplot(122)
        layout.addWidget(self.canvas)

        # Stats label
        self.stats_label = QLabel("")
        layout.addWidget(self.stats_label)

    def update_data(self, data: dict):
        """Update analysis with new data."""
        self.rewards = data.get('rewards_history', self.rewards)

        if len(self.rewards) == 0:
            return

        self.ax1.clear()
        self.ax2.clear()

        # Left: Moving average
        self.ax1.plot(self.rewards, alpha=0.5, label='Raw')
        if len(self.rewards) >= 20:
            ma = np.convolve(self.rewards, np.ones(20)/20, mode='valid')
            self.ax1.plot(range(20, len(self.rewards)), ma, 'r-', linewidth=2, label='20-ep avg')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # Right: Distribution
        self.ax2.hist(self.rewards, bins=20, alpha=0.7, edgecolor='black')
        self.ax2.axvline(np.mean(self.rewards), color='r', linestyle='--', label=f'Mean: {np.mean(self.rewards):.1f}')
        self.ax2.axvline(np.max(self.rewards), color='g', linestyle='--', label=f'Best: {np.max(self.rewards):.1f}')
        self.ax2.set_xlabel('Reward')
        self.ax2.set_ylabel('Frequency')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)

        # Update stats
        recent = self.rewards[-100:] if len(self.rewards) >= 100 else self.rewards
        self.stats_label.setText(
            f"Best: {np.max(self.rewards):.1f} | "
            f"Mean: {np.mean(self.rewards):.1f} | "
            f"Recent 100 avg: {np.mean(recent):.1f} | "
            f"Std: {np.std(self.rewards):.1f}"
        )

        self.canvas.fig.tight_layout()
        self.canvas.draw()


class NetworkInfoWidget(QWidget):
    """Network information display."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)

        self.setLayout(layout)

    def update_info(self, smn_rl: SMN_RL):
        """Update network information."""
        info = f"""
=== SMN Network Info ===

Architecture: {smn_rl.network.arch_str}
Parameters: {smn_rl.network.param_count:,}

Configuration:
  n (simplex dim): {smn_rl.n}
  m (resolution): {smn_rl.m}
  n_in (input): {smn_rl.n_in}
  n_out (output): {smn_rl.n_out}

Algorithm: {smn_rl.algorithm}
  gamma: {smn_rl.agent.gamma}
  lr: {smn_rl.agent.optimizer.param_groups[0]['lr']:.6f}
"""
        if smn_rl.algorithm == 'dqn':
            info += f"""
DQN Settings:
  epsilon: {smn_rl.agent.epsilon:.4f}
  epsilon_decay: {smn_rl.agent.epsilon_decay}
  epsilon_min: {smn_rl.agent.epsilon_min}
"""
        else:
            info += f"""
REINFORCE Settings:
  entropy_coef: {smn_rl.agent.entropy_coef}
  action_type: {smn_rl.agent.action_type}
"""
        self.info_text.setText(info)


class CheckpointManagerWidget(QWidget):
    """Checkpoint management widget."""

    load_checkpoint = Signal(Path)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.checkpoint_dir: Optional[Path] = None

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Checkpoint list
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['File', 'Episode', 'Reward', 'Loss', 'Time'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()
        self.load_btn = QPushButton("📂 Load Selected")
        self.refresh_btn = QPushButton("🔄 Refresh")
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.refresh_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.load_btn.clicked.connect(self.load_selected)
        self.refresh_btn.clicked.connect(self.refresh)

    def update_checkpoints(self, checkpoint_dir: Path):
        """Update checkpoint list."""
        self.checkpoint_dir = checkpoint_dir
        self.refresh()

    def refresh(self):
        """Refresh checkpoint list."""
        self.table.setRowCount(0)
        if self.checkpoint_dir is None:
            return

        ckpt_mgr = CheckpointManager(self.checkpoint_dir)
        checkpoints = ckpt_mgr.find_all_checkpoints()

        for ckpt_path in checkpoints:
            try:
                ckpt = torch.load(ckpt_path, weights_only=False)
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(ckpt_path.name))
                self.table.setItem(row, 1, QTableWidgetItem(str(ckpt.get('episode', 'N/A'))))
                self.table.setItem(row, 2, QTableWidgetItem(f"{ckpt.get('reward', 0):.1f}"))
                self.table.setItem(row, 3, QTableWidgetItem(f"{ckpt.get('loss', 'N/A')}"))
                self.table.setItem(row, 4, QTableWidgetItem(ckpt.get('timestamp', 'N/A')))
            except Exception:
                pass

    def load_selected(self):
        """Load selected checkpoint."""
        row = self.table.currentRow()
        if row < 0 or self.checkpoint_dir is None:
            return

        filename = self.table.item(row, 0).text()
        self.load_checkpoint.emit(self.checkpoint_dir / filename)


# Need torch for checkpoint loading
try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# Log Panel - Bottom
# =============================================================================

class LogPanel(QTextEdit):
    """Terminal-style log panel."""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumHeight(150)
        self.setFont(QFont("Monospace", 10))
        self.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")

    def log(self, message: str):
        """Append log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.append(f"[{timestamp}] {message}")
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# =============================================================================
# Overwrite Warning Dialog
# =============================================================================

class OverwriteWarningDialog(QDialog):
    """Dialog for confirming checkpoint overwrite with diff display."""

    def __init__(self, current_info: dict, new_info: dict, parent=None):
        super().__init__(parent)
        self.current_info = current_info
        self.new_info = new_info
        self.result_button = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("⚠️  Confirm Overwrite")
        self.setMinimumWidth(600)

        layout = QVBoxLayout(self)

        # Warning header
        header = QLabel("⚠️  You are about to overwrite the current network")
        header.setStyleSheet("color: #f44336; font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

        # Diff table
        diff_text = self.build_diff_text()
        diff_label = QLabel(diff_text)
        diff_label.setStyleSheet("""
            QLabel {
                font-family: Monospace;
                font-size: 12px;
                background-color: #f5f5f5;
                padding: 10px;
                border: 1px solid #ddd;
            }
        """)
        layout.addWidget(diff_label)

        # Warning message
        warn = QLabel("⚠️  Overwriting will discard the current network state. Consider saving first.")
        warn.setStyleSheet("color: #ff9800;")
        layout.addWidget(warn)

        # Buttons
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾 Save Current First")
        self.cancel_btn = QPushButton("❌ Cancel")
        self.confirm_btn = QPushButton("✅ Confirm Overwrite")
        self.confirm_btn.setStyleSheet("background-color: #f44336; color: white;")

        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.confirm_btn)
        layout.addLayout(button_layout)

        # Connect
        self.save_btn.clicked.connect(self.on_save)
        self.cancel_btn.clicked.connect(self.reject)
        self.confirm_btn.clicked.connect(self.on_confirm)

    def build_diff_text(self) -> str:
        """Build diff text comparing current vs new checkpoint."""
        lines = ["<b>Comparison: Current → New</b>", ""]

        def fmt_val(v):
            return str(v) if v is not None else "N/A"

        lines.append(f"Episode:      {fmt_val(self.current_info.get('episode'))} → {fmt_val(self.new_info.get('episode'))}")
        lines.append(f"Best Reward:  {fmt_val(self.current_info.get('reward'))} → {fmt_val(self.new_info.get('reward'))}")
        lines.append(f"Loss:         {fmt_val(self.current_info.get('loss'))} → {fmt_val(self.new_info.get('loss'))}")

        curr_cfg = self.current_info.get('config', {})
        new_cfg = self.new_info.get('config', {})
        lines.append(f"Network (n):  {fmt_val(curr_cfg.get('n'))} → {fmt_val(new_cfg.get('n'))}")
        lines.append(f"Network (m):  {fmt_val(curr_cfg.get('m'))} → {fmt_val(new_cfg.get('m'))}")
        lines.append(f"Network (in): {fmt_val(curr_cfg.get('n_in'))} → {fmt_val(new_cfg.get('n_in'))}")
        lines.append(f"Network (out):{fmt_val(curr_cfg.get('n_out'))} → {fmt_val(new_cfg.get('n_out'))}")
        lines.append(f"Timestamp:    {fmt_val(self.current_info.get('timestamp'))} → {fmt_val(self.new_info.get('timestamp'))}")

        return "<br>".join(lines)

    def on_save(self):
        self.result_button = self.save_btn
        self.accept()

    def on_confirm(self):
        self.result_button = self.confirm_btn
        self.accept()

    def get_result(self) -> str:
        """Return 'save', 'confirm', or 'cancel'."""
        if self.result_button == self.save_btn:
            return 'save'
        elif self.result_button == self.confirm_btn:
            return 'confirm'
        return 'cancel'


# =============================================================================
# Main Window
# =============================================================================

class TrainingGUI(QMainWindow):
    """Main training GUI window."""

    def __init__(self, smn_rl: SMN_RL):
        super().__init__()
        self.smn_rl = smn_rl
        self.thread: Optional[TrainingThread] = None
        self.setWindowTitle("SMN Training GUI")
        self.setMinimumSize(1200, 800)
        self.init_ui()

    def init_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Splitter for left/right panels
        splitter = QSplitter(Qt.Horizontal)

        # Left: Control panel
        self.control_panel = ControlPanel(self.smn_rl)
        splitter.addWidget(self.control_panel)

        # Right: Output panel with tabs
        self.output_panel = OutputPanel()
        splitter.addWidget(self.output_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

        # Bottom: Log panel
        self.log_panel = LogPanel()
        main_layout.addWidget(self.log_panel)

        # Connect signals
        self.control_panel.train_clicked.connect(self.start_training)
        self.control_panel.stop_clicked.connect(self.stop_training)
        self.output_panel.checkpoint_widget.load_checkpoint.connect(self.load_checkpoint)

        # Update network info
        self.output_panel.update_network_info(self.smn_rl)

        self.log_panel.log("GUI initialized. Ready to train.")

    def start_training(self):
        """Start training thread."""
        config = self.control_panel.get_config()
        self.log_panel.log(f"Starting training: {config['algorithm']}, n={config['n']}, m={config['m']}")

        self.thread = TrainingThread(
            self.smn_rl,
            num_episodes=config['num_episodes'],
            max_steps=config['max_steps'],
            checkpoint_every=config['checkpoint_every'],
        )

        self.thread.update_signal.connect(self.on_training_update)
        self.thread.finished_signal.connect(self.on_training_finished)
        self.thread.error_signal.connect(self.on_training_error)

        self.thread.start()
        self.control_panel.set_training_mode(True)
        self.log_panel.log("Training started...")

    def stop_training(self):
        """Stop training thread."""
        if self.thread:
            self.thread.stop()
            self.control_panel.set_training_mode(False)
            self.log_panel.log("Training stopped by user.")

    def on_training_update(self, data: dict):
        """Handle training update."""
        episode = data.get('episode', 0)
        reward = data.get('reward', 0)
        loss = data.get('loss', 0)

        # Update output panels
        self.output_panel.update_training_curves(data)
        self.output_panel.update_reward_analysis(data)
        self.output_panel.update_checkpoints(self.smn_rl.checkpoint_mgr.checkpoint_dir)

        # Log every 10 episodes
        if episode % 10 == 0:
            extra = ""
            if 'epsilon' in data:
                extra = f" | ε={data['epsilon']:.3f}"
            self.log_panel.log(f"Episode {episode}/{self.control_panel.episodes_spin.value()} | Reward: {reward:.1f} | Loss: {loss:.4f}{extra}")

    def on_training_finished(self):
        """Handle training completion."""
        self.control_panel.set_training_mode(False)
        self.log_panel.log("Training completed!")
        self.thread = None

    def on_training_error(self, error: str):
        """Handle training error."""
        self.control_panel.set_training_mode(False)
        self.log_panel.log(f"ERROR: {error}")
        self.thread = None

    def load_checkpoint(self, path: Path):
        """Load checkpoint with overwrite warning."""
        # Get current state
        current_info = {
            'episode': self.smn_rl.training_history[-1]['episode'] if self.smn_rl.training_history else 0,
            'reward': self.smn_rl.training_history[-1]['reward'] if self.smn_rl.training_history else 0,
            'loss': self.smn_rl.training_history[-1]['loss'] if self.smn_rl.training_history else None,
            'config': {
                'n': self.smn_rl.n,
                'm': self.smn_rl.m,
                'n_in': self.smn_rl.n_in,
                'n_out': self.smn_rl.n_out,
            },
            'timestamp': datetime.now().isoformat(),
        }

        # Load new checkpoint info
        if torch is None:
            self.log_panel.log("ERROR: torch not available for checkpoint loading")
            return

        try:
            new_ckpt = torch.load(path, weights_only=False)
            new_info = {
                'episode': new_ckpt.get('episode', 0),
                'reward': new_ckpt.get('reward', 0),
                'loss': new_ckpt.get('loss'),
                'config': new_ckpt.get('config', {}),
                'timestamp': new_ckpt.get('timestamp', 'N/A'),
            }
        except Exception as e:
            self.log_panel.log(f"ERROR loading checkpoint: {e}")
            return

        # Show warning dialog if current state is non-trivial
        if current_info['episode'] > 0:
            dialog = OverwriteWarningDialog(current_info, new_info, self)
            dialog.exec()
            result = dialog.get_result()

            if result == 'cancel':
                self.log_panel.log("Checkpoint load cancelled.")
                return
            elif result == 'save':
                self.save_current_checkpoint()

        # Load checkpoint
        if self.smn_rl.algorithm == 'dqn':
            dqn_checkpoint = {
                'q_network': new_ckpt['state_dict'],
                'target_network': new_ckpt['state_dict'],
                'optimizer': new_ckpt.get('optimizer_state'),
                'epsilon': self.smn_rl.agent.epsilon,
            }
            self.smn_rl.agent.load_checkpoint_from_dict(dqn_checkpoint)
        else:
            reinforce_checkpoint = {
                'policy_network': new_ckpt['state_dict'],
                'optimizer': new_ckpt.get('optimizer_state'),
            }
            self.smn_rl.agent.load_checkpoint_from_dict(reinforce_checkpoint)

        self.log_panel.log(f"Loaded checkpoint: {path.name} (episode {new_info['episode']})")
        self.output_panel.update_network_info(self.smn_rl)

    def save_current_checkpoint(self):
        """Save current state."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Checkpoint",
            str(self.smn_rl.checkpoint_mgr.checkpoint_dir / "checkpoint.pt"),
            "PyTorch Checkpoint (*.pt)"
        )
        if path:
            self.smn_rl.save_checkpoint(path)
            self.log_panel.log(f"Saved checkpoint: {path}")

    def run(self):
        """Show window and start event loop."""
        self.show()


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Launch GUI standalone."""
    import gymnasium as gym

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Create default environment and SMN_RL
    env = gym.make('CartPole-v1')
    smn_rl = SMN_RL(
        env=env,
        algorithm='dqn',
        n=2, m=4,
        n_in=4, n_out=2,
        gamma=0.99,
        lr=1e-3,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    gui = TrainingGUI(smn_rl)
    gui.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
