import os
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from agent import Agent
from artifact_manager import ArtifactManager

class BaseCallback(keras.callbacks.Callback):
    """Base class for callbacks."""

    def __init__(self):
        """Initialize the base callback."""

        super().__init__()
        self.agent = None
        self.env = None
        self.artifact_manager = None

    def set_agent(self, agent: Agent):
        """Set the agent for the callback.

        Args:
            agent: The agent to set.
        """

        self.agent = agent

    def set_environment(self, env: Any):
        """Set the environment for the callback.

        Args:
            env: The environment to set.
        """

        self.env = env

    def set_artifact_manager(self, artifact_manager: ArtifactManager):
        """Set the artifact manager for the callback.

        Args:
            artifact_manager: The artifact manager to set.
        """

        self.artifact_manager = artifact_manager

class DefaultCallback(BaseCallback):
    """Default callback that prints out statistics and saves the trained
    model."""

    def __init__(self):
        """Initialize the default callback."""

        super().__init__()
        self.episode_returns = []
        self._get_model_path = None

    def _register_model_artifact(self):
        """Register the model artifact with the artifact manager."""

        if self.artifact_manager is None or self.agent is None:
            return

        # Check for existing model files upfront.
        try:
            model_filenames = self.agent.get_save_file_names()
        except NotImplementedError:
            # Agent does not implement save, skip registration.
            return

        try:
            self._get_model_path = self.artifact_manager.register_artifact(
                artifact_identifier="models",
                artifact_description="DefaultCallback_models",
                file_name_generator=lambda *args: "",          # Return empty string for directory path.
                check_conflicts=False                          # We do custom conflict checking.
            )

            # Custom conflict check for model files
            model_dir = self._get_model_path(ensure_dir=False)
            for filename in model_filenames.values():
                file_path = os.path.join(model_dir, filename)
                if os.path.exists(file_path):
                    raise FileExistsError(
                        f"\nError: Saved model artifacts already exist at '{model_dir}'\n"
                        "To prevent overwriting existing results, training is terminated.\n"
                        "Please either:\n"
                        "  - Use a different seed (--seed).\n"
                        "  - Use a different run_name in the config.\n"
                        "  - Remove or backup the existing artifacts."
                    )
        except FileExistsError as e:
            raise e

    def on_train_begin(self, logs: dict[str, Any] | None = None):
        """Register model artifact at the beginning of training."""

        self._register_model_artifact()

    def on_epoch_end(self, epoch, logs: dict[str, Any] | None = None):
        epochs_per_episode = logs.get("epochs_per_episode", 1)
        avg_losses = logs.get("avg_losses", {})
        with np.printoptions(precision=4, suppress=True):
            print(f"Epoch {epoch+1:02}/{epochs_per_episode:02}: " + ", ".join([f"{name}={value:.4f}" for name, value in avg_losses.items()]))

    def on_episode_end(self, episode, logs: dict[str, Any] | None = None):
        N_episodes = logs.get("N_episodes", 1)
        N_total_steps = logs.get("N_total_steps", 0)
        init_states = logs.get("init_state", None)
        returns = logs.get("returns", None)
        estimated_value = logs.get("estimated_value", None)

        if estimated_value is not None:
            self.episode_returns.append(float(estimated_value))

        with np.printoptions(precision=2, suppress=True):
            print(f"Episode: {episode+1:02}/{N_episodes:02}, Total Steps: {N_total_steps}, Init states: {init_states}, Returns: {returns}")

        print(f"Episode {episode+1:02}/{N_episodes:02}: Estimated true value of initial state: {estimated_value:.2f}")

    def on_train_end(self, logs: dict[str, Any] | None = None):
        """Save the trained model at the end of training."""

        if self._get_model_path is None:
            return

        try:
            self.agent.save(self._get_model_path())
        except NotImplementedError:
            pass

class ThresholdEarlyStop(BaseCallback):
    """Early stop training if running mean return is above a threshold."""

    def __init__(
        self,
        target_return: float,
        patience: int | None = None
    ):
        """Initialize the early stopping callback.

        Args:
            target_return: The target return value to trigger early stopping.
            patience: The number of previous episodes used for mean return
            calculation. If set to None, all episodes are considered. Defaults to None.
        """

        super().__init__()
        self.target_return = target_return
        self.patience = patience
        self.episode_returns = []
        self.early_stop = False

    def on_episode_end(self, episode, logs: dict[str, Any] | None = None):
        if logs is None:
            return

        estimated_value = logs.get("estimated_value")
        if estimated_value is not None:
            self.episode_returns.append(float(estimated_value))

        # Only consider stopping after minimum episodes.
        if episode < self.patience:
            return

        # Calculate running average.
        if self.patience is None:
            recent_returns = self.episode_returns
        else:
            recent_returns = self.episode_returns[-self.patience:]
        running_avg = np.mean(recent_returns) if recent_returns else -float("inf")

        # Check if we have reached target return.
        if self.target_return is not None and running_avg >= self.target_return:
            self.stop_training = True
            print(f"\nTarget return {self.target_return:.2f} reached at episode {episode + 1}")
            print(f"Running average return: {running_avg:.2f}")
