import numpy as np
import tensorflow as tf

from agent import Agent
from env import VahvistajaEnvironment

class BaseCallback(tf.keras.callbacks.Callback):
    """Base class for callbacks."""

    def __init__(self):
        """Initialize the base callback."""

        super().__init__()
        self.agent = None
        self.env = None

    def set_agent(self, agent: Agent):
        """Set the agent for the callback.

        Args:
            agent: The agent to set.
        """

        self.agent = agent

    def set_environment(self, env: VahvistajaEnvironment):
        """Set the environment for the callback.

        Args:
            env: The environment to set.
        """

        self.env = env

class DefaultCallback(BaseCallback):
    """Default callback that merely prints out some statistics on episode,
    epoch, and batch ends."""

    def on_epoch_end(self, epoch, logs=None):
        epochs_per_episode = logs.get("epochs_per_episode", 1)
        avg_losses = logs.get("avg_losses", {})
        with np.printoptions(precision=4, suppress=True):
            print(f"Epoch {epoch+1:02}/{epochs_per_episode:02}: " + ", ".join([f"{name}={value:.4f}" for name, value in avg_losses.items()]))

    def on_episode_end(self, episode, logs=None):
        N_episodes = logs.get("N_episodes", 1)
        N_total_steps = logs.get("N_total_steps", 0)
        init_states = logs.get("init_state", None)
        returns = logs.get("returns", None)
        estimated_value = logs.get("estimated_value", None)
        with np.printoptions(precision=2, suppress=True):
            print(f"Episode: {episode+1:02}/{N_episodes:02}, Total Steps: {N_total_steps}, Init states: {init_states}, Returns: {returns}")

        print(f"Episode {episode+1:02}/{N_episodes:02}: Estimated true value of initial state: {estimated_value:.2f}")

class ThresholdEarlyStop(BaseCallback):
    """Early stop training if running return is above a threshold."""

    def __init__(
        self,
        target_return: float=3000,
        patience: int=20
    ):
        super().__init__()
        self.target_return = target_return
        self.patience = patience
        self.episode_returns = []
        self.early_stop = False

    def on_episode_end(self, episode, logs=None):
        if logs is None:
            return

        estimated_value = logs.get("estimated_value")
        if estimated_value is not None:
            self.episode_returns.append(float(estimated_value))

        # Only consider stopping after minimum episodes.
        if episode < self.patience:
            return

        # Calculate running average.
        recent_returns = self.episode_returns[-self.patience:]
        running_avg = np.mean(recent_returns) if recent_returns else 0

        # Check if we have reached target return.
        if self.target_return is not None and running_avg >= self.target_return:
            self.stop_training = True
            print(f"\nTarget return {self.target_return:.2f} reached at episode {episode + 1}")
            print(f"Running average return: {running_avg:.2f}")
            return
