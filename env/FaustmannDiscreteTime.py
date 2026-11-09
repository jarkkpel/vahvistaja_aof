from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import register

import numpy as np

from .base import VahvistajaEnvironment
from callbacks import BaseCallback

class FaustmannEnv(VahvistajaEnvironment):
    """A Faustmann discrete time forest rotation environment."""

    def __init__(self, env_idx: int | None = None, discount_factor: float = 1.0):
        """Initialize the Faustmann discrete time forest rotation environment.

        Args:
            env_idx: An optional index identifying the environment instance.
            discount_factor: The discount factor used in the environment. Defaults to 1.0.
        """

        super().__init__(env_idx, discount_factor)

        self.Tmin = 1
        self.Tmax = 100
        self.observation_space = gym.spaces.Box(low=self.Tmin, high=self.Tmax, shape=(1,), dtype=np.int32)

        self.action_space = gym.spaces.Discrete(2)

        self.init_state = np.array([1], dtype=np.int32)
        self.state = self.init_state

        self.max_episode_steps = 20000  # Maximum steps per episode
        self.current_step = 0

    def step(self, action: np.ndarray) -> tuple:
        """Take an action in the environment.

        Args:
            action: The action to take.
        Raises:
            ValueError: If the action shape is incorrect.
        Returns:
            A tuple of (next_state, reward, terminated, truncated, info).
        """

        if len(action.shape) == 2:
            if action.shape != (1, 1):
                raise ValueError(f"Incorrect action shape {action.shape}, expected (1, ) or (1, 1).")
            action = action.flatten()
        elif len(action.shape) > 2:
            raise ValueError(f"Incorrect action shape {action.shape}, expected (1, ) or (1, 1).")

        self.current_step += 1

        harvestValue = 50.0 * 500.0 / (1.0 + np.exp(-(self.state[0] - 50.0) / 10.0)) - 1500.0

        # growing_reward = 0.0

        newstate = min((1 - action[0]) * self.state[0] + 1, self.Tmax)
        # payoff = harvestValue if action == 1 else growing_reward
        payoff = harvestValue if action == 1 else 0.0

        # truncation is applied, although not perhaps necessary
        truncated = ( self.current_step >= self.max_episode_steps )
        # Optionally terminate if state reaches Tmax
        # terminated = terminated or (self.state >= self.Tmax)

        # problem is infinite: there is no termination
        terminated = False

        self.state = np.array([newstate], dtype=np.int32)
        return self.state, np.float32(payoff), terminated, truncated, {}

    def get_random_state(self) -> np.ndarray:
        """Return a random state sampled from the observation space.

        Returns:
            np.ndarray: A random state.
        """

        return self.observation_space.sample()

    def reset(self, seed: int | None = None, options: dict | list | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment to a state.

        Args:
            seed: Optional seed for random number generator.
            options: Optional dictionary or list of dictionaries with reset options.
        Raises:
            ValueError: If the reset state does not belong to the observation space.
        Returns:
            Tuple of (observation, info).
        """

        super().reset(seed=seed)

        if options is not None:
            if self.env_idx is not None:
                new_state = options[self.env_idx]["reset_state"]
            else:
                new_state = options["reset_state"]
        else:
            new_state = self.init_state

        # Check that the state is of appropriate type and within bounds.
        if not self.observation_space.contains(new_state):
            raise ValueError(f"Reset state {new_state} is not valid according to the observation space.")

        self.state = new_state
        self.current_step = 0

        return self.state, {}

    def render(self):
        pass

    def close(self):
        pass

class FaustmannCallback(BaseCallback):
    """An optional callback for the Faustmann environment that displays the
    probability of harvest for all states.

    Optionally saves the probabilities to CSV files for later analysis and comparison.
    """

    def __init__(
        self,
        save_to_file: bool = False,
        save_interval: int = 10
    ):
        """Initialize the Faustmann callback.

        Args:
            save_to_file: Whether to save probabilities to a file. Defaults to False.
            save_interval: Save probabilities every N episodes. Defaults to 10.
        """

        super().__init__()
        self.save_to_file = save_to_file
        self.save_interval = save_interval
        self._get_artifact_path = None

    def _register_artifact(self):
        """Register the harvest probability artifact with the artifact manager."""

        if self.artifact_manager is None:
            return

        # Define the filename generator function.
        def filename_gen(episode: int) -> str:
            return f"episode_{episode:04d}.csv"

        # Register the artifact and get the path generator.
        try:
            self._get_artifact_path = self.artifact_manager.register_artifact(
                artifact_identifier="harvest_probabilities",
                artifact_description="FaustmannCallback_harvest_probabilities",
                file_name_generator=filename_gen,
                check_conflicts=True
            )
        except FileExistsError as e:
            raise e

    def on_episode_end(self, episode: int, logs: dict[str, Any] | None = None):
        """Display the probability of harvest for all states.

        Args:
            episode: The episode number.
            logs: Episode training logs.
        """

        A = self.env.get_attr("Tmin")[0]
        B = self.env.get_attr("Tmax")[0]
        states = np.arange(A, B + 1).astype(np.int32)
        states = states.reshape(-1, 1)
        distribution_params = self.agent.policy(states)
        distribution = self.agent.distribution(distribution_params)
        probs = distribution.prob(np.ones((states.shape[0], 1))).numpy().flatten()

        with np.printoptions(precision=2, suppress=True):
            print(f"Prob. to harvest {probs}.")

        # Optionally save to file.
        if self.save_to_file and (episode + 1) % self.save_interval == 0:
            # Register artifact on first save attempt.
            if self._get_artifact_path is None:
                self._register_artifact()

            if self._get_artifact_path is None:
                raise RuntimeError("Cannot save - artifact manager not available or conflicts exist.")

            # Get the artifact path using the registered callable.
            file_path = self._get_artifact_path(episode + 1)

            # Save the data.
            data = np.column_stack((states.flatten(), probs))
            np.savetxt(file_path, data, delimiter=",", header="state,harvest_probability", comments="", fmt="%d,%.6f")
