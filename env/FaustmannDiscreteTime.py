from typing import Union

import gymnasium as gym
from gymnasium.envs.registration import register

import numpy as np

from .base import VahvistajaEnvironment
from callbacks import BaseCallback

class FaustmannEnv(VahvistajaEnvironment):
    """A Faustmann discrete time forest rotation environment."""

    def __init__(self, env_idx: Union[int,None]=None, discount_factor: float=1.0):
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

    def reset(self, seed: Union[int,None] = None, options: Union[dict,list,None] = None):
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
        if seed is not None:
            self.seed(seed)

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
    probability of harvest for all states."""

    def on_episode_end(self, episode: int, logs: Union[dict,None]=None):
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
