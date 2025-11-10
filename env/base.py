from typing import Union

import gymnasium as gym
import numpy as np

class VahvistajaEnvironment(gym.Env):
    """A convenience base class to be inherited by all Vahvistaja environments."""

    def __init__(self, env_idx: Union[int,None]=None, discount_factor: float=1.0):
        """Initialize the environment.

        Args:
            env_idx: An optional index identifying the environment instance.
            discount_factor: The discount factor used in the environment. Defaults to 1.0.
        """

        super().__init__()
        self.env_idx = env_idx
        self.discount_factor = discount_factor

    def get_random_state(self) -> np.ndarray:
        """Return a random state sampled from the observation space.

        Returns:
            A random state.
        """

        raise NotImplementedError("Subclasses must implement the get_random_state method.")
