import gymnasium as gym
import numpy as np

class VahvistajaEnvironment(gym.Env):
    """A convenience base class to be inherited by all Vahvistaja environments."""

    def __init__(self, env_idx: int | None = None, discount_factor: float = 1.0):
        """Initialize the environment.

        Args:
            env_idx: An optional index identifying the environment instance.
            discount_factor: The discount factor used in the environment. Defaults to 1.0.
        """

        super().__init__()
        self.env_idx = env_idx
        self.discount_factor = discount_factor
        # Local RNG that does not interfere with global Numpy RNG.
        self._np_random = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment.

        Args:
            seed: Optional seed for the local random number generator.
            options: Optional reset options.
        """

        # Initialize or reseed the local RNG.
        if seed is not None:
            self._np_random = np.random.Generator(np.random.PCG64(seed))
        elif self._np_random is None:
            # Initialize with a random seed.
            self._np_random = np.random.Generator(np.random.PCG64())

        # Call parent reset which will seed observation_space and action_space
        return super().reset(seed=seed, options=options)

    def get_random_state(self) -> np.ndarray:
        """Return a random state sampled from the observation space.

        Returns:
            A random state.
        """

        raise NotImplementedError("Subclasses must implement the get_random_state method.")
