from typing import Union

import gymnasium as gym
from gymnasium.envs.registration import register

import numpy as np
import tensorflow as tf

from .base import VahvistajaEnvironment
from callbacks import BaseCallback

class MarketAgeClassEnv(VahvistajaEnvironment):
    """Market-level model."""

    """
    The idea is as follows:

    The forest is assumed to be divided into age classes {1, 2, ..., A_max}.
    The age classes represent a proportion of the total forest in each age
    group, and they sum up to X_tot. Therefore the state is continuous and
    represented as a vector of length A_max with elements in [0, X_tot] summing
    to X_tot.

    The action specifies how large a proportion of the forest in each age class
    is cut. Thus the action is a vector of length A_max taking values in [0, 1].

    For each age class, there is a harvest value for the proportion 1.0. These
    values are saved in the vector self.qvec, which is constant. Given an
    action A in state S, the harvest value for each age class is given as
    A * S * self.qvec (i.e., we S*self.qvec represents how much harvest value
    is currently possible to have from each age class, and A*S*self.qvec
    represents the harvest value after the proportional cut decisions have been
    made). The total harvest value h is the sum of the age class harvest
    values. The actual reward is determined by an integral on the harvest value
    minus regrowth costs.

    After the forest has been cut, the state is updated as follows. All
    harvested proportion is put to the youngest age class. All the other
    proportions are harvested shifted towards the oldest age class, which is a
    sink class.

    The above is the "relative action" approach. An "absolute action" approach
    is to have the action vector components directly take values in [0, X_tot].
    This approach needs some constaints to not overcut the forest and is not
    considered here except on a brief comment level.
    """

    def __init__(self, env_idx: Union[int,None]=None, discount_factor: float=1.0):
        """Initialize the Market-level age class environment.

        Args:
            env_idx: An optional index identifying the environment instance.
            discount_factor: The discount factor used in the environment. Defaults to 1.0.
        """

        super().__init__(env_idx, discount_factor)

        self.A_max = 15  # The number of age classes.
        self.X_tot = 1.0 # Age class values sum up to this.

        self.observation_space = gym.spaces.Box(low=0.0, high=self.X_tot, shape=(self.A_max,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.A_max,), dtype=np.float32)
        # "Absolute action" approach:
        # self.action_space = gym.spaces.Box(low=0.0, high=self.X_tot, shape=(self.A_max,), dtype=np.float32)

        self.eps = 1e-8

        # Initial state.
        # We sample positive values for age classes 1-8 (uniformly random), zeros for the rest.
        self.init_state = np.zeros(self.A_max, dtype=np.float32)
        uniform_values = np.random.uniform(0, 1, 8)
        uniform_values = uniform_values / np.sum(uniform_values) * self.X_tot
        self.init_state[:8] = uniform_values
        self.state = self.init_state.copy()

        # Payoff constants.
        self.p0 = 50
        self.q0 = 60
        self.Creg = 1500  # Regrowth cost for proportion 1.0.
        self.DelasP = 1.0 # Demand elasticity of price.

        # Age class harvest vector.
        self.dt = 10 # Controls how different the harvest values per age class are. The larger this is, the more alike they are.
        self.qvec = 500.0 / (1.0 + np.exp(-(self.dt * np.arange(1, self.A_max + 1) - 50.0) / 10.0))

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
            if action.shape != (1, self.A_max):
                raise ValueError(f"Incorrect action shape {action.shape}, expected ({self.A_max}, ) or (1, {self.A_max}).")
            action = action.flatten()
        elif len(action.shape) > 2:
            raise ValueError(f"Incorrect action shape {action.shape}, expected ({self.A_max}, ) or (1, {self.A_max}).")

        self.current_step += 1

        h = np.sum(action * self.state * self.qvec) # Total harvest value.

        # Payoff is the integral of p0 * (x/q_0)^D from q_0 to h where D is self.DelasP (demand elasticitiy of price).
        # We add eps to avoid 0 values when nothing is harvested. We want to signal a very low payoff in this case.
        # TODO: When self.DelasP != 1, the value h = 0 seems to have quite
        # different effect than the case in self.DelasP == 1.
        if self.DelasP == 1:
            payoff = self.p0 * self.q0 * np.log((h + self.eps)/self.q0)
        else:
            payoff = self.p0 * self.q0 * ( (h/self.q0)**(1-self.DelasP) - 1 )
        # We subtract regrowth costs from the payoff.
        payoff = payoff - self.Creg * sum(action * self.state)

        new_state = np.zeros(self.A_max, dtype=np.float32)
        # All harvested forest is put to the youngest age class.
        new_state[0] = sum(action * self.state)
        # Values except the last one are shifted forward.
        for a in range(self.A_max - 2):
            new_state[a+1] = (1 - action[a])*self.state[a]
        # The last state is the sink state that has harvested forest from indices -1 and -2.
        new_state[self.A_max - 1] = (1 - action[self.A_max - 2])*self.state[self.A_max - 2] + (1 - action[self.A_max - 1])*self.state[self.A_max - 1]

        """
        # "Absolute action" approach:
        newstate[0] = sum(action)
        for a in range(self.A_max - 2):
            newstate[a+1] = self.state[a] - action[a]
        newstate[self.A_max - 1] = self.state[self.A_max - 2] + self.state[self.A_max - 1] - action[self.A_max - 2] - action[self.A_max - 1]
        """

        # Truncation is applied, although not perhaps necessary.
        truncated = ( self.current_step >= self.max_episode_steps )
        # Problem is infinite: there is no termination.
        terminated = False

        self.state = new_state

        return self.state, np.float32(payoff), terminated, truncated, {}

    def get_random_state(self) -> np.ndarray:
        """Return a random state sampled from the observation space.

        Returns:
            np.ndarray: A random state.
        """

        state = self.observation_space.sample()
        state = state / np.sum(state) * self.X_tot
        return state

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
        if np.abs(np.sum(new_state) - self.X_tot) > 1e-5:
            raise ValueError(f"Reset state {new_state} elements do not sum to {self.X_tot}.")

        self.state = new_state
        self.current_step = 0

        return self.state, {}

    def render(self):
        pass

    def close(self):
        pass

