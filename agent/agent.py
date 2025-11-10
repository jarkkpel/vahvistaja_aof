from typing import Callable, Union, Tuple

import gymnasium as gym
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np

from util import ClippedDistribution

class Agent:
    """A base class for all agents."""

    def __init__(self, action_space: gym.Space, observation_space: gym.Space, distribution: Union[Callable, str, None] = None):
        """Initialize the agent.

        Args:
            action_space: The action space of the environment.
            observation_space: The observation space of the environment.
            distribution: The action distribution callable or string name for continuous action spaces. Defaults to None.
        """

        self.action_space = action_space
        self.observation_space = observation_space
        self._init_spaces()
        self.params_per_component, self._distribution = self._resolve_distribution(distribution)
        self.action_distribution_params_dim = self.params_per_component*self.action_dim

    def _init_spaces(self):
        """Initialize action and observation space properties."""

        def check_continuous(space):
            if isinstance(space, gym.spaces.Box):
                if space.dtype in [np.float32, np.float64]:
                    return True, int(space.shape[0])
                else:
                    return False, int(space.shape[0])
            elif isinstance(space, gym.spaces.Discrete):
                return False, 1
            else:
                raise ValueError(f"Unsupported space type: {type(space)}.")

        self.state_continuous, self.state_dim = check_continuous(self.observation_space)
        self.action_continuous, self.action_dim = check_continuous(self.action_space)

    def _resolve_distribution(self, distribution: Union[Callable, str, None]) -> Tuple[int, Union[Callable, None]]:
        """Resolve distribution parameter to a callable.

        Args:
            distribution: The distribution to resolve.

        Returns:
            Tuple containing:
                - How many parameters are needed to estimate a component of the action.
                - Distribution callable or None.

        Raises:
            ValueError: If string distribution name is not recognized.
        """

        if callable(distribution):
            return 1, distribution

        if self.action_continuous and distribution is None:
            raise ValueError("For continuous action spaces, a distribution must be provided.")

        if distribution is None:
            # Categorical distribution.
            return int(self.action_space.n), None

        if not self.action_continuous:
            raise ValueError("Only categorical distribution is supported for discrete action spaces.")

        if isinstance(distribution, str):
            # We need to clip some distributions to avoid edge cases that lead
            # to Inf and NaN values in log_prob calculations.
            distribution_map = {
                "beta": (2, lambda params: ClippedDistribution(base_distribution=tfd.Beta(concentration1=tf.nn.softplus(params[:, 0, ...]) + 1e-6, concentration0=tf.nn.softplus(params[:, 1, ...]) + 1e-6), low=0.001, high=0.999), ),
                "normal": (2, lambda params:tfd.Normal(loc=params[:, 0, ...], scale=tf.nn.softplus(params[:, 1, ...]) + 1e-6), ),
                "uniform": (2, lambda params: tfd.Uniform(low=params[:, 0, ...], high=params[:, 1, ...]), ),
                "exponential": (1, lambda params: tfd.Exponential(rate=tf.nn.softplus(params[:, 0, ...]) + 1e-6), ),
                "gamma": (1, lambda params: tfd.Gamma(concentration=tf.nn.softplus(params[:, 0, ...]) + 1e-6, rate=tf.nn.softplus(params[:, 1, ...]) + 1e-6), ),
                "lognormal": (2, lambda params: tfd.LogNormal(loc=params[:, 0, ...], scale=tf.nn.softplus(params[:, 1, ...]) + 1e-6), )
            }

            distribution_name = distribution.lower()
            if distribution_name not in distribution_map:
                available = ", ".join(distribution_map.keys())
                raise ValueError(f"Unknown distribution '{distribution}'. Available distributions: {available}.")

            return distribution_map[distribution_name]

        raise ValueError(f"Distribution must be None, callable, or string, got {type(distribution)}.")

    @classmethod
    def get_save_file_names(cls) -> dict:
        """Get the filenames for saving the agent.

        Returns:
            Dictionary mapping model component names to filenames.
        """

        raise NotImplementedError("Method get_save_file_names not implemented.")

    def save(self, save_path: str):
        """Save the agent.

        Args:
            save_path: The path to save the agent to.
        """

        raise NotImplementedError("Method Save not implemented.")

    @classmethod
    def load(cls, load_path: str, agent_parameters: dict) -> object:
        """Load a saved agent.

        Args:
            load_path: The path where the agent was saved to.
            agent_parameters: The parameters needed to initialize the agent.
        Returns:
            The loaded agent instance.
        """

        raise NotImplementedError("Method load not implemented.")

    def distribution(self, distribution_params: tf.Tensor) -> tfd.Distribution:
        """Returns the action distribution given the distribution parameters.

        Args:
            distribution_params: The parameters of the action distribution.
        Returns:
            The action distribution.
        """

        if len(distribution_params.shape) == 1:
            distribution_params = tf.reshape(distribution_params, (1, -1))

        if self.action_continuous:
            # Continuous actions, return a distribution of desired type with
            # the computed parameters.
            distribution_params = tf.reshape(distribution_params, (distribution_params.shape[0], self.params_per_component, -1, ))
            distribution = self._distribution(distribution_params)
        else:
            # Discrete actions, return Categorical distribution with logits.
            # Reshape to ensure correct distribution shape.
            distribution_params = tf.reshape(distribution_params, (distribution_params.shape[0], 1, distribution_params.shape[1]))
            distribution = tfd.Categorical(logits=distribution_params)

        return distribution

    def policy(self, state: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        """Returns action distribution parameters for the given state.

        Args:
            state: The state of the environment in shape (state_dim,), (N_steps, state_dim), or (N_steps, N_traces, state_dim).
        Returns:
            The action distribution parameters for the given state in corresponding shape.
        """

        raise NotImplementedError("Policy method not implemented.")

    def value(self, state: Union[tf.Tensor, np.ndarray]) -> float:
        """Returns the estimated state value for the given state.

        Args:
            state: The state of the environment in shape (state_dim,), (N_steps, state_dim), or (N_steps, N_traces, state_dim).
        Returns:
            The value estimate for the given state in corresponding shape.
        """

        raise NotImplementedError("Value method not implemented.")

    @tf.function
    def train_on_batch(self, *args: tf.Tensor, **kwargs) -> dict:
        """Train the agent on a batch of data.

        This method should be implemented by subclasses with their specific
        signature.

        Args:
            *args: Variable number of TensorFlow tensors containing the training data.
            **kwargs: Additional keyword arguments as needed by specific implementations.
        Returns:
            Dictionary containing training metrics (e.g., loss values).
        """

        raise NotImplementedError("Method train_on_batch not implemented.")
