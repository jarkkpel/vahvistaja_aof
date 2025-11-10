import importlib.util
import os
import random
from pathlib import Path

import numpy as np

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

import tensorflow as tf
from tensorflow_probability import distributions as tfd

import yaml

def normalize_path(path: str) -> str:
    """Normalize a filesystem path by expanding user and environment variables.

    Args:
        path: The path to normalize.

    Returns:
        The normalized path.
    """

    if not path:
        return ""
    path_str = os.path.expandvars(os.path.expanduser(str(path)))
    return str(Path(path_str))

def join_paths(*paths: str) -> str:
    """Join multiple filesystem paths into a single normalized path.

    Args:
        *paths: The paths to join.
    Returns:
        The joined and normalized path.
    """

    if not paths:
        return ""
    normalized_paths = [normalize_path(p) for p in paths if p]
    if not normalized_paths:
        return ""
    result = Path(normalized_paths[0])
    for path_part in normalized_paths[1:]:
        result = result / path_part
    return str(result)

def set_seed(seed: int):
    """Set the random seed for all relevant libraries to ensure reproducibility.

    Args:
        seed: The seed value to set.
    """

    # Set environment variables for deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)

def load_config(config_file: str):
    """Load a YAML configuration file.

    Args:
        config_file: The path to the configuration file.
    Returns:
        The loaded configuration.
    Raises:
        ValueError: If the config file is not in YAML format.
    """

    if not (config_file.endswith(".yaml") or config_file.endswith(".yml")):
        raise ValueError("Config file must be in YAML format.")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_file}: {str(e)}")

    return config

def ensure_directory(path: str):
    """Ensure that the directory for the given path exists.

    Args:
        path: The path to the directory.
    """

    path = normalize_path(path)
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_method(file: str, name: str) -> object:
    """Loads the object of given name from the specified file.

    Args:
        file: The path to the file.
        name: The name of the object to load.
    Returns:
        The loaded object.
    """

    file = normalize_path(file)
    module_name = file.replace(os.sep, ".").rstrip(".py")
    spec = importlib.util.spec_from_file_location(module_name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, name)
    return cls

def create_environment(env_name: str, N_envs: int=1, discount_factor: float=1.0) -> AsyncVectorEnv:
    """Creates and sets up the environment (possibly multiple).

    Args:
        env_name: The name of the environment to create.
        N_envs: Number of parallel environments to create. Defaults to 1.
        discount_factor: The discount factor to be used. Defaults to 1.
    Returns:
        The created environments as an AsyncVectorEnv.
    """

    env_fn = [lambda env_idx=i: gym.make(env_name, env_idx=env_idx, discount_factor=discount_factor) for i in range(N_envs)]
    vec_env = AsyncVectorEnv(env_fn)
    return vec_env

class ClippedDistribution:
    """A distribution wrapper that clips sampled values to a specified range.

    This wrapper performs hard clipping on samples from the base distribution,
    ensuring all sampled values fall within [low, high]. The log_prob method
    returns -inf for values outside the valid range.

    This is strictly not a probability distribution as the clipping can remove
    probability mass. As long as the loss of mass is small, this should not
    matter too much.

    Uses composition rather than inheritance to avoid the complexities of
    inheriting tfp.Distribution properly.
    """

    def __init__(self, base_distribution, low, high):
        """Initialize the clipped distribution.

        Args:
            base_distribution: The underlying distribution to sample from.
            low: Lower bound for clipping (scalar or tensor).
            high: Upper bound for clipping (scalar or tensor).
        """

        self._base_distribution = base_distribution
        self._low = tf.convert_to_tensor(low, dtype=base_distribution.dtype)
        self._high = tf.convert_to_tensor(high, dtype=base_distribution.dtype)

    def sample(self, sample_shape=(), seed=None):
        """Sample values and clip them to the valid range."""

        samples = self._base_distribution.sample(sample_shape, seed=seed)
        return tf.clip_by_value(samples, self._low, self._high)

    def log_prob(self, value):
        """Compute log probability, returning -inf for out-of-bounds values."""

        # Check if value is within bounds.
        in_bounds = tf.logical_and(
            tf.greater_equal(value, self._low),
            tf.less_equal(value, self._high)
        )

        # Return -inf for out-of-bounds values, otherwise use base distribution.
        base_log_prob = self._base_distribution.log_prob(value)
        return tf.where(
            in_bounds,
            base_log_prob,
            tf.constant(-float('inf'), dtype=self._base_distribution.dtype)
        )

    def kl_divergence(self, other):
        """Compute KL divergence from this distribution to another."""

       # If other is also a ClippedDistribution, use its base distribution
        if isinstance(other, ClippedDistribution):
            return self._base_distribution.kl_divergence(other._base_distribution)
        else:
            return self._base_distribution.kl_divergence(other)

    def entropy(self):
        """Compute entropy of the distribution."""

        return self._base_distribution.entropy()

    def __getattr__(self, name):
        """Delegate any other attribute/method access to the base distribution."""

        return getattr(self._base_distribution, name)
