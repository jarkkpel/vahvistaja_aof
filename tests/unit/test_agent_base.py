import os
import pytest
import sys

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import gymnasium as gym

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agent.agent import Agent


class TestAgent:
    """Test the base Agent class functionality."""

    @pytest.mark.ml
    @pytest.mark.unit
    def test_agent_initialization(self):
        # Test continuous observation space, continuous action space.
        # One-dimensional.
        # ---------------------------------------------------------------------
        observation_space = gym.spaces.Box(low=0, high=10, dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float32)
        with pytest.raises(ValueError, match="For continuous action spaces, a distribution must be provided."):
            agent = Agent(observation_space=observation_space, action_space=action_space)
        with pytest.raises(ValueError, match="Unknown distribution"):
            agent = Agent(observation_space=observation_space, action_space=action_space, distribution="foobar")
        agent = Agent(observation_space=observation_space, action_space=action_space, distribution="normal")
        assert agent.state_continuous
        assert agent.state_dim == 1
        assert agent.action_continuous
        assert agent.action_dim == 1
        assert agent.params_per_component == 2
        assert agent.action_distribution_params_dim == 2*1
        params = tf.convert_to_tensor([0.0, 1.0])
        distribution = agent.distribution(params)
        assert isinstance(distribution, tfd.Normal)
        sample = distribution.sample().numpy()
        assert sample.shape == (1, 1)

        # Test continuous observation space, continuous action space.
        # Multidimensional.
        # ---------------------------------------------------------------------
        observation_space = gym.spaces.Box(low=0, high=10, shape=(5, ), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
        agent = Agent(observation_space=observation_space, action_space=action_space, distribution="exponential")
        assert agent.state_continuous
        assert agent.state_dim == 5
        assert agent.action_continuous
        assert agent.action_dim == 3
        assert agent.params_per_component == 1
        assert agent.action_distribution_params_dim == 1*3
        params = tf.convert_to_tensor([1.0, 1.0, 1.0])
        distribution = agent.distribution(params)
        assert isinstance(distribution, tfd.Exponential)
        sample = distribution.sample().numpy()
        assert sample.shape == (1, 3)

        # Test discrete observation space, discrete action space.
        # ---------------------------------------------------------------------
        observation_space = gym.spaces.Discrete(5)
        action_space = gym.spaces.Discrete(2)
        with pytest.raises(ValueError, match="Only categorical distribution is supported for discrete action spaces."):
            agent = Agent(observation_space=observation_space, action_space=action_space, distribution="normal")
        agent = Agent(observation_space=observation_space, action_space=action_space)
        assert not agent.state_continuous
        assert agent.state_dim == 1
        assert not agent.action_continuous
        assert agent.action_dim == 1
        assert agent.params_per_component == 2
        assert agent.action_distribution_params_dim == 2*1
        assert agent._distribution is None

        # Test continuous observation space, discrete action space.
        # ---------------------------------------------------------------------
        observation_space = gym.spaces.Box(low=0, high=10, shape=(5, ), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        agent = Agent(observation_space=observation_space, action_space=action_space)
        assert agent.state_continuous
        assert agent.state_dim == 5
        assert not agent.action_continuous
        assert agent.action_dim == 1
        assert agent.params_per_component == 2
        assert agent.action_distribution_params_dim == 2*1
        assert agent._distribution is None

        # Test discrete observation space, continuous action space.
        # ---------------------------------------------------------------------
        observation_space = gym.spaces.Discrete(5)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
        agent = Agent(observation_space=observation_space, action_space=action_space, distribution="normal")
        assert not agent.state_continuous
        assert agent.state_dim == 1
        assert agent.action_continuous
        assert agent.action_dim == 3
        assert agent.params_per_component == 2
        assert agent.action_distribution_params_dim == 2*3
        params = tf.convert_to_tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        distribution = agent.distribution(params)
        assert isinstance(distribution, tfd.Normal)
        sample = distribution.sample().numpy()
        assert sample.shape == (1, 3)

    @pytest.mark.unit
    def test_distribution_creation(self):
        """Test the creation of distributions from parameters."""

        # Continuous distribution.
        # ---------------------------------------------------------------------
        observation_space = gym.spaces.Box(low=0, high=10, shape=(5, ), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
        agent = Agent(observation_space=observation_space, action_space=action_space, distribution="normal")
        params = tf.convert_to_tensor(np.array([0.0, 1.0]*3).reshape(1, -1, 2))
        assert isinstance(agent.distribution(params), tfd.Normal)

        # Disrete distribution.
        # ---------------------------------------------------------------------
        observation_space = gym.spaces.Discrete(5)
        action_space = gym.spaces.Discrete(2)
        agent = Agent(observation_space=observation_space, action_space=action_space)
        params = tf.convert_to_tensor(np.arange(2, dtype=np.float32).reshape(1, -1))
        assert isinstance(agent.distribution(params), tfd.Categorical)
