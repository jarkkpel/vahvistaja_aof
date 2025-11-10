import pytest
import numpy as np
import tensorflow as tf
import gymnasium as gym
import tempfile
import shutil
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agent.ppo import PPOAgent, PPOSimulationTrace


class TestPPOAgent:
    """Test the PPO agent implementation."""

    @pytest.fixture
    def observation_space(self):
        return gym.spaces.Box(low=0, high=10, shape=(5, ), dtype=np.float32)

    @pytest.fixture
    def continuous_action_space(self):
        return gym.spaces.Box(low=-2**10, high=2**10, shape=(3, ), dtype=np.float32)

    @pytest.fixture
    def discrete_action_space(self):
        return gym.spaces.Discrete(3)

    @pytest.fixture
    def ppo_agent_discrete(self, discrete_action_space, observation_space):
        """Create a PPO agent with discrete action space."""

        return PPOAgent(
            action_space=discrete_action_space,
            observation_space=observation_space
        )

    @pytest.fixture
    def ppo_agent_continuous(self, continuous_action_space, observation_space):
        """Create a PPO agent with continuous action space."""

        return PPOAgent(
            action_space=continuous_action_space,
            observation_space=observation_space,
            distribution="normal"
        )

    @pytest.mark.ml
    def test_value_shape(self, ppo_agent_continuous):
        """Test the input and output shapes of the PPO agent."""

        agent = ppo_agent_continuous
        batch_size = 4
        state_dim = agent.state_dim

        # Check that both numpy and tensor inputs work and give the same
        # results.
        state = np.random.random(size=(batch_size, state_dim)).astype(np.float32)
        sve_1 = agent.value(state)
        assert isinstance(sve_1, tf.Tensor)
        assert sve_1.shape == (batch_size, )
        sve_2 = agent.value(tf.convert_to_tensor(state))
        assert isinstance(sve_2, tf.Tensor)
        np.testing.assert_array_almost_equal(sve_1, sve_2)

        # Check single state input in two different input formats.
        state = np.random.random(size=(1, state_dim)).astype(np.float32)
        sve_1 = agent.value(state)
        sve_2 = agent.value(state.reshape(-1))
        assert isinstance(sve_1, tf.Tensor)
        assert isinstance(sve_2, tf.Tensor)
        np.testing.assert_array_almost_equal(sve_1, sve_2)

        # Check for unsupported input.
        with pytest.raises(Exception):
            agent.value("invalid")

        # Check batch of states in 3D array format.
        N_traces = 2
        states_3d = np.random.random(size=(batch_size, N_traces, state_dim)).astype(np.float32)
        values = agent.value(states_3d)
        assert isinstance(values, tf.Tensor)
        assert values.shape == (batch_size, N_traces)

    @pytest.mark.ml
    def test_policy_shape(self, ppo_agent_continuous):
        """Test the input and output shapes of the PPO agent."""

        agent = ppo_agent_continuous
        batch_size = 4
        state_dim = agent.state_dim
        action_distribution_params_dim = agent.action_distribution_params_dim

        # Check that both numpy and tensor inputs work and give the same
        # results.
        state = np.random.random(size=(batch_size, state_dim)).astype(np.float32)
        dp_1 = agent.policy(state)
        assert isinstance(dp_1, tf.Tensor)
        assert dp_1.shape == (batch_size, action_distribution_params_dim)
        dp_2 = agent.policy(tf.convert_to_tensor(state))
        assert isinstance(dp_2, tf.Tensor)
        np.testing.assert_array_almost_equal(dp_1, dp_2)

        # Check single state input in two different input formats.
        state = np.random.random(size=(1, state_dim)).astype(np.float32)
        dp_1 = agent.policy(state)
        dp_2 = agent.policy(state.reshape(-1))
        assert isinstance(dp_1, tf.Tensor)
        assert isinstance(dp_2, tf.Tensor)
        np.testing.assert_array_almost_equal(dp_1, dp_2)

        # Check for unsupported input.
        with pytest.raises(Exception):
            agent.value("invalid")

        # Check batch of states in 3D array format.
        N_traces = 2
        states_3d = np.random.random(size=(batch_size, N_traces, state_dim)).astype(np.float32)
        values = agent.policy(states_3d)
        assert isinstance(values, tf.Tensor)
        assert values.shape == (batch_size, N_traces, action_distribution_params_dim)

    @pytest.mark.ml
    def test_policy_distribution_continuous(self, ppo_agent_continuous):
        """Test that the distribution parameters and distributions in the continuous case."""

        agent = ppo_agent_continuous
        batch_size = 4
        state_dim = agent.state_dim

        state = np.random.random(size=(batch_size, state_dim))
        distribution_params = agent.policy(state)
        distribution = agent.distribution(distribution_params)
        action = distribution.sample().numpy()
        action = np.clip(action, agent.action_space.low, agent.action_space.high)
        for a in action:
            assert agent.action_space.contains(a)

    @pytest.mark.ml
    def test_policy_distribution_discrete(self, ppo_agent_discrete):
        """Test that the distribution parameters and distributions in the discrete case."""

        agent = ppo_agent_discrete
        batch_size = 4
        state_dim = agent.state_dim

        state = np.random.random(size=(batch_size, state_dim))
        distribution_params = agent.policy(state)
        distribution = agent.distribution(distribution_params)
        action = distribution.sample().numpy().flatten()
        for a in action:
            assert agent.action_space.contains(a)

    @pytest.mark.ml
    def test_train_on_batch_shapes(self, ppo_agent_continuous):
        """Test training on batch with correct shapes."""

        agent = ppo_agent_continuous
        batch_size = 4
        state_dim = agent.state_dim
        action_dim = agent.action_dim
        action_distribution_params_dim = agent.action_distribution_params_dim

        # Create dummy training data.
        states = tf.random.uniform((batch_size, state_dim), dtype=tf.float32)
        actions = tf.random.uniform((batch_size, action_dim), minval=agent.action_space.low, maxval=agent.action_space.high, dtype=tf.float32)
        action_dist_params = tf.random.uniform((batch_size, action_distribution_params_dim), dtype=tf.float32)
        advantages = tf.random.uniform((batch_size,), dtype=tf.float32)
        returns = tf.random.uniform((batch_size,), dtype=tf.float32)

        losses = agent.train_on_batch(states, actions, action_dist_params, advantages, returns)
        assert "actor_loss" in losses
        assert "critic_loss" in losses
        assert isinstance(losses["actor_loss"], tf.Tensor)
        assert isinstance(losses["critic_loss"], tf.Tensor)

    @pytest.mark.ml
    def test_save_and_load(self, ppo_agent_discrete):
        """Test saving and loading the agent."""

        agent = ppo_agent_discrete

        # Build the models by calling them with dummy data.
        dummy_state = np.random.random(size=(1, agent.state_dim)).astype(np.float32)
        _ = agent.policy(dummy_state)
        _ = agent.value(dummy_state)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test saving.
            agent.save(temp_dir)
            assert os.path.exists(os.path.join(temp_dir, "actor.keras"))
            assert os.path.exists(os.path.join(temp_dir, "critic.keras"))

            # Test loading.
            agent_params = {
                "action_space": agent.action_space,
                "observation_space": agent.observation_space,
                "GAE_lambda": agent.GAE_lambda,
                "PPO_clip": agent.PPO_clip,
                "actor_lr": agent.actor_lr,
                "critic_lr": agent.critic_lr
            }
            loaded_agent = PPOAgent.load(temp_dir, agent_params)
