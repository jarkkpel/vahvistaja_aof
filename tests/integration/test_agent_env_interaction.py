import pytest
import numpy as np
import tensorflow as tf
import gymnasium as gym
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agent.ppo import PPOAgent, PPOSimulationTrace
from env.FaustmannDiscreteTime import FaustmannEnv
from env.MarketACD import MarketAgeClassEnv
from callback_manager import CallbackManager
from callbacks import BaseCallback
from simulation_trace import simulate_trace
from util import create_environment


class TestAgentEnvironmentInteraction:
    """Integration tests for agent-environment interaction."""

    @pytest.fixture
    def faustmann_env(self):
        """Create a Faustmann environment."""

        return FaustmannEnv()

    @pytest.fixture
    def market_acd_env(self):
        """Create a MarketACD environment."""

        return MarketAgeClassEnv()

    @pytest.fixture
    def faustmann_agent(self, faustmann_env):
        """Create a PPO agent for Faustmann environment."""

        return PPOAgent(
            action_space=faustmann_env.action_space,
            observation_space=faustmann_env.observation_space
        )

    @pytest.fixture
    def market_acd_agent(self, market_acd_env):
        """Create a PPO agent for MarketACD environment."""

        return PPOAgent(
            action_space=market_acd_env.action_space,
            observation_space=market_acd_env.observation_space,
            distribution="beta"
        )

    @pytest.mark.ml
    def test_faustmann_agent_env_step(self, faustmann_env, faustmann_agent):
        """Test that an agent can sample an action and step the Faustmann environment."""

        env = faustmann_env
        agent = faustmann_agent

        state, info = env.reset()

        # Get action distribution parameters and sample an action.
        action_dist_params = agent.policy(state)
        distribution = agent.distribution(action_dist_params)
        action = distribution.sample().numpy()

        # Step the environment with the sampled action.
        next_state, reward, terminated, truncated, info = env.step(action)

    @pytest.mark.ml
    def test_market_acd_agent_env_step(self, market_acd_env, market_acd_agent):
        """Test that an agent can sample an action and step the MarketACD environment."""

        env = market_acd_env
        agent = market_acd_agent

        state, info = env.reset()

        # Get action distribution parameters and sample an action.
        action_dist_params = agent.policy(state)
        distribution = agent.distribution(action_dist_params)
        action = distribution.sample().numpy()

        # Step the environment with the sampled action.
        next_state, reward, terminated, truncated, info = env.step(action)


class TestTrainingIntegration:
    """Integration tests for the training pipeline."""

    @pytest.fixture
    def training_env(self):
        """Create a vectorized environment for training."""

        env = create_environment("Faustmann-v0", N_envs=1)
        yield env
        env.close()

    @pytest.fixture
    def training_agent(self, training_env):
        """Create a PPO agent for training."""

        return PPOAgent(
            action_space=training_env.get_attr("action_space")[0],
            observation_space=training_env.get_attr("observation_space")[0]
        )

    @pytest.mark.ml
    def test_train_runs_multiple_episodes(self, training_env, training_agent):
        """Test that train function completes multiple episodes successfully."""

        from train import train

        env = training_env
        agent = training_agent

        # Run very short training.
        final_episode = train(
            agent,
            env,
            PPOSimulationTrace,
            N_episodes=2,
            N_steps=4,
            reset_prob=0.0,
            epochs_per_episode=1,
            batch_size=2
        )

        assert final_episode >= 0

    @pytest.mark.ml
    def test_train_with_early_stopping(self, training_env, training_agent):
        """Test that train function respects early stopping from callbacks."""

        from train import train

        env = training_env
        agent = training_agent

        # Create callback manager with immediate stopping callback.
        callback_manager = CallbackManager(env, agent)

        class ImmediateStopCallback(BaseCallback):
            def on_episode_end(self, episode, logs=None):
                if episode >= 1:  # Stop after second episode.
                    self.stop_training = True

        callback_manager.add_callback(ImmediateStopCallback())

        final_episode = train(
            agent,
            env,
            PPOSimulationTrace,
            N_episodes=10,  # Request 10 but should stop early.
            N_steps=4,
            reset_prob=0.0,
            epochs_per_episode=1,
            batch_size=2,
            callback_manager=callback_manager
        )

        # Should stop before completing all episodes.
        assert final_episode == 1


class TestSimulationTraceIntegration:
    """Integration tests for simulate_trace function."""

    @pytest.fixture
    def trace_params(self):
        """Parameters for creating a simulation trace."""

        return {
            "N_traces": 2,
            "N_steps": 5,
        }

    @pytest.fixture
    def faustmann_vec_env(self, trace_params):
        """Create a vectorized Faustmann environment."""

        N_traces = trace_params["N_traces"]
        env = create_environment("Faustmann-v0", N_envs=N_traces)
        yield env
        env.close()

    @pytest.fixture
    def faustmann_agent_for_trace(self, faustmann_vec_env):
        """Create a PPO agent for Faustmann environment."""

        return PPOAgent(
            action_space=faustmann_vec_env.get_attr("action_space")[0],
            observation_space=faustmann_vec_env.get_attr("observation_space")[0]
        )

    @pytest.fixture
    def faustmann_single_env(self):
        """Create a single Faustmann environment."""

        env = create_environment("Faustmann-v0", N_envs=1)
        yield env
        env.close()

    @pytest.fixture
    def faustmann_agent_single(self, faustmann_single_env):
        """Create a PPO agent for single Faustmann environment."""

        return PPOAgent(
            action_space=faustmann_single_env.get_attr("action_space")[0],
            observation_space=faustmann_single_env.get_attr("observation_space")[0]
        )

    @pytest.mark.ml
    def test_simulate_trace_correct_shape(self, trace_params, faustmann_vec_env, faustmann_agent_for_trace):
        """Test that simulate_trace creates trace with correct dimensions.

        Rationale: Verifies trace creation from actual environment interaction.
        """

        N_traces = trace_params["N_traces"]
        N_steps = trace_params["N_steps"]

        env = faustmann_vec_env
        agent = faustmann_agent_for_trace

        init_states = list(env.call("get_random_state"))
        trace, final_value = simulate_trace(
            PPOSimulationTrace,
            env,
            agent,
            N_steps=N_steps,
            init_state=init_states
        )

        assert trace.N_traces == N_traces
        assert trace.length == N_steps
        assert trace.states.shape == (N_steps, N_traces, agent.state_dim)
        assert isinstance(final_value, (float, np.floating))

    @pytest.mark.ml
    def test_simulate_trace_respects_init_state(self, faustmann_single_env, faustmann_agent_single):
        """Test that simulate_trace starts from provided initial states."""

        env = faustmann_single_env
        agent = faustmann_agent_single

        custom_init = [np.array([5], dtype=np.int32)]
        trace, _ = simulate_trace(
            PPOSimulationTrace,
            env,
            agent,
            N_steps=3,
            init_state=custom_init
        )

        # First state should match init state
        np.testing.assert_array_equal(trace.states[0, 0], custom_init[0])


class TestPPOAgentIntegration:
    """Integration tests for PPO agent with environments."""

    @pytest.fixture
    def faustmann_env_for_agent(self):
        """Create a Faustmann environment for agent testing."""

        env = FaustmannEnv()
        yield env

    @pytest.fixture
    def ppo_agent_faustmann(self, faustmann_env_for_agent):
        """Create a PPO agent for Faustmann environment."""

        return PPOAgent(
            action_space=faustmann_env_for_agent.action_space,
            observation_space=faustmann_env_for_agent.observation_space
        )

    @pytest.mark.ml
    def test_ppo_gradient_flow(self, ppo_agent_faustmann):
        """Test that PPO agent gradients flow correctly during training."""

        agent = ppo_agent_faustmann

        # Build models.
        dummy_state = np.random.random((1, agent.state_dim)).astype(np.float32)
        _ = agent.policy(dummy_state)
        _ = agent.value(dummy_state)

        # Get initial weights.
        initial_actor_weights = [w.numpy().copy() for w in agent.actor.trainable_weights]
        initial_critic_weights = [w.numpy().copy() for w in agent.critic.trainable_weights]

        # Train on dummy batch.
        batch_size = 4
        states = tf.random.uniform((batch_size, agent.state_dim), dtype=tf.float32)
        actions = tf.random.uniform((batch_size, agent.action_dim), minval=0, maxval=1, dtype=tf.float32)
        action_dist_params = agent.policy(states)
        advantages = tf.random.uniform((batch_size,), dtype=tf.float32)
        returns = tf.random.uniform((batch_size,), dtype=tf.float32)

        agent.train_on_batch(states, actions, action_dist_params, advantages, returns)

        # Verify weights changed.
        actor_changed = any(
            not np.array_equal(w.numpy(), init_w)
            for w, init_w in zip(agent.actor.trainable_weights, initial_actor_weights)
        )
        critic_changed = any(
            not np.array_equal(w.numpy(), init_w)
            for w, init_w in zip(agent.critic.trainable_weights, initial_critic_weights)
        )

        assert actor_changed, "Actor weights did not change after training."
        assert critic_changed, "Critic weights did not change after training."

    @pytest.mark.ml
    def test_ppo_batch_size_independence(self, ppo_agent_faustmann):
        """Test that PPO produces consistent results regardless of batch size."""

        agent = ppo_agent_faustmann

        # Same state evaluated individually and in batch.
        state = np.random.random((1, agent.state_dim)).astype(np.float32)

        single_policy = agent.policy(state)
        single_value = agent.value(state)

        # Same state repeated in batch.
        batch_state = np.repeat(state, 5, axis=0)
        batch_policy = agent.policy(batch_state)
        batch_value = agent.value(batch_state)

        # First element should match.
        np.testing.assert_array_almost_equal(
            single_policy.numpy()[0],
            batch_policy.numpy()[0],
            decimal=5
        )
        np.testing.assert_almost_equal(
            single_value.numpy()[0],
            batch_value.numpy()[0],
            decimal=5
        )
