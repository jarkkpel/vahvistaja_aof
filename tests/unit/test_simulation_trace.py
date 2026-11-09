import os
import pytest
import sys

import numpy as np
import tensorflow as tf

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agent.ppo import PPOSimulationTrace


class MockAgent:
    """Mock agent for testing trace functionality."""
    def __init__(self):
        self.normalize_advantages = True
        self.normalize_returns = True
        self.GAE_lambda = 0.9

        self.epsilon = 1e-8

        self.state_dim = 3
        self.action_dim = 2
        self.action_distribution_params_dim = 4


class TestPPOSimulationTrace:
    """Test the PPO simulation trace implementation."""

    @pytest.fixture
    def trace_params(self):
        """Parameters for creating a simulation trace."""

        return {
            "N_traces": 2,
            "length": 5,
            "discount_factor": 0.9
        }

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        return MockAgent()

    @pytest.fixture
    def ppo_trace(self, trace_params, mock_agent):
        """Create a PPO simulation trace for testing."""

        return PPOSimulationTrace(agent=mock_agent, **trace_params)

    def test_trace_initialization(self, ppo_trace, trace_params):
        """Test trace initialization."""

        trace = ppo_trace

        N_traces = trace_params["N_traces"]
        length = trace_params["length"]
        state_dim = trace.agent.state_dim
        action_dim = trace.agent.action_dim
        action_distribution_params_dim = trace.agent.action_distribution_params_dim
        discount_factor = trace_params["discount_factor"]

        assert trace.N_traces == N_traces
        assert trace.length == length
        assert trace.state_dim == state_dim
        assert trace.action_dim == action_dim
        assert trace.discount_factor == discount_factor

        assert trace.states.shape == (length, N_traces, state_dim)
        assert trace.actions.shape == (length, N_traces, action_dim)
        assert trace.rewards.shape == (length, N_traces)
        assert trace.terminateds.shape == (length, N_traces)
        assert trace.action_distribution_params.shape == (length, N_traces, action_distribution_params_dim)
        assert trace.returns.shape == (N_traces, )

    def test_trace_append_access(self, ppo_trace, trace_params):
        """Test appending data to the trace and accessing the data."""

        trace = ppo_trace

        N_traces = trace_params["N_traces"]
        state_dim = trace.agent.state_dim
        action_dim = trace.agent.action_dim
        action_distribution_params_dim = trace.agent.action_distribution_params_dim
        discount_factor = trace_params["discount_factor"]

        # Create sample data with random values.
        state = np.random.random((N_traces, state_dim))
        action = np.random.random((N_traces, action_dim))
        reward = np.random.random(N_traces)
        action_dist_params = np.random.random((N_traces, action_distribution_params_dim))
        terminated = np.array([False] * N_traces)

        # Append data for all timesteps.
        for t in range(trace.length):
            trace.append(t, state, action, reward, action_dist_params, terminated)

        # Check that data was stored correctly.
        np.testing.assert_array_almost_equal(trace.states[0], state)
        np.testing.assert_array_almost_equal(trace.actions[0], action)
        np.testing.assert_array_almost_equal(trace.rewards[0], reward)
        np.testing.assert_array_almost_equal(trace.terminateds[0], terminated)
        np.testing.assert_array_almost_equal(trace.action_distribution_params[0], action_dist_params)

        # Check that returns are computed correctly.
        powers = np.array([discount_factor**t for t in range(trace.length)])
        correct_returns = np.sum(trace["rewards"] * powers[:, np.newaxis], axis=0)
        np.testing.assert_array_almost_equal(trace["returns"], correct_returns)

        # Test access for valid keys.
        assert np.array_equal(trace["states"], trace.states)
        assert np.array_equal(trace["actions"], trace.actions)
        assert np.array_equal(trace["rewards"], trace.rewards)
        assert np.array_equal(trace["terminateds"], trace.terminateds)
        assert np.array_equal(trace["action_distribution_params"], trace.action_distribution_params)
        assert np.array_equal(trace["returns"], trace.returns)

        # Test access for an invalid key.
        with pytest.raises(KeyError):
            trace["invalid_key"]

    def test_compute_advantages_and_returns(self, ppo_trace, trace_params):
        """Test advantage and return computation."""

        trace = ppo_trace

        N_traces = trace_params["N_traces"]
        state_dim = trace.agent.state_dim
        action_dim = trace.agent.action_dim
        action_distribution_params_dim = trace.agent.action_distribution_params_dim

        # Fill trace with artificial data.
        for t in range(trace.length):
            state = np.random.random((N_traces, state_dim))
            action = np.random.random((N_traces, action_dim))
            reward = np.array([float(t), float(t + 1)])
            action_dist_params = np.random.random((N_traces, action_distribution_params_dim))
            terminated = np.array([False]*N_traces)

            trace.append(t, state, action, reward, action_dist_params, terminated)

        # Artificial state value estimates.
        trace["state_value_estimates"] = np.array([[10, 15], [10, 14], [9, 12], [7, 9], [4, 5]])

        # Compute advantages and returns.
        trace.compute_advantages_and_returns(GAE_lambda=0.9, next_state_value=None)

        # Hand-computed correct estimated advantages and returns.
        correct_estimated_advantages = np.array([[-2.4008464, -3.22821], [-1.72944, -2.25705], [-1.024, -1.305], [-0.4, -0.5], [0, 0]])
        correct_estimated_returns = correct_estimated_advantages + trace["state_value_estimates"]

        np.testing.assert_array_almost_equal(trace["estimated_advantages"], correct_estimated_advantages)
        np.testing.assert_array_almost_equal(trace["estimated_returns"], correct_estimated_returns)

    def test_normalize_for_training(self, trace_params, mock_agent):
        """Test normalization of advantages and returns for training."""

        # Normalize only advantages.
        mock_agent.normalize_advantages = True
        mock_agent.normalize_returns = False

        trace = PPOSimulationTrace(agent=mock_agent, **trace_params)
        N_traces = trace_params["N_traces"]
        state_dim = trace.agent.state_dim
        action_dim = trace.agent.action_dim
        action_distribution_params_dim = trace.agent.action_distribution_params_dim

        # Fill trace with artificial data.
        for t in range(trace.length):
            state = np.random.random((N_traces, state_dim))
            action = np.random.random((N_traces, action_dim))
            reward = np.array([float(t), float(t + 1)])
            action_dist_params = np.random.random((N_traces, action_distribution_params_dim))
            terminated = np.array([False]*N_traces)

            trace.append(t, state, action, reward, action_dist_params, terminated)

        trace["state_value_estimates"] = np.array([[10, 15], [10, 14], [9, 12], [7, 9], [4, 5]])
        trace.compute_advantages_and_returns(GAE_lambda=0.9, next_state_value=None)

        original_advantages = trace["estimated_advantages"].copy()
        original_returns = trace["estimated_returns"].copy()

        # Test normalizing advantages only.
        stats = trace.normalize()
        assert "advantages" in stats
        assert "returns" not in stats
        assert "mean" in stats["advantages"] and "std" in stats["advantages"]
        assert not np.array_equal(trace["estimated_advantages"], original_advantages)  # Should be normalized.
        assert np.array_equal(trace["estimated_returns"], original_returns)            # Should be unchanged.
