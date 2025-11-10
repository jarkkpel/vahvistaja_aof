import pytest
import numpy as np
import gymnasium as gym
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from env.FaustmannDiscreteTime import FaustmannEnv
from env.MarketACD import MarketAgeClassEnv


class TestFaustmannEnv:
    """Unit tests for the Faustmann environment."""

    @pytest.fixture
    def env(self):
        """Create a Faustmann environment."""

        return FaustmannEnv()

    def test_initialization(self, env):
        """Test that the environment initializes correctly."""

        assert env.current_step == 0

        # Check the spaces.
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (1,)
        assert env.observation_space.low == env.Tmin
        assert env.observation_space.high == env.Tmax
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 2

        # Check the initial state.
        assert env.state.shape == (1,)
        assert env.state[0] == 1

    def test_reset_default(self, env):
        """Test reset with default parameters."""

        state, info = env.reset()

        assert isinstance(state, np.ndarray)
        assert state.shape == env.observation_space.shape
        assert state[0] == 1
        assert env.current_step == 0

    def test_reset_with_seed(self, env):
        """Test reset with seed for reproducibility."""

        state = np.array([42], dtype=np.int32)
        state1, _ = env.reset(options={"reset_state": state})
        state2, _ = env.reset(options={"reset_state": state})
        np.testing.assert_array_equal(env.state, state)
        np.testing.assert_array_equal(state1, state2)
        assert env.current_step == 0

    def test_reset_invalid_state(self, env):
        """Test that reset raises error for invalid state."""

        invalid_state = np.array([env.Tmax + 1], dtype=np.int32)  # Out of bounds.
        options = {"reset_state": invalid_state}
        with pytest.raises(ValueError, match="not valid according to the observation space"):
            env.reset(options=options)

    def test_step_no_harvest(self, env):
        """Test stepping with action 0 (no harvest)."""

        env.reset()
        initial_state = env.state.copy()
        # Test with shape (1, ).
        action = np.array([0])
        next_state, reward, terminated, truncated, info = env.step(action)
        # Test with shape (1, 1).
        action = np.array([[0]])
        next_state, reward, terminated, truncated, info = env.step(action)

        # State should increment by 1 when not harvesting.
        assert next_state[0] == initial_state[0] + 2
        # No harvest means reward should be 0.
        assert reward == 0.0
        assert env.current_step == 2

    def test_step_harvest(self, env):
        """Test stepping with action 1 (harvest)."""

        state = int((env.Tmax - env.Tmin) // 2)
        options = {"reset_state": np.array([state], dtype=np.int32)}
        env.reset(options=options)
        action = np.array([1])

        next_state, reward, terminated, truncated, info = env.step(action)

        # State should reset to 1 when harvesting.
        assert next_state[0] == env.Tmin

    def test_step_invalid_action_shape(self, env):
        """Test that step raises error for invalid action shapes."""

        # Invalid 2D shape.
        action = np.array([[0, 1]])
        with pytest.raises(ValueError, match="Incorrect action shape"):
            env.step(action)

        # Invalid 3D shape.
        action = np.array([[[0]]])
        with pytest.raises(ValueError, match="Incorrect action shape"):
            env.step(action)

    def test_get_random_state(self, env):
        """Test get_random_state method."""

        random_state = env.get_random_state()
        assert env.observation_space.contains(random_state)
        assert random_state.shape == (1,)

    def test_faustmann_max_steps_boundary(self, env):
        """Test Faustmann environment at maximum age boundary."""

        max_state = np.array([env.Tmax], dtype=np.int32)

        env.reset(options={"reset_state": max_state})
        assert env.state[0] == env.Tmax

    def test_faustmann_consecutive_harvests(self, env):
        """Test that Faustmann environment handles consecutive harvest
        actions."""

        env.reset()

        _, r1, _, _, _ = env.step(np.array([1]))
        _, r2, _, _, _ = env.step(np.array([1]))
        assert env.state[0] == env.Tmin


class TestMarketAgeClassEnv:
    """Unit tests for the MarketACD environment."""

    @pytest.fixture
    def env(self):
        """Create a MarketACD environment."""

        return MarketAgeClassEnv()

    def test_initialization(self, env):
        """Test that the environment initializes correctly."""

        assert env.current_step == 0

        # Check the spaces.
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (env.A_max,)
        assert np.array_equal(env.observation_space.low, np.array([0]*env.A_max))
        assert np.array_equal(env.observation_space.high, np.array([env.X_tot]*env.A_max))
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (env.A_max,)
        assert np.array_equal(env.action_space.low, np.array([0]*env.A_max))
        assert np.array_equal(env.action_space.high, np.array([1]*env.A_max))

    def test_initial_state_sum(self, env):
        """Test that initial state sums to X_tot."""

        assert np.isclose(np.sum(env.init_state), env.X_tot, atol=1e-6)

    def test_reset_default(self, env):
        """Test reset with default parameters."""

        state, info = env.reset()

        assert state.shape == env.observation_space.shape
        assert env.current_step == 0
        assert np.isclose(np.sum(state), env.X_tot, atol=1e-5)

    def test_reset_with_seed(self, env):
        """Test reset with seed for reproducibility."""

        state = np.random.uniform(size=env.A_max).astype(np.float32)
        state = state / np.sum(state) * env.X_tot
        state1, _ = env.reset(options={"reset_state": state})
        state2, _ = env.reset(options={"reset_state": state})
        np.testing.assert_array_equal(env.state, state)
        np.testing.assert_array_almost_equal(state1, state2)
        assert np.isclose(np.sum(state), env.X_tot, atol=1e-5)
        assert env.current_step == 0

    def test_reset_invalid_state(self, env):
        """Test that reset raises error for invalid states."""

        # State out of bounds.
        invalid_state = np.ones(env.A_max, dtype=np.float32) * 2.0
        options = {"reset_state": invalid_state}
        with pytest.raises(ValueError, match="not valid according to the observation space"):
            env.reset(options=options)

        # State does not sum to X_tot.
        invalid_state = np.ones(env.A_max, dtype=np.float32)  # Sums to A_max*state_dim, not 1.0.
        options = {"reset_state": invalid_state}
        with pytest.raises(ValueError, match="do not sum to"):
            env.reset(options=options)

    def test_step_no_harvest(self, env):
        """Test stepping with zero action (no harvest)."""

        state_before, _ = env.reset()
        # Test with shape (env.A_max).
        action = np.zeros(env.A_max, dtype=np.float32)
        next_state, reward, terminated, truncated, info = env.step(action)
        # Test with shape (1, env.A_max).
        action = np.zeros((1, env.A_max), dtype=np.float32)
        next_state, reward, terminated, truncated, info = env.step(action)

        assert env.current_step == 2

    def test_step_full_harvest(self, env):
        """Test stepping with maximum harvest."""

        env.reset()
        action = np.ones(env.A_max, dtype=np.float32)

        next_state, reward, terminated, truncated, info = env.step(action)

        assert next_state[0] == env.X_tot
        np.testing.assert_array_equal(next_state[1:], np.zeros(env.A_max - 1))

    def test_step_invalid_action_shape(self, env):
        """Test that step raises error for invalid action shapes."""

        # Invalid 2D shape.
        action = np.zeros((2, env.A_max), dtype=np.float32)
        with pytest.raises(ValueError, match="Incorrect action shape"):
            env.step(action)

        # Invalid 3D shape.
        action = np.zeros((1, 1, env.A_max), dtype=np.float32)
        with pytest.raises(ValueError, match="Incorrect action shape"):
            env.step(action)

    def test_get_random_state(self, env):
        """Test get_random_state method."""

        random_state = env.get_random_state()
        assert env.observation_space.contains(random_state)
        assert random_state.shape == (env.A_max,)
        assert np.isclose(np.sum(random_state), env.X_tot, atol=1e-5)

    def test_sink_state_behavior(self, env):
        """Test that oldest age class acts as sink."""

        # Create state with all forest in oldest age class.
        custom_state = np.zeros(env.A_max, dtype=np.float32)
        custom_state[-1] = env.X_tot
        options = {"reset_state": custom_state}
        env.reset(options=options)

        # Do not harvest.
        action = np.zeros(env.A_max, dtype=np.float32)
        next_state, _, _, _, _ = env.step(action)

        # Should remain in oldest age class.
        assert next_state[-1] == env.X_tot

    def test_age_class_shifting(self, env):
        """Test that age classes shift correctly."""

        # Put all forest in youngest age class.
        custom_state = np.zeros(env.A_max, dtype=np.float32)
        custom_state[0] = env.X_tot
        options = {"reset_state": custom_state}
        env.reset(options=options)

        # Do not harvest.
        action = np.zeros(env.A_max, dtype=np.float32)
        next_state, _, _, _, _ = env.step(action)

        # Forest should shift from age class 0 to age class 1
        assert next_state[0] == 0.0
        assert np.isclose(next_state[1], env.X_tot, atol=1e-5)
