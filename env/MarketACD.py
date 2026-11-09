from typing import Any

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

    def __init__(self, env_idx: int | None = None, discount_factor: float = 1.0):
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

        # Initial state will be set in self.reset.
        self.init_state = None
        self.state = None

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

    def _initialize_state(self):
        """Initialize the state."""

        if self.init_state is None:
            # We sample positive values for age classes 1-8 (uniformly random),
            # zeros for the rest.
            self.init_state = np.zeros(self.A_max, dtype=np.float32)
            uniform_values = self._np_random.uniform(0, 1, 8)
            uniform_values = uniform_values / np.sum(uniform_values) * self.X_tot
            self.init_state[:8] = uniform_values

    def get_random_state(self) -> np.ndarray:
        """Return a random state sampled from the observation space.

        Returns:
            np.ndarray: A random state.
        """

        state = self.observation_space.sample()
        state = state / np.sum(state) * self.X_tot
        return state

    def reset(self, seed: int | None = None, options: dict | list | None = None) -> tuple[np.ndarray, dict]:
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
        # Initialize state on first reset (after seeding via super().reset()).
        self._initialize_state()

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

class MarketACDCallback(BaseCallback):
    """An optional callback for the MarketACD environment that tracks and saves
    the following:
    - beta distribution parameters for a fixed state,
    - training losses,
    - estimated returns for the fixed state,
    - state visit frequencies in datasets throughout training (optional).
    On each episode end, prints out action beta distribution parameters and
    their running means for a certain fixed state. In addition, prints mean
    action for that state and estimated state value.
    """

    def __init__(self, track_visit_counts=False, bins=10):
        super().__init__()

        self.track_visit_counts = track_visit_counts
        self.bins = bins

        # To be set on the first call to on_episode_end.
        self.state = None
        self.A = None
        self.B = None

        self.losses = {"critic_loss": [], "actor_loss": []}
        self.return_estimates = []
        if self.track_visit_counts:
            self.visit_counts = np.zeros((15, self.bins), dtype=np.int32)
        else:
            self.visit_counts = None

    def on_dataset_creation(self, dataset: tf.data.Dataset):
        """Track the states visited in datasets throughout the training."""

        if not self.track_visit_counts: return

        states = np.array([row[0].numpy() for row in dataset])

        # Vectorized binning: clip values to [0, 1), then multiply by bins and convert to int.
        bin_indices = np.clip(states, 0, 1 - 1e-10) * self.bins
        bin_indices = bin_indices.astype(np.int32)

        for age_class in range(bin_indices.shape[1]):
            np.add.at(self.visit_counts[age_class], bin_indices[:, age_class], 1)

    def on_epoch_end(self, episode: int, logs: dict[str, Any] | None = None):
        self.losses["critic_loss"].append(logs["avg_losses"].get("critic_loss", None))
        self.losses["actor_loss"].append(logs["avg_losses"].get("actor_loss", None))

    def on_episode_end(self, episode: int, logs: dict[str, Any] | None = None):
        if self.state is None:
            # The state has X_tot/8 for age classes 1-8, zeros for the rest.
            self.state = np.zeros((1, self.env.get_attr("A_max")[0]), dtype=np.float32)
            self.state[0, 0:8] = self.env.get_attr("X_tot")[0] / 8.0

        # Save distribution parameters for the fixed state.
        action_distribution_params = self.agent.policy(self.state)
        distribution = self.agent.distribution(action_distribution_params)

        # Detect distribution type and extract parameters.
        dist_name = getattr(self.agent, "distribution_name", None)
        if dist_name and "beta" in dist_name.lower():
            # Beta distribution.
            A = distribution.concentration1.numpy().flatten()
            B = distribution.concentration0.numpy().flatten()
            if self.A is None:
                self.A = A.reshape(1, -1)
                self.B = B.reshape(1, -1)
            else:
                self.A = np.concatenate((self.A, A.reshape(1, -1)), axis=0)
                self.B = np.concatenate((self.B, B.reshape(1, -1)), axis=0)

            value = self.agent.value(self.state).numpy().flatten()[0]
            mean_actions = A / (A + B)

            with np.printoptions(precision=2, suppress=True):
                print(f"Episode {episode+1:02}: Beta distribution parameters for fixed state:")
                print(f"  Alpha: {A}")
                print(f"  Beta:  {B}")
                print(f"  Mean actions: {mean_actions}")
                print(f"  Mean parameters so far:")
                print(f"    Alpha {np.mean(self.A, axis=0)} ({np.std(self.A, axis=0)})")
                print(f"    Beta {np.mean(self.B, axis=0)} ({np.std(self.B, axis=0)})")
                print(f"  Estimated state value: {value:.2f}")
        elif dist_name and "logistic_normal" in dist_name.lower():
            # Logistic normal distribution - extract loc and scale from underlying Normal
            loc = distribution.distribution.loc.numpy().flatten()
            scale = distribution.distribution.scale.numpy().flatten()
            if self.A is None:
                self.A = loc.reshape(1, -1)
                self.B = scale.reshape(1, -1)
            else:
                self.A = np.concatenate((self.A, loc.reshape(1, -1)), axis=0)
                self.B = np.concatenate((self.B, scale.reshape(1, -1)), axis=0)

            value = self.agent.value(self.state).numpy().flatten()[0]
            # Mean of logistic normal: approximate using sigmoid(loc) for small scale.
            mean_actions = 1.0 / (1.0 + np.exp(-loc))

            with np.printoptions(precision=2, suppress=True):
                print(f"Episode {episode+1:02}: Logistic-Normal distribution parameters for fixed state:")
                print(f"  Loc:   {loc}")
                print(f"  Scale: {scale}")
                print(f"  Mean actions: {mean_actions}")
                print(f"  Mean parameters so far:")
                print(f"    Loc   {np.mean(self.A, axis=0)} ({np.std(self.A, axis=0)})")
                print(f"    Scale {np.mean(self.B, axis=0)} ({np.std(self.B, axis=0)})")
                print(f"  Estimated state value: {value:.2f}")
        else:
            # Unknown distribution type.
            raise ValueError(f"Unknown or unsupported distribution type: {dist_name}. Supported types are 'beta', 'beta_restricted', and 'logistic_normal'.")

        # Save estimated value for the fixed state.
        self.return_estimates.append(value)

    def on_train_end(self, logs: dict[str, Any] | None = None):
        """Save the collected distribution parameters and state visit
        frequencies to a file."""

        if self.artifact_manager is None:
            return

        # Register the artifacts and get the path generators.
        try:
            get_distribution_path = self.artifact_manager.register_artifact(
                artifact_identifier="distribution_parameters",
                artifact_description="MarketACD_distribution_parameters",
                file_name_generator=lambda x: "distribution_parameters.csv",
                check_conflicts=True
            )
            get_return_estimate_path = self.artifact_manager.register_artifact(
                artifact_identifier="return_estimates",
                artifact_description="MarketACD_return_estimates",
                file_name_generator=lambda x: "return_estimates.csv",
                check_conflicts=True
            )
            get_loss_path = self.artifact_manager.register_artifact(
                artifact_identifier="training_losses",
                artifact_description="MarketACD_training_losses",
                file_name_generator=lambda x: "training_losses.csv",
                check_conflicts=True
            )
            if self.track_visit_counts:
                get_visit_freq_path = self.artifact_manager.register_artifact(
                    artifact_identifier="visit_frequencies",
                    artifact_description="MarketACD_visit_frequencies",
                    file_name_generator=lambda x: "visit_frequencies.csv",
                    check_conflicts=True
                )
        except FileExistsError as e:
            raise e

        # Save distribution parameters.
        file_path = get_distribution_path(None)
        data = np.column_stack((self.A, self.B))

        # Set headers based on distribution type
        dist_name = getattr(self.agent, 'distribution_name', None)
        if dist_name and 'logistic_normal' in dist_name.lower():
            header_A = ",".join([f"loc{i+1}" for i in range(self.A.shape[1])])
            header_B = ",".join([f"scale{i+1}" for i in range(self.B.shape[1])])
        else:
            # Default to beta naming
            header_A = ",".join([f"A{i+1}" for i in range(self.A.shape[1])])
            header_B = ",".join([f"B{i+1}" for i in range(self.B.shape[1])])

        header = header_A + "," + header_B
        np.savetxt(file_path, data, delimiter=",", header=header, comments="", fmt=",".join("%.6f" for _ in range(self.A.shape[1] + self.B.shape[1])))

        # Save return estimates.
        file_path = get_return_estimate_path(None)
        header = "return_estimate"
        data = np.array(self.return_estimates).reshape(-1, 1)
        np.savetxt(file_path, data, delimiter=",", header=header, comments="", fmt="%.6f")

        # Save training losses.
        file_path = get_loss_path(None)
        header = "critic_loss,actor_loss"
        data = np.column_stack((np.array(self.losses["critic_loss"]), np.array(self.losses["actor_loss"])))
        np.savetxt(file_path, data, delimiter=",", header=header, comments="", fmt="%.6f,%.6f")

        # Save state frequencies.
        if self.track_visit_counts:
            file_path = get_visit_freq_path(None)
            header = ",".join([f"Bin{i}" for i in range(self.bins)])
            visit_freq = self.visit_counts / np.sum(self.visit_counts, axis=1, keepdims=True)
            np.savetxt(file_path, visit_freq, delimiter=",", header=header, comments="", fmt=",".join("%.6f" for _ in range(self.bins)))
