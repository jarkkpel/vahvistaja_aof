from typing import Type

import numpy as np
import tensorflow as tf

from agent import Agent
from env import VahvistajaEnvironment

class SimulationTrace:
    """A class to store multiple simulation traces."""

    # The idea is that we store all parallel traces separately, but return them
    # as one big batch once requested. This class is supposed to be subclassed
    # depending on the particular agent model.

    def __init__(
        self,
        agent: Agent,
        N_traces: int,
        length: int,
        discount_factor: float = 1.0
    ):
        """Initialize the simulation trace.

        Args:
            agent: The agent used for training.
            N_traces: Number of parallel traces.
            length: Length of each trace.
            discount_factor: Discount factor used in the environment. Defaults to 1.0.
        """

        self.agent = agent
        self.N_traces = N_traces
        self.length = length
        self.state_dim = agent.state_dim
        self.action_dim = agent.action_dim
        self.action_distribution_params_dim = agent.action_distribution_params_dim
        self.discount_factor = discount_factor

        self.epsilon = agent.epsilon

        # Used for saving normalization statistics.
        self.normalization_stats = {}

        # Inhereting class should set these up.
        self.states = None
        self.actions = None
        self.rewards = None
        self.terminateds = None
        self.action_distribution_params = None
        self.returns = None
        self.state_value_estimates = None

    def __getitem__(self, key: str) -> np.ndarray:
        match key:
            case "states":
                return self.states
            case "actions":
                return self.actions
            case "rewards":
                return self.rewards
            case "terminateds":
                return self.terminateds
            case "action_distribution_params":
                return self.action_distribution_params
            case "returns":
                return self.returns
            case "state_value_estimates":
                return self.state_value_estimates
            case _:
                raise KeyError(f"Unknown key: {key}")

    def __setitem__(self, key: str, value: np.ndarray):
        match key:
            case "state_value_estimates":
                self.state_value_estimates = value
            case _:
                raise KeyError(f"Unknown or read-only key: {key}")

    def append(
        self,
        t: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        action_distribution_params: np.ndarray,
        terminated: np.ndarray
    ):
        """Append data for time step t.

        Args:
            t: Time step index.
            state: State at time step t. Shape (N_traces, state_dim).
            action: Action taken at time step t. Shape (N_traces, action_dim).
            reward: Reward received at time step t. Shape (N_traces,).
            action_distribution_params: Action distribution parameters at time step t. Shape (N_traces, action_distribution_params_dim).
            terminated: Whether the episode terminated at time step t. Shape (N_traces,).
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def preprocess_trace(self, agent: Agent, next_state_value: np.ndarray):
        """Preprocess the trace data before batching it for training.

        Args:
            agent: The agent used for training.
            next_state_value: The estimated value of the state following the last state in the trace. Shape (N_traces,).
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def get_as_dataset(self) -> tf.data.Dataset:
        """Return the trace data as a TensorFlow dataset for batched training.

        Returns:
            The dataset containing the trace data.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def normalize(self) -> dict:
        """Normalize trace data for training and return normalization statistics.

        This method should normalize relevant trace data (e.g., advantages, returns)
        according to the agent's normalization settings. The normalized data should
        be stored back in the trace object, and statistics about the normalization
        should be returned.

        Returns:
            Dictionary containing normalization statistics with keys like 'advantages',
            'returns', etc. Each entry should contain 'mean' and 'std' values used
            for normalization.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def unnormalize(self, normalized_values: tf.Tensor | np.ndarray, what: str) -> np.ndarray:
        """Unnormalize internally normalized values.

        Args:
            normalized_values: The normalized values to unnormalize.
            what: A string indicating what type of values to unnormalize and return.
        Returns:
            The unnormalized values.
        """

        raise NotImplementedError("Subclasses must implement this method.")

def simulate_trace(
    st_class: Type[SimulationTrace],
    env: VahvistajaEnvironment,
    agent: Agent,
    N_steps: int,
    init_state: np.ndarray
) -> tuple[SimulationTrace, np.ndarray]:
    """Computes a simulation trace for the given number of steps.

    Arguments:
        st_class: The SimulationTrace class to use.
        env: The environment to use (possibly multiple).
        agent: The agent whose policy and value function to use.
        N_steps: Number of steps for the episode.
        init_state: Initial state (possibly multiple).

    Returns:
        An instance of SimulationTrace."""

    discount_factor = env.get_attr("discount_factor")[0]

    # Reset the environments.
    options_list = [{"reset_state": s} for s in init_state]
    state, _ = env.reset(options=options_list)

    # Create the trace object.
    N_traces = env.num_envs
    trace = st_class(
        agent=agent,
        N_traces=N_traces,
        length=N_steps,
        discount_factor=discount_factor
    )

    for t in range(N_steps):
        # Get the agent's action.
        action_distribution_params = agent.policy(state).numpy()
        action_distribution = agent.distribution(action_distribution_params)
        action = action_distribution.sample().numpy()

        # Take the action.
        next_state, reward, terminated, _, _ = env.step(action)

        # Append to the trace.
        trace.append(
            t,
            state=state,
            action=action,
            reward=reward,
            action_distribution_params=action_distribution_params,
            terminated=terminated
        )

        # Update state.
        state = next_state

        # TODO: Figure this out for multiple environments.
        #if terminated:
        #    # TODO: This currently never happens. Should we reset or
        #    # terminate the episode?
        #    state, _ = env.reset()

    # Compute the estimated (unnormalized) state-value function values in a
    # batch for efficiency.
    state_value_estimates = agent.value(trace["states"], normalize=False)
    trace["state_value_estimates"] = state_value_estimates.numpy()

    # Estimate the value of the next state.
    final_state_value_estimate = agent.value(state, normalize=False).numpy()[0]

    return trace, final_state_value_estimate
