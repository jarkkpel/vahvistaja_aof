"""
Proximal Policy Optimization (PPO) agent implementation. Follows largely the paper

    J. Schulman et al.: Proximal policy optimization algorithms (2017). https://arxiv.org/abs/1707.06347

and implements the agent according to the actor-critic style.
"""

from typing import Union

import numpy as np
import os

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
tfd = tfp.distributions

from simulation_trace import SimulationTrace

from agent import Agent

@tf.keras.utils.register_keras_serializable(package="agent.ppo")
class Actor(tf.keras.Model):
    """The actor model that outputs action logits."""

    def __init__(self, num_actions, **kwargs):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        #self.layer1 = layers.Dense(64, activation='leaky_relu')
        #self.layer2 = layers.Dense(64, activation='leaky_relu')
        self.layer1 = layers.Dense(64, activation='tanh')
        self.layer2 = layers.Dense(64, activation='tanh')
        #self.bn1 = layers.BatchNormalization()
        #self.bn2 = layers.BatchNormalization()
        self.logits = layers.Dense(num_actions, activation='linear', name='policy_logits')

    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        #x = self.layer1(self.bn1(state))
        #x = self.layer2(self.bn2(x))
        return self.logits(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_actions": self.num_actions
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="agent.ppo")
class Critic(tf.keras.Model):
    """The critic model that estimates the state-value function."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.layer1 = layers.Dense(64, activation='leaky_relu')
        #self.layer2 = layers.Dense(64, activation='leaky_relu')
        self.layer1 = layers.Dense(64, activation='tanh')
        self.layer2 = layers.Dense(64, activation='tanh')
        #self.bn1 = layers.BatchNormalization()
        #self.bn2 = layers.BatchNormalization()
        self.value = layers.Dense(1, activation='linear', name='value')

    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        #x = self.layer1(self.bn1(state))
        #x = self.layer2(self.bn2(x))
        return self.value(x)

class PPOAgent(Agent):
    """Proximal Policy Optimization (PPO) agent."""

    def __init__(
        self,
        action_space,
        observation_space,
        distribution=None,
        GAE_lambda=0.9,
        PPO_clip=0.25,
        actor_lr=3e-4,
        critic_lr=1e-4,
        momentum=0.9,
        kl_beta=0.001,
        use_kl_divergence=False
    ):
        """Initialize the PPO agent.

        Args:
            action_space: The action space of the environment.
            observation_space: The observation space of the environment.
            distribution: The action distribution callable or string name for continuous action spaces. Defaults to None.
            GAE_lambda: The lambda parameter for Generalized Advantage Estimation (GAE). Defaults to 0.9.
            PPO_clip: The clipping parameter for PPO. Defaults to 0.25.
            actor_lr: The learning rate for the actor network. Defaults to 3e-4.
            critic_lr: The learning rate for the critic network. Defaults to 1e-4.
            momentum: The momentum for the optimizer. Defaults to 0.9.
            kl_beta: The coefficient for the KL divergence/entropy penalty. Defaults to 0.001.
            use_kl_divergence: Whether to use KL divergence (True) or entropy (False) for regularization. Defaults to True.
        """

        super().__init__(action_space=action_space, observation_space=observation_space, distribution=distribution)
        self.GAE_lambda = GAE_lambda                # GAE lambda parameter.
        self.PPO_clip = PPO_clip                    # PPO clipping parameter.
        self.actor_lr = actor_lr                    # Actor learning rate.
        self.critic_lr = critic_lr                  # Critic learning rate.
        self.momentum = momentum                    # ema_momentum in Adam.
        self.kl_beta = kl_beta                      # Soft coefficient for the KL divergence/entropy penalty.
        self.use_kl_divergence = use_kl_divergence  # Whether to use KL divergence (True) or entropy (False) for regularization.

        self.actor = Actor(self.action_distribution_params_dim)
        self.critic = Critic()

        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)

        self.eps = np.finfo(np.float32).eps.item()

    @classmethod
    def get_save_file_names(cls) -> dict:
        """Get the filenames for saving the agent.

        Returns:
            Dictionary with keys 'actor' and 'critic' mapping to their respective filenames.
        """

        return {"actor": "actor.keras", "critic": "critic.keras"}

    def save(self, save_path: str):
        """Save the agent.

        Args:
            save_path: The path to save the agent to.
        """

        file_names = self.get_save_file_names()
        actor_file_name = os.path.join(save_path, file_names["actor"])
        critic_file_name = os.path.join(save_path, file_names["critic"])

        self.actor.save(actor_file_name)
        self.critic.save(critic_file_name)

    @classmethod
    def load(cls, load_path: str, agent_parameters: dict) -> object:
        """Load a saved agent.

        Args:
            load_path: The path where the agent was saved to.
            agent_parameters: The parameters needed to initialize the agent.
        Returns:
            The loaded agent instance.
        """

        file_names = cls.get_save_file_names()
        actor_file_name = os.path.join(load_path, file_names["actor"])
        critic_file_name = os.path.join(load_path, file_names["critic"])

        # Create a new agent.
        agent = cls(**agent_parameters)

        # Load the saved models.
        agent.actor = tf.keras.models.load_model(actor_file_name)
        agent.critic = tf.keras.models.load_model(critic_file_name)

        return agent

    def _preprocess_state(self, state: Union[tf.Tensor, np.ndarray]) -> tuple[tf.Tensor, tuple]:
        """Preprocess state input to consistent tensor format.

        Args:
            state: The state of the environment in shape (b, d), (d,), or (b, k, d).
        Returns:
            Tuple containing:
                - The preprocessed state tensor in shape (b, d) or (b*k, d) when applicable.
                - The original shape of the input state.
        """

        # Store the original shape.
        if isinstance(state, np.ndarray):
            original_shape = state.shape
        elif isinstance(state, tf.Tensor):
            original_shape = tuple(state.shape.as_list())
        else:
            original_shape = None

        if isinstance(state, tf.Tensor):
            state_tensor = tf.cast(state, tf.float32)
            if len(state_tensor.shape) == 1:
                state_tensor = tf.expand_dims(state_tensor, axis=0) # shape: (1, d)
            elif len(state_tensor.shape) > 2:
                state_tensor = tf.reshape(state_tensor, (-1, state_tensor.shape[-1])) # shape: (b*k, d)
        elif isinstance(state, np.ndarray):
            if state.ndim == 1:
                state = state.reshape(1, -1) # shape: (1, d)
            elif state.ndim > 2:
                state = state.reshape(-1, state.shape[-1]) # shape: (b*k, d)
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}.")

        return state_tensor, original_shape

    def _reshape_to_original(self, output: tf.Tensor, original_shape: tuple) -> tf.Tensor:
        """Reshape output back to match the original input's batch dimensions.

        Args:
            output: The output tensor in shape (b*k, d) or (b, d).
            original_shape: The original shape of the input state.
        Returns:
            The reshaped output matching the original batch dimensions.
        """

        if original_shape is not None and len(original_shape) == 3:
            # Reshape from (b*k, d) to (b, k, d).
            return tf.reshape(output, (original_shape[0], original_shape[1], -1))

        return output

    @tf.function
    def _policy_inference(self, state_tensor):
        return self.actor(state_tensor)

    @tf.function
    def _value_inference(self, state_tensor):
        return self.critic(state_tensor)

    def policy(self, state: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        """Returns action distribution parameters for the given state.

        Args:
            state: The state of the environment in shape (state_dim,), (N_steps, state_dim), or (N_steps, N_traces, state_dim).
        Returns:
            The action distribution parameters for the given state with shape matching the input.
        """

        state_tensor, original_shape = self._preprocess_state(state)
        action_distribution_params = self._policy_inference(state_tensor)
        action_distribution_params = self._reshape_to_original(action_distribution_params, original_shape)

        return action_distribution_params

    def value(self, state):
        """Returns the estimated state value for the given state or batch of states.

        Args:
            state: The state of the environment in shape (state_dim,), (N_steps, state_dim), or (N_steps, N_traces, d).
        Returns:
            The value estimate for the given state with shape:
                - (N_steps,) for input shape (N_steps, state_dim)
                - (1,) for input shape (state_dim,)
                - (N_steps, N_traces) for input shape (N_steps, N_traces, state_dim)
        """

        state_tensor, original_shape = self._preprocess_state(state)
        value_estimates = self._value_inference(state_tensor)
        value_estimates = self._reshape_to_original(value_estimates, original_shape)
        value_estimates = tf.squeeze(value_estimates, axis=-1)

        return value_estimates

    @tf.function
    def train_on_batch(self, states: tf.Tensor, actions: tf.Tensor, action_distribution_params: tf.Tensor, advantages: tf.Tensor, returns: tf.Tensor, *args: tf.Tensor, **kwargs):
        """Train the actor and the critic on a batch of training data.

        Args:
            states: Batch of states, shape (batch_size, state_dim).
            actions: Batch of actions taken, shape (batch_size, action_dim).
            action_distribution_params: Batch of action distribution parameters used when actions were taken, shape (batch_size, action_distribution_params_dim).
            advantages: Batch of estimated advantages, shape (batch_size,).
            returns: Batch of estimated returns, shape (batch_size,).
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).
        Returns:
            Dictionary containing the actor and critic losses.
        """

        # states shape: (batch_size, state_dim)
        # actions shape: (batch_size, action_dim)
        # action_distribution_params shape: (batch_size, action_distribution_params_dim)
        # advantages shape: (batch_size, )
        # returns shape: (batch_size, )

        # Train the critic.
        # ---------------------------------------------------------------------
        # Compute loss and track the gradient.
        with tf.GradientTape() as critic_tape:
            state_value_estimates = tf.squeeze(self.critic(states)) # shape: (batch_size, )

            # MSE loss.
            critic_loss = tf.reduce_mean(tf.square(returns - state_value_estimates))

        # Compute and apply the gradient.
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # Train the actor.
        # ---------------------------------------------------------------------
        # Compute loss and track the gradient.
        with tf.GradientTape() as actor_tape:
            # Get new action distribution parameters from the actor model.
            new_action_distribution_params = self.actor(states) # shape: (batch_size, action_distribution_params_dim)

            new_dist = self.distribution(new_action_distribution_params)
            old_dist = self.distribution(action_distribution_params)

            # Calculate the probability ratio.
            new_log_probs = new_dist.log_prob(actions)
            old_log_probs = old_dist.log_prob(actions)
            # For multi-dimensional actions, sum log probabilities across
            # action dimensions to get the total log probability of the joint
            # action.
            new_log_probs = tf.reduce_sum(new_log_probs, axis=-1)
            old_log_probs = tf.reduce_sum(old_log_probs, axis=-1)
            ratio = tf.exp(new_log_probs - old_log_probs)

            # Calculate the clipped surrogate objective.
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.PPO_clip, 1 + self.PPO_clip)
            policy_objective = tf.minimum(ratio * advantages, clipped_ratio * advantages)

            # Choose between KL divergence and entropy.
            if self.use_kl_divergence:
                # Compute the average of the observed Kullback-Leibler divergences.
                # For multi-dimensional actions, sum KL divergences across action dimensions.
                kl_per_action = old_dist.kl_divergence(new_dist)
                kl_per_action = tf.reduce_sum(kl_per_action, axis=-1)
                kl_divergence = tf.reduce_mean(kl_per_action)
                regularization_term = kl_divergence
            else:
                # Compute the entropy of the new distribution.
                entropy = tf.reduce_mean(tf.reduce_sum(new_dist.entropy(), axis=-1))
                regularization_term = entropy

            # The Actor loss is the negative of the objective to be maximized plus the regularization term.
            actor_loss = -(tf.reduce_mean(policy_objective) + self.kl_beta * regularization_term)

        # Calculate gradients for the Actor
        actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        return {"actor_loss": actor_loss, "critic_loss": critic_loss}

class PPOSimulationTrace(SimulationTrace):
    """Simulation trace data structure for PPO agent training."""

    def __init__(
            self,
            N_traces,
            length,
            state_dim,
            action_dim,
            action_distribution_params_dim,
            discount_factor=1
        ):
        """Initialize the simulation trace.

        Args:
            N_traces: Number of parallel traces.
            length: Length of each trace.
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
            action_distribution_params_dim: Dimension of the action distribution parameters.
            discount_factor: Discount factor used in the environment. Defaults to 1.
        """

        super().__init__(N_traces, length, state_dim, action_dim, action_distribution_params_dim, discount_factor)

        self.states = np.zeros((length, N_traces, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((length, N_traces, action_dim), dtype=np.float32)
        self.rewards = np.zeros((length, N_traces), dtype=np.float32)
        self.terminateds = np.zeros((length, N_traces), dtype=np.bool)
        self.action_distribution_params = np.zeros((length, N_traces, self.action_distribution_params_dim), dtype=np.float32)

        self.returns = np.zeros(N_traces, dtype=np.float32)

        # These are computed on demand.
        self.state_value_estimates = None
        self.estimated_advantages = None
        self.estimated_returns = None

        self.estimated_advantages_mean = 0.0
        self.estimated_advantages_std = 1.0
        self.estimated_returns_mean = 0.0
        self.estimated_returns_std = 1.0

    def __getitem__(self, key):
        match key:
            case "state_value_estimates":
                value = super().__getitem__(key)
                if value is None:
                    raise ValueError("State value estimates have not been computed.")
                return value
            case "estimated_advantages":
                if self.estimated_advantages is None:
                    raise ValueError("Advantages have not been computed.")
                return self.estimated_advantages
            case "estimated_returns":
                if self.estimated_returns is None:
                    raise ValueError("Returns have not been computed.")
                return self.estimated_returns
            case _:
                try:
                    return super().__getitem__(key)
                except KeyError:
                    raise KeyError(f"Unknown key: {key}")

    def append(self, t: int, state: np.ndarray, action: np.ndarray, reward: np.ndarray, action_distribution_params: np.ndarray, terminated: np.ndarray):
        """Append data for time step t.

        Args:
            t: Time step index.
            state: State at time step t. Shape (N_traces, state_dim).
            action: Action taken at time step t. Shape (N_traces, action_dim).
            reward: Reward received at time step t. Shape (N_traces,).
            action_distribution_params: Action distribution parameters at time step t. Shape (N_traces, action_distribution_params_dim).
            terminated: Whether the episode terminated at time step t. Shape (N_traces,).
        """

        self.states[t] = state
        self.actions[t] = action
        self.rewards[t] = reward
        self.terminateds[t] = terminated
        self.action_distribution_params[t] = action_distribution_params

        self.returns += reward * self.discount_factor**t

    def preprocess_trace(self, agent: Agent, next_state_value: np.ndarray):
        """Preprocess the trace data before batching it for training.

        Args:
            agent: The agent used for training.
            next_state_value: The estimated value of the state following the last state in the trace. Shape (N_traces,).
        """

        self.compute_advantages_and_returns(GAE_lambda=agent.GAE_lambda, next_state_value=next_state_value)

    def get_as_dataset(self):
        """Return the trace data as a TensorFlow dataset for batched training.

        Returns:
            The dataset containing the trace data.
        """

        if self.estimated_advantages is None or self.estimated_returns is None:
            raise ValueError("Advantages and returns have not been computed.")

        # More efficient flattening - flatten all arrays in one operation
        # and create tensors directly with proper shapes to minimize reshaping in train_on_batch
        total_samples = self.length * self.N_traces

        # Flatten and create tensors in one step with target shapes
        states_flat = tf.convert_to_tensor(self.states.reshape(total_samples, -1), dtype=tf.float32)
        actions_flat = tf.convert_to_tensor(self.actions.reshape(total_samples, -1), dtype=tf.float32)
        action_distribution_params_flat = tf.convert_to_tensor(self.action_distribution_params.reshape(total_samples, -1), dtype=tf.float32)
        advantages_flat = tf.convert_to_tensor(self.estimated_advantages.reshape(total_samples), dtype=tf.float32)
        returns_flat = tf.convert_to_tensor(self.estimated_returns.reshape(total_samples), dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((
            states_flat,
            actions_flat,
            action_distribution_params_flat,
            advantages_flat,
            returns_flat
        ))

        return dataset

    def unnormalize(self, normalized_values: Union[tf.Tensor,np.ndarray], what: str) -> np.ndarray:
        """Unnormalize internally normalized values.

        Args:
            normalized_values: The normalized values to unnormalize.
            what: A string indicating what type of values to unnormalize and return.
        Returns:
            The unnormalized values.
        """

        if what == "advantages":
            if self.estimated_advantages_mean is None or self.estimated_advantages_std is None:
                raise ValueError("Advantages have not been computed.")
            return normalized_values * (self.estimated_advantages_std + 1e-8) + self.estimated_advantages_mean
        elif what == "returns":
            if self.estimated_returns_mean is None or self.estimated_returns_std is None:
                raise ValueError("Returns have not been computed.")
            return normalized_values * (self.estimated_returns_std + 1e-8) + self.estimated_returns_mean
        else:
            raise ValueError(f"Unknown what: {what}. Must be either 'advantages' or 'returns'.")

    def compute_advantages_and_returns(self, GAE_lambda: float = 1, next_state_value: Union[np.ndarray,float] = 0.0):
        """Given that the trace has been completed and the next state value
        function value (can be an estimate, default 0), estimates its advantage
        values and returns. The advantages are estimated in the generalized
        advantage estimation (GAE) fashion. The returns are estimated as one
        step estimates with the discounted part estimated by the provided state
        value estimates.

        Args:
            GAE_lambda: The lambda parameter for GAE. Defaults to 1.
            next_state_value: The estimated value of the state following the
            last state in the trace. Defaults to 0.0."""

        rewards = self.rewards
        terminateds = self.terminateds
        if self.state_value_estimates is None:
            raise ValueError("State value estimates have not been computed.")
        state_value_estimates = self.state_value_estimates

        self.estimated_advantages = np.zeros((self.length, self.N_traces), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self.length)):
            # mask is 0 if terminated at time t, 1 otherwise
            mask = 1.0 - terminateds[t].astype(np.float32)

            next_value_estimates = state_value_estimates[t+1] if t < self.length - 1 else next_state_value

            # Compute the TD error, which estimates the advantage.
            delta = rewards[t] + self.discount_factor*next_value_estimates*mask - state_value_estimates[t]

            # Calculate the GAE estimate for the advantage.
            gae = delta + self.discount_factor*GAE_lambda*mask * gae

            self.estimated_advantages[t] = gae

        self.estimated_returns = self.estimated_advantages + state_value_estimates

        # Save as normalized and save means and stds for later unnormalization.
        self.estimated_advantages_mean = np.mean(self.estimated_advantages)
        self.estimated_advantages_std = np.std(self.estimated_advantages)
        self.estimated_advantages = (self.estimated_advantages - self.estimated_advantages_mean) / (self.estimated_advantages_std + 1e-8)
        self.estimated_returns_mean = np.mean(self.estimated_returns)
        self.estimated_returns_std = np.std(self.estimated_returns)
        self.estimated_returns = (self.estimated_returns - self.estimated_returns_mean) / (self.estimated_returns_std + 1e-8)
