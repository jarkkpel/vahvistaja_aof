import argparse
import os
from typing import Union

# Set environment variables for deterministic TensorFlow behavior.
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tqdm

from agent import Agent
from env import VahvistajaEnvironment
from artifact_manager import ArtifactManager
from callbacks import DefaultCallback
from callback_manager import CallbackManager
from simulation_trace import SimulationTrace, simulate_trace
from util import *

def setup_environment(config):
    """Set up and create an environment based on the config."""

    env_name = config["environment"].get("name", None)                  # Name of the environment to use.
    if env_name is None:
        raise ValueError("Config must contain an 'environment.name' field.")
    N_env = config["environment"].get("N_env", 1)                       # How many parallel environments to use.
    discount_factor = config["environment"].get("discount_factor", 0.9) # Discount factor for future rewards.

    env = create_environment(env_name, N_envs=N_env, discount_factor=discount_factor)

    return env

def create_agent(config, env):
    """Creates and sets up the agent based on the config."""

    model_file = config["agent"]["file"]
    class_name = config["agent"]["class_name"]
    model_class = load_method(model_file, class_name)
    model_params = config["agent"].get("params", {})
    model_params["action_space"] = env.get_attr("action_space")[0]
    model_params["observation_space"] = env.get_attr("observation_space")[0]

    agent = model_class(**model_params)

    return agent

def train(
        agent: Agent,
        env: VahvistajaEnvironment,
        st_class: SimulationTrace,
        N_episodes: int,
        N_steps: int,
        reset_prob: float,
        epochs_per_episode: int,
        batch_size: int,
        callback_manager: Union[CallbackManager,None]=None
    ):
    """Trains the agent on the environment.

    Args:
        agent: The agent to train.
        env: The environment (possibly multiple) to train on.
        st_class: The SimulationTrace class to use in trace creation.
        N_episodes: Number of episodes to train.
        N_steps: Number of simulation steps per episode per environment.
        reset_prob: Probability of resetting to a random state.
        epochs_per_episode: Number of training epochs per episode.
        batch_size: Batch size for training.
        callback_manager: Callback manager to use during training. Defaults to None.
    """

    if N_episodes <= 0:
        raise ValueError("N_episodes must be positive.")
    if N_steps <= 0:
        raise ValueError("N_steps must be positive.")
    if not (0.0 <= reset_prob <= 1.0):
        raise ValueError("reset_prob must be in [0, 1].")
    if epochs_per_episode <= 0:
        raise ValueError("epochs_per_episode must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if callback_manager is None:
        callback_manager = CallbackManager(env, agent, [])

    # We compute traces and train on them sequentially.

    callback_manager.on_train_begin()

    N_env = env.num_envs
    env_init_state = env.get_attr("init_state")[0] # Initial state of the environment.

    N_total_steps = 0              # Total number of steps taken in the environment.
    episodes_estimated_return = [] # Tracks the estimated return of the initial state for each episode.
    min_episodes_criterion = 20    # How many previous estimated episode returns are considered when deciding convergence.

    episodes = tqdm.trange(N_episodes)
    for episode in episodes:
        callback_manager.on_episode_begin(episode)

        # Set the episode initial state. We start from a random initial state a
        # certain proportion of the time.
        init_state = list(env.call("get_random_state"))
        for i in range(N_env):
            if np.random.rand() >= reset_prob:
                init_state[i] = env_init_state

        # Simulate an episode trace.
        trace, final_state_value_estimate = simulate_trace(st_class, env, agent, N_steps, init_state=init_state)
        N_total_steps += trace.N_traces*trace.length

        # Preprocessing of the trace before batching.
        try:
            trace.preprocess_trace(agent, final_state_value_estimate)
        except NotImplementedError:
            pass

        # Create dataset.
        dataset = trace.get_as_dataset()

        # Train the agent on the trace.
        for epoch in range(epochs_per_episode):
            callback_manager.on_epoch_begin(epoch)

            # Shuffle the dataset.
            ds = dataset.shuffle(buffer_size=N_steps).batch(batch_size, drop_remainder=True)

            total_losses = {}
            batch_count = 0

            # Iterate over the batches and train the agent.
            for batch in ds:
                callback_manager.on_train_batch_begin(batch_count)

                losses = agent.train_on_batch(*batch)

                if len(total_losses) == 0:
                    for loss_name in losses:
                        total_losses[loss_name] = 0.0
                for loss_name in losses:
                    total_losses[loss_name] += losses[loss_name].numpy()

                batch_count += 1

                callback_manager.on_train_batch_end(batch_count, logs=losses)

            avg_losses = {loss_name: total / batch_count if batch_count > 0 else 0.0 for loss_name, total in total_losses.items()}

            logs = {
                "epochs_per_episode": epochs_per_episode,
                "avg_losses": avg_losses
            }
            callback_manager.on_epoch_end(epoch, logs=logs)

            # Check for early stopping at epoch level.
            if callback_manager.should_stop_training():
                break

        # Get the agent's current estimate of the true value for the initial state.
        normalized_estimated_value = agent.value(env.get_attr("init_state")[0]).numpy()[0]
        estimated_value = trace.unnormalize(normalized_estimated_value, "returns")

        # Progress is measured by the estimated value of the initial state.
        episodes_estimated_return.append(estimated_value)
        running_return = np.mean(episodes_estimated_return[-min_episodes_criterion:])

        # Display episode return and running return in the progress bar.
        episodes.set_postfix(episode_return=trace["returns"], running_return=running_return)

        logs = {
            "N_episodes": N_episodes,
            "N_total_steps": N_total_steps,
            "init_states": init_state,
            "returns": trace["returns"],
            "estimated_value": estimated_value,
        }
        callback_manager.on_episode_end(episode, logs=logs)

        # Check for early stopping at episode level.
        if callback_manager.should_stop_training():
            break

    # Training finished
    callback_manager.on_train_end()
    env.close()

    return episode, running_return

def main(config, seed=None):
    use_gpu = config.get("use_gpu", False)
    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")

    # Set seed for reproducibility.
    seed = seed if seed is not None else 0
    set_seed(seed)

    # Set up and create the environment.
    # -------------------------------------------------------------------------
    env = setup_environment(config) # The environment.
    N_env = env.num_envs            # Number of parallel environments.

    # Episode / training batch settings.
    # -------------------------------------------------------------------------
    N_episodes = config.get("N_episodes", 500)                # How many episodes to train.
    N_steps = config.get("N_steps", 1024)                     # How many steps per episode.
    if not N_steps % N_env == 0:
        raise ValueError(f"N_steps ({N_steps}) must be divisible by N_env ({N_env}).")
    # Divide number of steps by the number of environments if specified in config (default: True).
    if config.get("divide_steps", True):
        N_steps = int(N_steps // N_env)
    reset_prob = config.get("reset_prob", 0.0)                # What proportion of episodes should start from a random state (default: 0).
    epochs_per_episode = config.get("epochs_per_episode", 10) # How many training epochs per episode (default: 10).
    batch_size = config.get("batch_size", 32)                 # Trace batch size (default: 32).

    # Get the appropriate SimulationTrace class.
    # -------------------------------------------------------------------------
    SimulationTraceClass = load_method(config["simulation_trace"]["file"], config["simulation_trace"]["class_name"])

    # Create the agent.
    # -------------------------------------------------------------------------
    agent = create_agent(config, env)

    # Set up artifact manager and check for existing artifacts.
    # -------------------------------------------------------------------------
    artifact_manager = ArtifactManager(config, seed)
    artifact_manager.ensure_no_conflicts(agent)

    # Set up the callbacks.
    # -------------------------------------------------------------------------
    callback_manager = CallbackManager(env=env, agent=agent)

    # Add default callback first.
    default_callback = DefaultCallback()
    callback_manager.add_callback(default_callback)

    # Add configured callbacks.
    if "callbacks" in config:
        for callback_data in config["callbacks"]:
            if not "class_name" in callback_data:
                raise ValueError("Each callback must have 'class_name' field.")
            # Use callbacks.py as default file if not specified.
            callback_file = callback_data.get("file", "callbacks.py")
            callback_class = load_method(callback_file, callback_data["class_name"])
            callback_params = callback_data.get("params", {})
            callback_instance = callback_class(**callback_params)
            callback_manager.add_callback(callback_instance)

    # Train the agent on the environment.
    # -------------------------------------------------------------------------
    episode, running_return = train(
        agent,
        env,
        SimulationTraceClass,
        N_episodes,
        N_steps,
        reset_prob,
        epochs_per_episode,
        batch_size,
        callback_manager
    )

    if episode < N_episodes - 1:
        print(f"\nSolved at episode {episode+1}: average return: {running_return:.2f}!")
    else:
        print(f"\nNot solved after {N_episodes} episodes: average return: {running_return:.2f}.")

    # Save the model.
    artifact_manager.save_training_artifacts(agent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--config", type=str, default=join_paths("config", "default_faustmann.yaml"), help="Path to the config file")
    parser.add_argument("--config", type=str, default=join_paths("config", "default_marketacd.yaml"), help="Path to the config file")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (default 0)")
    args = parser.parse_args()

    config = load_config(args.config)
    if config is None:
        raise ValueError("No config found.")
    main(config, seed=args.seed if args.seed >= 0 else None)
