# Documentation
This is a brief code documentation, and it is subject to change.

# Solving a Forest Rotation Problem
The purpose of the script `train.py` is to solve a forest rotation problem
defined by a configuration file. Default configurations are found in `config/`
in YAML format. In the next sections, we go through the various parts of a
configuration file.

## General Settings
```yaml
run_name: default
output_path: "../runs/{run_name}"
use_gpu: true
```
This part specifies the run name that is used when saving artifacts (see below)
related to a run of `train.py`. The variable `output_path` defines where these
artifacts are saved. The setting `use_gpu` specifies whether to use GPU
acceleration or not (setting it to true does not guarantee that GPU is used; it
needs to be properly configured).

## Environment Settings
```yaml
environment:
  name:            "Faustmann-v0"
  discount_factor: 0.97
  N_env:           16
```
This specifies that the registered environment 'Faustmann-v0' should be used.
The registered environments are defined in `env/__init__.py`. The parameter `discount_factor` specifies (globally) the discount factor to be used for the RL problem. The parameter `N_env` (default 1) specifies how many parallel environment instances are used in trace generation in training.

## Agent Settings
```yaml
agent:
  file: "agent/ppo.py"
  class_name: "PPOAgent"
  params:
    GAE_lambda: 0.90
    actor_lr:   0.0003
    critic_lr:  0.0001
    PPO_clip:   0.25
    kl_beta:    0.001
    momentum:   0.9
```
This part defines the agent to be used (see below for details). The parameters
simply define what file and what class should be instantiated and with what
parameters.

## Simulation Trace Settings
```yaml
simulation_trace:
  file: "agent/ppo.py"
  class_name: "PPOSimulationTrace"
```
A simulation trace is an object that records the results of the agent (possibly
several) when started from a state. Different kinds of agents might need to
store different sorts of information, so this class can be customized as
necessary. The above settings simply specify what class should be loaded and
from which file. See below for more on simulation traces.

## Training Settings
```yaml
N_episodes: 500
N_steps: 1024
divide_steps: true
reset_prob: 0.9
epochs_per_episode: 10
batch_size: 64
```
Finally, the above specifies the details of the agent training. The number
`N_episodes` specifies for how many episodes the agent is trained (this can be
shortened by using callbacks; see below). The number `N_steps` controls how
many steps a simulation from a state is run. If `divide_steps` is true, the
actual number of steps run is `N_steps / N_env`, where `N_env` is the number of
parallel environments as defined above. The number `reset_prob` indicates what
proportion of episode initial states should be randomly sampled from the
environment's state space. The parameter `epochs_per_episode` indicates how
many training epochs are run per episode, and `batch_size` indicates into how
many batches a simulation trace should be divided.

## Callback Settings
```yaml
callbacks:
  - file: env/FaustmannDiscreteTime.py
    class_name: FaustmannCallback
  - class_name: ThresholdEarlyStop
    params:
      target_return: 3000
      patience: 20
```
Callbacks are objects whose methods are called at various stages of the training. They can be used to collect information throughout the training, to modify the training process while it is ongoing, or to stop the training early if certain criteria are met. See below for more details. The above settings define two callbacks `FaustmannCallback` loaded from the specified file and `ThresholdEarlyStop` defined in the default callback module `callbacks.py`. Parameters for the callbacks can be specified in the `params` section.

# Environments
Here we describe the environment classes and how custom environments should be
implemented. The environments used are essentially Gymnasium environments [1]
with minor extra functionality. For convenience, we offer the class
`VahvistajaEnvironment` that implements this extra functionality. The user needs
to ensure that their class constructor takes parameters `env_idx` (default
`None`) and `discount_factor` (default `1.0`) and passes them to the superclass
constructor. Moreover, the user has to implement the method `get_random_state()` that returns a sample from the environment's observation space (state space). The training code instantiates several parallel environments and groups them as one `VectorEnv` object. Whenever the environments are reset, each environment gets the reset information for all parallel environments. The attribute `self.env_idx` should be used to pick the correct reset information for the specific environment instance. See the file `env/FaustmannDiscreteTime.py` for an example implementation.

[1] M. Towers et al.: Gymnasium: A standard interface for reinforcement learning environments. arXiv:2407.17032 (2024).

# Agents
The agent is on object that inherits from the `Agent` class. The agent takes the following initialization parameters:
- action_space: The action space of the environment.
- observation_space: The observation space (state space) of the environment.
- distribution: The action distribution to be used by the agent.

The distribution argument is a string that specifies a continuous action
distribution to be used (e.g., `"normal"`, `"beta"`). If it is `None`, a
discrete categorical distribution is assumed.

An agent needs to define the following methods:
- policy(state): Returns the action distribution parameters for the given state(s).
- value(state): Returns the state value estimate for the given state(s). By
  default the returned value is not normalized in any way. A normalized value
  can be returned by setting the argument `normalized=True` in case the agent
  implements value normalization.
- train_on_batch(*args): Trains the agent on a batch of training data. The
  training data consists of Tensorflow tensors from a simulation trace
  `get_as_dataset` method call.
- get_save_file_names: Returns the file names for the files in which the agent
  should be saved.
- load(load_path): Load the agent from a file.
- save(save_path): Save the agent to a file.

For the exact argument types and return types, see the `Agent` class in
`agent/agent.py`. An agent for the Proximal Policy Optimization Algorithm is imeplemented in `agent/ppo.py` as `PPOAgent`.

# Simulation Traces
Simulation trace is an object that records the actions, rewards, etc. of an agent when started from a state. A simulation trace class should inherit from the `SimulationTrace` class defined in `simulation_trace.py`. This base class initializer takes the following arguments:
- agent: The agent whose actions are being recorded.
- N_traces: Number of parallel traces to be recorded (corresponds to the number of parallel environments).
- length: Common length of each trace.
- discount_factor: Discount factor used in the environment(s).

The simulation trace class needs to implement the following methods:
- append(t, state, action, reward, action_distribution_params, done): Appends
  data from simulation time step t to the trace. The arguments are Numpy arrays
  of appropriate dimensions. For example, `state` has shape `(N_traces,
  state_dim)`.
- get_as_dataset(): Returns the recorded data as a Tensorflow dataset whose
  batches can be used for training the agent via the agent's `train_on_batch`
  method.

Optionally, the simulation trace class can implement the following methods:
- preprocess_trace(agent, final_state_value_estimate): Preprocess the recorded data before the method `get_as_dataset` is called. The array `final_state_value_estimate` is a Numpy array of shape `(N_traces,)` that contains the agent's estimate of the agent's final state's value.

The recorded data is accessed via the bracket notation, e.g., `trace["states"]` returns the recorded states as a Numpy array of shape `(length, N_traces, state_dim)`.

For an example implementation of a simulation trace, see `agent/ppo.py` where the class `PPOSimulationTrace` for the Proximal Policy Optimization algorithm is defined.

# Callbacks
Callbacks are objects whose methods are called at various stages of the
training, and they can thus be used to record information during training,
modify the training process, or stop the training early if certain criteria are
met.

A callback class should inherit from the `Callback` class defined in
`callbacks.py`. The methods that are called during the training are as follows:
- on_train_begin(): Called at the beginning of training.
- on_train_end(): Called at the end of training.
- on_episode_begin(episode): Called at the beginning of each episode.
- on_episode_end(episode, logs): Called at the end of each episode. The `logs`
  argument is a dictionary containing information about the episode, e.g., the
  estimated return of the initial state.
- on_epoch_begin(epoch): Called at the beginning of each training epoch.
- on_epoch_end(epoch, logs): Called at the end of each training epoch. The
  `logs` argument is a dictionary information about the epoch, e.g., average
  epoch training losses.
- on_train_batch_begin(batch): Called at the beginning of each training batch.
- on_train_batch_end(batch, logs): Called at the end of each training batch.
  The `logs` argument is a dictionary containing information about the batch,
  e.g., training losses.

A callback has access to the agent and the environment are accessible via
`self.agent` and `self.env`. A callback can terminate the training early by
setting the attributo `self.stop_training` to `True`. Example callbacks are
defined in `callbacks.py`.

# Training Artifacts
Artifact manager (a single instance of the `ArtifactManager` class) manages the
save paths for training artifacts such as trained models and training logs. The
artifact manager is exposed to callbacks, which should register the artifacts
they wish to save via the manager's `register_artifact` method. The method
takes the following arguments:
- artifact_identifier: Identifier for the artifact (e.g., 'harvest_probabilities', 'models'). Used as part of the artifact's path.
- artifact_description: Artifact description.
- file_name_generator: Function that takes arbitrary arguments and returns a
  filename.
- check_conflicts: Whether to check for existing files in the artifact
  directory.

The idea is that the callback provides a callable `file_name_generator` that
takes arbitrary arguments and returns file names/local paths. The return value
of the `register_artifact` wraps this function so that the file names/local
paths point to a save location dictated by the configuration file. For example,
suppose that `file_name_generator` takes as its argument an integer `i` and
returns a file name `model_{i}.keras`. Then the wrapped function (with the same
arguments) returns a path
`{global_save_path}/{artifact_identifier}/model_{i}.keras`, so the callback
does not need to know about what `global_save_path` is. Additionally, the
manager checks if files already exist in
`{global_save_path}/{artifact_identifier}` to avoid overwriting files and it
ensures that the this path  exists.

# Example Run
Running
```bash
python3 train.py --config config/default_faustmann.yaml --seed 0
```
trains an agent on the environment as defined in the configuration file
`default_faustmann.yaml`. In this case, an actor-critic agent is trained using
Proximal Policy Optimization algorithm to solve the Faustmann forest rotation
problem. The random seed is set to `0` for reproducibility. At the end of the
training, the trained agent is saved in the folder `../runs/default/0`.
Training progress (losses, agent action probabilities, etc.) are displayed
during the training script execution.
