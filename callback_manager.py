from typing import Union

from agent import Agent
from callbacks import BaseCallback
from env import VahvistajaEnvironment

class CallbackManager:
    """Manages a list of callbacks and provides convenient methods to call them."""

    def __init__(self, env: VahvistajaEnvironment, agent: Agent, callbacks: Union[list,None]=None):
        """Initialize the callback manager.

        Args:
            env: The environment.
            agent: The agent.
            callbacks: An optional list of callbacks to manage.
        """

        self.env = env
        self.agent = agent
        self.callbacks = callbacks or []
        self.stop_training = False       # Flag for early stopping.

    def add_callback(self, callback: BaseCallback):
        """Add a single callback.

        Args:
            callback: The callback to add.
        """

        self.callbacks.append(callback)
        callback.set_agent(self.agent)
        callback.set_environment(self.env)

    def _call_method(self, method_name, *args, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                getattr(callback, method_name)(*args, **kwargs)
                # Check if any callback has requested early stopping.
                if hasattr(callback, "stop_training") and callback.stop_training:
                    self.stop_training = True

    def on_train_begin(self, logs=None):
        self._call_method("on_train_begin", logs)

    def on_train_end(self, logs=None):
        self._call_method("on_train_end", logs)

    def on_episode_begin(self, episode, logs=None):
        self._call_method("on_episode_begin", episode, logs)

    def on_episode_end(self, episode, logs=None):
        self._call_method("on_episode_end", episode, logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._call_method("on_epoch_begin", epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self._call_method("on_epoch_end", epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        self._call_method("on_train_batch_begin", batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self._call_method("on_train_batch_end", batch, logs)

    def should_stop_training(self):
        return self.stop_training

    def reset_stop_flag(self):
        self.stop_training = False

    def __len__(self):
        return len(self.callbacks)

    def __iter__(self):
        return iter(self.callbacks)

    def __getitem__(self, index):
        return self.callbacks[index]
