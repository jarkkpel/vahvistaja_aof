import gymnasium as gym
from gymnasium.envs.registration import register

from .base import VahvistajaEnvironment
from .FaustmannDiscreteTime import FaustmannEnv
from .MarketACD import MarketAgeClassEnv

register(
    id="Faustmann-v0",
    entry_point="env:FaustmannEnv",
)
register(
    id="MarketACD-v0",
    entry_point="env:MarketAgeClassEnv",
)
