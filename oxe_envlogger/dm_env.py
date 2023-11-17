#!/usr/bin/env python3

import numpy as np

import gym
import dm_env
from dm_env import specs

from typing import Any, Dict, List, Tuple, Callable
from oxe_envlogger.data_type import from_space_to_spec


class GymReturn:
    """
    Since EnvLogger uses dm_env.Environment interface,
    we need to convert the return values of dm_env.TimeStep to return values of gym.Env
    to ensure consistency.
    """

    def convert_step(val: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        This function ensures that the return value of gym.Env.step is consistent
        :arg val either a dm_env.TimeStep or a tuple of (obs, reward, truncate, terminate, info)
        :return obs, reward, truncate, terminate, info
        """
        if isinstance(val, dm_env.TimeStep):
            obs = val.observation
            reward = val.reward
            truncate = False
            terminate = val.last()
            info = {}
        else:
            obs, reward, truncate, terminate, info = val
        return obs, reward, truncate, terminate, info

    def convert_reset(val: Any) -> Tuple[np.ndarray, dict]:
        """
        This function ensures that the return value of gym.Env.reset is consistent
        :arg val either a dm_env.TimeStep or a tuple of (obs, info)
        :return obs, info
        """
        if isinstance(val, dm_env.TimeStep):
            obs = val.observation
            info = {}
        else:
            obs, info = val
        return obs, info


class DummyDmEnv():
    """
    This is a dummy class to use so that the dm_env.Environment interface
    can still receive the observation_space and action_space. So that the
    logging can be done in the dm_env.Environment interface.

    https://github.com/google-deepmind/dm_env
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 step_callback: Callable,
                 reset_callback: Callable,
                 ):
        """
        :param observation_space: gym observation space
        :param action_space: gym action space
        :param step_callback: callback function to call gymenv.step
        :param reset_callback: callback function to call gymenv.reset
        """
        self.step_callback = step_callback
        self.reset_callback = reset_callback
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, action) -> dm_env.TimeStep:
        # Note that dm_env.step doesn't accept additional arguments
        val = self.step_callback(action)
        obs, reward, terminate, truncate, info = val
        reward = float(reward)
        if terminate:
            ts = dm_env.termination(reward=reward, observation=obs)
        elif truncate:
            ts = dm_env.truncation(reward=reward, observation=obs)
        else:
            ts = dm_env.transition(reward=reward, observation=obs)
        return ts

    def reset(self) -> dm_env.TimeStep:
        # Note that dm_env.reset doesn't accept additional arguments
        obs, _ = self.reset_callback()
        ts = dm_env.restart(obs)
        return ts

    def observation_spec(self):
        return from_space_to_spec(self.observation_space, "observation")

    def action_spec(self):
        return from_space_to_spec(self.action_space, "action")

    def reward_spec(self):
        return specs.Array(
            shape=(),
            dtype=np.float64,
            name='reward',
        )

    def discount_spec(self):
        return specs.Array(
            shape=(),
            dtype=np.float64,
            name='discount',
        )

class DmEnvWrapper(gym.Wrapper):
    """
    This class wraps gym.Env with dm_env.Environment
    EnvLogger uses dm_env.Environment interface, which requires the use of `spec`
    and `timestep` as return values of `step` and `reset` methods.
    https://github.com/google-deepmind/dm_env
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action) -> dm_env.TimeStep:
        val = self.env.step(action)
        obs, reward, terminate, truncate, info = val
        reward = float(reward)
        if terminate:
            ts = dm_env.termination(reward=reward, observation=obs)
        elif truncate:
            ts = dm_env.truncation(reward=reward, observation=obs)
        else:
            ts = dm_env.transition(reward=reward, observation=obs)
        return ts

    def reset(self) -> dm_env.TimeStep:
        obs, _ = self.env.reset()
        ts = dm_env.restart(obs)
        return ts

    def observation_spec(self):
        return from_space_to_spec(self.env.observation_space, "observation")

    def action_spec(self):
        return from_space_to_spec(self.env.action_space, "action")

    def reward_spec(self):
        return specs.Array(
            shape=(),
            dtype=np.float64,
            name='reward',
        )

    def discount_spec(self):
        return specs.Array(
            shape=(),
            dtype=np.float64,
            name='discount',
        )
