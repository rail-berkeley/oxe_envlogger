#!/usr/bin/env python3

import numpy as np

import gym
import dm_env
from dm_env import specs

from typing import Any, Dict, List, Tuple, Callable, Optional
from oxe_envlogger.data_type import from_space_to_spec, enforce_type_consistency
import copy


class GymReturn:
    """
    Since EnvLogger uses dm_env.Environment interface,
    we need to convert the return values of dm_env.TimeStep to return values of gym.Env
    to ensure consistency.
    """

    def convert_step(val: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        This function ensures that the return value of gym.Env.step is consistent
        :param  val either a dm_env.TimeStep or a tuple of (obs, reward, truncate, terminate, info)
        :return obs, reward, truncate, terminate, info
        """
        if isinstance(val, dm_env.TimeStep):
            obs = val.observation
            reward = val.reward
            truncate = False
            terminate = val.last()
            info = {}
        else:
            obs, reward, terminate, truncate, info = val
        # NOTE: since logging is done async, this is important to ensure the obs is
        # immutable even after wrapper, thus deepcopy
        obs = copy.deepcopy(obs)
        return obs, reward, terminate, truncate, info

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
        # NOTE: since logging is done async, this is important to ensure the obs is
        # immutable even after wrapper, thus deepcopy
        obs = copy.deepcopy(obs)
        return obs, info


class DummyDmEnv():
    """
    This is a dummy class to use so that the dm_env.Environment interface
    can still receive the observation_space and action_space. So that the
    logging can be done in the dm_env.Environment interface.

    https://github.com/google-deepmind/dm_env
    """

    def __init__(self, gym_env: gym.Env):
        """
        :param gym_env: gym.Env instance
        """
        self.gym_env = gym_env
        self.observation_space = gym_env.observation_space
        self.action_space = gym_env.action_space
        # # NOTE experimental feature, this to pass in kwargs to the step
        #       and reset function as user might have some kwargs defined
        #       in the step(action, ...) and reset(...) api
        self.step_kwargs = {}
        self.reset_kwargs = {}
        
        # NOTE experimental feature, this to pass in custom callback when
        #      step and reset is called. This is useful for logging
        self.custom_step_env_callback = None
        self.custom_reset_env_callback = None

    def step(self, action) -> dm_env.TimeStep:
        """
        Standard dm_env.step interface
        Note: dm_env.step doesn't accept additional arguments
        """
        if self.custom_step_env_callback:
            val = self.custom_step_env_callback(action)
        else:
            val = self.gym_env.step(action, **self.step_kwargs)

        # NOTE: to support previous version fo gym api where it doesnt
        # return truncate and terminate.
        # https://github.com/openai/gym/releases/tag/0.26.0
        if len(val) == 5:
            obs, reward, terminate, truncate, info = val
        else:
            obs, reward, done, info = val
            terminate = done
            truncate = False  # NOTE: truncate is not supported in gym<0.26.0

        # ensure type consistency, with reward and discount below is float32 comply with spec
        reward = np.float32(reward)
        obs = enforce_type_consistency(self.observation_space, obs)

        if terminate:
            ts = dm_env.termination(reward=reward, observation=obs)
        elif truncate:
            ts = dm_env.truncation(reward=reward, observation=obs, discount=np.float32(1.0))
        else:
            ts = dm_env.transition(reward=reward, observation=obs, discount=np.float32(1.0))
        return ts

    def reset(self) -> dm_env.TimeStep:
        """
        Standard dm_env.reset interface
        Note: dm_env.reset doesn't accept additional arguments
        """
        if self.custom_reset_env_callback:
            obs = self.custom_reset_env_callback()
        else:
            obs = self.gym_env.reset(**self.reset_kwargs)

        # NOTE: api change in gym 0.26.0
        # https://github.com/openai/gym/releases/tag/0.26.0
        if isinstance(obs, tuple):
            obs, info = obs
        obs = enforce_type_consistency(self.observation_space, obs)
        ts = dm_env.restart(obs)
        return ts

    def observation_spec(self):
        return from_space_to_spec(self.observation_space, "observation")

    def action_spec(self):
        return from_space_to_spec(self.action_space, "action")

    def reward_spec(self):
        return specs.Array(
            shape=(),
            dtype=np.float32,
            name='reward',
        )

    def discount_spec(self):
        return specs.Array(
            shape=(),
            dtype=np.float32,
            name='discount',
        )

    def close(self):
        if hasattr(self.gym_env, "close"):
            self.gym_env.close()
