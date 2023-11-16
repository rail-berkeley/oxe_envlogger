#!/usr/bin/env python3

import numpy as np

import gym
import envlogger
import dm_env
from dm_env import specs
from envlogger.backends import tfds_backend_writer
import tensorflow_datasets as tfds
import tensorflow as tf

from typing import Any, Dict, List, Tuple, Optional


class DMEnvWrapper(gym.Wrapper):
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
        obs, reward, truncate, terminate, info = val
        if terminate:
            ts = dm_env.termination(reward=reward, observation=obs)
        else:
            ts = dm_env.transition(reward=reward, observation=obs)
        return ts

    def reset(self) -> dm_env.TimeStep:
        obs, _ = self.env.reset()
        ts = dm_env.restart(obs)
        return ts

    def observation_spec(self):
        return specs.Array(
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype,
            name='observation',
        )

    def action_spec(self):
        return specs.Array(
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype,
            name='action',
        )

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


def step_return(val: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """
    This function ensures that the return value of gym.Env.step is consistent
    :val either a dm_env.TimeStep or a tuple of (obs, reward, truncate, terminate, info)
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


def reset_return(val: Any) -> Tuple[np.ndarray, dict]:
    """
    This function ensures that the return value of gym.Env.reset is consistent
    :val either a dm_env.TimeStep or a tuple of (obs, reward, truncate, terminate, info)
    """
    if isinstance(val, dm_env.TimeStep):
        obs = val.observation
        info = {}
    else:
        obs, info = val
    return obs, info


# define metadata info
MetadataInfo = Dict[str, tfds.features.FeatureConnector]


def make_env_writer(
    dataset_name: str,
    env: gym.Env,
    directory: str,
    max_episodes_per_file: int,
    step_metadata_info: Optional[MetadataInfo] = None,
    episode_metadata_info: Optional[MetadataInfo] = None,
    version: str = '0.1.0',
) -> tfds_backend_writer.TFDSBackendWriter:
    """
    Makes a TFDSBackendWriter for a given environment.
    args:
        dataset_name: name of the dataset
        env: gym.Env instance
        directory: directory to store the trajectories
        max_episodes_per_file: maximum number of episodes to store in a single file
        step_metadata_info: dict of custom metadata information for each step
        episode_metadata_info: dict of custom metadata information for each episode
    """
    # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/rlds/rlds_base.py
    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name=dataset_name,
        observation_info=tfds.features.Tensor(
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
            encoding=tfds.features.Encoding.ZLIB
        ),
        action_info=tfds.features.Tensor(
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
            encoding=tfds.features.Encoding.ZLIB
        ),
        reward_info=tf.float64,
        discount_info=tf.float64,
        step_metadata_info=step_metadata_info,
        episode_metadata_info=episode_metadata_info,
        version=version,
    )

    writer = tfds_backend_writer.TFDSBackendWriter(
        data_directory=directory,
        split_name='train',
        max_episodes_per_file=max_episodes_per_file,
        ds_config=dataset_config,
        version=version,
        )
    return writer
