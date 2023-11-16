#!/usr/bin/env python3

import os
import numpy as np

import gym
import envlogger
import dm_env
from dm_env import specs
from envlogger.backends import tfds_backend_writer
import tensorflow_datasets as tfds
import tensorflow as tf

from typing import Any, Dict, List, Tuple, Optional, Callable


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


# Define MetadataInfo and MetadataCallback types
# https://github.com/google-deepmind/envlogger/blob/dc2c6a30be843eeb3fe841737f763ecabcb3a30a/envlogger/environment_logger.py#L45-L48

MetadataInfo = Dict[str, tfds.features.FeatureConnector]
MetadataCallback = Callable[[dm_env.TimeStep, Any, dm_env.Environment], Any]


def make_env_logger(
    dataset_name: str,
    env: dm_env.Environment,
    directory: str,
    max_episodes_per_file: int,
    step_metadata: Optional[Tuple[MetadataInfo, MetadataCallback]] = None,
    episode_metadata: Optional[Tuple[MetadataInfo, MetadataCallback]] = None,
    version: str = '0.1.0',
) -> dm_env.Environment:
    """
    Makes an env for a given environment. If custom data needs to be logged,
    add it by passing in `step_metadata` and `episode_metadata`.
    args:
        dataset_name: name of the dataset
        env: gym.Env instance
        directory: directory to store the trajectories
        max_episodes_per_file: maximum number of episodes to store in a single file
        step_metadata: tuple of (step_metadata_info, step_fn)
        episode_metadata: tuple of (episode_metadata_info, episode_fn)
    """
    # ensure directory exists
    assert os.path.exists(directory), f"{directory} does not exist"

    if step_metadata is None:
        step_metadata_info, step_fn = None, None
    else:
        assert len(
            step_metadata) == 2, "should be a tuple of (step_metadata_info, step_fn)"
        step_metadata_info, step_fn = step_metadata

    if episode_metadata is None:
        episode_metadata_info, episode_fn = None, None
    else:
        assert len(
            episode_metadata) == 2, "should be a tuple of (episode_metadata_info, episode_fn)"
        episode_metadata_info, episode_fn = episode_metadata

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
    env = envlogger.EnvLogger(env,
                              step_fn=step_fn,
                              episode_fn=episode_fn,
                              backend=writer
                              )
    return env
