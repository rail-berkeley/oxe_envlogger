#!/usr/bin/env python3

import os
import numpy as np

import gym
import envlogger
import dm_env
from envlogger.backends import tfds_backend_writer
import tensorflow_datasets as tfds
import tensorflow as tf

from typing import Any, Dict, List, Tuple, Optional, Callable
from oxe_envlogger.data_type import from_space_to_feature, populate_docs
from oxe_envlogger.dm_env import GymReturn, DummyDmEnv, DmEnvWrapper

# Define MetadataInfo and MetadataCallback types
# https://github.com/google-deepmind/envlogger/blob/dc2c6a30be843eeb3fe841737f763ecabcb3a30a/envlogger/environment_logger.py#L45-L48

MetadataInfo = Dict[str, tfds.features.FeatureConnector]
MetadataCallback = Callable[[dm_env.TimeStep, Any, dm_env.Environment], Any]
DocField = Dict[str, str]


class OXEEnvLogger(gym.Wrapper):
    """
    This is the easiest way to wrap gym.Env with EnvLogger. If 
    custom data needs to be logged,
    add it by passing in `step_metadata` and `episode_metadata`.
    """

    def __init__(
            self,
            env: gym.Env,
            dataset_name: str,
            directory: str,
            max_episodes_per_file: int,
            step_metadata: Optional[Tuple[MetadataInfo,
                                          MetadataCallback]] = None,
            episode_metadata: Optional[Tuple[MetadataInfo,
                                             MetadataCallback]] = None,
            doc_field: DocField = {},
            version: str = '0.1.0',
    ):
        """
        args:
            dataset_name: name of the dataset
            env: gym.Env instance
            directory: directory to store the trajectories
            max_episodes_per_file: maximum number of episodes to store in a single file
            step_metadata: tuple of (step_metadata_info, step_fn)
            episode_metadata: tuple of (episode_metadata_info, episode_fn)
            doc_field: dict of doc field for the dict tree in obs and action space
            version: version of the dataset
        """
        super().__init__(env)
        self.dataset_name = dataset_name
        self.directory = directory
        self.max_episodes_per_file = max_episodes_per_file
        self.step_metadata = step_metadata
        self.episode_metadata = episode_metadata
        self.version = version

        def step_callback(action):
            return self.env.step(action)

        def reset_callback():
            return self.env.reset()

        self.dm_env = DummyDmEnv(
            observation_space=env.observation_space,
            action_space=env.action_space,
            step_callback=step_callback,
            reset_callback=reset_callback,
        )

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
        self.dataset_config = tfds.rlds.rlds_base.DatasetConfig(
            name=dataset_name,
            observation_info=from_space_to_feature(env.observation_space, doc_field),
            action_info=from_space_to_feature(env.action_space, doc_field),
            reward_info=tf.float64,
            discount_info=tf.float64,
            step_metadata_info=step_metadata_info,
            episode_metadata_info=episode_metadata_info,
            version=version,
        )

        self.dataset_config = populate_docs(
            self.dataset_config, doc_field=doc_field)

        writer = tfds_backend_writer.TFDSBackendWriter(
            data_directory=directory,
            split_name='train',
            max_episodes_per_file=max_episodes_per_file,
            ds_config=self.dataset_config,
            version=version,
        )
        self.dm_env = envlogger.EnvLogger(self.dm_env,
                                          step_fn=step_fn,
                                          episode_fn=episode_fn,
                                          backend=writer
                                          )

    def step(self, action, **kwargs) -> Tuple:
        """
        Return Tuple:
            obs, reward, truncate, terminate, info
        """
        val = self.dm_env.step(action, **kwargs)
        return GymReturn.convert_step(val)

    def reset(self, **kwargs) -> Tuple:
        """
        Return Tuple:
            Observation and Info
        """
        val = self.dm_env.reset(**kwargs)
        return GymReturn.convert_reset(val)


##############################################################################

def make_env_logger(
    env: DmEnvWrapper,
    dataset_name: str,
    directory: str,
    max_episodes_per_file: int,
    step_metadata: Optional[Tuple[MetadataInfo, MetadataCallback]] = None,
    episode_metadata: Optional[Tuple[MetadataInfo, MetadataCallback]] = None,
    version: str = '0.1.0',
    doc_field: DocField = {},
) -> DmEnvWrapper:
    """
    NOTE: the env here is dm_env.Environment, not gym.Env
    This provides a flexible way to enable logging for dm_env.Environment.

    Usage: 
        env = DmEnvWrapper(env)
        env = make_env_logger(
            env,
            dataset_name,
            directory=FLAGS.output_dir,
            max_episodes_per_file=500,
            step_metadata=step_metadata,
            episode_metadata=episode_metadata,
        )
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
        observation_info=from_space_to_feature(env.observation_space, doc_field),
        action_info=from_space_to_feature(env.action_space, doc_field),
        reward_info=tf.float64,
        discount_info=tf.float64,
        step_metadata_info=step_metadata_info,
        episode_metadata_info=episode_metadata_info,
        version=version,
    )

    dataset_config = populate_docs(dataset_config, doc_field=doc_field)

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
