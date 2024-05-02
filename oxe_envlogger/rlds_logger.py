#!/usr/bin/env python3

import dm_env
from dm_env import TimeStep, StepType
from envlogger import step_data
from typing import Dict, Any, Optional

import tensorflow as tf
import tensorflow_datasets as tfds
from envlogger.backends import tfds_backend_writer
from oxe_envlogger.data_type import from_space_to_feature, populate_docs, enforce_type_consistency

import gym
import os
import numpy as np

MetadataInfo = Dict[str, tfds.features.FeatureConnector]
DocField = Dict[str, str]


class RLDSStepType:
    RESTART = 0
    TRANSITION = 1
    TRUNCATION = 2
    TERMINATION = 3


class RLDSLogger:
    """
    Simple wrapper for RLDS logging.
    Refer to https://github.com/google-deepmind/envlogger/blob/main/envlogger/environment_logger.py
    """

    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            dataset_name: str,
            directory: str,
            max_episodes_per_file: int,
            max_steps_per_episode: Optional[int] = None,
            step_metadata_info: MetadataInfo = {},
            doc_field: DocField = {},
            version: str = '0.1.0',
    ):
        """
        args:
            dataset_name: name of the dataset
            observation_space: gym observation space
            action_space: gym action space
            directory: directory to store the trajectories
            max_episodes_per_file: maximum number of episodes to store in a single file
            max_steps_per_episode: maximum number of steps per episode, default to None
            step_metadata_info: dict of step metadata for the dict tree in obs and action space
            doc_field: dict of doc field for the dict tree in obs and action space
            version: version of the dataset
        """
        self.dataset_name = dataset_name
        self.directory = directory
        self.max_episodes_per_file = max_episodes_per_file
        self.version = version
        self.observation_space = observation_space
        self.action_space = action_space
        self.max_steps_per_episode = max_steps_per_episode
        self._curr_steps = 0
        self.step_metadata_info = step_metadata_info
        self._prev_custom_data = None

        # check if directory exists else create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Create new directory: {directory}")

        # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/rlds/rlds_base.py
        self.dataset_config = tfds.rlds.rlds_base.DatasetConfig(
            name=dataset_name,
            observation_info=from_space_to_feature(
                observation_space, doc_field),
            action_info=from_space_to_feature(action_space, doc_field),
            reward_info=np.float32,
            discount_info=np.float32,
            step_metadata_info=step_metadata_info,
            version=version,
        )

        self.dataset_config = populate_docs(
            self.dataset_config, doc_field=doc_field)

        # backend writer
        self.writer = tfds_backend_writer.TFDSBackendWriter(
            data_directory=directory,
            split_name='train',
            max_episodes_per_file=max_episodes_per_file,
            ds_config=self.dataset_config,
            version=version,
        )

    def __call__(self,
                 action: Any,
                 obs: Any,
                 reward: float,
                 metadata: Dict[str, Any] = {},
                 step_type: RLDSStepType = RLDSStepType.TRANSITION,
                 ):
        """
        args:
            action: gym action
            obs: gym observation
            reward: gym reward
            step_type: gym step type defined in RLDSStepType
        """
        # explicitly enforce type consistency
        obs = enforce_type_consistency(self.observation_space, obs)
        action = enforce_type_consistency(self.action_space, action)
        reward = np.float32(reward)

        # record step
        if step_type == RLDSStepType.TERMINATION:
            # NOTE: explicitly set discount to custom float32 since dm_env.termination() is using float64
            # https://github.com/google-deepmind/dm_env/blob/91b46797fea731f80eab8cd2c8352a0674141d89/dm_env/_environment.py#L226
            # ts = dm_env.termination(reward=reward, observation=obs), 
            ts = TimeStep(StepType.LAST, reward, np.float32(0.0), obs)
        elif step_type == RLDSStepType.TRUNCATION:
            ts = dm_env.truncation(reward=reward, observation=obs, discount=np.float32(1.0))
        elif step_type == RLDSStepType.RESTART:
            ts = dm_env.restart(obs)
        elif step_type == RLDSStepType.TRANSITION:
            ts = dm_env.truncation(reward=reward, observation=obs, discount=np.float32(1.0))
        else:
            raise NotImplementedError

        self._curr_steps += 1
        if self.max_steps_per_episode and \
            self._curr_steps >= self.max_steps_per_episode:
            # we will restart the episode if user
            # specifies max_steps_per_episode
            step_type = RLDSStepType.RESTART
            self._curr_steps = 0

        # This is to record step metadata
        custom_data = None
        if self.step_metadata_info:
            # we will use the cached metadata if user didn't provide during
            # the following steps
            if metadata:
                custom_data = metadata
                self._prev_custom_data = metadata
            else:
                custom_data = self._prev_custom_data

        self.writer.record_step(
            step_data.StepData(
                timestep=ts,
                action=action,
                custom_data=custom_data,
            ),
            is_new_episode=step_type == RLDSStepType.RESTART,
        )

    def close(self):
        # flush the writer
        # manually close the writer
        if hasattr(self, 'writer'):
            self.writer._write_and_reset_episode()
            self.writer._sequential_writer.close_all()
            del self.writer

    def __del__(self):
        # NOTE: early termination of writer wont export the dataset
        self.close()
