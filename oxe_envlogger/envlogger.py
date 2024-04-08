#!/usr/bin/env python3

from oxe_envlogger.data_type import get_tfds_feature
import sys
import os
import numpy as np

# try:
#     import gymnasium as gym
# except ImportError:
#     print("gynasium is not installed, use gym instead")
import gym

import envlogger
from envlogger.backends import tfds_backend_writer
import tensorflow_datasets as tfds
import tensorflow as tf

from typing import Any, Dict, Tuple, Optional
from oxe_envlogger.data_type import from_space_to_feature, populate_docs
from oxe_envlogger.dm_env import GymReturn, DummyDmEnv
from abc import ABC, abstractmethod


def print_yellow(x): return print("\033[93m" + x + "\033[0m")


class EnvLoggerBase(ABC):
    @abstractmethod
    def step(self, action, **kwargs) -> Tuple:
        """Standard gym step()
        :return Tuple:     obs, reward, truncate, terminate, info
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, **kwargs) -> Tuple:
        """Standard gym reset()
        :return Tuple:   obs, info
        """
        raise NotImplementedError

    @abstractmethod
    def set_step_metadata(self, metadata: Dict[str, Any]):
        """Set the step metadata to log, call this before step() and reset()
        :param metadata: dict of metadata to log
        """
        raise NotImplementedError

    @abstractmethod
    def set_episode_metadata(self, metadata: Dict[str, Any]):
        """Set the episode metadata to log, call this before reset()
        :param metadata: dict of metadata to log
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close the logger, flush the data to disk
        """
        raise NotImplementedError


##############################################################################

# Define MetadataInfo and MetadataCallback types
# https://github.com/google-deepmind/envlogger/blob/dc2c6a30be843eeb3fe841737f763ecabcb3a30a/envlogger/environment_logger.py#L45-L48
"""
example of MetadataInfo:
    step_metadata_info = {'timestamp': tfds.features.Tensor(
        shape=(), dtype=tf.float32, doc="Timestamp for the step.")}
"""
MetadataInfo = Dict[str, tfds.features.FeatureConnector]
DocField = Dict[str, str]


class OXEEnvLogger(gym.Wrapper, EnvLoggerBase):
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
            step_metadata_info: Optional[MetadataInfo] = None,
            episode_metadata_info: Optional[MetadataInfo] = None,
            doc_field: DocField = {},
            version: str = '0.1.0',
    ):
        """
        args:
            dataset_name: name of the dataset
            env: gym.Env instance
            directory: directory to store the trajectories
            max_episodes_per_file: maximum number of episodes to store in a single file
            step_metadata_info: description of the step metadata specs
            episode_metadata_info: description of the episode metadata specs
            doc_field: dict of doc field for the dict tree in obs and action space
            version: version of the dataset
        """
        super().__init__(env)
        self.dataset_name = dataset_name
        self.directory = directory
        self.max_episodes_per_file = max_episodes_per_file
        self.version = version
        self.step_metadata_info = step_metadata_info
        self.episode_metadata_info = episode_metadata_info

        # check if directory exists else create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Create new directory: {directory}")

        # Metadata step and episode functions callback
        step_metadata_fn = None
        self.step_metadata_elements = {}
        if step_metadata_info is not None:
            def step_metadata_fn(timestep, action, env):
                assert self.step_metadata_elements, "step metadata not set"
                return self.step_metadata_elements

        episode_metadata_fn = None
        self.episode_metadata_elements = {}
        if episode_metadata_info is not None:
            def episode_metadata_fn(timestep, action, env):
                assert self.episode_metadata_elements, "episode metadata not set"
                return self.episode_metadata_elements

        # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/rlds/rlds_base.py
        self.dataset_config = tfds.rlds.rlds_base.DatasetConfig(
            name=dataset_name,
            observation_info=from_space_to_feature(
                env.observation_space, doc_field),
            action_info=from_space_to_feature(env.action_space, doc_field),
            reward_info=np.float32,
            discount_info=np.float32,
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
            store_ds_metadata=True,
        )

        dm_env = DummyDmEnv(env)
        self.dm_env = envlogger.EnvLogger(dm_env,
                                          step_fn=step_metadata_fn,
                                          episode_fn=episode_metadata_fn,
                                          backend=writer
                                          )
        print("Done wrapping environment with EnvironmentLogger.")

    def step(self, action, **kwargs) -> Tuple:
        """Refer to abstract method in EnvLoggerBase"""
        self.dm_env.step_kwargs = kwargs  # experimental
        val = self.dm_env.step(action)
        self.dm_env.step_kwargs = {}
        return GymReturn.convert_step(val)

    def reset(self, **kwargs) -> Tuple:
        """Refer to abstract method in EnvLoggerBase"""
        self.dm_env.reset_kwargs = kwargs  # experimental
        val = self.dm_env.reset()
        self.dm_env.reset_kwargs = {}
        return GymReturn.convert_reset(val)

    def set_step_metadata(self, metadata: Dict[str, Any]):
        """Refer to abstract method in EnvLoggerBase"""
        assert len(metadata) == len(
            self.step_metadata_info), "metadata definition mismatch"
        assert set(metadata.keys()) == set(self.step_metadata_info.keys())
        self.step_metadata_elements = metadata

    def set_episode_metadata(self, metadata: Dict[str, Any]):
        """Refer to abstract method in EnvLoggerBase"""
        assert len(metadata) == len(
            self.episode_metadata_info), "metadata definition mismatch"
        assert set(metadata.keys()) == set(self.episode_metadata_info.keys())
        self.episode_metadata_elements = metadata

    def close(self):
        self.dm_env.close()


##############################################################################


class AutoOXEEnvLogger(gym.Wrapper, EnvLoggerBase):
    """
    This provides the most flexible way to wrap gym.Env with EnvLogger. With
    type introspection, AutoOXEEnvLogger automatically obtains the specs of 
    the observation, action, metadata specs from the gym.Env instance.
    The optimal size of shards is automatically calculated based on the 
    first episode.
    """

    def __init__(
            self,
            env: gym.Env,
            dataset_name: str,
            directory: str = None,
            optimal_shard_size: int = 200,
    ):
        """
        args:
            dataset_name: name of the dataset
            env: gym.Env instance
            directory: directory to store the trajectories
            optimal_shard_size: optimal size of each tfrecord file in MB
        """
        super().__init__(env)
        self.dataset_name = dataset_name
        self.directory = directory

        # check if directory is specified, else create it
        if directory is None:
            directory = os.getcwd() + f"/logs/{dataset_name}"

        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Create new directory: {directory}")
        self.directory = directory

        self.has_init_logger = False
        self._temp_init_step_data = []
        self._temp_init_episode_data = []
        self.default_version = '0.1.0'
        self.optimal_shard_size = optimal_shard_size
        self.step_metadata_elements = {}
        self.episode_metadata_elements = {}

    def __init_logger(self):
        """
        This function is called when the first episode is observed.
        It creates the metadata specs and the tfrecord writer.
        """
        print(self._temp_init_step_data[0])
        print_yellow("Initializing logger...")
        # convert to mb and times a factor of 2 to accomodata short episode of traj_0
        epi_size = (
            sys.getsizeof(self._temp_init_step_data) +
            sys.getsizeof(self.episode_metadata_elements)
        ) * 2.0 / 1024.0 / 1024.0
        max_episodes_per_file = max(int(self.optimal_shard_size / epi_size), 1)
        print_yellow(f"Size of first episode: {epi_size} MB, "
                     f"with shard size: {max_episodes_per_file} episodes")

        # Take last metadata for episode and step
        self.episode_metadata_elements = self.episode_metadata_elements.copy()
        self.step_metadata_elements = self.step_metadata_elements.copy()

        assert len(self._temp_init_step_data) > 0, "no step data!"

        # NOTE: we will only take the first element of the list
        first_step_data = self._temp_init_step_data[0]
        observation_info = get_tfds_feature(first_step_data['obs'])
        action_info = get_tfds_feature(first_step_data['action'])

        step_metadata_info = {key: get_tfds_feature(value)
                              for key, value in first_step_data['metadata'].items()}
        episode_metadata_info = {key: get_tfds_feature(value)
                                 for key, value in self.episode_metadata_elements.items()}

        # TODO: check consistency within the step data, else throw error

        # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/rlds/rlds_base.py
        self.dataset_config = tfds.rlds.rlds_base.DatasetConfig(
            name=self.dataset_name,
            observation_info=observation_info,
            action_info=action_info,
            reward_info=np.float32,
            discount_info=np.float32,
            step_metadata_info=step_metadata_info,
            episode_metadata_info=episode_metadata_info,
            version=self.default_version,
        )

        writer = tfds_backend_writer.TFDSBackendWriter(
            data_directory=self.directory,
            split_name='train',
            max_episodes_per_file=max_episodes_per_file,
            ds_config=self.dataset_config,
            version=self.default_version,
            store_ds_metadata=True,
        )

        # metadata callback handler
        step_metadata_fn = None
        episode_metadata_fn = None

        if self.step_metadata_elements:
            def step_metadata_fn(timestep, action, env):
                return self.step_metadata_elements
        if self.episode_metadata_elements:
            def episode_metadata_fn(timestep, action, env):
                return self.episode_metadata_elements

        self.dm_env = DummyDmEnv(self.env)
        self.dm_env = envlogger.EnvLogger(self.dm_env,
                                          step_fn=step_metadata_fn,
                                          episode_fn=episode_metadata_fn,
                                          backend=writer
                                          )

    def _save_first_episode(self):
        """
        This is an internal function that saves the stored first episode
        after the logger is initialized.
        NOTE: this saves the entire first episode, and the first step of the
                second episode. (when reset() called twice)
        """
        print_yellow("Saving first episode...")
        assert len(self._temp_init_episode_data) == 2, "triggered only during 2nd reset()"

        # Helper function to save episode
        def save_episode(episode_data, step_data_list=None):
            self.episode_metadata_elements = episode_data['episode_metadata']
            self.step_metadata_elements = episode_data['step_metadata']
            self.dm_env.custom_reset_env_callback = episode_data['obs']
            self.dm_env.reset()

            if not step_data_list:
                return

            for _data in step_data_list:
                self.step_metadata_elements = _data['metadata']
                self.dm_env.custom_step_env_callback = lambda action: (
                    _data['obs'], _data['reward'], _data['terminate'], _data['truncate']
                )
                self.dm_env.step(_data['action'])

        # save the first episode and its trajectory
        save_episode(self._temp_init_episode_data[0], self._temp_init_step_data)

        # save the first step of the second episode
        save_episode(self._temp_init_episode_data[1])

        # Reset environment callbacks and clear temporary data
        self.dm_env.custom_step_env_callback = None
        self.dm_env.custom_reset_env_callback = None
        self._temp_init_step_data = None
        self._temp_init_episode_data = None

        print_yellow("Done saving first episode.")

    def reset(self, **kwargs):
        """Refer to abstract method in EnvLoggerBase"""
        if not self.has_init_logger:
            obs = self.env.reset(**kwargs)
            if isinstance(obs, tuple):
                obs, info = obs
            else:
                info = {}
            self._temp_init_episode_data.append(
                dict(
                    obs=obs,
                    episode_metadata=self.episode_metadata_elements.copy(),
                    step_metadata=self.step_metadata_elements.copy(),
                )
            )
            if len(self._temp_init_episode_data) < 2:
                print("collecting first episode to initialize logger...")
            else:
                self.__init_logger()
                self._save_first_episode()
                self.has_init_logger = True
                print_yellow("Done initializing logger.")
            return obs, info
        # Normal mode after init
        self.dm_env.reset_kwargs = kwargs  # experimental
        val = self.dm_env.reset()
        self.dm_env.reset_kwargs = {} # reset custom kwargs
        return GymReturn.convert_reset(val)

    def step(self, action, **kwargs):
        """Refer to abstract method in EnvLoggerBase"""
        if not self.has_init_logger:
            ret_tuple = self.env.step(action, **kwargs)
            self._temp_init_step_data.append(
                dict(
                    obs=ret_tuple[0],
                    action=action,
                    metadata=self.step_metadata_elements.copy(),
                    reward=ret_tuple[1],
                    terminate=ret_tuple[2],
                    truncate=ret_tuple[3],
                )
            )
            # NOTE: truncate is not supported in gym<0.26.0
            if len(ret_tuple) == 5:
                obs, reward, done, truncate, info = ret_tuple
            else:
                obs, reward, done, info = ret_tuple
                truncate = False
            return obs, reward, done, truncate, info
        self.dm_env.step_kwargs = kwargs  # experimental
        val = self.dm_env.step(action)
        self.dm_env.step_kwargs = {} # reset custom kwargs
        return GymReturn.convert_step(val)

    def set_step_metadata(self, metadata: Dict[str, Any]):
        """Refer to abstract method in EnvLoggerBase"""
        self.step_metadata_elements = metadata

    def set_episode_metadata(self, metadata: Dict[str, Any]):
        """Refer to abstract method in EnvLoggerBase"""
        self.episode_metadata_elements.update(metadata)

    def close(self):
        self.dm_env.close()
