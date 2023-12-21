#!/usr/bin/env python3

import numpy as np
import gym
import tensorflow as tf
import tensorflow_datasets as tfds
from dm_env import specs
from typing import Dict, Any


def get_gym_spaces(data_sample: Any) -> gym.spaces.Space:
    """
    Get the data type as Gym space of a provided data sample.
    This function currently only supports common types like Box and Dict type.

    :Args:  data_sample (Any): The data sample to be converted.
    :Returns:   gym.spaces.Space
    """
    # Case for numerical data (numpy arrays, lists of numbers, etc.)
    if isinstance(data_sample, (np.ndarray, list, tuple)):
        # Ensure it's a numpy array to get shape and dtype
        data_sample = np.array(data_sample)
        return gym.spaces.Box(low=-np.inf, high=np.inf,
                              shape=data_sample.shape, dtype=data_sample.dtype)

    # Case for dictionary data
    elif isinstance(data_sample, dict):
        # Recursively convert each item in the dictionary
        return gym.spaces.Dict({key: get_gym_spaces(value)
                                for key, value in data_sample.items()})
    else:
        raise TypeError("Unsupported data type for Gym spaces conversion.")


def from_space_to_feature(space_def: gym.Space,
                          doc_field: Dict[str, str] = {},
                          ) -> tfds.features.FeatureConnector:
    """
    from gym.Space to tfds.features.FeatureConnector
    https://www.tensorflow.org/datasets/api_docs/python/tfds/features

    The doc field is used to describe the tree
    """
    # check if observation space is dict
    if isinstance(space_def, gym.spaces.Dict):
        feature = tfds.features.FeaturesDict({
            key: tfds.features.Tensor(shape=space.shape,
                                      dtype=space.dtype,
                                      doc=doc_field.get(key, None)
                                      )
            for key, space in space_def.spaces.items()
        })
    else:
        feature = tfds.features.Tensor(
            shape=space_def.shape,
            dtype=space_def.dtype,
        )
    return feature


def from_space_to_spec(space_def: gym.Space, name: str) -> specs:
    """
    From gym.Space to dm_env.specs
    https://github.com/google-deepmind/dm_env/blob/master/dm_env/specs.py
    """
    # check if observation space is dict
    if isinstance(space_def, gym.spaces.Dict):
        spec = {
            key: specs.Array(
                shape=space.shape,
                dtype=space.dtype,
                name=key,
            )
            for key, space in space_def.spaces.items()
        }
    else:
        spec = specs.Array(
            shape=space_def.shape,
            dtype=space_def.dtype,
            name=name,
        )
    return spec


def populate_docs(dataset_config: tfds.rlds.rlds_base.DatasetConfig,
                  doc_field: Dict[str, str] = {},
                  ):
    """
    Populate the doc within the dataset_config
    """
    if dataset_config.observation_info and "observation" in doc_field:
        doc = tfds.features.Documentation(desc=doc_field["observation"])
        dataset_config.observation_info._doc = doc
    if dataset_config.action_info and "action" in doc_field:
        doc = tfds.features.Documentation(desc=doc_field["action"])
        dataset_config.action_info._doc = doc
    if dataset_config.reward_info and "reward" in doc_field:
        doc = tfds.features.Documentation(desc=doc_field["reward"])
        dataset_config.reward_info._doc = doc
    if dataset_config.discount_info and "discount" in doc_field:
        doc = tfds.features.Documentation(desc=doc_field["discount"])
        dataset_config.discount_info._doc = doc
    return dataset_config
