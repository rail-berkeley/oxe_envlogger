#!/usr/bin/env python3

import numpy as np
try:
    import gymnasium as gym
except ImportError:
    print("gynasium is not installed, use gym instead")
    import gym

import tensorflow as tf
import tensorflow_datasets as tfds
from dm_env import specs
from typing import Dict, Any


def get_gym_space(data_sample: Any) -> gym.spaces.Space:
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
        return gym.spaces.Dict({key: get_gym_space(value)
                                for key, value in data_sample.items()})
    elif isinstance(data_sample, (int, float, str)):
        return gym.spaces.Discrete(1)
    else:
        raise TypeError("Unsupported data type for Gym spaces conversion.")


def get_tfds_feature(data_sample: Any) -> tfds.features.FeatureConnector:
    """
    Get the data type as tfds feature of a provided data sample.
    Similar to the get_gym_space function above
    """
    if isinstance(data_sample, (np.ndarray, list, tuple)):
        data_sample = np.array(data_sample)
        return tfds.features.Tensor(shape=data_sample.shape,
                                    dtype=data_sample.dtype)
    elif isinstance(data_sample, dict):
        return tfds.features.FeaturesDict({key: get_tfds_feature(value)
                                           for key, value in data_sample.items()})
    elif isinstance(data_sample, (int, np.int32, np.int64)):
        return tfds.features.Tensor(shape=(), dtype=np.int64)
    elif isinstance(data_sample, (float, np.float32, np.float64)):
        return tfds.features.Tensor(shape=(), dtype=np.float64)
    else:
        raise TypeError(f"Unsupported data type for tfds features conversion, {type(data_sample)}")


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


def enforce_type_consistency(space: gym.Space, data: Any) -> Any:
    """
    Enforce type consistency between the data and the space
    """
    if isinstance(space, gym.spaces.Dict):
        for key, space in space.spaces.items():
            data[key] = enforce_type_consistency(space, data[key])
    else:
        if not isinstance(data, np.ndarray): # might want to not convert to np array
            data = np.array(data, dtype=space.dtype)
        assert space.shape == data.shape, f" mismatch shape"
        data = data.astype(space.dtype)
    return data
