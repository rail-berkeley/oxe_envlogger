#!/usr/bin/env python3

import numpy as np
import gym
import tensorflow as tf
import tensorflow_datasets as tfds
from dm_env import specs


def from_space_to_feature(space_def: gym.Space) -> tfds.features.FeatureConnector:
    # check if observation space is dict
    if isinstance(space_def, gym.spaces.Dict):
        observation_info = tfds.features.FeaturesDict({
            key: tfds.features.Tensor(shape=space.shape, dtype=space.dtype)
            for key, space in space_def.spaces.items()
        })
    else:
        observation_info = tfds.features.Tensor(
            shape=space_def.shape,
            dtype=space_def.dtype,
        )
    return observation_info


def from_space_to_spec(space_def: gym.Space, name: str) -> specs:
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
