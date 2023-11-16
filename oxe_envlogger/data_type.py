#!/usr/bin/env python3

import numpy as np
import gym
import tensorflow as tf
import tensorflow_datasets as tfds
from dm_env import specs
from typing import Dict


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
