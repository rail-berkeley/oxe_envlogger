#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import os
from typing import Tuple, Dict, List, Any, Optional
from tensorflow_datasets.core import SequentialWriter


def find_datasets(log_dirs=['logs']) -> List[str]:
    """
    Finds dataset directories within specified log directories.

    Args:
        log_dirs (list of str): Directories to search for datasets.

    Returns:
        list of str: A list of dataset directories.
    """
    datasets_dirs = []
    for log_dir in log_dirs:
        if os.path.isdir(log_dir):
            for dir in os.listdir(log_dir):
                dir_path = os.path.join(log_dir, dir)
                if os.path.isdir(dir_path):
                    datasets_dirs.append(dir_path)
    return datasets_dirs


def get_datasets(
    dataset_dirs: List[str],
) -> Tuple[List[tf.data.Dataset], List[tfds.core.DatasetInfo]]:
    """
    Load datasets from given directories.
    args:
        dataset_dirs: list of dataset directories
    returns:
        datasets: list of datasets
        dataset_infos: list of dataset info
    """
    datasets = []
    dataset_infos = []
    for dataset_dir in dataset_dirs:
        temp_d = tfds.builder_from_directory(dataset_dir)
        dataset = temp_d.as_dataset(split='all')
        datasets.append(dataset)
        dataset_infos.append(temp_d.info)
    return datasets, dataset_infos


def merge_datasets(datasets: List[tf.data.Dataset]) -> tf.data.Dataset:
    """
    Merges datasets and returns the merged dataset.

    Args:
        datasets (list of tf.data.Dataset): A list of datasets to merge.

    Returns:
        tf.data.Dataset: A merged dataset.
    """
    # Assuming the datasets are compatible, we can concatenate them.
    merged_dataset = datasets[0]
    for dataset in datasets[1:]:
        merged_dataset = merged_dataset.concatenate(dataset)
    return merged_dataset


def save_rlds_dataset(
    dataset: tf.data.Dataset,
    dataset_info: tfds.core.DatasetInfo,
    max_episodes_per_shard: int = 1000,
    overwrite: bool = False,
):
    # Convert observation and action spaces to TFDS features
    writer = SequentialWriter(
        dataset_info, max_episodes_per_shard, overwrite=overwrite
    )
    writer.initialize_splits(["train"], fail_if_exists=False)

    def recursive_dict_to_numpy(d):
        """
        convert each values in the step dict to numpy array
        TODO: THIS IS A HACK!!! since the sequence writer uses
        https://github.com/tensorflow/datasets/blob/ab25353173afa5fa01d03759a5e482c82d4fd889/tensorflow_datasets/core/features/tensor_feature.py#L149
        API, which will throw an error if the input is not a numpy array
        """
        for k, v in d.items():
            if isinstance(v, dict):
                recursive_dict_to_numpy(v)
            else:
                d[k] = v.numpy()
        return d

    # Write episodes to disk
    for episode in dataset:
        # Manage non-"steps" keys data, e.g. language_embedding, metadata, etc.
        episodic_data_dict = {
            key: episode[key] for key in episode.keys() if key != "steps"
        }
        episodic_data_dict = recursive_dict_to_numpy(episodic_data_dict)
        steps = []
        for step in episode["steps"]:

            step = recursive_dict_to_numpy(step)
            steps.append(step)

        # write the episode to the dataset
        writer.add_examples(
            {"train": [{"steps": steps, **episodic_data_dict}]}
        )
    writer.close_all()
