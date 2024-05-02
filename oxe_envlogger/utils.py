#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import os
from typing import Tuple, List, Set, Callable, Optional, Dict, Any
from tensorflow_datasets.core import SequentialWriter
import tqdm


def find_datasets(log_dirs=['logs']) -> Set[str]:
    """
    Finds dataset directories within specified log directories
    that contain 'features.json'.

    Args:
        log_dirs (list of str): Directories to search for datasets.

    Returns:
        list of str: A list of dataset directories.
    """
    dataset_dirs = set()

    def search_dirs(dir_path: str):
        # Iterate over each item in the current directory
        for item in os.listdir(dir_path):
            full_path = os.path.join(dir_path, item)
            # If the item is a directory, recursively search it
            if os.path.isdir(full_path):
                search_dirs(full_path)
            # If the item is 'features.json', add its directory to the list
            elif item == 'features.json':
                dataset_dirs.add(os.path.dirname(full_path))
                break

    # Iterate over each directory in log_dirs and search it
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            search_dirs(log_dir)
        else:
            print(f"Warning: The directory {log_dir} does not exist.")
    return dataset_dirs


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
    eps_filtering_fn: Optional[Callable[[int, Dict[str, Any]], bool]] = None,
):
    """
    Save the dataset to disk in the RLDS format.
    
    Args:
        dataset: tf.data.Dataset: The dataset to save.
        dataset_info: tfds.core.DatasetInfo: Information about the dataset.
        max_episodes_per_shard: int: Maximum number of episodes per shard.
        overwrite: bool: Whether to overwrite existing files.
        eps_filtering_fn: Optional[Callable[[int, Dict[str, Any]], bool]]: A function that
            takes an episode index and the episode data and returns a boolean indicating
            whether to include the episode in the saved dataset. Return False to skip
    """
    writer = SequentialWriter(
        dataset_info, max_episodes_per_shard, overwrite=overwrite
    )
    writer.initialize_splits(["train"], fail_if_exists=False)

    def recursive_dict_to_numpy(d):
        """
        convert each values in the dict to numpy array
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
    for idx, episode in enumerate(tqdm.tqdm(dataset, desc="Writing episodes")):
        # Manage non-"steps" keys data, e.g. language_embedding, metadata, etc.
        episodic_data_dict = {
            key: episode[key] for key in episode.keys() if key != "steps"
        }
        
        # Skip this episode if filter returns False
        if eps_filtering_fn and not eps_filtering_fn(idx, episodic_data_dict):
            continue

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
