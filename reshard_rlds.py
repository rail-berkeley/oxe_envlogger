#!/usr/bin/env python3

import os
import argparse
import copy
from oxe_envlogger.utils import find_datasets, save_rlds_dataset
import tensorflow_datasets as tfds


def print_yellow(x): return print("\033[93m {}\033[00m" .format(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds_dirs", type=str, nargs='+', required=True)
    parser.add_argument("--output_rlds",  type=str, required=True)
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--shard_size", type=int, default=1500, help="Max episodes per shard")
    args = parser.parse_args()

    # Recursively find all datasets in the given directories
    dataset_dirs = find_datasets(args.rlds_dirs)
    print_yellow(f"Found {dataset_dirs} rlds datasets to merge.")

    # Create a merged dataset of all the datasets in the given directories
    builder = tfds.builder_from_directories(list(dataset_dirs))
    merged_dataset = builder.as_dataset(split='all')

    dataset_info = builder.info
    total_size = dataset_info.dataset_size
    recommended_shard_size = round(200*1024*1024*len(merged_dataset)/total_size)

    print_yellow(f"Total size of datasets: {total_size/1024.0} kb")
    print_yellow(f"Merging {len(dataset_dirs)} datasets with "
                 f"total {len(merged_dataset)} episodes.")
    print_yellow(f"!!NOTE!! It is recommended to keep tfrecord size at "
                 f"around 200MB. Thus the recommended shard size should "
                 f"be around {recommended_shard_size} episodes. ")

    def update_data_dir(target_dir, dataset_info):
        # Bug in update_data_dir() method, need to update the identity object as well
        # https://github.com/tensorflow/datasets/pull/5297
        # dataset_info.update_data_dir(target_dir) # not supported in MultiSplitInfo()
        dataset_info._identity.data_dir = target_dir

    # Create a new dataset info with the updated data_dir
    dataset_info = copy.deepcopy(dataset_info)
    update_data_dir(args.output_rlds, dataset_info)
    assert dataset_info.data_dir == args.output_rlds
    # print(dataset_info)

    if not os.path.exists(args.output_rlds):
        os.makedirs(args.output_rlds)

    print_yellow(f"Writing episodes to disk: [{args.output_rlds}]")
    save_rlds_dataset(
        dataset=merged_dataset,
        dataset_info=dataset_info,
        max_episodes_per_shard=args.shard_size,
        overwrite=args.overwrite
    )
    print("Updated dataset info: ", dataset_info)
    print_yellow(f"Saved rlds dataset to: {args.output_rlds}")
