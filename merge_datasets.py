#!/usr/bin/env python3

import os
import argparse
import copy
from oxe_envlogger.utils import \
    find_datasets, get_datasets, merge_datasets, save_rlds_dataset


def print_yellow(x): return print("\033[93m {}\033[00m" .format(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dirs", type=str, nargs='+')
    parser.add_argument("--output_dir", type=str, default='merged_logs')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--shard_size", type=int, default=1500, help="Max episodes per shard")
    args = parser.parse_args()

    dataset_dirs = find_datasets(args.log_dirs)
    datasets, dataset_infos = get_datasets(dataset_dirs)
    merged_dataset = merge_datasets(datasets)
    print_yellow(f"Merging {len(datasets)} datasets with \
                   total {len(merged_dataset)} episodes.")

    def update_data_dir(target_dir, dataset_info):
        # Bug in update_data_dir() method, need to update the identity object as well
        # https://github.com/tensorflow/datasets/pull/5297
        dataset_info.update_data_dir(target_dir)
        dataset_info._identity.data_dir = target_dir

    dataset_info = copy.deepcopy(dataset_infos[0])
    update_data_dir(args.output_dir, dataset_info)
    print(dataset_info)
    print_yellow(f"Saving merged dataset to [{args.output_dir}]")
    assert dataset_info.data_dir == args.output_dir

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print_yellow(f"Writing episodes to disk... with {len(merged_dataset)} episodes.")
    save_rlds_dataset(
        dataset=merged_dataset,
        dataset_info=dataset_info,
        max_episodes_per_shard=args.shard_size,
        overwrite=args.overwrite
    )
    print("new ", dataset_info)
    print_yellow(f"Saved merged dataset to {args.output_dir}")
