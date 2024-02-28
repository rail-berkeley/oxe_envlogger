import os
import argparse
import copy
from typing import Tuple, Dict, List, Any, Optional
from oxe_envlogger.utils import \
    print_yellow, find_datasets, get_datasets, merge_datasets, save_rlds_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dirs", type=str, nargs='+')
    parser.add_argument("--output_dir", type=str, default='merged_logs')
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args()

    dataset_dirs = find_datasets(args.log_dirs)
    datasets, dataset_infos = get_datasets(dataset_dirs)
    merged_dataset = merge_datasets(datasets)
    print_yellow(f"Merging {len(datasets)} datasets with {len(merged_dataset)} episodes.")

    def update_data_dir(target_dir, dataset_info):
        # Bug in update_data_dir() method, need to update the identity object as well
        # https://github.com/tensorflow/datasets/blob/v4.9.3/tensorflow_datasets/core/dataset_info.py#L528
        dataset_info.update_data_dir(target_dir)
        dataset_info._identity.data_dir = target_dir

    dataset_info = copy.deepcopy(dataset_infos[0])
    update_data_dir(args.output_dir, dataset_info)
    print_yellow(dataset_info)
    print_yellow(f"Saving merged dataset to [{args.output_dir}]")
    assert dataset_info.data_dir == args.output_dir

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    save_rlds_dataset(merged_dataset, dataset_info,
                      max_episodes_per_shard=1000,
                      overwrite=args.overwrite
                      )
    print(f"Saved merged dataset to {args.output_dir}")
