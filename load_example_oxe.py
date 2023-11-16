"""
This provides a simple example of how contribute to the Open X Embodiment (OXE) project.

Create a PR to https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/robotics

Example Notebook:
https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb
"""

import os
import tensorflow_datasets as tfds
from tensorflow_datasets.robotics import dataset_importer_builder


print_yellow = lambda x: print("\033[93m {}\033[00m" .format(x))

class HalfCheetah(dataset_importer_builder.DatasetImporterBuilder):
    """DatasetBuilder for example `halfcheetah` dataset."""

    # NOTE: This is the name of the dataset folder, ideally we would like to opensource this
    _GCS_BUCKET = os.path.expanduser("~/tensorflow_datasets")

    def get_description(self):
        return 'gym cheetah environment'

    def get_citation(self):
        return """@article{}"""

    def get_homepage(self):
        return 'https://github.com/google-deepmind/open_x_embodiment'

    def get_relative_dataset_location(self):
        return 'half_cheetah/0.1.0'


if __name__ == "__main__":
    # b = tfds.robotics.rtx.JacoPlay()
    # temp update the env TF_DATA_DIR to point to the local dataset
    b = HalfCheetah()

    print_yellow(b.get_dataset_location())
    print_yellow(b._info())

    ds = b.as_dataset(split='train[:10]').shuffle(10)   # take only first 10 episodes

    """
    features=FeaturesDict({
        'language_embedding': Tensor(shape=(5,), dtype=float32),
        'steps': Dataset({
            'action': Tensor(shape=(6,), dtype=float32),
            'discount': float64,
            'is_first': bool,
            'is_last': bool,
            'is_terminal': bool,
            'observation': Tensor(shape=(17,), dtype=float64),
            'reward': float64,
            'timestamp': int64,
        }),
    }),
    """
    
    is_last = False
    for episode in ds.take(1):  # Take only the first episode for demonstration
        # Unpack the episode
        language_embedding, steps = episode['language_embedding'], episode['steps']
        
        # Iterate through steps in the episode
        for step in steps:
            # Accessing step data
            timestamp = step['timestamp'].numpy()
            reward = step['reward'].numpy()
            is_last = step['is_last'].numpy()
            print(f"timestamp: {timestamp}, Reward: {reward}, Is Last: {is_last}")

    assert is_last, "The last step should be the last step of the episode"

    print("Done")
