"""
This provides a simple example of how contribute to the Open X Embodiment (OXE) project.

Create a PR to https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/robotics

Example Notebook:
https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb
"""

import os
import tensorflow_datasets as tfds
from tensorflow_datasets.robotics import dataset_importer_builder


def print_yellow(x): return print("\033[93m {}\033[00m" .format(x))


class HalfCheetah(dataset_importer_builder.DatasetImporterBuilder):
    """
    DatasetBuilder for example `halfcheetah` dataset.

    Check:
    https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/robotics/rtx/rtx.py
    """

    # NOTE: This path to the datasets directory
    _GCS_BUCKET = os.path.expanduser("~/tensorflow_datasets")

    def get_description(self):
        return 'gym cheetah environment'

    def get_citation(self):
        return """@article{awesomepaper2023
            title   = {Awesome paper title},
            journal = {Journal of Awesomeness},
        }
        """

    def get_homepage(self):
        return 'https://github.com/google-deepmind/open_x_embodiment'

    def get_relative_dataset_location(self):
        return 'half_cheetah/0.1.0'


##############################################################################

if __name__ == "__main__":
    # b = tfds.robotics.rtx.JacoPlay()
    b = HalfCheetah()

    print_yellow(b.get_dataset_location())
    print_yellow(b._info())

    ds = b.as_dataset(split='train[:10]').shuffle(10)  # take 10 episodes

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
    step_count = 0
    for episode in ds.take(1):  # Take only the first episode for demonstration
        # Unpack the episode
        language_embedding, steps = episode['language_embedding'], episode['steps']
        assert len(
            steps) == 1001, "Each episode should have 1001 steps for HalfCheetah env"

        # Iterate through steps in the episode
        for step in steps:

            # Accessing step data
            timestamp = step['timestamp'].numpy()
            reward = step['reward'].numpy()
            is_last = step['is_last'].numpy()
            print(
                f"timestamp: {timestamp}, Reward: {reward}, Is Last: {is_last}")

    assert is_last, "The last step should be the last step of the episode"

    print("Done")
