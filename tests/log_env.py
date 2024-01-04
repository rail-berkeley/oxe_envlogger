#!/usr/bin/env python3

import tensorflow_datasets as tfds
from oxe_envlogger.envlogger import OXEEnvLogger, AutoOXEEnvLogger

import numpy as np
import gym

import tensorflow as tf
import time


def main(auto_logger=False):
    print("##########################################")
    print("## Testing with auto_logger: ", auto_logger)

    env = gym.make("CartPole-v1")

    # generate a random name for the dataset
    random_name_suffix = np.random.randint(1000)
    ds_name = f"test_env_{random_name_suffix}"

    if auto_logger:
        env = AutoOXEEnvLogger(
            env=env,
            dataset_name=ds_name,
            directory="logs",
        )
    else:
        step_metadata_info = {'timestamp': tfds.features.Tensor(shape=(),
                                                                dtype=np.float32,
                                                                doc="Timestamp for the step.")}
        episode_metadata_info = {'language_embedding': tfds.features.Tensor(
            shape=(5,),
            dtype=np.float32,
            doc="Language embedding for the episode.")}

        env = OXEEnvLogger(
            env=env,
            dataset_name=ds_name,
            directory="logs",
            max_episodes_per_file=10,
            step_metadata_info=step_metadata_info,
            episode_metadata_info=episode_metadata_info,
        )

    print("Done setting up logger")

    num_of_episodes = 3
    for i in range(num_of_episodes):
        env.set_episode_metadata({"language_embedding": np.random.rand(5).astype(np.float32)})
        env.set_step_metadata({"timestamp": time.time()})
        print("episode", i)
        env.reset(seed=i)

        for i in range(10):
            env.set_step_metadata({"timestamp": time.time()})
            obs, reward, done, trunc, info = env.step(env.action_space.sample())
            # print(obs, reward, done, trunc, info)

            if done or trunc:
                break

    env.close()  # explicitly close the env logger to flush the data to disk

    print("Done logging")

    ##############################################################################
    # Test Cases
    ##############################################################################

    ds_builder = tfds.builder_from_directory("logs")
    dataset = ds_builder.as_dataset(split='all')

    assert ds_builder.info.name == ds_name, f"dataset name should be {ds_name}"

    # print len of dataset
    print("size of dataset", len(list(dataset)))
    assert len(list(dataset)) == num_of_episodes, f"There should be 3 episodes in the dataset"

    for episode in dataset.take(num_of_episodes):  # Take only the first episode for demonstration
        steps = episode['steps']

        num_steps = len(list(steps))

        assert num_steps == 11, "epi should have 1 + 10 steps, rerun since it can terminate early"
        assert "language_embedding" in episode.keys()

        # Iterate through steps in the episode
        for i, step in enumerate(steps):
            if i == 0:
                assert step['is_first'].numpy() == True, "first step is is_first=True"
            else:
                assert step['is_first'].numpy() == False, "non-first step is is_first=False"

            assert "timestamp" in step.keys(), "step should have timestamp metadata"


if __name__ == "__main__":
    print("Testing OXEEnvLogger")
    main(auto_logger=False)
    main(auto_logger=True)
    print("Done All")
