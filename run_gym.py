#!/usr/bin/env python3

import time
import numpy as np
from absl import app, flags, logging

import gym

from oxe_envlogger.env_logger import DmEnvWrapper, make_env_logger, GymReturn
import tensorflow_datasets as tfds
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to log.')
flags.DEFINE_string('output_dir', 'datasets/',
                    'Path in a filesystem to record trajectories.')
flags.DEFINE_string('env_name', 'CartPole-v1', 'Name of the environment.')
flags.DEFINE_boolean('enable_envlogger', False, 'Enable envlogger.')

##############################################################################


def wrap_env_logger(env: gym.Env):
    """Simple util to wrap gym.Env to dm_env.Environment."""
    logging.info('Wrapping environment with EnvironmentLogger...')
    env = DmEnvWrapper(env)

    # This function will be called at every step of the environment.
    def step_fn(unused_timestep, unused_action, unused_env):
        return {
            'timestamp': time.time(),
        }

    # This function will be called at every episode of the environment.
    def episode_fn(unused_timestep, unused_action, unused_env):
        return {
            'language_embedding': np.random.random((5,)).astype(np.float32),
        }

    # Crude way to use snake case for dataset name
    if "Cheetah" in FLAGS.env_name:
        dataset_name = "half_cheetah"
    else:
        dataset_name = FLAGS.env_name

    # Define tuple of MetadataInfo and MetadataCallback
    step_metadata = ({'timestamp': tf.int64}, step_fn)
    episode_metadata = (
        {'language_embedding': tfds.features.Tensor(shape=(5,), dtype=tf.float32),},
        episode_fn
    )

    # make env logger
    env = make_env_logger(
        env,
        dataset_name,
        directory=FLAGS.output_dir,
        max_episodes_per_file=500,
        step_metadata=step_metadata,
        episode_metadata=episode_metadata,
    )
    logging.info('Done wrapping environment with EnvironmentLogger.')
    return env

##############################################################################


def main(unused_argv):
    logging.info(f'Creating gym environment...')

    env = gym.make(FLAGS.env_name)
    logging.info(f'Done creating {FLAGS.env_name} environment.')

    if FLAGS.enable_envlogger:
        env = wrap_env_logger(env)

    logging.info('Training an agent for %r episodes...', FLAGS.num_episodes)

    for i in range(FLAGS.num_episodes):
        logging.info('episode %r', i)
        timestep = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            timestep = env.step(action)

            # this ensure that the return value of gym.Env.step is consistent
            obs, reward, _, done, _ = GymReturn.convert_step(timestep)

    logging.info(
        'Done training a random agent for %r episodes.', FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(main)
