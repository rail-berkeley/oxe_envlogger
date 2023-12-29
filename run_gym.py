#!/usr/bin/env python3

import time
import numpy as np
from absl import app, flags, logging

import gym

from oxe_envlogger.envlogger import OXEEnvLogger
import tensorflow_datasets as tfds
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_episodes', 10, 'Number of episodes to log.')
flags.DEFINE_string('output_dir', 'datasets/',
                    'Path in a filesystem to record trajectories.')
flags.DEFINE_string('env_name', 'CartPole-v1', 'Name of the environment.')
flags.DEFINE_boolean('enable_envlogger', True, 'Enable envlogger.')


##############################################################################

def wrap_env_logger(env: gym.Env):
    """Simple util to wrap gym.Env to dm_env.Environment."""
    logging.info('Wrapping environment with EnvironmentLogger...')

    # Define tuple of MetadataInfo and MetadataCallback
    step_metadata_info = {'timestamp': tfds.features.Tensor(shape=(),
                                                            dtype=np.float32,
                                                            doc="Timestamp for the step.")}
    episode_metadata_info = {'language_embedding': tfds.features.Tensor(
        shape=(5,),
        dtype=np.float32,
        doc="Language embedding for the episode.")}

    # make env logger
    env = OXEEnvLogger(
        env,
        dataset_name=FLAGS.env_name,
        directory=FLAGS.output_dir,
        max_episodes_per_file=500,
        step_metadata_info=step_metadata_info,
        episode_metadata_info=episode_metadata_info,
        doc_field={
            "proprio": "Proprioception of the robot arm.",
            "image_0": "RGB image from the camera.",
        },
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

        # example to log custom metadata during new episode
        if FLAGS.enable_envlogger:
            env.set_episode_metadata({
                "language_embedding": np.random.random((5,)).astype(np.float32)
            })
            env.set_step_metadata({"timestamp": time.time()})

        logging.info('episode %r', i)
        obs, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()

            # example to log custom step metadata
            if FLAGS.enable_envlogger:
                env.set_step_metadata({"timestamp": time.time()})

            obs, reward, _, done, _ = env.step(action)

    logging.info(
        'Done training a random agent for %r episodes.', FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(main)
