#!/usr/bin/env python3

import time
import numpy as np
from absl import app, flags, logging

import gym
from oxe_envlogger.envlogger import AutoOXEEnvLogger

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_episodes', 10, 'Number of episodes to log.')
flags.DEFINE_string('output_dir', 'datasets/',
                    'Path in a filesystem to record trajectories.')
flags.DEFINE_string('env_name', 'CartPole-v1', 'Name of the environment.')
flags.DEFINE_boolean('enable_envlogger', False, 'Enable envlogger.')


##############################################################################


def main(unused_argv):
    logging.info(f'Creating gym environment...')

    env = gym.make(FLAGS.env_name)
    logging.info(f'Done creating {FLAGS.env_name} environment.')

    if FLAGS.enable_envlogger:
        env = AutoOXEEnvLogger(
            env=env,
            dataset_name=FLAGS.env_name,
            directory=FLAGS.output_dir,
        )

    logging.info('Training an agent for %r episodes...', FLAGS.num_episodes)

    for i in range(FLAGS.num_episodes):

        # example to log custom metadata during new episode
        if FLAGS.enable_envlogger:
            env.set_episode_metadata({
                "language_embedding": np.random.random((5,)).astype(np.float32)
            })
            env.set_step_metadata({"timestamp": time.time()})

        logging.info('episode %r', i)
        env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()

            # example to log custom step metadata
            if FLAGS.enable_envlogger:
                env.set_step_metadata({"timestamp": time.time()})

            return_step = env.step(action)

            # NOTE: to handle gym.Env.step() return value change in gym 0.26
            if len(return_step) == 5:
                obs, reward, terminated, truncated, info = return_step
            else:
                obs, reward, terminated, info = return_step
                truncated = False

    logging.info(
        'Done training a random agent for %r episodes.', FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(main)
