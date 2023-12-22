import tensorflow_datasets as tfds
from oxe_envlogger.data_type import get_gym_space
from oxe_envlogger.rlds_logger import RLDSLogger, RLDSStepType
import numpy as np
from absl import app, flags, logging

def main(_):
    obs_sample = {
        "state1": np.array([4, 5, 6]),
        "state2": np.array([1, 2, 3]),
    }
    action_sample = np.array([0, 1, 2, 4])

    # 1. Create RLDSLogger
    logger = RLDSLogger(
        observation_space=get_gym_space(obs_sample),
        action_space=get_gym_space(action_sample),
        dataset_name="test",
        directory="logs",
        max_episodes_per_file=1,
    )

    # 2. Log data
    for i in range(3):
        for i in range(10):
            if i == 0:
                step_type = RLDSStepType.RESTART
            else:
                step_type = RLDSStepType.TRANSITION
            logger(action_sample, obs_sample, 1.0, step_type=step_type)
        logger(action_sample, obs_sample, 1.0, step_type=RLDSStepType.TERMINATION)

    logger.close()
    print("Done logging")

    # Test Cases
    ##############################################################################
    dataset = tfds.builder_from_directory("logs").as_dataset(split='all')

    # print len of dataset
    print("size of dataset", len(list(dataset)))
    assert len(list(dataset)) == 3, "There should be 3 episodes in the dataset"

    for episode in dataset.take(3):  # Take only the first episode for demonstration
        steps = episode['steps']
        print("step size", len(list(steps)))

        # Iterate through steps in the episode
        for i, step in enumerate(steps):
            
            if i == 0:
                assert step['is_first'].numpy() == True, "first step is is_first=True"
            else:
                assert step['is_first'].numpy() == False, "non-first step is is_first=False"

            assert len(step["observation"]) == 2, "observation should have 2 keys"

            print(" is first and last: ", step['is_first'].numpy(), step['is_last'].numpy()) 
            is_last = step['is_last']

    assert is_last, "The last step should be the last step of the episode"

    print("Done All")

if __name__ == "__main__":   
    app.run(main)
