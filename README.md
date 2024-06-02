# oxe_envlogger

**Make data collection for robot learning easy ➡️ More reusable datasets. 🤖📈**

Env logger for robots. For [open-x-embodiment](https://robotics-transformer-x.github.io/), aka OXE

## Installation

This package wraps the [envlogger](https://github.com/google-deepmind/envlogger) lib to make it compatible with OXE and openai gym env.

```bash
sudo apt install libgmp-dev

cd oxe_envlogger
pip install -e .
```

---

## Quick Run

1. A simple example is provided to show how to record a [gym env](https://www.gymlibrary.dev/api/core/).

Without env_logger
```bash
python run_gym.py --env_name="HalfCheetah-v4" 
```

2. Now, with env_logger

```bash
# create directory
mkdir -p datasets/half_cheetah/0.1.0

# Run a gym environment  `--enable_envlogger`
python run_gym.py --env_name="HalfCheetah-v4" --enable_envlogger --output_dir="datasets/half_cheetah/0.1.0"
```

3. Check the recorded [tfds](https://www.tensorflow.org/datasets/api_docs/python/tfds) data

This stores the data in `datasets/half_cheetah` directory. Check the data format in `cat datasets/half_cheetah/0.1.0/features.json`

## Try load 

After collecting data with the cheetah env, we will now try to load the data.
Here is a simple example of a dataset class which is compatible with tensorflow_dataset's [RTX](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/robotics)

```bash
cp -rf datasets/half_cheetah/ ~/tensorflow_datasets/

python load_example_oxe.py
```

## Usage

It is extremely easy to use `OXEEnvLogger`. Just add the following lines to your code to wrap your env with the logger. For more detailed example, check `tests/log_env.py` or `run_gym.py`.

**1. AutoOXEEnvLogger** (with type introspection)

```py
from oxe_envlogger.envlogger import AutoOXEEnvLogger

env = YOUR_GYM_ENV
env = AutoOXEEnvLogger(
    env,
    YOUR_DATASET_NAME,
)
```

The above code will automatically obtain the observation and action space from the env, and also support custom metadata. The optimal shard size is 
automatically calculated based on the first episode. Use `set_step_metadata()` and `set_episode_metadata()` to set the metadata for each step and episode respectively. Log is saved in `logs/YOUR_DATASET_NAME` by default.

**2. OXEEnvLogger** (without type introspection)

For more fine-grained type casting of the data, use `OXEEnvLogger`. This requires
the correct type cast of `self.action_space` and `self.observation_space` in the gym.env class.

```py
from oxe_envlogger.envlogger import OXEEnvLogger

env = YOUR_GYM_ENV
env = OXEEnvLogger(
    env,
    YOUR_DATASET_NAME,
    directory=YOUR_OUTPUT_DIR
    max_episodes_per_file=500,
    # step_metadata_info=YOUR_METADATA, # optional
    # episode_metadata_info=YOUR_METADATA, # optional
)
```

**3. RLDSLogger**

Or, you can use the RLDSLogger to log the data manually. For more detailed example, check `tests/log_rlds.py`

```py
import numpy as np
from oxe_envlogger.data_type import get_gym_space
from oxe_envlogger.rlds_logger import RLDSLogger, RLDSStepType

# 0. sample data
obs_sample = {"state1": np.array([4, 5, 6]), "state2": np.array([1, 2, 3]),}
action_sample = np.array([0, 1, 2, 4])

# 1. Create RLDSLogger
logger = RLDSLogger(
        observation_space=get_gym_space(obs_sample),
        action_space=get_gym_space(action_sample),
        dataset_name="test",
        directory="logs",
        max_episodes_per_file=1,
)

# 2. log data
logger(action_sample, obs_sample, 1.0, step_type=RLDSStepType.RESTART)
logger(action_sample, obs_sample, 1.0)
logger(action_sample, obs_sample, 1.0, step_type=RLDSStepType.TERMINATION)
logger.close() # this is important to flush the current data to disk
```

Notes: type casting is very important in the env logger. For example, the defined `action_space` and `observation_space` in the env should be provided/returned as what they are defined. Otherwise, the logger will raise an error. Also, take note of specific types like `float32`/`float64`, `int32`/`int64` etc.

## Utilities

**Script to merge and reshard the rlds datasets**

```bash
# --rlds_dirs: directory(s) that contain nested directories of rlds logs
# --output_rlds: directory to store the merged rlds logs
# --shard_size: number of episodes per file
# --overwrite: overwrite the output directory to start from `.tfrecord-00000`
python3 reshard_rlds.py --overwrite --rlds_dirs all_logs --output_rlds output_rlds --shard_size 15
```

Note: The log_dirs can be in a form of `--rlds_dirs all_logs` or `--rlds_dirs all_logs/logs1 all_logs/logs2 all_logs/logs3`. When merging, all logs should have the same format.

**Test scripts for `oxe_envlogger`**

```bash
python tests/log_rlds.py
python tests/log_env.py
```
