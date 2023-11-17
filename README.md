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

Just add the following lines to your code to wrap your env with the logger. For more detailed example, check `run_gym.py`

```py
from oxe_envlogger.envlogger import OXEEnvLogger

env = YOUR_GYM_ENV
env = OXEEnvLogger(
    env,
    YOUR_DATASET_NAME,
    directory=YOUR_OUTPUT_DIR
    max_episodes_per_file=500,
)
```
