# oxe_envlogger

> Make data collection for robot learning easy ➡️ more reusable datasets. 🤖📈

Env logger for robots. Related to [open-x-embodiment](https://robotics-transformer-x.github.io/)

## Installation

```bash
cd oxe_envlogger
pip install -e .
```

## Quick Run

1. A simple example is provided to show how to record a gym env.

Without env_logger
```bash
python run_gym.py --env_name="HalfCheetah-v4" 
```

2. Now, with env_logger `--enable_envlogger`

```bash
# create directory
mkdir -p datasets/half_cheetah/0.1.0

# Run a gym environment
python run_gym.py --env_name="HalfCheetah-v4" --enable_envlogger --output_dir="datasets/half_cheetah/0.1.0"
```

3. Check the recorded `tfds` data

This stores the data in `datasets/half_cheetah` directory. Check the data format in `cat datasets/half_cheetah/0.1.0/features.json`

## Try load 

After collecting data with the cheetah env, we will now try to load the data.
Here is a simple example of a dataset class which is compatible with tensorflow_dataset's [RTX](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/robotics)

```bash
cp -rf datasets/half_cheetah/ ~/tensorflow_datasets/

python load_example_oxe.py
```
