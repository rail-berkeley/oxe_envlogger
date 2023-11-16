# oxe_envlogger

> Make data collection for robot learning easy ➡️ more reusable datasets. 🤖📈

Env logger for robots. Related to [open-x-embodiment](https://robotics-transformer-x.github.io/)

## Installation

```bash
cd oxe_envlogger
pip install -e .
```

## Quick Run

Without env_logger  
```bash
python run_gym.py --env_name="HalfCheetah-v4" 
```


```bash
# create 
mkdir -p datasets/half_cheetah/0.1.0

# Run a gym environment
python run_gym.py --env_name="HalfCheetah-v4" --enable_envlogger --output_dir="datasets/half_cheetah/0.1.0"
```

This will store the data in `datasets/half_cheetah` directory. check the dataformat in `cat datasets/half_cheetah/0.1.0/features.json`



## Try load 

A simple example to create a dataset which is compatible with tensorflow_dataset's [RTX](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/robotics)

```bash
cp -rf datasets/half_cheetah/ ~/tensorflow_datasets/

python load_example_oxe.py
```
