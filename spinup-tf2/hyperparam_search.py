import spinup
import numpy as np
import tensorflow as tf
import gym
import time
import neuroflight_trainer.gyms
import os
import random
random.seed(0)
from train_nf1_ddpg import train_nf1
# from spinup.utils.run_utils import setup_logger_kwargs

num_step = 300_000

hyperparams_possibilities = {
    "seed":            lambda: int(time.time()* 1e5) % int(1e6),
    "steps_per_epoch": lambda: 10000,
    "replay_size":     lambda: random.choice([100_000, 500_000, 1000_000, 5000_000]),
    "gamma":           lambda: random.choice([0.8, 0.9, 0.95, 0.99]),
    "polyak":          lambda: random.choice([0.5, 0.9, 0.95, 0.99, 0.995]),
    "pi_lr":           lambda: random.choice([5e-4, 1e-3, 3e-3, 5e-3]),
    "q_lr":            lambda: random.choice([5e-4, 1e-3, 3e-3, 5e-3]),
    "batch_size":      lambda: random.choice([50, 100, 200, 400]),
    "act_noise":       lambda: random.choice([0.01, 0.05, 0.1, 0.2]),
    "max_ep_len":      lambda: 10000,
}


def sample(hyperparams):
    sampled_params = {key: sampler() for key, sampler in hyperparams.items()}
    sampled_params["epochs"] = num_step//sampled_params["steps_per_epoch"]
    return sampled_params


def search():
    for i in range(100):
        hyperparams = sample(hyperparams_possibilities)
        with open('hyperparams.txt', 'a') as file:
            os.system("killall gzserver")
            file.write(f"{hyperparams}: {train_nf1(**hyperparams)}\n")


if __name__ == '__main__':
    search()