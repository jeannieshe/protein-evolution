import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import cProfile
import pstats

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import A2C

from protein_evolution.fitness_functions import fitness_ESM, fitness_ESM_DMS
from protein_evolution.callbacks import *
from protein_evolution.environments import ProteinEnv

from datetime import datetime

NUM_ENVS = 64

if __name__ == "__main__":
    with open('data/aav_wt.txt', 'r') as file:
        wt = file.readline().strip()

    def make_env():
        # Provide your own initial sequence + fitness_fn
        return ProteinEnv(wt, fitness_ESM_DMS, 'data/aav_dms.csv')

    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])

    model = A2C(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=0.0007, # Parameter default
        n_steps=6, 
        gae_lambda=0.5,
        gamma=0.99,
        # A2C updates in one go, cannot set n_epochs=10,
        max_grad_norm=0.5, # To try to mimic PPO's clip_range=0.2,
        verbose=1,
        device="cpu"
    )

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H:%M:%S")

    total_timesteps = 10000 #10000
    print(f'Using {total_timesteps} total timesteps')
    tqdm_cb = TQDMCallback(total_timesteps=total_timesteps, algo='A2C')
    logger_cb = ProteinRLLogger(check_freq=1, save_path=f'logs/{formatted_date}')
    callback = CallbackList([tqdm_cb, logger_cb])

    profiler = cProfile.Profile()
    profiler.enable()

    model.learn(total_timesteps=total_timesteps, callback=callback)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 time consumers

    model.save(f'logs/{formatted_date}/a2c_pretraining')