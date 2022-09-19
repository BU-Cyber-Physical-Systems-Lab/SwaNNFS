import numpy as np
import gym
import shutil
import time
import os
from signal import signal, SIGINT

import neuroflight_trainer.gyms
from neuroflight_trainer.validation.fc_logging_utils import FlightLog
import tensorflow as tf

def existing_actor():
    return tf.keras.models.load_model(
        "/data/neuroflight/CODE/gymfc-nf1/training_data/results/tf2_ddpg_cff12c6b_s820484_t220611-080203/checkpoints/ckpt_61/actor"
    )

def save_flight():
    num_episodes=1
    env = gym.make("gymfc_perlin_discontinuous-v3")
    env.noise_sigma = 1
    save_location = "."
    actor = existing_actor()
    print("saving to:", save_location)
    flight_log = FlightLog(save_location)
    num_total_steps = 0
    rewards_sum = 0
    for episode_index in range(num_episodes):
        env.ep_counter = 0
        env.seed(episode_index)
        ob = env.reset()
        env.validation = True
        done = False
        while not done:
            ac = actor(np.array([ob]))[0].numpy()
            ob, composed_reward, done, ep_info = env.step(ac)
            done = env.sim_time > 4
            num_total_steps += 1
            rewards_sum += composed_reward
            flight_log.add(ob, 
                           composed_reward, 
                           ep_info['rewards_dict'],
                           #ac,
                           env.y,
                           env.imu_angular_velocity_rpy, 
                           env.omega_target,
                           env.esc_motor_angular_velocity,
                           dbg=env.dbg)
        flight_log.save(episode_index, ob.shape[0])
    return rewards_sum/num_total_steps

if __name__ == '__main__':
  save_flight()