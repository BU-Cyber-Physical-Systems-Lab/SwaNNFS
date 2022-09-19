import transmitter
import receiver
import time
import serial
from threading import Thread
import xbee
import datetime
from anchor_live_ddpg import live_ddpg
import copy
from gym import spaces
import numpy as np
import pickle
import os
from multiprocessing import Process, Queue, Value
import tensorflow as tf
import obs_utils
import matplotlib.pyplot as plt
import ctypes

class Paths:
    def __init__(self):
        self.shared_ckpt_index = Value('d', 0)
        self.shared_traj_index = Value('d', 0)
        self.timestr = time.strftime("y%Ym%md%d-h%Hm%Ms%S")

    def ckpt_path(self):
        path = os.path.join("runs", self.timestr, f"ckpt_{self.shared_ckpt_index.value}")
        os.makedirs(path, exist_ok=True)
        return path

    def traj_path(self):
        print(self.ckpt_path())
        path = os.path.join(self.ckpt_path(), f"traj_{self.shared_traj_index.value}")
        os.makedirs(path, exist_ok=True)
        return path


paths = Paths()

def clear_queue(q):
    while not q.empty():
        q.get()

def existing_actor_critic(*args, **kwargs):
    actor = tf.keras.models.load_model("actor")
    critic = tf.keras.models.load_model("critic")
    return actor, critic 

def live_training(nn_queue, obs_queue):
    action_space = spaces.Box(-np.ones(4), np.ones(4), dtype=np.float32)
    observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)
    def on_save(actor, critic, epoch):
        paths.shared_ckpt_index.value = epoch
        paths.shared_traj_index.value = 0
        save_path = paths.ckpt_path()
        critic.save(os.path.join(save_path, "critic"))
        actor.save(os.path.join(save_path, "actor"))
        print("saving to", save_path)
        print("Sending new actor!")
        converted = tf.lite.TFLiteConverter.from_keras_model(actor).convert()
        with open("sent.tflite", "wb") as f:
            f.write(converted)
        nn_queue.put(converted)

    live_ddpg(
        obs_queue,
        observation_space,
        action_space,
        actor_critic=existing_actor_critic,
        on_save=on_save,
        anchor_q=tf.keras.models.load_model("critic")
    )


def plot_traj(traj):
    fig = plt.figure(constrained_layout=True)
    rpy_subfig = [["r"], ["p"], ["y"]]
    act_subfig = [["a1"], ["a2"], ["a3"], ["a4"]]
    ax = fig.subplot_mosaic([[rpy_subfig, act_subfig]])
    unrolled = np.array(list(map(lambda obs_act: obs_utils.unroll_obs(obs_act[0]), traj))).T
    if len(traj) > 50:
        try:
            obs_utils.convert_traj_to_flight_log(traj, paths.traj_path())
            er = unrolled[0]
            ep = unrolled[1]
            ey = unrolled[2]
            r = unrolled[3]
            p = unrolled[4]
            y = unrolled[5]
            sr = er + r
            sp = ep + p
            sy = ey + y

            ax["r"].plot(r)
            ax["p"].plot(p)
            ax["y"].plot(y)

            ax["r"].plot(sr)
            ax["p"].plot(sp)
            ax["y"].plot(sy)

            ax["a1"].plot(unrolled[-4])
            ax["a2"].plot(unrolled[-3])
            ax["a3"].plot(unrolled[-2])
            ax["a4"].plot(unrolled[-1])
            plt.savefig(os.path.join(paths.traj_path(), "traj.pdf"))
        except:
            print("err")
    else:
        print("trajectory too short: ", len(traj))

def save_traj(traj):
    plot_traj(traj)
    pickle.dump( traj, open( os.path.join(paths.traj_path(), "traj.p"), "wb" ) )
    paths.shared_traj_index.value += 1


def tranceive(ser, nn_queue, obs_queue, circular_buffer_size=2000):
    check = True
    obs_dropped_count = 0
    next_obs = None
    current_traj = []
    while True:
        while nn_queue.empty():
            check, obs = receiver.receive_obs(ser, check, keep_checking=True, debug=False)
            if next_obs:
                if obs.iter + 1 == next_obs.iter: # trajectories are reversed
                    copied_next_obs=copy.deepcopy(next_obs)
                    current_traj.append((obs, copied_next_obs.prev_action))
                    obs_queue.put((obs, copied_next_obs.prev_action, copied_next_obs))
                    if obs_queue.qsize() > circular_buffer_size:
                        obs_queue.get()
                else:
                    # print(current_traj)
                    current_traj.reverse()
                    save_traj(current_traj)
                    current_traj = []
                    obs_dropped_count += 1
                    print("obs dropped:", obs_dropped_count)
            next_obs = obs
        NN_to_send = nn_queue.get_nowait() # nn_queue should always be non-empty
        xbee.empty_read_buffer(ser)
        print("TRANSMITTING")
        transmitter.transmit(ser, NN_to_send)
        xbee.empty_read_buffer(ser)
        print("DEBUG: bytes on input buffer: ", ser.in_waiting)
        print("DEBUG: sending b")
        ser.write(b'b') # Telling drone to send metadata of the NN
        print("DEBUG: Sent the 'b' command; waiting for metadata")
        
        transmitter.receive_metadata(ser)
        time.sleep(.1)
        xbee.empty_read_buffer(ser) #EXPECT BUFFER TO BE EMPTY
        
        print("DEBUG: bytes on input buffer: ", ser.in_waiting)
        print("DEBUG: sending c")
        ser.write(b'c') # Telling drone to re-start sending obs data by convention, drone waits for 'c'
        print("DEBUG: Sent the 'C' command; waiting for OBSdata")
        clear_queue(nn_queue)
        check = True

if __name__ == '__main__':
    ser = xbee.init()
    print("Number of pending bytes on imput buffer:", ser.in_waiting)
    print("Reading the dummy byte from drone: ")
    # print(ser.read(4))
    nn_queue = Queue()
    obs_queue = Queue()
    # time.sleep(3)

    Process(target=live_training, args=(nn_queue, obs_queue)).start()
    tranceive(ser, nn_queue, obs_queue)
    #Thread(target=lambda: receiver.keep_receiving(ser)).start()