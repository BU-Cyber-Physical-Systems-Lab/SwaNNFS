import pickle
import matplotlib.pyplot as plt
import numpy as np

def unroll_rpy(rpy):
    return np.array([rpy.roll, rpy.pitch, rpy.yaw])

def unroll_act(act):
    return np.array([
        act.top_left,
        act.top_right,
        act.bottom_left,
        act.bottom_right,
    ])

def unroll_obs(obs):
    return np.concatenate([
        unroll_rpy(obs.error),
        unroll_rpy(obs.ang_vel),
        unroll_rpy(obs.ang_acc),
        unroll_act(obs.prev_action)
    ])

def plot_traj(traj):
    fig = plt.figure(constrained_layout=True)
    rpy_subfig = [["r"], ["p"], ["y"]]
    act_subfig = [["a1"], ["a2"], ["a3"], ["a4"]]
    ax = fig.subplot_mosaic([[rpy_subfig, act_subfig]])
    unrolled = np.array(list(map(lambda obs_act: unroll_obs(obs_act[0]), traj))).T
    try:
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
        plt.show()
    except:
        print("err")

traj = pickle.load( open( "traj.p", "rb" ) )
plot_traj(traj)