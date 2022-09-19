import tensorflow as tf
import numpy as np
import ctypes
from neuroflight_trainer.validation.fc_logging_utils import FlightLog

class MyStructure(ctypes.Structure):
    def __repr__(self) -> str:
        values = ", ".join(f"{name}={value}"
            for name, value in self._asdict().items())
        return f"<{self.__class__.__name__}: {values}>"
    def _asdict(self) -> dict:
        return {field[0]: getattr(self, field[0])
            for field in self._fields_}


class rpy_t (MyStructure):
    _pack_ = 1
    _fields_ = [
        ("roll", ctypes.c_float),     #4B
        ("pitch", ctypes.c_float),    #4B
        ("yaw", ctypes.c_float)       #4B
    ]
    
class action_t (MyStructure):
    _pack_ = 1
    _fields_ = [
        ("top_left", ctypes.c_float),     #4B
        ("top_right", ctypes.c_float),    #4B
        ("bottom_left", ctypes.c_float),  #4B
        ("bottom_right", ctypes.c_float),  #4B
    ]

class observation_t (MyStructure):
    _pack_ = 1
    _fields_ = [
        ("error",   rpy_t),                   #12B
        ("ang_vel", rpy_t),                   #12B 
        ("ang_acc", rpy_t),                   #12B 
        ("prev_action", action_t),            #16B     
        ("iter", ctypes.c_uint16),            #16B 
        ("delta_micros", ctypes.c_uint16)     #16B 
]

class checked_observation_t (MyStructure):
    _pack_ = 1
    _fields_ = [
        ("observation", observation_t),
        ("crc", ctypes.c_uint16)
]

@tf.function
def geo(l,axis=0):
    return tf.exp(tf.reduce_mean(tf.math.log(l),axis=axis))

@tf.function
def p_mean(l, p, slack=0.0, axis=1):
    slacked = l + slack
    if(len(slacked.shape) == 1): #enforce having batches
        slacked = tf.expand_dims(slacked, axis=0)
    batch_size = slacked.shape[0]
    zeros = tf.zeros(batch_size, l.dtype)
    ones = tf.ones(batch_size, l.dtype)
    handle_zeros = tf.reduce_all(slacked > 1e-20, axis=axis) if p <=1e-20 else tf.fill((batch_size,), True)
    escape_from_nan = tf.where(tf.expand_dims(handle_zeros, axis=axis), slacked, slacked*0.0 + 1.0)
    handled = (
            geo(escape_from_nan, axis=axis)
        if p == 0 else
            tf.reduce_mean(escape_from_nan**p, axis=axis)**(1.0/p)
        ) - slack
    res = tf.where(handle_zeros, handled, zeros)
    return res

@tf.function
def p_to_min(l, p=0, q=0):
    deformator = p_mean(1.0-l, q)
    return p_mean(l, p)*deformator + (1.0-deformator)*tf.reduce_min(l)

# @tf.function
# def with_mixer(actions): #batch dimension is 0
#     return actions-tf.reduce_min(actions,axis=1)

# def mixer_diff_dfl(a1,a2):
#     return tf.abs(with_mixer(a1)-with_mixer(a2))/2.0

@tf.function
def weaken(weaken_me, weaken_by):
    return (weaken_me + weaken_by)/(1.0 + weaken_by)

@tf.custom_gradient
def scale_gradient(x, scale):
  grad = lambda dy: (dy * scale, None)
  return x, grad


def to_positive(r):
    return np.clip(1-r,0,1)

def closeness_rw(true_error):
    return p_mean(70.0/(70.0+np.abs(true_error)),0)

# def acc_rw(motor_acc):
#     return with_importance(p_mean(to_positive(np.abs(motor_acc)),0),-0.7)

# def rewards_scalar(rewards_list):
#     closeness, keep_middle, acc = rewards_list
#     return p_mean([closeness,acc], 0)

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


def rewards_fn(obs, act):
    # motor_acc = unroll_act(obs.prev_action) - unroll_act(act)
    closeness = closeness_rw(unroll_rpy(obs.error))
    return closeness


def convert_traj_to_flight_log(traj, save_location):
    print("saving to:", save_location)
    flight_log = FlightLog(save_location)
    last_ob = None
    print(traj)
    for ob, ac in traj:
        last_ob = ob
        print(last_ob)
        rw = rewards_fn(ob,ac).numpy()[0]
        ang_vel = unroll_rpy(ob.ang_vel)
        target = unroll_rpy(ob.error) + ang_vel
        flight_log.add(unroll_obs(ob), 
                       rw, 
                       {"closeness": rw},
                       #ac,
                       unroll_act(ac)*0.5+0.5,
                       ang_vel, 
                       target,
                       np.array([1.0,1.0,1.0,1.0]))
    print(last_ob)
    flight_log.save(0, unroll_obs(last_ob).shape[0])