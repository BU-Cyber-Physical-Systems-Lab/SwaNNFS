import tensorflow as tf
import numpy as np
import Pendulum
import matplotlib.pyplot as plt
import pickle

saved = tf.saved_model.load("pendulum/actor")
actor = lambda x: saved(np.array([x]))[0]
env = Pendulum.PendulumEnv(g=10., color=(0.0, 0.8,0.2))
env.seed(123)
o = env.reset()
high = env.action_space.high
low = env.action_space.low
os = []
for _ in range(200):
    o, r, d, i, = env.step(actor(o)*(high - low)/2.0 + (high + low)/2.0)
    os.append(o)
    # env.render()


with open("up_leaning.p", "wb") as file:
	pickle.dump(np.array(os), file)

plt.plot(np.array(os)[:,1])
plt.show()