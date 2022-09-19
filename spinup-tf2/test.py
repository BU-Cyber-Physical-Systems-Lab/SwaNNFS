import tensorflow as tf
import gym
import numpy as np

env = gym.make("gymfc-perlin_discontinuous-v3")
actor = tf.keras.models.load_model("pretty_please")
o = env.reset()

for i in range(400):
	a = actor(np.array([o]))[0].numpy()
	print("action", a)
	o, r, d, info = env.step(a)
	#env.render()
