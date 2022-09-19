### SwaNNFlight Framework

The SwaNNFlight Framework is divided into several different code bases:
* spinup-tf2 defines our custom version of ddpg which we term in the paper psiDDPGx.
* Pendulum defines the pendulum environment and files for calling ddpg
* gymfc-nf1 defines the environment used for training the quadrotor's policies
* swannflight defines the fork of betaflight 3.3x which has all the necessary tools for swapping neural networks on-the-fly
* extra-data contains a plot showing the agent behaving erradicly while adapting without anchors

The hyperparameters used to train the quadrotor agents are defined inside spinup-tf2/train_nf1_ddpg.py by the HyperParameters class

