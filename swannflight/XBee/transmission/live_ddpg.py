from dataclasses import asdict, dataclass
from typing import NamedTuple
import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.ddpg import core
from spinup.utils.logx import EpochLogger
import obs_utils
import pickle

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class HyperParams:
    def __init__( self,
            ac_kwargs={"actor_hidden_sizes":(32,32), "critic_hidden_sizes":(400,300)},
            seed=int(time.time()* 1e5) % int(1e6),
            steps_per_epoch=1000,
            replay_size=int(1e6),
            gamma=0.9,
            polyak=0.995,
            pi_lr=1e-5,
            q_lr=1e-5,
            batch_size=300,
            train_every=50,
            train_steps=30,
        ):
        self.ac_kwargs = ac_kwargs
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.batch_size = batch_size
        self.train_every = train_every
        self.train_steps = train_steps
    
"""

Deep Deterministic Policy Gradient (DDPG)

"""
def live_ddpg(obs_queue, obs_space, act_space, hp: HyperParams=HyperParams(),actor_critic=core.mlp_actor_critic, logger_kwargs=dict(), save_freq=1, on_save=lambda *_:(), anchor_q=None):
    """

    Args:
        obs_queue : a queue containing (observation1, action, observation2) objects

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """


    logger = EpochLogger(**logger_kwargs)
    logger.save_config(hp.__dict__)

    tf.random.set_seed(hp.seed)
    np.random.seed(hp.seed)

    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    max_q_val = 1.0/(1.0-hp.gamma)

    # Main outputs from computation graph
    with tf.name_scope('main'):
        pi_network, q_network = actor_critic(obs_space, act_space, **hp.ac_kwargs)


    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=hp.replay_size)
    anchor_replay = pickle.load( open( "replay.p", "rb" ) )

    # Separate train ops for pi, q
    pi_optimizer = tf.keras.optimizers.Adam(learning_rate=hp.pi_lr)

    @tf.function
    def pi_update(obs1, obs2, anchor_obs1):
        with tf.GradientTape() as tape:
            pi = pi_network(obs1)
            pi2 = pi_network(obs2)
            anchor_pi = pi_network(anchor_obs1)
            q_c = tf.reduce_mean(q_network(tf.concat([anchor_obs1, anchor_pi], axis=-1))**0.5)**2.0

            noise = tf.random.normal(
                [13], mean=0.0, stddev=np.array([
                    1.0,1.0,1.0,
                    1.0,1.0,1.0,
                    0.1,0.1,0.1,
                    0.0,0.0,0.0,0.0]),
            )
            pi_bar = pi_network(obs1+noise)
            var_sum = sum(map(lambda v: tf.reduce_mean(tf.abs(v)),pi_network.trainable_variables))
            pi_weight_c = tf.stack([10.0/(10.0+var_sum)])**2.0
            before_tanh_c = tf.stack([(20.0/(20.0+sum(pi_network.losses)))**4.0])
            # objective for regularizing the output of the nn as well as the weights
            # tf.print(pi_network.trainable_variables)
            # tf.print(pi_network.losses)

            temporal_c = obs_utils.p_mean(obs_utils.p_mean((0.2/(0.2+tf.abs(pi-pi2)))**2.0, 1.0), 1.0)
            # objective for minimizing subsequent action differences

            spatial_c = obs_utils.p_mean(obs_utils.p_mean(0.1/(0.1+tf.abs(pi-pi_bar)), 0.), 0.)
            # objective representing similar inputs should map to similar outputs

            center_c = obs_utils.p_mean(obs_utils.p_mean(1.0 - (tf.abs(pi+0.76)/1.761),0.),0.)
            # objective for maintaining an output of -0.8

            reg_c = tf.squeeze(obs_utils.p_mean(tf.stack([spatial_c, temporal_c, before_tanh_c],axis=1), 0.0))
            # all_c = p_to_min(tf.stack([q_c, reg_c]), q=0.0)
            all_c = obs_utils.p_mean(tf.stack([tf.stack([obs_utils.scale_gradient(q_c, 3e2)]), obs_utils.scale_gradient(before_tanh_c, 3.0), spatial_c, obs_utils.scale_gradient(temporal_c, 1.0)], axis=1), 0.0)
            # all_c = q_c + 0.008*pi_diffs_c + 0.005*pi_bar_c + 0.025*center_c
            # if debug:
            tf.print("temporal_c", temporal_c)
            tf.print("spatial_c", spatial_c)
            tf.print("pi_weight_c", pi_weight_c)
            tf.print("before_tanh_c", before_tanh_c)
            tf.print("reg_c", reg_c)
            tf.print("q_c", q_c)
            pi_loss = 1.0 - all_c
        grads = tape.gradient(pi_loss, pi_network.trainable_variables)
        grads_and_vars = zip(grads, pi_network.trainable_variables)
        pi_optimizer.apply_gradients(grads_and_vars)
        return pi_loss, center_c, q_c, 0, reg_c

    start_time = time.time()
    t = 0
    # Main loop: collect experience in env and update/log each epoch
    while True:
        (o, a, o2) = obs_queue.get()
        t+=1
        r = obs_utils.rewards_fn(o, a)
        d = False
        # Store experience to replay buffer

        replay_buffer.store(
            obs_utils.unroll_obs(o),
            obs_utils.unroll_act(a),
            r,
            obs_utils.unroll_obs(o2),
            d
        )

        if t % hp.train_every == 0:
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for _ in range(hp.train_steps):
                batch = replay_buffer.sample_batch(hp.batch_size)
                anchor_obs_batch = anchor_replay.sample_batch(hp.batch_size)
                anchor_obs = tf.constant(anchor_obs_batch['obs1'])
                obs1 = tf.constant(batch['obs1'])
                obs2 = tf.constant(batch['obs2'])
                acts = tf.constant(batch['acts'])
                rews = tf.constant(batch['rews'])
                dones = tf.constant(batch['done'])
                # Q-learning update
                # loss_q, q_vals = q_update(obs1, obs2, acts, rews, dones)
                # logger.store(LossQ=loss_q)

                # Policy update
                pi_loss, avoid_extremes, qs, safe_qs, reg = pi_update(obs1, obs2, anchor_obs)
                logger.store(
                    LossPi=pi_loss.numpy(),
                    NormQ=qs,
                    NormSafe=safe_qs,
                    AvoidExtremes=avoid_extremes,
                    Reg=reg
                )

        # End of epoch wrap-up
        if t > 0 and t % hp.steps_per_epoch == 0:
            epoch = t // hp.steps_per_epoch

            # Save model
            if (epoch % save_freq == 0):
                on_save(pi_network, q_network, epoch//save_freq)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('NormQ', average_only=True)
            logger.log_tabular('NormSafe', average_only=True)
            logger.log_tabular('AvoidExtremes', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('Reg', average_only=True)
            logger.dump_tabular()