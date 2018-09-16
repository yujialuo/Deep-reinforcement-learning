#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python dagger.py experts/Walker2d-v2.pkl Walker2d-v2 --render --num_rollouts 40

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import pickle
from sklearn.utils import shuffle


DATA_PATH = 'expert_data/'
MODEL_PATH = 'models/'
NUM_LAYER_UNITS = 64
TRAIN_ITER = 3000
BATCH_SIZE = 2048
NUM_TRAIN_BATCH_ITERS = 10


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    # Get Data
    data_file = os.path.join(DATA_PATH, "{}.pkl".format(args.envname))
    with open(data_file, "rb") as f:
        data = pickle.loads(f.read())
    expert_obs = data['observations']
    expert_acts = data['actions']
    obs_dim = expert_obs.shape[1]
    act_dim = expert_acts.shape[2]
    expert_acts = expert_acts.reshape(-1, act_dim)
    expert_obs, expert_acts = shuffle(expert_obs, expert_acts, random_state=0)

    # Build Model
    x = tf.placeholder(tf.float32, [None, obs_dim])
    y = tf.placeholder(tf.float32, [None, act_dim])
    w = tf.Variable(tf.truncated_normal([obs_dim, NUM_LAYER_UNITS]))
    b = tf.Variable(tf.truncated_normal([NUM_LAYER_UNITS]))
    h = tf.tanh(tf.matmul(x, w) + b)
    w2 = tf.Variable(tf.truncated_normal([NUM_LAYER_UNITS, NUM_LAYER_UNITS]))
    b2 = tf.Variable(tf.truncated_normal([NUM_LAYER_UNITS]))
    h2 = tf.tanh(tf.matmul(h, w2) + b2)
    w3 = tf.Variable(tf.truncated_normal([NUM_LAYER_UNITS, act_dim]))
    b3 = tf.Variable(tf.truncated_normal([act_dim]))
    y_hat = tf.matmul(h2, w3) + b3
    loss = tf.reduce_sum(tf.losses.mean_squared_error(y_hat, y))
    train_target = tf.train.AdamOptimizer().minimize(loss)

    # Train and Deploy Model
    train_i = 0
    total_mean = []
    total_std = []
    model_path = MODEL_PATH + "cloned_model_{}".format(args.envname)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf_util.initialize()
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        for i in range(NUM_TRAIN_BATCH_ITERS):
            # Train Model
            for j in range(TRAIN_ITER):
                random_i = np.random.choice(len(expert_obs), BATCH_SIZE)
                obatch, abatch = expert_obs[random_i], expert_acts[random_i]
                sess.run(train_target, feed_dict={x: obatch, y: abatch})
            train_i += TRAIN_ITER

            # Deploy Model
            print('train iter', train_i)
            print(sess.run(loss, feed_dict={x: obatch, y: abatch}))

            returns = []
            observations = []
            actions = []

            for i in range(args.num_rollouts):
                # print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    observations.append(obs)
                    action_exp = policy_fn(obs[None, :])
                    actions.append(action_exp)
                    action = sess.run(y_hat, feed_dict={x: np.array([obs]), y: np.array([expert_acts[0]])})
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            # Augment Data
            np.append(expert_obs, np.array(observations), axis=0)
            actions = np.array(actions).reshape(-1, expert_acts.shape[1])
            np.append(expert_acts, actions, axis=0)

            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
            total_mean.append(np.mean(returns))
            total_std.append(np.std(returns))

            if train_i % 10000 == 0:
                saver.save(sess, model_path, global_step=train_i)

    print("total_mean", total_mean)
    print("total_std", total_std)


if __name__ == '__main__':
    main()
