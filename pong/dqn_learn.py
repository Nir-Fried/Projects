import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from matplotlib import pyplot as plt
from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}

def dqn_learing(
    env,
    q_func,
    optimizer_spec,
    exploration,
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000
    ):
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1: 
        input_arg = env.observation_space.shape[0]
    else: 
        img_h, img_w, img_c = env.observation_space.shape 
        input_arg = frame_history_len * img_c # number of most recent frames to produce the input to the network
    num_actions = env.action_space.n

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold: 
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0 # normalize the observation
            # with torch.no_grad() variable is only used in inference mode, i.e. don’t save the history
            with torch.no_grad():
                return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    # Initialize target q function and q function

    Q = q_func(input_arg, num_actions)
    
    Q_target = q_func(input_arg, num_actions)
    Q_target.load_state_dict(Q.state_dict()) # copy Q's weights and biases to Q_target
    
    if USE_CUDA:
        print("cuda")
        Q = Q.cuda()
        Q_target = Q_target.cuda()

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    for t in count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            break

        ### 2. Step the env and store the transition
        
        # if this is the first iteration, the action is random
        if t == 0:
            action = np.random.randint(num_actions)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            idx = replay_buffer.store_frame(obs)
            obs = replay_buffer.encode_recent_observation()
            
        else:
            action = select_epilson_greedy_action(Q, obs, t)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            
            replay_buffer.store_effect(idx, action, reward, done) # store the effect of the action in the replay buffer
            idx = replay_buffer.store_frame(obs) # store the frame in the replay buffer
            obs = replay_buffer.encode_recent_observation() # encode the most recent frame to be fed to the network


        ### 3. Perform experience replay and train the network.
       
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)): # if the replay buffer can sample a batch of transitions
            
            # 3.a sample a batch of transitions
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size) # get a batch of transitions from the replay buffer
            obs_batch = Variable(torch.from_numpy(obs_batch) / 255.0) 
            act_batch = Variable(torch.from_numpy(act_batch).long())
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch) / 255.0)
            done_mask = Variable(torch.from_numpy(done_mask).type(torch.BoolTensor))
            bellman_values = Variable(torch.from_numpy(rew_batch))

            if USE_CUDA:
                obs_batch = obs_batch.cuda()
                act_batch = act_batch.cuda()
                next_obs_batch = next_obs_batch.cuda()
                bellman_values = bellman_values.cuda()

            # 3.b compute the bellman error
            Q_target_values = Q_target(next_obs_batch).max(1)[0] # will return the max value of each row
            Q_target_values[done_mask == 1.0] = 0.0 # if the episode is done, the Q value is 0
            bellman_values += gamma * Q_target_values # Q(s,a) = r + gamma * max(Q(s',a'))
            bellman_values = bellman_values.unsqueeze(1) # add a dimension to the tensor since the loss function expects a 2D tensor

            current_Q = Q(obs_batch).gather(1, act_batch.unsqueeze(1)) # Q(s,a)
            bellman_error = bellman_values - current_Q # bellman error

            # clip the bellman error between [-1,1] 
            bellman_error = Variable(-1 * torch.clip(bellman_error, -1, 1))

            if USE_CUDA:
                bellman_error = bellman_error.cuda()


            # 3.c train the model
            optimizer.zero_grad()
            current_Q.backward(bellman_error.data) 
            optimizer.step()

            # 3.d periodically update the target network
            num_param_updates += 1
            if num_param_updates % target_update_freq == 0: 
                Q_target.load_state_dict(Q.state_dict())
                Q_target.eval() 

        ### 4. Log progress and keep track of statistics
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        Statistic["mean_episode_rewards"].append(mean_episode_reward)
        Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("---------------------------------------")
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            # Dump statistics to pickle
            with open('statistics.pkl', 'wb') as f:
                pickle.dump(Statistic, f)
                print("Saved to %s" % 'statistics.pkl')
