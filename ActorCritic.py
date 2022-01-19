# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:33:08 2022

@author: notfu
"""

import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from src.tetris import Tetris
from src.deep_q_network import ActorCritic
from tensorboardX import SummaryWriter
#%%
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument("--num_epochs", type=int, default=3000)
parser.add_argument("--width", type=int, default=10, help="The common width for all images")
parser.add_argument("--height", type=int, default=20, help="The common height for all images")
parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
parser.add_argument("--log_path", type=str, default="tensorboard")
args = parser.parse_args()
#%%
env = Tetris(width=args.width, height=args.height, block_size=args.block_size)
torch.cuda.manual_seed(123)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
#%%
model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()
writer = SummaryWriter(args.log_path)
#%%
def select_action(x):

    probs, value = model(x.cuda())#reshape
    #print(probs, value)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action_no = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action_no), value))
    # print(m.log_prob(action_no))
    # the action to take (left or right)
    return action_no

def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values
    # print(len(model.rewards))
    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    # print(returns)
    # returns = (returns - returns.mean()) / (returns.std() + eps)
    # print(returns)
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)
        # print(value.data)
        # print(R)
        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value.data, R.clone().detach().requires_grad_(True).cuda()))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

def main():
    running_reward = 10
    last_score = 0
    # run inifinitely many episodes
    # for i_episode in count(1):

        # reset environment and episode reward
    state = env.reset()
    ep_reward = 0
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()
    # for each episode, only run 9999 steps so that we don't 
    # infinite loop while learning
    epoch = 0
    while epoch < args.num_epochs:
        next_steps = env.get_next_states() #num of next_steps = 9, 17, 34
        # 從curr_piece當下的方塊推算可能有的動作數量(從num_rotate可知道此方塊可被旋轉幾次)，可能的STEPS數量都是9, 17, 34
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        # x 的最後一維度是oreginal state, 前面是possible next states
        x = torch.cat((next_states, state.unsqueeze(0)))
        
        # select action from policy
        action_no = select_action(x)
        next_state = next_states[action_no, :]
        action = next_actions[action_no]
        # take the action
        reward, done = env.step(action, render=True)
        # print(reward, done)
        # if args.render:
        #     env.render()

        model.rewards.append(reward)
        ep_reward += reward
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue
        epoch += 1
        # update cumulative reward
        # running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            args.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)
        # log results
        # if i_episode % args.log_interval == 0:
        #     print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
        #           i_episode, ep_reward, running_reward))
    
        # check if we have "solved" the cart pole problem
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()
#%%

torch.save(model, "tetris_fixreward")