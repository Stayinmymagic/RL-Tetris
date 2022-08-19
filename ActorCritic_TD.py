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
from src.Model import Actor, Critic
from tensorboardX import SummaryWriter
#%%
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
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
parser.add_argument("--save_interval", type=int, default=1000)
parser.add_argument("--replay_memory_size", type=int, default=30000,
                     help="Number of epoches between testing phases")
parser.add_argument("--saved_path", type=str, default="trained_policy_networks")
args = parser.parse_args()
#%%
# create env
env = Tetris(width=args.width, height=args.height, block_size=args.block_size)

# newwork
policy_network = Actor()
state_value_network = Critic()
# optimizer
policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-3)
state_value_optimizer = optim.Adam(state_value_network.parameters(), lr=1e-3)

writer = SummaryWriter(args.log_path)
#%%
def select_action(x):

    if torch.cuda.is_available():
        x = x.cuda()
    probs = policy_network(x)#reshape

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action_no = m.sample()

    # save to action buffer
    # policy_network.saved_actions.append(SavedAction(m.log_prob(action_no), value))

    return action_no, m.log_prob(action_no)

    

def main():
    # running_reward = 10

    # run inifinitely many episodes
    # for i_episode in count(1):

        # reset environment and episode reward
    state = env.reset()
    ep_reward = 0
    if torch.cuda.is_available():
        policy_network.cuda()
        state = state.cuda()
    # for each episode, only run 9999 steps so that we don't 
    # infinite loop while learning
    epoch = 0
    I = 1
    while epoch < args.num_epochs:
        # if epoch > 2900:
        #     RENDER = True
        # else:
        #     RENDER = False
        next_steps = env.get_next_states() #num of next_steps = 9, 17, 34
        # 從curr_piece當下的方塊推算可能有的動作數量(從num_rotate可知道此方塊可被旋轉幾次)，
        # 可能的STEPS數量都是9, 17, 34 =>所以state size (10 or 18 or 35, 4)
        # 4是tetris board的特徵
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        # x 的最後一維度是oreginal state, 前面是possible next states
        x = torch.cat((next_states, state.unsqueeze(0)))
        
        # select action from policy
        action_no, lp = select_action(x)
        
        next_state = next_states[action_no, :]
        
        action = next_actions[action_no]
        # take the action
        reward, done = env.step(action, render=False)
        # get state value of current state
        state_value = state_value_network(x)
        # get state value of next state
        next_state_value = state_value_network(next_states)
        # if args.render:
        #     env.render()
        
        # backprop
        #if terminal state, next state val is 0
        if done:
            next_state_value = torch.tensor([0]).float().unsqueeze(0)
        #calculate value function loss with MSE
        val_loss = F.mse_loss(reward + args.gamma * next_state_value, state_value)
        val_loss *= I
        
        #calculate policy loss
        advantage = reward + args.gamma * next_state_value.item() - state_value.item()
        policy_loss = -lp * advantage
        policy_loss *= I
        
        #Backpropagate policy
        policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        policy_optimizer.step()
        
        #Backpropagate value
        state_value_optimizer.zero_grad()
        val_loss.backward()
        state_value_optimizer.step()
        
        ep_reward += reward
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            I = 1
            if torch.cuda.is_available():
                state = state.cuda()
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
            # if epoch > 0 and epoch % args.save_interval == 0:
            #     torch.save(policy_network, "{}/tetris_{}".format(args.saved_path, epoch))
        else:
            state = next_state
            I *= args.gamma
            continue
        epoch += 1

    torch.save(policy_network, "{}/tetris".format(args.saved_path))



if __name__ == '__main__':
    main()
