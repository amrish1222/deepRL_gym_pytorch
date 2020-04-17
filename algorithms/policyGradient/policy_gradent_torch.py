# -*- coding: utf-8 -*-

# @file    policy_gradient_torch.py
# @Author  Amrish Baskaran (amrish1222)
# @copyright  MIT
# @brief  network architecture and agent for policy gradient

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, fcl_dims_list, num_actions):
        super(PolicyNetwork,self).__init__()
        
        self.input_dims = input_dims
        self.lr = lr
        self.fcl_dims_list = fcl_dims_list + [num_actions]
        self.fcl_list = []
        
        
        for i in range(len(self.fcl_dims_list)):
            if i == 0:
                in_dims = self.input_dims
            else:
                in_dims = self.fcl_dims_list[i-1]
                
            out_dims = self.fcl_dims_list[i]
            self.fcl_list.append(nn.Linear(in_dims, out_dims))
            self.add_module("hidden_layer"+str(i), self.fcl_list[-1])
        
            
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Device : {self.device}')
        
        self.to(self.device)
        
    def forward(self, observation):
        state = torch.Tensor(observation).to(self.device)
        x = state
        for i,fcl in enumerate(self.fcl_list):
            if i ==  len(self.fcl_list) -1 :
                x = fcl(x)
            else:
                x = F.relu(fcl(x))
        
        return x
    
class Agent(object):
    def __init__(self, lr, input_dims, gamma = 0.99, num_actions = 4,
                 fcl_dims_list = [256,256]):
        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []
        self.done_memory = []
        self.policy = PolicyNetwork(lr, input_dims, fcl_dims_list, num_actions)
        
    def choose_action(self, observation):
        probs = F.softmax(self.policy.forward(observation), dim= 0)
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        
        return action.item()
    
    def store_memory(self, reward, done):
        self.reward_memory.append(reward)
        self.done_memory.append(done)
        
    def learn(self):
        self.policy.optimizer.zero_grad()
        G = deque()
        discounted_reward = 0
        for reward, done in zip(reversed(self.reward_memory), reversed(self.done_memory)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            G.appendleft(discounted_reward)
        G = np.asarray(G, dtype = np.float64)
        
        G = torch.tensor(G).to(self.policy.device)
        G = (G - G.mean()) / (G.std() + 1e-5)
        
        loss = 0
        
        for g, log_prob in zip(G, self.action_memory):
            loss += -g * log_prob
        
        loss.backward()
        self.policy.optimizer.step()
        
        self.action_memory = []
        self.reward_memory = []
        self.done_memory = []
        
        