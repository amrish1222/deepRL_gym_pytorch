# -*- coding: utf-8 -*-

# @file    lunarLander_v2.py
# @Author  Amrish Baskaran (amrish1222)
# @copyright  MIT
# @brief  training program for the gym.lunarLander-v2

import gym
from policy_gradent_torch import Agent
from gym import wrappers
import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm
import keyboard

def getKeyPress(act):
    if keyboard.is_pressed('['):
        act = 1
    elif keyboard.is_pressed(']'):
        act = 2
    return act

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(lr = 0.01, input_dims=8, gamma=0.99, num_actions= 4, fcl_dims_list= [128,128])
    
    score_history = []
    score = 0
    num_episodes = 2500
    avg = []
    keypress = 1    
#    env = wrappers.Monitor(env, 'tmp/lunarLander', video_callable = lambda)
    
    for episode in tqdm(range(num_episodes)):
        
        done = False
        score = 0
        observation = env.reset()
        
        
        while not done:
            keypress = getKeyPress(keypress)
            if keypress == 1:
                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_memory(reward, done)
            observation = observation_
            score += reward
        
        score_history.append(score)
        agent.learn()
      
        if len(score_history) > 100:
            avg.append(mean(score_history[-100:]))
        elif len(score_history) <= 1:
            pass
        else:
            avg.append(mean(score_history))
        if episode % 100 == 1 and len(avg) > 0:
            plt.plot(avg)
            plt.pause(0.0001)