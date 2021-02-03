import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils
from torch.distributions import Categorical

from typing import *
from gym import spaces

from matplotlib import pyplot as plt

'''
class Actor:
    Actor Module - Neural Network for approximating the policy
'''
class Actor(nn.Module):
    def __init__(self, n_state, n_action):
        super(Actor, self).__init__()
        self.hidden1 = nn.Linear(n_state, 24)
        self.out = nn.Linear(24, n_action)

    def forward(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        x = self.hidden1(x)
        x = torch.tanh(x)
        x = self.out(x)
        # out = Categorical(F.softmax(x, dim=-1))
        out = F.softmax(x, dim=-1)
        return out 

'''
class Critic:
    Critic Module - Neural Network for approximating the value
'''
class Critic(nn.Module):
    def __init__(self, n_state):
        super(Critic, self).__init__()
        self.hidden1 = nn.Linear(n_state, 24)
        self.hidden2 = nn.Linear(24, 24)
        self.out = nn.Linear(24, 1)

    def forward(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        x = self.hidden1(x)
        x = torch.tanh(x)
        x = self.hidden2(x)
        x = torch.tanh(x)
        out = self.out(x)
        return out

'''
class CartPoleAgent:
    RL agent for learning CartPole task
    This agent learn the task based on Actor-critic
'''
class CartPoleAgent:
    def __init__(self, n_state, n_action):
        self.actor: Actor = Actor(n_state=n_state, n_action=n_action)
        self.critic: Critic = Critic(n_state=n_state)

        self.n_action = n_action
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        #torch_utils.clip_grad_norm_(self.actor.parameters(), 5.0)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        #torch_utils.clip_grad_norm_(self.critic.parameters(), 5.0)

    def get_action(self, state):
        policy = self.actor(state)
        
        '''
        action = policy.sample()
        action = action.numpy()[0]
        '''

        
        policy = policy.detach().numpy().squeeze(0)
        # print(policy)
        action = np.random.choice(self.n_action, 1, p=policy)[0]
        
        # print(action)
        return action

    def train_model(self, state, action, reward, next_state, done):
        #print('Train Start')
        self.actor_optimizer.zero_grad(), self.critic_optimizer.zero_grad()
        agent.actor.train(), agent.critic.train()
        policy, value, next_value = self.actor(state), self.critic(state), self.critic(next_state)
        # print(policy.shape, value.shape)
        # print(f'policy: {policy}, {policy.shape}')
        # print(f'value: {value}, next_value: {next_value}')

        # target = torch.from_numpy(reward + (1 - done) * self.discount_factor * next_value.detach().numpy()[0]).reshape(1, 1)
        #target = torch.from_numpy(reward + (1 - done) * self.discount_factor * next_value.detach().numpy()[0])
        target = reward + (1 - done) * self.discount_factor * next_value
        # print(f'target: {target}, {target.shape}, {target.dtype}')
        #print(f'reward = {reward}')
        '''
        action_prob = policy[0][action]
        '''

        
        one_hot_action = torch.zeros(n_action)
        one_hot_action[action] = policy[0][action]
        action_prob = one_hot_action.reshape(2)

        # action_prob = torch.from_numpy(one_hot_action).reshape(2)
        action_prob = torch.sum(action_prob)
        
        #logprob = policy.log_prob(torch.Tensor([action]))

        #print(f'action_prob: {action_prob}, {action_prob.shape}')
        advantage = (target - value.item()).detach().reshape(1)
        # print(f'advantage: {advantage}, {advantage.shape}, {advantage.dtype}')

        actor_loss = -(torch.log(action_prob + 1e-5)*advantage)
        # actor_loss = -(logprob*advantage.detach())

        #new_target = target.detach()
        #critic_loss = 0.5 * torch.square(new_target - value[0])
        critic_loss = 0.5 * torch.square(target.detach() - value[0])

        #critic_loss = (target.detach() - value.item()).pow(2)
        # critic_loss = advantage.pow(2).mean()
        #critic_loss = torch.mean(critic_loss)
        
        loss = 0.2 * actor_loss + critic_loss
        loss.backward()
        #print(f'loss: {loss.item()}')

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return loss.item()



env: gym.Env = gym.make('CartPole-v1')
state = env.reset()

n_state: int = env.observation_space.shape[0]
n_action: int = env.action_space.n 
print(env.observation_space)
print(n_state, n_action)
print(state)

agent: CartPoleAgent = CartPoleAgent(n_state, n_action)

'''
action = agent.get_action(state)
print(action)
next_state, reward, done, _ = env.step(action)
print(f'next_state: {next_state}, reward: {reward}, done: {done}')
agent.train_model(state, action, reward, next_state, done)
'''

scores = []
losses = []
for i in range(500):
    state = env.reset()
    score = 0
    done = False
    loss = 0
    j = 0
    while not done:
        env.render()
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        score += reward
        reward = 0.1 if not done or score >= 500 else -1
        
        loss += agent.train_model(state, action, reward, next_state, done)
        j += 1
        #print(state, reward, done, info)
        
        state = next_state
        if done:
            break
    print(f'episode[{i+1}] score = {score}, avg_loss {loss/j}')
    scores.append(score)
    losses.append(loss/j)

env.close()

plt.plot(list(range(500)), losses, color='red', label='score')
plt.legend()
plt.xlim(0, 500)
plt.show()

