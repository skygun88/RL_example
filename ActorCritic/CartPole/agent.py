import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as torch_utils
from model import * 

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
        policy = policy.detach().numpy().squeeze(0)
        action = np.random.choice(self.n_action, 1, p=policy)[0]
        return action

    def save_model(self, dir_path):
        torch.save(self.actor.state_dict(), dir_path+'actor.pt')
        torch.save(self.critic.state_dict(), dir_path+'critic.pt')

    def load_model(self, dir_path):
        self.actor.load_state_dict(torch.load(dir_path+'actor.pt'))
        self.critic.load_state_dict(torch.load(dir_path+'critic.pt'))

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()

    def train_model(self, state, action, reward, next_state, done):
        self.actor_optimizer.zero_grad(), self.critic_optimizer.zero_grad()
        self.actor.train(), self.critic.train()
        policy, value, next_value = self.actor(state), self.critic(state), self.critic(next_state)

        target = reward + (1 - done) * self.discount_factor * next_value
        
        one_hot_action = torch.zeros(self.n_action)
        one_hot_action[action] = policy[0][action]
        action_prob = one_hot_action.reshape(2)
        action_prob = torch.sum(action_prob)
        advantage = (target - value.item()).detach().reshape(1)
        # print(f'advantage: {advantage}, {advantage.shape}, {advantage.dtype}')

        actor_loss = -(torch.log(action_prob + 1e-5)*advantage)
        critic_loss = 0.5 * torch.square(target.detach() - value[0])
        loss = 0.2 * actor_loss + critic_loss
        loss.backward()
        #print(f'loss: {loss.item()}')

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return loss.item()