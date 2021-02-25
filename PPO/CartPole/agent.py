import numpy as np
import math
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
        np_state = np.array(state).reshape(1, -1)
        policy = self.actor(np_state)
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

    def train_model_from_memory(self, memory, batch_size):
        self.actor.train(), self.critic.train()
        n_batch = math.ceil(len(memory)/batch_size)
        for batch_idx in range(n_batch):
            self.actor_optimizer.zero_grad(), self.critic_optimizer.zero_grad()
            if batch_idx == n_batch-1:
                curr_memory = memory[batch_idx*batch_size:]
            else:
                curr_memory = memory[batch_idx*batch_size:batch_idx*batch_size+batch_size]
            ''' memory[n, 5] = [states, actions, rewards, next_states, masks]'''
            states = np.array(tuple(map(lambda x: x[0], curr_memory)))
            actions = np.array(tuple(map(lambda x: x[1], curr_memory)))
            rewards = np.array(tuple(map(lambda x: x[2], curr_memory)))
            next_states = np.array(tuple(map(lambda x: x[3], curr_memory)))
            masks = np.array(tuple(map(lambda x: x[4], curr_memory)))

            torch_masks = torch.from_numpy(masks)
            torch_rewards = torch.from_numpy(rewards)


            policies = self.actor(states)
            values = self.critic(states)
            next_values = self.critic(next_states)
            values = values.squeeze()
            next_values = next_values.squeeze()

            one_hot_actions = torch.zeros(len(actions), self.n_action)

            for i in range(len(actions)):
                one_hot_actions[i][actions[i]] = 1
            #print(one_hot_actions)
            one_hot_actions = policies * one_hot_actions 
            #print(one_hot_actions)
            action_prob = torch.sum(one_hot_actions, dim=1)
            log_prob = torch.log(action_prob+1e-5)
            target = torch_rewards +  torch_masks * self.discount_factor * next_values
            advantage = (target - values).detach()
            actor_loss = torch.sum(-log_prob*advantage)
            critic_loss = torch.sum(0.5 * torch.square(target.detach() - values))
            loss = 0.2 * actor_loss + critic_loss
            loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return