import os
import gym
import random
from matplotlib import pyplot as plt
from model import *
from agent import CartPoleAgent

''' Path MACROS '''
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
WEIGHT_PATH = CURRENT_PATH+'/weight/'
RESULT_PATH = CURRENT_PATH+'/result/'

def environment_setup(env_name='CartPole-v1'):
    env: gym.Env = gym.make(env_name)
    n_state: int = env.observation_space.shape[0]
    n_action: int = env.action_space.n 
    return env, n_state, n_action


def collect_memroy(env, agent, m_size=1000, verbose=True):
    state = env.reset()
    avg_score = 0
    ''' memory '''
    memory = []

    i = 0
    ''' collection loop '''
    while len(memory) < m_size:
        state = env.reset()
        score = 0
        done = False
        i += 1
        while not done:
            if verbose:
                env.render()
            ''' Get actions based on agent NN '''
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = 0.1 if not done or score >= 500 else -1
            
            mask = 0 if done else 1

            ''' collect each time step on memory '''
            memory.append((state[:], action, reward, next_state[:], mask))
            '''
            if len(memory) >= m_size:
                break
            '''
            state = next_state
        avg_score += score
    env.close()
    
    avg_score = avg_score/i
    print(f'Average score of current collection step = {avg_score}')
    return memory, avg_score



def train(env, n_state, n_action, verbose=True, max_iteration=100, max_memory=5000, memory_size=1000, batch_size=50):
    state = env.reset()
    agent: CartPoleAgent = CartPoleAgent(n_state, n_action)

    ''' Replay memory '''
    replay_memory = []
    train_memory = []

    scores = []
    losses = []
    ''' Training loop '''
    for i in range(max_iteration):
        memory, avg_score = collect_memroy(env, agent, 1000, verbose)
        if avg_score >= 400:
            break

        
        ''' Push new memory '''
        replay_memory.extend(memory)
        if len(replay_memory) > max_memory:
            replay_memory = replay_memory[len(replay_memory)-max_memory:]
        
        ''' Random sampling'''
        train_memory = random.sample(replay_memory, memory_size)
        
        ''' Train the model '''
        agent.train_model_from_memory(train_memory, batch_size)
        agent.save_model(WEIGHT_PATH)

    agent.save_model(WEIGHT_PATH)

    

def draw_plot(scores, losses):
    plt.plot(list(range(len(scores))), scores, color='red', label='score')
    plt.legend()
    plt.xlim(0, len(scores))
    plt.savefig(RESULT_PATH+'cartpole_actor_critic_score.png', dpi=300)
    plt.clf()

    plt.plot(list(range(len(losses))), losses, color='red', label='loss')
    plt.legend()
    plt.xlim(0, len(losses))
    plt.savefig(RESULT_PATH+'cartpole_actor_critic_loss.png', dpi=300)
    plt.clf()

if __name__ == '__main__':
    env, n_state, n_action = environment_setup(env_name='CartPole-v1')
    train(env=env, 
            n_state=n_state, 
            n_action=n_action, 
            verbose=False, 
            max_iteration=100, 
            max_memory=5000, 
            memory_size=1000, 
            batch_size=50
    )

