import os
import gym
from matplotlib import pyplot as plt
from model import *
from agent import CartPoleAgent

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
WEIGHT_PATH = CURRENT_PATH+'/weight/'
RESULT_PATH = CURRENT_PATH+'/result/'

def environment_setup(env_name='CartPole-v1'):
    env: gym.Env = gym.make(env_name)
    n_state: int = env.observation_space.shape[0]
    n_action: int = env.action_space.n 
    return env, n_state, n_action

def test(env, n_state, n_action, verbose=True, max_iteration=10):
    state = env.reset()
    agent: CartPoleAgent = CartPoleAgent(n_state, n_action)
    agent.load_model(WEIGHT_PATH)
    agent.set_eval()

    for i in range(max_iteration):
        state = env.reset()
        score, t = 0, 0
        done = False
        while not done:
            if verbose:
                env.render()
            ''' Get actions based on agent NN '''
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = 0.1 if not done or score >= 500 else -1
            t += 1
            state = next_state
            if done:
                break
        print(f'[{i+1}] score = {score}')
    env.close()


if __name__ == '__main__':
    env, n_state, n_action = environment_setup(env_name='CartPole-v1')
    test(env=env, n_state=n_state, n_action=n_action, verbose=True, max_iteration=5)

