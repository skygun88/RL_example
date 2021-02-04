import os
import gym
from matplotlib import pyplot as plt
from model import *
from agent import CartPoleAgent

''' Path MACROS '''
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
WEIGHT_PATH = CURRENT_PATH+'/weight/'
RESULT_PATH = CURRENT_PATH+'/result/'
TRAJECTORY_PATH = CURRENT_PATH+'/trajectory/'

def environment_setup(env_name='CartPole-v1'):
    env: gym.Env = gym.make(env_name)
    n_state: int = env.observation_space.shape[0]
    n_action: int = env.action_space.n 
    return env, n_state, n_action

def test(env, n_state, n_action, verbose=True, n_trajectories=20):
    state = env.reset()
    agent: CartPoleAgent = CartPoleAgent(n_state, n_action)
    agent.load_model(WEIGHT_PATH)
    agent.set_eval()
    trajectories = []
    recorded = 0

    while recorded < n_trajectories:
        state = env.reset()
        score, t = 0, 0
        done = False
        trajectory = []
        while not done:
            if verbose:
                env.render()
            ''' Get actions based on agent NN '''
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append(list(state)+[action])

            score += reward
            t += 1
            state = next_state
            if done:
                break
        print(f'Score = {score}')
        if score >= 500:
            trajectories.append(trajectory)
            recorded += 1
            print(f'Recorded {recorded}/{n_trajectories}')
            
    env.close()

    for i in range(n_trajectories):
        with open(TRAJECTORY_PATH + f'CartPole_trajectory_{i}.csv', 'w') as f:
            curr_trajectory = trajectories[i]
            for step in curr_trajectory:
                str_step = tuple(map(lambda x: str(x), step))
                f.write(','.join(str_step)+'\n')
            f.close()


if __name__ == '__main__':
    env, n_state, n_action = environment_setup(env_name='CartPole-v1')
    test(env=env, n_state=n_state, n_action=n_action, verbose=True, n_trajectories=5)

