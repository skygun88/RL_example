import gym
import readchar
import numpy as np
import os

''' Path MACROS '''
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
TRAJECTORY_PATH = CURRENT_PATH+'/trajectory/'

''' MACROS '''
LEFT = 0
RIGHT = 1

''' Key mapping '''
arrow_keys = {
    'a': LEFT,
    'd': RIGHT}

def record_user_trajectory(n_trajectories=20):
    env = gym.make('CartPole-v1')
    trajectories = []
    for episode in range(n_trajectories): # n_trajectories : 20
        trajectory = []
        t = 0

        env.reset()
        while True: 
            env.render()
            print(f'[{episode+ 1}] timestep = {t}')

            key = readchar.readkey()
            if key not in arrow_keys.keys():
                print(f'Pressed {key}')
                continue
            print(f'Pressed {arrow_keys[key]}')

            action = arrow_keys[key]
            state, reward, done, _ = env.step(action)

            if done or t >= 500: # trajectory_length : 130
                break

            trajectory.append((state[0], state[1], state[2], state[3], action))
            t += 1

        trajectories.append(trajectory)

    for i in range(n_trajectories):
        with open(TRAJECTORY_PATH + f'CartPole_trajectory_{i}.csv', 'w') as f:
            curr_trajectory = trajectories[i]
            for step in curr_trajectory:
                str_step = tuple(map(lambda x: str(x), step))
                f.write(','.join(str_step)+'\n')
            f.close()

if __name__ == '__main__':
    n_trajectories=20
    record_user_trajectory(n_trajectories)