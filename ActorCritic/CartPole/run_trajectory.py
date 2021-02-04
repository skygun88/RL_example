import os
import gym

''' Path MACROS '''
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
TRAJECTORY_PATH = CURRENT_PATH+'/trajectory/'

def environment_setup(env_name='CartPole-v1'):
    env: gym.Env = gym.make(env_name)
    n_state: int = env.observation_space.shape[0]
    n_action: int = env.action_space.n 
    return env, n_state, n_action

def run_trajectories(env):
    state = env.reset()

    print(f'There are {len(os.listdir(TRAJECTORY_PATH))} trajectories')

    for filename in os.listdir(TRAJECTORY_PATH):
        with open(TRAJECTORY_PATH+filename, 'r') as f:
            lines = f.readlines()
            f.close()

        trajectory = list(map(lambda x: list(map(lambda y: float(y), x.split(',')[:-1])), lines))
        print(f'Record: {filename}')
        for step in trajectory:
            env.env.state = step
            env.render()
    env.close()

if __name__ == '__main__':
    env, _, _ = environment_setup(env_name='CartPole-v1')
    run_trajectories(env)


