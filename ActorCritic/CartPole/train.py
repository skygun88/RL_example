import os
import gym
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

def train(env, n_state, n_action, verbose=True, max_iteration=500):
    state = env.reset()
    agent: CartPoleAgent = CartPoleAgent(n_state, n_action)

    scores = []
    losses = []
    ''' Training loop '''
    for i in range(max_iteration):
        state = env.reset()
        score, loss, t = 0, 0, 0
        done = False
        while not done:
            if verbose:
                env.render()
            ''' Get actions based on agent NN '''
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = 0.1 if not done or score >= 500 else -1
            
            ''' Train the model through gradient descent '''
            loss += agent.train_model(state, action, reward, next_state, done)
            t += 1
            
            state = next_state
            if done:
                break
        
        ''' Collect score & loss data'''
        print(f'[{i+1}] score = {score} | avg_loss {loss/t}')
        scores.append(score)
        losses.append(loss/t)
        
        ''' If max score occurs in last 10 times, finish the training loop '''
        if len(scores) > 9 and (sum(scores[-10:])/10 >= 500):
            break

        ''' Save current model & Draw the score & loss plot '''
        if i % 50 == 0 and i > 0:
            agent.save_model(WEIGHT_PATH)
            draw_plot(scores, losses)
    env.close()

    ''' Save well trained model & Draw the result score & loss plot '''
    agent.save_model(WEIGHT_PATH)
    draw_plot(scores, losses)
    

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
    train(env=env, n_state=n_state, n_action=n_action, verbose=True, max_iteration=500)

