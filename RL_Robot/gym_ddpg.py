import numpy as np
import gym
import time
from ddpg import *


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200

ENV_NAME = 'CartPole-v0'

TEST = 10

def main():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)
    env.seed(1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DDPG(state_dim, action_dim)

    for episode in range(MAX_EPISODES):
        state = env.reset()
        for step in range(MAX_EP_STEPS):
            # Add exploration noise
            print(state)
            action = agent.noise_action(state)
            print(action)
            next_state, reward, done, info = env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                for j in xrange(env.spec.timestep_limit):
                    # env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
    env.monitor.close()

if __name__ == '__main__':
    main()
