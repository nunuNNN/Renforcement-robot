from ddpg import *
import gym
from env import ObstacleEnv
from gym.envs.registration import register

EPISODES = 100000
TEST = 10

def main():
    register(
        id='Obstacle-v0',
        entry_point=ObstacleEnv,
        max_episode_steps=1000,
        reward_threshold=100.0,
        )
    env = gym.make('Obstacle-v0')
    agent = DDPG(14, 2)

    for episode in xrange(EPISODES):
        train_reward = 0
        state = env.reset()
        while True:
            action = agent.noise_action(state[0])
            next_state, reward, done, info = env.step(action)
            agent.perceive(state[0], action[0], reward, next_state[0], done)
            state = next_state
            train_reward += reward
            if done:
                break

        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                while True:
                    # env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            test_ave_reward = total_reward / TEST
            print('test_episode: ', episode, 'Evaluation Average Reward:', test_ave_reward)


if __name__ == '__main__':
    main()
