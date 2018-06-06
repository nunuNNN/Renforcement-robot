from actor_network_bn import ActorNetwork
import tensorflow as tf
import gym
from env import ObstacleEnv
from gym.envs.registration import register
import rospy

import csv

EPISODES = 100000


def main():
    register(
        id='Obstacle-v0',
        entry_point=ObstacleEnv,
        max_episode_steps=1000,
        reward_threshold=100.0,
        )
    env = gym.make('Obstacle-v0')

    sess = tf.InteractiveSession()
    agent = ActorNetwork(sess, 14, 2)

    rate = rospy.Rate(10.0)

    for episode in xrange(EPISODES):
        state = env.reset()
        # state_p = state[0][:4]
        while True:
            action = agent.action(state[0])
            next_state, reward, done, info = env.step(action)
            # next_state_p = next_state[0][:4]

            # with open("/home/ld/test_csv/state.csv", "a") as csvFile:
            #     writer = csv.writer(csvFile)
            #     writer.writerow(next_state[0])
            # with open("/home/ld/test_csv/vel.csv", "a") as csvFile:
            #     writer = csv.writer(csvFile)
            #     writer.writerow(action)
            # state_p = next_state_p
            state = next_state
            if done:
                break
            rate.sleep()


if __name__ == '__main__':
    main()
