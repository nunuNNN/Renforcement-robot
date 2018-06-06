import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network_bn import ActorNetwork
from replay_buffer import ReplayBuffer
import threading

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99


class DDPG:
    """docstring for DDPG"""
    def __init__(self, state_dim, action_dim):
        """name for uploading resuults"""
        self.name = 'DDPG'
        self.time_step = 0
        # self.atten_rate = 1

        """Randomly initialize actor network and critic network"""
        """and both their target networks"""
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim)

        """initialize replay buffer"""
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        """Initialize a random process the Ornstein-Uhlenbeck process for action exploration"""
        self.exploration_noise = OUNoise(self.action_dim)
        """Initialize a Treading"""
        self.threading = threading.Thread(target=self.train, name='LoopThread--DDPG')

    def train(self):
        # if self.time_step ==0:
        #     print("Begins Training!!!")
        #print("Training Begins")
        self.time_step += 1
        """Sample a random minibatch of N transitions from replay buffer"""
        """take out BATCH_SIZE sets of data"""
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        """resize the action_batch shape to  [BATCH_SIZE, self.action_dim]"""
        action_batch = np.resize(action_batch, [BATCH_SIZE, self.action_dim])

        """Calculate y_batch(reward)"""
        next_action_batch = self.actor_network.target_action(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])

        """Update critic by minimizing the loss L (training)"""
        self.critic_network.train(y_batch, state_batch, action_batch)

        """Update the actor policy using the sampled gradient:"""
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch, state_batch)

        """Update the target networks"""
        self.actor_network.update_target()
        self.critic_network.update_target()
        #print("Training Finished")

    def noise_action(self, state):
        """Select action a_t according to the current policy and exploration noise"""
        action = self.actor_network.action(state)
        exp_noise = self.exploration_noise.noise()
        action += exp_noise
        # action[0] = np.clip(action[0], 0, 1)
        # action[1] = np.clip(action[1], -1, 1)
        return action

    def action(self, state):
        action = self.actor_network.action(state)
        # action[0] = np.clip(action[0], 0, 1)
        # action[1] = np.clip(action[1], -1, 1)
        return action

    def perceive(self, state, action, reward, next_state, done):
        """Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)

        """Store transitions to replay start size then start training"""
        # if self.replay_buffer.count() % 1000 == 0:
        #     print("The buffer count is ", self.replay_buffer.count())
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.train()
            # self.atten_rate *= 0.99995
            if not self.threading.is_alive():
                self.threading = threading.Thread(target=self.train, name='LoopThread--DDPG')
                self.threading.start()
            """SAVE NETWORK"""
            if self.time_step % 100 == 0:
                print("Training_time_step:", self.time_step)
            if self.time_step % 1000 == 0:
                print("!!!!!!!save model success!!!!!!!!")
                self.actor_network.save_network(self.time_step)
                self.critic_network.save_network(self.time_step)

        """Re-iniitialize the random process when an episode ends"""
        if done:
            self.exploration_noise.reset()

