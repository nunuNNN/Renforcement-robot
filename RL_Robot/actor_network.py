import tensorflow as tf
import math


"""####################!!Hyper   Parameters!!###################"""
LAYER1_SIZE = 256  #Each layer of neurons
LAYER2_SIZE = 128
LAYER3_SIZE = 128

TUA = 0.001
LEARNING_RATE = 1e-4

class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self, sess, state_dim, action_dim):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        """"create actor network"""
        self.state_input, \
        self.action_output, \
        self.net = self.create_network(state_dim)

        """create actor target network"""
        self.target_state_input, \
        self.target_action_output, \
        self.target_update, \
        self.target_net = self.create_target_network(state_dim, self.net)
        #
        """define training rules"""
        self.create_training_method()

        """define test training rules"""
        # self.create_teat_train()

        """init"""
        self.sess.run(tf.initialize_all_variables())

        """update the para of target network"""
        self.update_target()

        """loading network model"""
        self.load_network()

        """loading test network model"""
        # self.load_test_network()

        """Visualization weights and  biases"""
        # self.merged = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter("logs/", sess.graph)

    """#########################!!I N I T    F U N C T I O N!!################################"""
    def create_training_method(self):
        self.q_gradient_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!", self.action_output)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@", self.net)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", -self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,
                                                                                   self.net))
        print("############################", zip(self.parameters_gradients, self.net))

    def create_network(self, state_dim):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE
        layer3_size = LAYER3_SIZE

        state_input = tf.placeholder(tf.float32, [None, state_dim])

        W1 = self.variable([state_dim, layer1_size], state_dim)
        # tf.summary.histogram('/w1', W1)
        b1 = self.variable([layer1_size], state_dim)
        # tf.summary.histogram('/b1', b1)
        W2 = self.variable([layer1_size, layer2_size], layer1_size)
        # tf.summary.histogram('/w2', W2)
        b2 = self.variable([layer2_size], layer1_size)
        # tf.summary.histogram('/b2', b2)
        W3 = self.variable([layer2_size, layer3_size], layer2_size)
        # tf.summary.histogram('/w3', W2)
        b3 = self.variable([layer3_size], layer2_size)
        # tf.summary.histogram('/b3', b3)
        W_lin = tf.Variable(tf.random_uniform([layer3_size, 1], -3e-3, 3e-3))
        # tf.summary.histogram('/w_lin', W_lin)
        b_lin = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))
        # tf.summary.histogram('/b_lin', b_lin)
        W_ang = tf.Variable(tf.random_uniform([layer3_size, 1], -3e-3, 3e-3))
        # tf.summary.histogram('/w_ang', W_ang)
        b_ang = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))
        # tf.summary.histogram('/b_ang', b_ang)

        layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
        layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
        layer_lin = tf.nn.sigmoid(tf.matmul(layer3, W_lin) + b_lin)
        layer_ang = tf.nn.tanh(tf.matmul(layer3, W_ang) + b_ang)

        action_output = tf.concat([layer_lin, layer_ang], 1)
        # tf.summary.histogram('/action_output', action_output)

        return state_input, action_output, [W1, b1, W2, b2, W3, b3, W_lin, b_lin, W_ang, b_ang]

    def create_target_network(self, state_dim, net):
        state_input = tf.placeholder(tf.float32, [None, state_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1-TUA)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + target_net[3])
        layer3 = tf.nn.relu(tf.matmul(layer2, target_net[4]) + target_net[5])
        layer_lin = tf.nn.sigmoid(tf.matmul(layer3, target_net[6]) + target_net[7])
        layer_ang = tf.nn.tanh(tf.matmul(layer3, target_net[8]) + target_net[9])

        action_output = tf.concat([layer_lin, layer_ang], 1)

        return state_input, action_output, target_update, target_net

    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1/math.sqrt(f), 1/math.sqrt(f)))

    def update_target(self):
        self.sess.run(self.target_update)

    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step):
        #print("save actor-network...", time_step)
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step=time_step)


    """#########################!!F U N C T I O N    F U N C T I O N!!##########################"""
    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch
        })

    def actions(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state
        })

    def action(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch
        })[0]

    def target_action(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch
        })


    """#########################!!T E S T     F U N C T I O N!!##########################"""
    def create_teat_train(self):
        self.action_batch = tf.placeholder(tf.float32, [None, self.action_dim])
        self.loss = tf.reduce_mean(tf.square(self.action_output - self.action_batch))
        # tf.reduce_sum(
        #     tf.square(self.action_output - self.action_batch), reduction_indices=[1]
        # )
        self.train_step = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        tf.summary.scalar('loss', self.loss)

    def test_tarin(self, action_batch, state_batch):
        self.sess.run(self.train_step, feed_dict={
            self.action_batch: action_batch,
            self.state_input: state_batch
        })

    def test_loss(self, action_batch, state_batch):
        return self.sess.run(self.loss, feed_dict={
            self.action_batch: action_batch,
            self.state_input: state_batch
        })

    def save_test_network(self, time_step):
        print("save actor-test-network...", time_step)
        self.saver.save(self.sess, 'saved_actor_test_networks/' + 'actor-test-network', global_step=time_step)

    def load_test_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_test_networks")

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old actor test network weights")

    def visualization(self, action_batch, state_batch, episode):
        result = self.sess.run(self.merged, feed_dict={
            self.action_batch: action_batch,
            self.state_input: state_batch
        })
        self.writer.add_summary(result, episode)

def main():
    sess = tf.InteractiveSession()
    actor_network = ActorNetwork(sess, 14, 2)

    vel_train = [[0.58, 0.043]]
    transition = [[10, 4.724, 10, 10, 4.577, 5.326, 2.484, 1.828, 1.585, 1.868, 2.521, -0.228, 0.804, 0.339]]
    actor_network.test_tarin(vel_train, transition)

    sess.close()

if __name__ == '__main__':
    main()
