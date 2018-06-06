import tensorflow as tf
import math


"""####################!!Hyper   Parameters!!###################"""
LAYER1_SIZE = 256  #Each layer of neurons
LAYER2_SIZE = 128
LAYER3_SIZE = 128

TAU = 0.001
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
        self.net, \
        self.is_training = self.create_network(state_dim, action_dim)

        """create actor target network"""
        self.target_state_input, \
        self.target_action_output, \
        self.target_update, \
        self.target_is_training = \
            self.create_target_network(state_dim, action_dim, self.net)
        """define training rules"""
        self.create_training_method()

        """define test training rules"""
        # self.create_test_train()

        """init"""
        self.sess.run(tf.initialize_all_variables())

        """update the para of target network"""
        self.update_target()

        """loading network model"""
        self.load_network()

        """loading test network model"""
        # self.load_test_network()

        """Visualization weights and  biases"""
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("logs/", sess.graph)


    """#########################!!I N I T    F U N C T I O N!!################################"""
    def create_training_method(self):
        self.q_gradient_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,
                                                                                   self.net))

    # def create_network(self, state_dim):
    #     layer1_size = LAYER1_SIZE
    #     layer2_size = LAYER2_SIZE
    #     layer3_size = LAYER3_SIZE
    #
    #     state_input = tf.placeholder(tf.float32, [None, state_dim])
    #     is_training = tf.placeholder(tf.bool)
    #
    #     W1 = self.variable([state_dim, layer1_size], state_dim)
    #     b1 = self.variable([layer1_size], state_dim)
    #     W2 = self.variable([layer1_size, layer2_size], layer1_size)
    #     b2 = self.variable([layer2_size], layer1_size)
    #     W3 = self.variable([layer2_size, layer3_size], layer2_size)
    #     b3 = self.variable([layer3_size], layer2_size)
    #     W_lin = tf.Variable(tf.random_uniform([layer3_size, 1], -3e-3, 3e-3))
    #     b_lin = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))
    #     W_ang = tf.Variable(tf.random_uniform([layer3_size, 1], -3e-3, 3e-3))
    #     b_ang = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))
    #
    #     layer0_bn = self.batch_norm_layer(state_input, training_phase=is_training,
    #                                       scope_bn='batch_norm_0', activation=tf.identity)
    #     layer1 = tf.matmul(layer0_bn, W1) + b1
    #     layer1_bn = self.batch_norm_layer(layer1, training_phase=is_training,
    #                                       scope_bn='batch_norm_1', activation=tf.nn.relu)
    #     layer2 = tf.matmul(layer1_bn, W2) + b2
    #     layer2_bn = self.batch_norm_layer(layer2, training_phase=is_training,
    #                                       scope_bn='batch_norm_2', activation=tf.nn.relu)
    #     layer3 = tf.matmul(layer2_bn, W3) + b3
    #     layer3_bn = self.batch_norm_layer(layer3, training_phase=is_training,
    #                                       scope_bn='batch_norm_3', activation=tf.nn.relu)
    #
    #     layer_lin = tf.nn.sigmoid(tf.matmul(layer3_bn, W_lin) + b_lin)
    #     layer_lin_bn = self.batch_norm_layer(layer_lin, training_phase=is_training,
    #                                          scope_bn='batch_norm_lin', activation=tf.nn.relu)
    #     layer_ang = tf.nn.tanh(tf.matmul(layer3_bn, W_ang) + b_ang)
    #     layer_ang_bn = self.batch_norm_layer(layer_ang, training_phase=is_training,
    #                                          scope_bn='batch_norm_ang', activation=tf.nn.relu)
    #
    #     action_output = tf.concat([layer_lin_bn, layer_ang_bn], 1)
    #
    #     return state_input, action_output, \
    #            [W1, b1, W2, b2, W3, b3, W_lin, b_lin, W_ang, b_ang], \
    #            is_training
    #
    # def create_target_network(self, state_dim, net):
    #     state_input = tf.placeholder(tf.float32, [None, state_dim])
    #     is_training = tf.placeholder(tf.bool)
    #
    #     ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
    #     target_update = ema.apply(net)
    #     target_net = [ema.average(x) for x in net]
    #
    #     layer0_bn = self.batch_norm_layer(state_input, training_phase=is_training,
    #                                       scope_bn='target_batch_norm_0', activation=tf.identity)
    #     layer1 = tf.matmul(layer0_bn, target_net[0]) + target_net[1]
    #     layer1_bn = self.batch_norm_layer(layer1, training_phase=is_training,
    #                                       scope_bn='target_batch_norm_1', activation=tf.nn.relu)
    #     layer2 = tf.matmul(layer1_bn, target_net[2]) + target_net[3]
    #     layer2_bn = self.batch_norm_layer(layer2, training_phase=is_training,
    #                                       scope_bn='target_batch_norm_2', activation=tf.nn.relu)
    #     layer3 = tf.matmul(layer2_bn, target_net[4]) + target_net[5]
    #     layer3_bn = self.batch_norm_layer(layer3, training_phase=is_training,
    #                                       scope_bn='target_batch_norm_3', activation=tf.nn.relu)
    #
    #     layer_lin = tf.nn.sigmoid(tf.matmul(layer3_bn, target_net[6]) + target_net[7])
    #     layer_lin_bn = self.batch_norm_layer(layer_lin, training_phase=is_training,
    #                                          scope_bn='target_batch_norm_lin', activation=tf.nn.relu)
    #     layer_ang = tf.nn.tanh(tf.matmul(layer3_bn, target_net[8]) + target_net[9])
    #     layer_ang_bn = self.batch_norm_layer(layer_ang, training_phase=is_training,
    #                                          scope_bn='target_batch_norm_ang', activation=tf.nn.relu)
    #
    #     action_output = tf.concat([layer_lin, layer_ang], 1)
    #
    #     return state_input, action_output, target_update, target_net,is_training

    def create_network(self, state_dim, action_dim):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_dim])
        is_training = tf.placeholder(tf.bool)

        W1 = self.variable([state_dim, layer1_size], state_dim)
        b1 = self.variable([layer1_size], state_dim)
        W2 = self.variable([layer1_size, layer2_size], layer1_size)
        b2 = self.variable([layer2_size], layer1_size)
        W3 = tf.Variable(tf.random_uniform([layer2_size, action_dim], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([action_dim], -3e-3, 3e-3))

        layer0_bn = self.batch_norm_layer(state_input, training_phase=is_training, scope_bn='batch_norm_0',
                                          activation=tf.identity)
        layer1 = tf.matmul(layer0_bn, W1) + b1
        layer1_bn = self.batch_norm_layer(layer1, training_phase=is_training, scope_bn='batch_norm_1',
                                          activation=tf.nn.relu)
        layer2 = tf.matmul(layer1_bn, W2) + b2
        layer2_bn = self.batch_norm_layer(layer2, training_phase=is_training, scope_bn='batch_norm_2',
                                          activation=tf.nn.relu)

        action_output = tf.tanh(tf.matmul(layer2_bn, W3) + b3)

        return state_input, action_output, [W1, b1, W2, b2, W3, b3], is_training

    def create_target_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        is_training = tf.placeholder(tf.bool)
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer0_bn = self.batch_norm_layer(state_input, training_phase=is_training, scope_bn='target_batch_norm_0',
                                          activation=tf.identity)

        layer1 = tf.matmul(layer0_bn, target_net[0]) + target_net[1]
        layer1_bn = self.batch_norm_layer(layer1, training_phase=is_training, scope_bn='target_batch_norm_1',
                                          activation=tf.nn.relu)
        layer2 = tf.matmul(layer1_bn, target_net[2]) + target_net[3]
        layer2_bn = self.batch_norm_layer(layer2, training_phase=is_training, scope_bn='target_batch_norm_2',
                                          activation=tf.nn.relu)

        action_output = tf.tanh(tf.matmul(layer2_bn, target_net[4]) + target_net[5])

        return state_input, action_output, target_update, is_training

    def batch_norm_layer(self, x, training_phase, scope_bn, activation=None):
        return tf.cond(training_phase,
                       lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                                                            updates_collections=None, is_training=True, reuse=None,
                                                            scope=scope_bn, decay=0.9, epsilon=1e-5),
                       lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                                                            updates_collections=None, is_training=False, reuse=True,
                                                            scope=scope_bn, decay=0.9, epsilon=1e-5))

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
        print("save actor-network...", time_step)
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step=time_step)



    """#########################!!F U N C T I O N    F U N C T I O N!!##########################"""
    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch,
            self.is_training: True
        })

    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch,
            self.is_training: True
        })

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state],
            self.is_training: False
        })[0]

    def target_action(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_is_training: True
        })

    """#########################!!T E S T     F U N C T I O N!!##########################"""
    def create_test_train(self):
        self.action_batch = tf.placeholder(tf.float32, [None, self.action_dim])
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.action_output - self.action_batch), reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def test_tarin(self, action_batch, state_batch):
        self.sess.run(self.train_step, feed_dict={
            self.action_batch: action_batch,
            self.state_input: state_batch,
            self.is_training: True
        })

    def test_loss(self, action_batch, state_batch):
        return self.sess.run(self.loss, feed_dict={
            self.action_batch: action_batch,
            self.state_input: state_batch,
            self.is_training: False
        })

    def save_test_network(self, time_step):
        print("save actor-bn-test-network...", time_step)
        self.saver.save(self.sess, 'saved_actor_bn_test_networks/' + 'actor-bn-test-network', global_step=time_step)

    def visualization(self, action_batch, state_batch, episode):
        result = self.sess.run(self.merged, feed_dict={
            self.action_batch: action_batch,
            self.state_input: state_batch
        })
        self.writer.add_summary(result, episode)

    def load_test_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_bn_test_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old actor test network weights")
