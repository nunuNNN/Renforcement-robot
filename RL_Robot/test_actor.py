# -*- coding:utf-8 -*-
import numpy as np
from actor_network import *
import tensorflow as tf

EPISODES = 10000
state_dim = 14
action_dim = 2


"""create pipeline of csv"""
def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
    """slice the data"""
    vel_x, vel_y, \
    pose_x, pose_y, \
    laser1, laser2, laser3, laser4, laser5, laser6, laser7, laser8, laser9, laser10, \
    next_vel_x, next_vel_y = \
        tf.decode_csv(value, defaults)

    laser = \
        tf.stack([laser1/5, laser2/5, laser3/5, laser4/5, laser5/5, laser6/5, laser7/5, laser8/5, laser9/5, laser10/5])
    pose = tf.stack([pose_x, pose_y])
    vel = tf.stack([vel_x, vel_y])
    next_vel = tf.stack([next_vel_x, next_vel_y])

    """shuffle the sequence of data"""
    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    laser_batch = tf.train.shuffle_batch(
        [laser], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    pose_batch = tf.train.shuffle_batch(
        [pose], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    vel_batch = tf.train.shuffle_batch(
        [vel], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    next_vel_batch = tf.train.shuffle_batch(
        [next_vel], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return laser_batch, pose_batch, vel_batch, next_vel_batch


#-----------------------------------------------------------------------------------------#
laser_train_new, pose_train_new, vel_train_new, next_vel_train_new = \
    create_pipeline('train_data.csv', 100, num_epochs=10000)
laser_test, pose_test, vel_test, next_vel_test = create_pipeline('test_data.csv', 800)
#-----------------------------------------------------------------------------------------#

with tf.Session() as sess:
    local_init = tf.local_variables_initializer()  # local variables like epoch_num, batch_size
    sess.run(local_init)

    """Start populating the filename queue."""
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    init = tf.global_variables_initializer()
    sess.run(init)

    """ Retrieve a single instance:"""
    laser_test, pose_test, vel_test, next_vel_test = sess.run([laser_test, pose_test, vel_test, next_vel_test])
    transition_test = np.hstack((laser_test, pose_test, vel_test))
    # transition_test = np.hstack((pose_test, vel_test))

    actor_network = ActorNetwork(sess, state_dim, action_dim)

    for episode in range(8001):
        laser_train, pose_train, vel_train, next_vel_train = \
            sess.run([laser_train_new, pose_train_new, vel_train_new, next_vel_train_new])
        transition = np.hstack((laser_train, pose_train, vel_train))
        # transition = np.hstack((pose_train, vel_train))
        # print(transition)

        actor_network.test_tarin(next_vel_train, transition)
        _loss = actor_network.test_loss(next_vel_train, transition)

        # print(transition)
        # print(actor_network.actions(transition))

        if episode % 100 == 0:
            print("epoch:%d\tloss:%.5f" % (episode, _loss))

        if episode % 1000 == 0:
            actor_network.save_test_network(episode)
            # actor_network.visualization(next_vel_train, transition, episode)

    __loss = actor_network.test_loss(next_vel_test, transition_test)
    print("the test loss is: ", __loss)


    coord.request_stop()
    coord.join(threads)
    sess.close()
