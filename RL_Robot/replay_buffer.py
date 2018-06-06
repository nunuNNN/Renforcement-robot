from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


"""#########################!!T E S T    F U N C T I O N!!################################"""
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

if __name__ == '__main__':
    rb = ReplayBuffer(10000)

    for i in range(10000):
        state_rm = random_int_list(10, -10, 14)
        action_rm = random_int_list(1, -1, 2)
        reward_rm = random_int_list(50, -50, 1)
        new_state_rm = random_int_list(1, -1, 14)
        done_rm = random_int_list(1, -1, 1)

        rb.add(state_rm, action_rm, reward_rm, new_state_rm, done_rm)


    minibatch = rb.get_batch(6)

    import numpy as np
    state_batch = np.asarray([data[0] for data in minibatch])
    print(state_batch)
    action_batch = np.asarray([data[1] for data in minibatch])
    print(action_batch)
    reward_batch = np.asarray([data[2] for data in minibatch])
    print(reward_batch)
    next_state_batch = np.asarray([data[3] for data in minibatch])
    print(next_state_batch)
    done_batch = np.asarray([data[4] for data in minibatch])
    print(done_batch)


