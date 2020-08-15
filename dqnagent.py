from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, BatchNormalization, Conv3D, MaxPooling3D, InputLayer
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD
import keras
from keras.models import load_model
from collections import deque
import time
import random
import numpy as np
import tensorflow as tf
from tm_env import *
import os
from tqdm import tqdm
from presskeys import W, S, A, D, enter, esc
from math import log
import time
# from keras.losses import huber_loss
from keras import backend as K
# from keystrokes import key_check


REPLAY_MEMORY_SIZE = 5000
MODEL_NAME = '32-64-128-128D_StridedCONVnet_long_track'
MINIBATCH_SIZE = 64
DISCOUNT = 0.95
ALPHA = 0.1
UPDATE_TARGET_EVERY = 1000
# MIN_REWARD = -200  # For model savepwawawawa
# MEMORY_FRACTION = 0.20
SAVE_FROM_EPISODE = 100

# Environment settings
EPISODES = 5000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99944
MIN_EPSILON = 0.10

#  Stats settings
AGGREGATE_STATS_EVERY = 25  # episodes
SHOW_PREVIEW = False
MIN_REPLAY_MEMORY_SIZE = 1000


# For stats
ep_rewards = []

# For more repetitive results when calling random
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metricsa
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):
        # main model - train every step
        self.model = self.create_model((6, 60, 80))
        # fix the targets
        self.target_model = self.create_model((6, 60, 80))
        # self.target_model = load_model('models/[THIRD-SAFETY]32-64-128-128D_StridedCONVnet_left_huber_-3.216471629705268max_-70.7795296128333avg_-217.76575878519014min_0.11epsilon_1999episode_1587752890.model', custom_objects={'huber_loss': self.huber_loss})
        # self.model = load_model('models/[THIRD-SAFETY]32-64-128-128D_StridedCONVnet_left_huber_-3.216471629705268max_-70.7795296128333avg_-217.76575878519014min_0.11epsilon_1999episode_1587752890.model', custom_objects={'huber_loss': self.huber_loss})
        # self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir='logs/{}-{}'.format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0

    def huber_loss(self, a, b, in_keras=True):
        error = a - b
        quadratic_term = (error * error) / 2
        linear_term = abs(error) - (1 / 2)
        use_linear_term = (abs(error) > 1.5)
        if in_keras:
            # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
            use_linear_term = K.cast(use_linear_term, 'float32')
        return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term

    def create_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation='relu', strides = (4, 4), padding = 'same', input_shape = input_shape, data_format = 'channels_first'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (5, 5), activation='relu', strides = (2, 2), padding = 'same', data_format = 'channels_first'))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (3, 3), activation='relu', strides = (1, 1), padding = 'same', data_format = 'channels_first'))
        model.add(BatchNormalization())

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(7, activation='linear'))
        model.summary()
        exit(0)
        model.compile(loss=self.huber_loss, optimizer=Adam(lr=1e-5, clipnorm=1.0), metrics=['accuracy'])

        return model


    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]


    def train(self, terminal_state, step):
        global UPDATE_TARGET_EVERY
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            print('Filling up Replay Memory...', len(self.replay_memory))
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_state = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_state)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[self.tensorboard] if step == 1 else None)
        # update target model or not
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0



def training_agent():
    global epsilon
    agent = DQNAgent()
    game = TM()
    collisions = []
    for episode in tqdm(range(EPISODES), unit='episodes'):
        print('Exploration Rate: ',round(epsilon, 2))
        print('Replay memory size: ', len(agent.replay_memory))
        episode_reward=0
        agent.tensorboard.step = episode
        step=1
        done = False
        stuck=False
        complete=False
        num_collisions = 0
        game.new_episode()
        curr_state, speed_list, timestamp_list, collided = game.live_data(initial_call=True)


        while not done:
            if np.random.random() > epsilon:
                print('Exploiting...')
                qlist = agent.get_qs(curr_state)
                action = np.argmax(qlist)
                print(qlist)

            else:
                print('Exploring...')
                action = np.random.randint(0, 7)

            game.make_action(action)

            next_frame, next_speed, next_timestamp, collided = game.live_data()

            # next_state=np.append(curr_state[:, :, 1:], np.reshape(next_frame,(*next_frame.shape, 1)), axis=2)

            next_state = np.concatenate((curr_state[1:], [next_frame]), axis = 0)

            next_speed_list=deque(speed_list, maxlen=6)
            next_timestamp_list=deque(timestamp_list, maxlen=6)
            next_speed_list.append(next_speed[0])
            next_timestamp_list.append(next_timestamp[0])

            if step > 6:
                complete = game.check_completion(next_timestamp_list)
                stuck = game.check_stuck(next_speed_list, next_timestamp_list)

            if complete:
                print('YAYYYY!!! RACE COMPLETED!!!')
                print('I will now reset the race...')
                ReleaseAllKeys()
                done=True
                time.sleep(7)
                PressKey(enter)
                ReleaseKey(enter)
                PressKey(enter)
                ReleaseKey(enter)
                PressKey(enter)
                ReleaseKey(enter)
                PressKey(enter)
                ReleaseKey(enter)
                time.sleep(7)
                PressKey(enter)
                ReleaseKey(enter)
                PressKey(enter)
                ReleaseKey(enter)
                PressKey(enter)
                ReleaseKey(enter)
                PressKey(enter)
                ReleaseKey(enter)

            elif stuck:
                done = True
                print('OOPS I GOT STUCK!!')


            elif collided:
                print('OOPS I COLLIDED!!')
                num_collisions += 1
            else:
                if action == 3:
                    diff = abs(next_speed_list[-2] - next_speed_list[-1])
                    if diff == 0 or diff == 1:
                        game.reward = -0.5
                    else:
                        game.reward = -log(diff)
                else:
                    game.get_reward(next_speed_list)




            agent.update_replay_memory([curr_state, action, game.reward, next_state, done])
            agent.train(done, step)
            curr_state=np.copy(next_state)
            speed_list=deque(next_speed_list, maxlen=6)
            timestamp_list=deque(next_timestamp_list, maxlen=6)
            step+=1
            episode_reward+=game.reward

        print('Current Episode Reward: ', episode_reward)

        # ### FOR GENERATING STATS and WRITING IT TO TENSORBOARD
        ep_rewards.append(episode_reward)
        collisions.append(num_collisions)
        if not (episode+1) % AGGREGATE_STATS_EVERY or (episode+1) == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            avg_collisions = sum(collisions[-AGGREGATE_STATS_EVERY:])/len(collisions[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon, avg_collisions = avg_collisions)

            # Save model, but only when min reward is greater or equal a set value
            if (episode + 1) % 100 == 0:

                agent.model.save('models/{}_{}max_{}avg_{}min_{}epsilon_{}episode_{}.model'.format(MODEL_NAME, max_reward, average_reward, min_reward,
                                                                    round(epsilon, 2), episode, int(time.time())))
        # DECAY EPSILON
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

training_agent()

