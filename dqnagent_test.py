from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, BatchNormalization, Conv3D, MaxPooling3D, InputLayer
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import keras
from keras.models import load_model
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from collections import deque
import time
import random
import numpy as np
import tensorflow as tf
from tm_env import *
import os
from tqdm import tqdm
from presskeys import W, S , A, D, enter, esc
from keystrokes import  key_check
import keras.backend as K


REPLAY_MEMORY_SIZE = 20000
MODEL_NAME = 'straight_path_race_model_test_final'
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 2
MIN_REWARD = -100  # For model save
# MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 50

# Exploration settings
epsilon = 0  # not a constant, going to be decayed
EPSILON_DECAY = 0.9977
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 1  # episodes
SHOW_PREVIEW = False
MIN_REPLAY_MEMORY_SIZE = 100


# For stats
ep_rewards = []

# For more repetitive results when calling random
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)
# tf.random.set_seed(1)# tf 2.0

# Create models folder
# if not os.path.isdir('models'):
#     os.makedirs('models')



# class ModifiedTensorBoard(TensorBoard):
#     # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.step = 1
#         self.writer = tf.summary.FileWriter(self.log_dir)
#         # self.writer = tf.summary.create_file_writer(self.log_dir)
#
#     # Overriding this method to stop creating default log writer
#     def set_model(self, model):
#         pass
#
#     # Overrided, saves logs with our step number
#     # (otherwise every .fit() will start writing from 0th step)
#     def on_epoch_end(self, epoch, logs=None):
#         self.update_stats(**logs)
#
#     # Overrided
#     # We train for one batch only, no need to save anything at epoch end
#     def on_batch_end(self, batch, logs=None):
#         pass
#
#     # Overrided, so won't close writer
#     def on_train_end(self, _):w
#         pass
#
#     # Custom method for saving own metrics
#     # Creates writer, writes custom metrics and closes writer
#     def update_stats(self, **stats):
#         self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):
        # main model - train every step
        # self.model = self.create_model((6, 60, 80))
        self.model=load_model('models/[SECOND]32-64-128-128D_StridedCONVnet_left_huber_-16.955265568467212max_-42.62267230648464avg_-105.88737257779276min_0.11epsilon_1999episode_1587647207.model', custom_objects={'huber_loss': self.huber_loss})
        # self.model.load_weights('models/[FIRST-PASS]_64-128-256-256D_decayR[.99888]_left_turn_1019.2701239881852max_457.44943267519056avg_-476.65726157636686min_0.2epsilon_1425episode_1586510098.model')
        # self.model.set_weights(self.loaded_wts.get_weights())
        # target model - .pre
        # dict every step
        # self.target_model = self.create_model((6, 60, 80))
        # self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # self.tensorboard = ModifiedTensorBoard(log_dir='logs/{}-{}'.format(MODEL_NAME, int(time.time())))
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
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', data_format = 'channels_first'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format = 'channels_first'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), activation='relu', data_format = 'channels_first'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format = 'channels_first'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), activation='relu', data_format = 'channels_first'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format = 'channels_first'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(7, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

        return model


    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state, step):
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
        # self.model.fit(np.reshape(np.array(X),(*np.array(X).shape,1)), np.array(y), batch_size=MINIBATCH_SIZE,
        #                verbose=0, shuffle=False, callbacks=[spwelf.tensorboard] if terminal_state else None)

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
    num_completed=0
    for episode in tqdm(range(EPISODES), unit='episodes'):
        print('Exploration Rate: ', round(epsilon, 2))
        print('Replay memory size: ', len(agent.replay_memory))
        episode_reward = 0
        # agent.tensorboard.step = episode
        step = 1
        done = False
        stuck = False
        complete = False
        game.new_episode()
        curr_state, speed_list, timestamp_list, collided = game.live_data(initial_call=True)

        while not done:
            if np.random.random() > epsilon:
                print('Exploiting...')
                x = agent.get_qs(curr_state)
                action = np.argmax(x)
                # print(x)

            else:
                print('Exploring...')
                action = np.random.randint(0, 7)

            game.make_action(action)
            print(key_check())
            next_frame, next_speed, next_timestamp, collided = game.live_data()

            next_state = np.concatenate((curr_state[1:], [next_frame]), axis = 0)
            next_speed_list = deque(speed_list, maxlen=6)
            next_timestamp_list = deque(timestamp_list, maxlen=6)
            next_speed_list.append(next_speed[0])
            next_timestamp_list.append(next_timestamp[0])
            #
            if step > 6:
                complete = game.check_completion(next_timestamp_list)
                stuck = game.check_stuck(next_speed_list, next_timestamp_list)
            #
            if stuck:
                done = True

            elif complete:
                print('YAYYYY!!! RACE COMPLETED!!!')

                print('I will now reset the race...')
                done = True
                time.sleep(3)
                PressKey(enter)
                ReleaseKey(enter)
                time.sleep(1)
                PressKey(enter)
                ReleaseKey(enter)
            elif collided:
                print('OOPS I COLLIDED!!')
            # else:
            #     game.get_reward(speed_list)
            #     if action == 3:
            #         game.reward = -game.reward

            # agent.update_replay_memory((curr_state, action, game.reward, next_state, done))
            # agent.train(done, step)
            curr_state = np.copy(next_state)
            # cv2.imwrite('test_states/%d.jpg'%step,next_state[-1])
            speed_list = deque(next_speed_list, maxlen=6)
            timestamp_list = deque(next_timestamp_list, maxlen=6)
            step += 1
            # episode_reward += game.reward
        # print('Current Episode Reward: ', episode_reward)

        # # ### FOR GENERATING STATS and WRITING IT TO TENSORBOARD
        # ep_rewards.append(game.cumulative_reward)
        # if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        #     average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #     min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #     max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #     agent.tensorboard.apdddddddddddddddssdsddddddddddddddsdsddddsaaaaupdate_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
        #                                    epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            # if average_reward >= MIN_REWARD:
            #     agent.model.save('models/{}__{}max_{}avg_{}min_{}epsilon_{}.model'.format(MODEL_NAME, max_reward, average_reward, min_reward,
            #                                                         epsilon, int(time.time())))

    ReleaseAllKeys()
    print('THE AGENT COMPLETED THE RACE %d times in %d races.'%(num_completed, EPISODES))

training_agent()

