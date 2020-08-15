import numpy as np
import cv2
from keras.models import load_model

import time
from generate_scr_data import *
from PIL import ImageGrab
from keystrokes import key_check
import random
from presskeys import PressKey, W, A, S, D, P, L, ReleaseKey, ReleaseAllKeys
from math import log
import time
from collections import deque
from math import e


# NUM_FRAMES = 6
COMPLETION_REWARD = 50
COLLISION_PENALTY = -10
STUCK_PENALTY = -25
COMPLETE_STOP_PENALTY = -2

class TM(object):
    def __init__(self):
        self.model=load_model('weights-80-0.018682-0.993.hdf5')
        self.collision_model=load_model('collision_model/weights-45-0.000115-1.000-ADDED_1.hdf5')
        self.reward=0

    def get_reward(self, speed_list):
        if speed_list[-1]==0:
            self.reward = COMPLETE_STOP_PENALTY
        else:
            self.reward = np.exp(speed_list[-1]/100)


    def new_episode(self):
        ReleaseAllKeys()
        PressKey(P)
        ReleaseKey(P)
        self.reward=0
        # PressKey(W)
        time.sleep(3)


    def make_action(self, action):
        '''
        0: W
        1: WA
        2: WD
        3: S
        4: NO ACTION
        5: A
        6: D
        :param action:
        :return:
        '''

        if action==0:
            PressKey(W)
            ReleaseKey(A)
            ReleaseKey(S)
            ReleaseKey(D)
        #WA
        elif action==1:
            PressKey(W)
            PressKey(A)
            ReleaseKey(S)
            ReleaseKey(D)
        #WD
        elif action==2:
            PressKey(W)
            PressKey(D)
            ReleaseKey(A)
            ReleaseKey(S)
        #S
        elif action==3:
            PressKey(S)
            ReleaseKey(W)
            ReleaseKey(D)
            ReleaseKey(A)


        #NO ACTION
        elif action==4:
            ReleaseAllKeys()

        #A
        elif action==5:
            PressKey(A)
            ReleaseKey(D)
            ReleaseKey(S)
            ReleaseKey(W)

        #D
        elif action==6:
            PressKey(D)
            ReleaseKey(A)
            ReleaseKey(S)
            ReleaseKey(W)

        print('Action: ', key_check())

    def check_stuck(self, speed_list, timestamp_list):
        if max(speed_list)<5 or timestamp_list[-1]>45:
            self.reward = STUCK_PENALTY
            return True
        else:
            return False

    def check_completion(self, timestamp_list):
        # if len(set(timestamp_list))==1:
        #     self.reward = COMPLETION_REWARD
        #     return True
        if timestamp_list[-1]==timestamp_list[-2] and timestamp_list[-2]==timestamp_list[-3]:
            self.reward = COMPLETION_REWARD
            return True
        else:
            return False

    def compute_time(self, timestamp_str):
        minute=int(timestamp_str.split(':')[0])
        seconds=int(timestamp_str[timestamp_str.index(':')+1:timestamp_str.index('.')])
        milliseconds=int(timestamp_str[-2:])
        return (minute*60)+seconds+(milliseconds/100)

    def live_data(self, initial_call=False):

        stack_dict=dict()
        speed_list = deque(maxlen=6)
        timestamp_list = deque(maxlen=6)
        i = 0
        collision_flag=False
        if initial_call:
            NUM_FRAMES=6
        else:
            NUM_FRAMES=1

        while (True):

            screen = np.array(ImageGrab.grab(bbox=(0, 40, 640, 520)))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            speed = screen[454:479, 589:639]

            speed_ones, speed_tens, speed_hundreds = split_cropped_speed(speed)
            speed_str = generate_speed_str(self.model, speed_ones, speed_tens, speed_hundreds)

            timestamp = crop_timestamp(screen)
            millisecond_ones, millisecond_tens, second_ones, second_tens, minute_ones = split_cropped_timestamp(
                timestamp)
            timestamp_str = generate_timestamp_str(self.model, millisecond_ones, millisecond_tens, second_ones, second_tens,
                                                   minute_ones)
            speed=int(speed_str)
            timestamp=self.compute_time(timestamp_str)

            # Crop speedometer and time readings
            screen = screen[:-50, :]

            if i!=NUM_FRAMES:
                img=cv2.resize(screen, (80, 60))/255.0
                # img = cv2.resize(screen, (80,60))
                stack_dict[i]=img
                speed_list.append(speed)
                timestamp_list.append(timestamp)

                if np.argmax(self.collision_model.predict(np.reshape(img,(-1, *img.shape, 1))))==0 and collision_flag==False:
                    collision_flag=True
                    self.reward = COLLISION_PENALTY

            else:
                if initial_call:
                    img_stack=np.stack((stack_dict[0], stack_dict[1], stack_dict[2],
                                        stack_dict[3], stack_dict[4], stack_dict[5]), axis=0)
                    return img_stack, speed_list, timestamp_list, collision_flag
                else:
                    return stack_dict[0], speed_list, timestamp_list, collision_flag
            i+=1
