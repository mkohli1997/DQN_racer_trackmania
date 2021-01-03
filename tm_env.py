# This code is the implementation for the TrackMania Forever's environment. This requires the game being played in windowed mode (640x480) in the top-left
# corner of the screen.

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


DIGIT_MODEL_PATH = "<path to the digit model>"
COLLISION_MODEL_PATH = "<path to the collision model>"
COMPLETION_REWARD = 50
COLLISION_PENALTY = -10
STUCK_PENALTY = -25
COMPLETE_STOP_PENALTY = -2

class TM(object):
    def __init__(self):
        self.model=load_model(DIGIT_MODEL_PATH)
        self.collision_model=load_model(COLLISION_MODEL_PATH)
        self.reward=0

    def get_reward(self, speed_list):
        """
        This function records the reward for the agent for the current speed.
        
        Args:
        speed_list: (list) speeds in the last 6 frames
        Returns:
        None
        """
        if speed_list[-1] == 0:
            self.reward = COMPLETE_STOP_PENALTY
        else:
            self.reward = np.exp(speed_list[-1]/100)


    def new_episode(self):
        """
        This function starts a new episode (new race) by invoking the key press "P"
        
        Args: None
        Returns: None
        """
        ReleaseAllKeys()
        PressKey(P)
        ReleaseKey(P)
        self.reward=0
        
        # Wait for the 3-second countdown to finish before ech race begins
        time.sleep(3)


    def make_action(self, action):
        """
        This function invokes key presses corresponding to the actions that are performed by the agent during gameplay.
        Args:
        action (int): Following is the int-key mapping for actions
                      0: W
                      1: WA
                      2: WD
                      3: S
                      4: NO ACTION
                      5: A
                      6: D
        Returns:
        None
        """

        if action == 0:
            PressKey(W)
            ReleaseKey(A)
            ReleaseKey(S)
            ReleaseKey(D)
        #WA
        elif action == 1:
            PressKey(W)
            PressKey(A)
            ReleaseKey(S)
            ReleaseKey(D)
        #WD
        elif action == 2:
            PressKey(W)
            PressKey(D)
            ReleaseKey(A)
            ReleaseKey(S)
        #S
        elif action == 3:
            PressKey(S)
            ReleaseKey(W)
            ReleaseKey(D)
            ReleaseKey(A)


        #NO ACTION
        elif action == 4:
            ReleaseAllKeys()

        #A
        elif action == 5:
            PressKey(A)
            ReleaseKey(D)
            ReleaseKey(S)
            ReleaseKey(W)

        #D
        elif action == 6:
            PressKey(D)
            ReleaseKey(A)
            ReleaseKey(S)
            ReleaseKey(W)

        print('Action: ', key_check())

        
    def check_stuck(self, speed_list, timestamp_list):
        """
        This function checks if the agent is stuck on the track. If stuck, record the penalty.
        
        Args:
        speed_list (list): Speeds in the last 6 frames.
        timestamp_list (list): Timer's reading in the last 6 frames.
        Returns:
        True: If stuck.
        False: If the agent is not stuck on the track.
        """
        
        # The agent is considered stuck if the agent spends more than 45 seconds on the track or max speed in the last
        # 6 frames is less than 5 km/hr. 
        if max(speed_list) < 5 or timestamp_list[-1] > 45:
            self.reward = STUCK_PENALTY
            return True
        else:
            return False

        
    def check_completion(self, timestamp_list):
        """
        This function checks if the agent has completed the race. If completed, record the reward.
        Args:
        timestamp_list (list): Timer's reading in the last 6 frames.
        Returns:
        True: If the agent completes the race.
        False: If the agent has not completed the race yet.
        """
        
        # When the race is completed, the timer stops at the current reading. We use this to detect the completion of the race.
        if len(set(timestamp_list)) == 1:
            self.reward = COMPLETION_REWARD
            return True
        else:
            return False
        

    def compute_time(self, timestamp_str):
        """
        This function computes time from the detected digits in the gameplay screen and converts into seconds
        
        Args: 
        timestamp_str (str): detected timestamp in str.
        Returns: timestamp in seconds.
        """
        
        minute = int(timestamp_str.split(':')[0])
        seconds = int(timestamp_str[timestamp_str.index(':')+1:timestamp_str.index('.')])
        milliseconds = int(timestamp_str[-2:])
        return (minute*60)+seconds+(milliseconds/100)

    
    def live_data(self, initial_call=False):
        """
        This function processes live data from the gameplay screen.
        
        Args:
        initial_call (boolean) (default=False): In the first call, speeds and timestamps from first 6 frames are appended to the queue.
                                                After that frames are processed 1 at a time.
        Returns:
        if initial_call == True:
            img_stack (numpy.ndarray): stack of image data from 6 frames
        else:
            stack_dict[0] (numpy.ndarray): image data from the latest single frame only
            
        speed_list (list): speeds from the 6 frames
        timestamp_list (list): timestamps from the 6 frames
        collision_flag (boolean): whether the agent collided in the latest frame
        """
        
        stack_dict = dict()
        speed_list = deque(maxlen=6)
        timestamp_list = deque(maxlen=6)
        i = 0
        collision_flag=False
        if initial_call:
            NUM_FRAMES = 6
        else:
            NUM_FRAMES = 1

        while True:
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
            speed = int(speed_str)
            timestamp = self.compute_time(timestamp_str)

            # Crop speedometer and time readings
            screen = screen[:-50, :]

            if i != NUM_FRAMES:
                img=cv2.resize(screen, (80, 60))/255.0
                # img = cv2.resize(screen, (80,60))
                stack_dict[i]=img
                speed_list.append(speed)
                timestamp_list.append(timestamp)

                # predict whether the agent collided or not
                if np.argmax(self.collision_model.predict(np.reshape(img,(-1, *img.shape, 1)))) == 0 and collision_flag == False:
                    collision_flag = True
                    self.reward = COLLISION_PENALTY

            else:
                if initial_call:
                    img_stack = np.stack((stack_dict[0], stack_dict[1], stack_dict[2],
                                        stack_dict[3], stack_dict[4], stack_dict[5]), axis=0)
                    return img_stack, speed_list, timestamp_list, collision_flag
                else:
                    return stack_dict[0], speed_list, timestamp_list, collision_flag
            i+=1
