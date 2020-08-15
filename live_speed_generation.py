import numpy as np
import cv2
from keras.models import load_model
import time
from generate_scr_data import *
from PIL import ImageGrab
from keystrokes import key_check
import random
from presskeys import PressKey, W, A, S, D, P, L, ReleaseKey

model=load_model('weights-80-0.018682-0.993.hdf5')

i=0
while(True):
    t1=time.time()
    screen =  np.array(ImageGrab.grab(bbox=(0,40,640,520)))
    screen=cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('full window', screen)
    speed=screen[455:480,589:639]
    cv2.imshow('speed crop',speed)

    speed_ones, speed_tens, speed_hundreds = split_cropped_speed(speed)
    speed_str = generate_speed_str(model, speed_ones, speed_tens, speed_hundreds)

    timestamp = crop_timestamp(screen)
    cv2.imshow('t', timestamp)
    millisecond_ones, millisecond_tens, second_ones, second_tens, minute_ones = split_cropped_timestamp(timestamp)
    timestamp_str = generate_timestamp_str(model, millisecond_ones, millisecond_tens, second_ones, second_tens, minute_ones)
    # cv2.imshow('full', screen)
    # print(screen.shape)
    # print(speed.shape)
    # cv2.imwrite('grabs/%d.jpg'%i,screen)

    # r=random.randint(0,6)
    # if r==0:
    #     PressKey(W)
    #     ReleaseKey(A)
    #     ReleaseKey(S)
    #     ReleaseKey(D)
    # elif r==1:
    #     PressKey(D)
    #     ReleaseKey(A)
    #     ReleaseKey(S)
    # elif r==2:
    #     PressKey(A)
    #     ReleaseKey(D)
    #     ReleaseKey(S)
    # elif r==3:
    #     PressKey(W)
    #     PressKey(D)
    #     ReleaseKey(A)
    #     ReleaseKey(S)
    # elif r==4:
    #     PressKey(W)
    #     PressKey(A)
    #     ReleaseKey(S)
    #     ReleaseKey(D)
    # elif r==5:
    #     PressKey(A)
    #     PressKey(S)
    #     ReleaseKey(W)
    #     ReleaseKey(D)
    # elif r==6:
    #     PressKey(D)
    #     PressKey(S)
    #     ReleaseKey(W)
    #     ReleaseKey(A)
    # # elif r==6:
    #     PressKey(P)
    # elif r==7:
    #     PressKey(L)
    print('Key pressed: ', key_check())
    print('Current Speed: ', speed_str)
    print('Current Timestamp: ',timestamp_str)

    # time.sleep(0.5)

    # elif r==6:
    #     ReleaseKey(P)
    # elif r==7:
    #     ReleaseKey(L)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    print(time.time()-t1)
    i+=1