import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
import os

# model=load_model('saved_weights_digit_recognition/weights-34-0.036266-0.989.hdf5')

# GAME_SCR_DIR='Stadium/Stadium/obstacles_no'

def crop_speed(img):
    return img[img.shape[0]-25:,img.shape[1]-50:]

def crop_timestamp(img):
    return img[img.shape[0]-25:, img.shape[1]-375:img.shape[1]-275]

def split_cropped_timestamp(timestamp):
    millisecond_ones = timestamp[:, timestamp.shape[1] - 15:timestamp.shape[1] - 3]
    millisecond_tens = timestamp[:, timestamp.shape[1] - 32:timestamp.shape[1] - 20]
    second_ones = timestamp[:, timestamp.shape[1] - 51:timestamp.shape[1] - 39]
    second_tens = timestamp[:, timestamp.shape[1] - 66:timestamp.shape[1] - 54]
    minute_ones = timestamp[:, timestamp.shape[1] - 85:timestamp.shape[1] - 73]

    return millisecond_ones, millisecond_tens, second_ones, second_tens, minute_ones

def generate_timestamp_str(model, millisecond_ones, millisecond_tens, second_ones, second_tens, minute_ones):

    # while True:
    #     cv2.imshow('i', millisecond_ones)
    #     cv2.imshow('ii', millisecond_tens)
    #     cv2.imshow('iii', second_ones)
    #     cv2.imshow('iv', second_tens)
    #     cv2.imshow('v', minute_ones)

        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break

    ret, thresh_second_ones = cv2.threshold(second_ones, 200, 255, cv2.THRESH_BINARY)

    thresh_second_ones = np.reshape(thresh_second_ones,
                                    (1, thresh_second_ones.shape[0], thresh_second_ones.shape[1], 1))
    second_ones = np.argmax(model.predict(thresh_second_ones / 255.0))

    ret, thresh_second_tens = cv2.threshold(second_tens, 200, 255, cv2.THRESH_BINARY)


    thresh_second_tens = np.reshape(thresh_second_tens,
                                    (1, thresh_second_tens.shape[0], thresh_second_tens.shape[1], 1))
    second_tens = np.argmax(model.predict(thresh_second_tens / 255.0))


    ret, thresh_minute_ones = cv2.threshold(minute_ones, 200, 255, cv2.THRESH_BINARY)
    thresh_minute_ones = np.reshape(thresh_minute_ones,
                                    (1, thresh_minute_ones.shape[0], thresh_minute_ones.shape[1], 1))
    minute_ones = np.argmax(model.predict(thresh_minute_ones / 255.0))

    ret, thresh_millisecond_ones = cv2.threshold(millisecond_ones, 200, 255, cv2.THRESH_BINARY)
    thresh_millisecond_ones = np.reshape(thresh_millisecond_ones,
                                         (1, thresh_millisecond_ones.shape[0], thresh_millisecond_ones.shape[1], 1))
    millisecond_ones = np.argmax(model.predict(thresh_millisecond_ones / 255.0))

    ret, thresh_millisecond_tens = cv2.threshold(millisecond_tens, 200, 255, cv2.THRESH_BINARY)
    thresh_millisecond_tens = np.reshape(thresh_millisecond_tens,
                                         (1, thresh_millisecond_tens.shape[0], thresh_millisecond_tens.shape[1], 1))
    millisecond_tens = np.argmax(model.predict(thresh_millisecond_tens / 255.0))

    timestamp_str = str(minute_ones) + ':' + str(second_tens) + str(second_ones) + '.' + str(millisecond_tens) + str(millisecond_ones)

    return timestamp_str

def split_cropped_speed(speed):
    ret, thresh = cv2.threshold(speed, 200, 255, cv2.THRESH_BINARY)
    ones = thresh[:, thresh.shape[1] - 20: thresh.shape[1]-8]
    tens = thresh[:, thresh.shape[1] - 33:thresh.shape[1] - 21]
    hundreds = thresh[:, thresh.shape[1] - 46:thresh.shape[1] - 34]

    # while True:
    #     # cv2.imshow('i', ones)
    #
    #     # cv2.imshow('ii', tens)
    #     # cv2.imshow('iii', hundreds)
    #     cv2.imshow('iv', thresh)
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break

    return ones, tens, hundreds

def generate_speed_str(model, ones, tens, hundreds):
    ones = np.reshape(ones, (1, ones.shape[0], ones.shape[1], 1))
    tens = np.reshape(tens, (1, tens.shape[0], tens.shape[1], 1))
    hundreds = np.reshape(hundreds, (1, hundreds.shape[0], hundreds.shape[1], 1))

    ones_prediction = np.argmax(model.predict(ones / 255.0))
    tens_prediction = np.argmax(model.predict(tens / 255.0))
    hundreds_prediction = np.argmax(model.predict(hundreds / 255.0))
    # print(tens_prediction)

    if tens_prediction==10:
        tens_prediction=0
    if hundreds_prediction==10:
        hundreds_prediction=0

    return str(hundreds_prediction)+str(tens_prediction)+str(ones_prediction)


# img=[]
# time=[]
# speed=[]
# for filename in os.listdir(GAME_SCR_DIR):
#     img=cv2.imread(GAME_SCR_DIR+'/'+filename, 0)
#     img=img[5:img.shape[0]-5,:]
#     speed=crop_speed(img)
#     timestamp=crop_timestamp(img)
#     millisecond_ones, millisecond_tens, second_ones, second_tens, minute_ones=split_cropped_timestamp(timestamp)
#
#     timestamp_str=generate_timestamp_str(millisecond_ones, millisecond_tens, second_ones, second_tens, minute_ones)
#
#     speed_ones, speed_tens, speed_hundreds=split_cropped_speed(speed)
#     speed_str=generate_speed_str(speed_ones, speed_tens, speed_hundreds)
#     print(filename+' : '+speed_str)
    # s=input()

    '''
    img.append(img)
    time.append(timestamp_str)
    speed.append(speed_str)
    
df=pd.DataFrame({'Image': img,
                'Timestamps': time,
                'Speeds': speed)
                
df.to_csv('game_scr_data.csv', index=False)
    '''
