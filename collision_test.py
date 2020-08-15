import numpy as np
import cv2
from keras.models import load_model
import time
from PIL import ImageGrab

model=load_model('collision_model/weights-45-0.000115-1.000-ADDED_1.hdf5')
i=0
print('Now Detecting...')
while(True):
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 640, 520)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    cv2.imshow('i', screen_gray[:-50,:])
    screen_gray = screen_gray[:-50, :]
    screen_resized = cv2.resize(screen_gray, (80,60))
    screen_resized_reshape = np.reshape(screen_resized, (1, 60, 80, 1))
    prediction = np.argmax(model.predict(screen_resized_reshape/255.0))
    if prediction==0:
        print('Collision detected!')
        print('Writing %d...' % i)
        cv2.imwrite('collision_test_write/%d.jpg' % i, screen)
        i += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break