import cv2
import numpy as np
import os
import time
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

x, y=[],[]
NUM_CLASSES=2
DATA_DIR='collision_data'
MODEL_NAME='TEST'

# LOAD DATA

for i in range(NUM_CLASSES):
    print('Loading data for class %d...'%i)
    for filename in os.listdir(DATA_DIR+'/'+'%d'%i):
        img=cv2.imread(DATA_DIR+'/%d/'%i+filename, 0)
        img=cv2.resize(img,(80,60))
        img=np.reshape(img,(img.shape[0], img.shape[1], 1))
        img=img/255.0
        x.append(img)
        y.append(i)

x=np.array(x)
y=np.array(y)
y=to_categorical(y,2)

x, y = shuffle(x, y)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20)

input_shape=x.shape[1:]
optimizer=Adam(lr=1e-4)

print('INPUT SHAPE:', input_shape)
print('\n\n DATA SHAPE: ', x.shape)

datagen_train=ImageDataGenerator(horizontal_flip=True)
datagen_test=ImageDataGenerator(horizontal_flip=True)

datagen_train.fit(x_train)
datagen_test.fit(x_val)

model=Sequential()

model.add(Conv2D(64, (5,5), strides=(1,1), input_shape=input_shape, activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), strides=(1,1), padding='valid', activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), strides=(1,1), padding='valid', activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

filepath="saved_weights_collision_recognition/weights-{epoch:02d}-{val_loss:4f}-{val_acc:.3f}-ADDED_1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tb_obj=TensorBoard(log_dir='logs/{}-{}'.format(MODEL_NAME, int(time.time())), write_graph=True, write_images=True, histogram_freq=10)

model.summary()
exit(0)
model.fit_generator(datagen_train.flow(x_train,y_train, batch_size=128),epochs=100, validation_data=datagen_test.flow(x_val, y_val),
                    steps_per_epoch=len(x_train)//128, validation_steps=len(x_val)//128, callbacks=[checkpoint, tb_obj], verbose=2)


