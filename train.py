import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion


TRAIN_DIR = 'C:\\Users\\Prime Focus Systems\\Pictures\\New DataSet\\train'
TEST_DIR = 'C:\\Users\\Prime Focus Systems\\Pictures\\New DataSet\\test'
IMG_SIZE = 64
LR = 1e-3  # 0.001

MODEL_NAME = 'flatten2-{}-{}.model'.format(LR,'6conv-basic')  # just so we remember which saved model is which, sizes must match


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == '500real' or word_label == '2000real':
        return [1, 0]
    #                             [no cat, very doggo]
    elif word_label == '500fake' or word_label == '2000fake':
        return [0, 1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.medianBlur(img, 5)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


train_data = create_train_data()

import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.medianBlur(img, 5)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
from keras.layers import Flatten

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = np.array(layer_shape[1:4], dtype = int).prod()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet, num_features = flatten_layer(convnet)

convnet = fully_connected(convnet, 1024, activation='relu')

convnet = fully_connected(convnet, 512, activation='relu')

convnet = fully_connected(convnet, 256, activation='relu')
convnet = dropout(convnet, 0.7)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
model.save(MODEL_NAME)


import os

#MODEL_NAME = 'fakeorreal-{}-{}.model'.format(LR,'4conv-basic')  # just so we remember which saved model is which, sizes must match
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-30000]
test = train_data[-30000:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=15
          , validation_set=({'input': test_x}, {'targets': test_y}),
         snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)

TEST_DIR = 'C:\\Users\\Prime Focus Systems\\Pictures\\New DataSet\\test'

test_data = process_test_data()

import os

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded! finally')
else:
    print("model not loaded")

import matplotlib.pyplot as plt

# if you need to create the data:
# test_data = process_test_data()
# if you already have some saved:
test_data = process_test_data()



for num, data in enumerate(test_data[:8000]):
    # cat: [1,0]
    # dog: [0,1]

    img_num = data[1]
    img_data = data[0]


    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out)==0:
        print("real"+" "+img_num)
    else:
        print("fake "+img_num)
