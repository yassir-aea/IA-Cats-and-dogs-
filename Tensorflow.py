import os
import matplotlib.pylab as plt
import numpy as np
import cv2
import random
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout , Activation , Flatten , Conv2D , MaxPooling2D
from keras.callbacks.tensorboard_v2 import TensorBoard
import time

NAME = "Cats-vs-dogs-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs\\{}'.format(NAME))


Data_dir = "C:\\Users\\Yassir\\Desktop\\images"

CATEGORIES = ['Dog' , 'Cat']

IMG_SIZE = 30
training_data = []

def Create_training_data():
    for categorie in CATEGORIES:
        path = os.path.join(Data_dir, categorie)  # path to categories
        class_num = CATEGORIES.index(categorie)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                print(str(img))

Create_training_data()

random.shuffle(training_data)

x=[]
y=[]

for features, label in training_data :
    x.append(features)
    y.append(label)
x=np.array(x).reshape(-1, IMG_SIZE,IMG_SIZE,1)


pickle_out = open("x.pickle" , "wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle" , "wb")
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in = open("x.pickle","rb")
x = pickle.load(pickle_in)
x[1]

X = x/255.0
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer= "adam", metrics=['accuracy'])
model.fit(x,y,batch_size=3, validation_split=0.1 , epochs=20 , callbacks=[tensorboard])































