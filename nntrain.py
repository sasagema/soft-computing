

from __future__ import print_function
import keras
import cv2
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from skimage.measure import label, regionprops

#pomeranje slike u gornji levi ugao
def move_to_upper_left_corner(image, x, y):

    moved = np.zeros((28, 28))
    moved[:(28-x), :(28-y)] = image[x:,y:]
    return moved


def prepare_data(data):

    prepared = np.empty_like(data)

    for i in range(0, len(data)):

        labeled = label(data[i].reshape(28, 28))
        contours = regionprops(labeled)

        min_x = contours[0].bbox[0]
        min_y = contours[0].bbox[1]

        for j in range(1, len(contours)):
            if contours[j].bbox[0] < min_x:
                min_x = contours[j].bbox[0]
            if contours[j].bbox[1] < min_y:
                min_y = contours[j].bbox[1]

        moved = move_to_upper_left_corner(data[i].reshape(28, 28), min_x, min_y)
        prepared[i] = moved

    return prepared


batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()




#cv2.imwrite('uzorak1.jpg', x_train[1])

input_shape = (784,)
x_train_prepared = prepare_data(x_train)
x_test_prepared = prepare_data(x_test)

x_train = x_train_prepared.reshape(60000, 784)
x_test = x_test_prepared.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, input_shape=input_shape, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(384, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#Test loss: 0.0783547559347
#Test accuracy: 0.9788
model.save('model2.h5')