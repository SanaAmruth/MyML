import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
#from keras.utils import to_categorical
from keras.utils import np_utils
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Convert y_train into one-hot format 
temp = []
for i in range(len(y_train)):
    temp.append(np_utils.to_categorical(y_train[i], num_classes=10))


y_train = np.array(temp)

temp = []
for i in range(len(y_test)):
    temp.append(np_utils.to_categorical(y_test[i], num_classes=10))

y_test = np.array(temp)

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(60, activation='ReLU'))
model.add(Dense(50, activation='ReLU'))
model.add(Dense(40, activation='ReLU'))
model.add(Dense(20, activation='ReLU'))
model.add(Dense(10, activation='softmax'))


model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])


model.fit(X_train, y_train, epochs=10, validation_data=(X_test,y_test))