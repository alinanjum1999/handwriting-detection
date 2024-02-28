import os
import cv2 #for computer vision to load and process images
import numpy as np
import matplotlib.pyplot as plt #visualization of digits 
import tensorflow as tf #machine learning part
import image 

# run the following command primarily for training the model 
################################################################
# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# #normalizing the data (scaling down from 0-255 -> 0-1)

# x_train = tf.keras.utils.normalize(x_train, axis=1)

# x_test = tf.keras.utils.normalize(x_test, axis=1)


# model= tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #flattens to a line of 28*28 pixels
# model.add(tf.keras.layers.Dense(128,activation='relu')) #relu - rectify linear units 
# model.add(tf.keras.layers.Dense(10,activation='softmax')) #softmax makes is so that all the 10 neurons activate into one

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train,epochs= 3)

# model.save('test.model')

# testing the model to see the loss and accuracy
################################

# model = tf.keras.models.load_model('test.model')

# loss, accuracy = model.evaluate(x_test, y_test)

# print('loss:',loss)
# print ('accuracy:',accuracy)

#loading handwritten digits to see the loss and accuracy of the model

model = tf.keras.models.load_model('test.model')

while os.path.isfile(f"digita/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digita/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"this digit is probably a {np.argmax(prediction)}")
        #argmax gives us the index of the most likely prediction
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print('Error')
    finally:
        {image_number} += 1