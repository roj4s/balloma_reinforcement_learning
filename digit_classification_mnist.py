import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint

def get_model(input_shape=(28, 28, 1)):
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
    return model

def train_on_mnist(model, output_to='saved_models/mnist'):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points
    # after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    checkpointer = ModelCheckpoint(filepath='{}_weights.hdf5'.format(output_to),
                                                     verbose=2,
                      save_best_only=True)
    model.fit(x=x_train,y=y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[checkpointer])


def test(model, img):
    return model.predict(img)

if __name__ == "__main__":
    import sys
    output_addr = sys.argv[1]
    model = get_model()
    #train_on_mnist(model, output_to=output_addr)

    # Only for testing
    import cv2
    import numpy as np
    model.load_weights('{}_weights.hdf5'.format(output_addr))
    img = cv2.imread(sys.argv[2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    #img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
    img = np.array([img])
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img /= 255
    print(img)
    print(np.argmax(test(model, img)))


