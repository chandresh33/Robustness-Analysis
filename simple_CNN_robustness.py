import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import KFold
import os

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_dataset():
    # load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # print('MNIST Dataset Shape:')
    # print('X_train: '+str(X_train.shape))
    # print('Y_train: '+str(Y_train.shape))
    # print('X_test: '+str(X_test.shape))
    # print('Y_test: '+str(Y_test.shape))

    # reshape images
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # one hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # return preprocessed data
    return x_train, y_train, x_test, y_test


def prep_pixels(train, test):
    # make the integer numbers to type float32
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    # normalise values in the range 0-1
    train_norm /= 255.0
    test_norm /= 255.0

    # return normalised images
    return train_norm, test_norm


def define_model():
    # define model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def evaluate_model(data_x, data_y):
    scores, histories = list(), list()

    # define model
    model = define_model()

    # select rows for train and test
    train_x, train_y, test_x, test_y = data_x[0:(len(data_x)*0.67)], data_y[0:(len(data_y)*0.67)],\
                                       data_x[(len(data_x)*0.67):len(data_x)-(len(data_x)*0.67)],\
                                       data_y[(len(data_y)*0.67):len(data_y)-(len(data_y)*0.67)]

    # fit model
    history = model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(test_x, test_y), verbose=0)

    # evaluate model
    _, acc = model.evaluate(test_x, test_y, verbose=0)
    print('> %.3f' % (acc * 100))

    # stores scores
    scores.append(acc)
    histories.append(history)

    model.save("2D_cnn_mnist.h5")

    return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')

        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')

    pyplot.show()


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores) * 100, np.std(scores) * 100, len(scores)))

    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()


# run the test harness for evaluating a model
def run_test_harness(graphics):
    # load dataset
    train_x, train_y, test_x, test_y = load_dataset()

    # prepare pixel data
    train_x, test_x = prep_pixels(train_x, test_x)

    # evaluate model
    scores, histories = evaluate_model(train_x, train_y)

    if graphics == 1:
        # learning curves
        summarize_diagnostics(histories)

        # summarise estimated performance
        summarize_performance(scores)


# run the whole model
run_test_harness(graphics=0)
