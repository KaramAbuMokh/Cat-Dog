import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import filter_images

if __name__ == '__main__':

    names, labels = filter_images.data()
    X = []
    fltered_labels = []
    for photo, label in zip(names, labels):
        try:
            img = cv2.imread('images/' + photo)
            if img is None:
                os.remove('images/' + photo)
            else:
                X.append(img)
                fltered_labels.append(label)
        except:
            print("file not found" + photo)

    X_train = X[:int(len(X) * 0.8)]
    y_cat_train = fltered_labels[:int(len(fltered_labels) * 0.8)]
    X_test = X[int(len(X) * 0.8):]
    y_cat_test = fltered_labels[int(len(fltered_labels) * 0.8):]

    X_train = np.array(X_train)
    y_cat_train = np.array(y_cat_train)
    X_test = np.array(X_test)
    y_cat_test = np.array(y_cat_test)

    from sklearn.utils import shuffle

    X_train, y_cat_train = shuffle(X_train, y_cat_train, random_state=0)
    X_test, y_cat_test = shuffle(X_test, y_cat_test, random_state=0)

    # scale the data
    X_test = X_test / 255
    X_train = X_train / 255

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
    import tensorflow_hub as hub
    import tensorflow as tf

    '''url='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
    mobile_net=hub.KerasLayer(url,input_shape=(224,224,3))
    mobile_net.trainaable=False

    model = Sequential()
    model.add(mobile_net)
    model.add(Conv2D(filters=64, kernel_size=(4, 4), input_shape=(128, 128, 3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(128, 128, 3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    '''

    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", output_shape=[1280],
                       trainable=False),  # Can be True, see below.
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.build([4, 224, 224, 3])

    model.compile(loss='categorical_crossentropy', optimizer='adam'
                  , metrics=['accuracy'])

    # see the summary of the model structure
    print(model.summary())

    # adding early stop
    from tensorflow.keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(monitor='val_loss', patience=2)

    # train the model
    model.fit(X_train, y_cat_train, epochs=10
              , validation_data=(X_test, y_cat_test)
              , callbacks=[early_stop]
              , batch_size=4)

    # save the model
    model.save('model to predict animals.h5')

    # save the history of the model
    models_history = pd.DataFrame(model.history.history)
    models_history.to_csv('history of the model.csv', index=False)

    # load the model
    from tensorflow.keras.models import load_model

    my_model = load_model('model to predict animals.h5')

    # load the history od the model
    models_history = pd.read_csv('history of the model.csv')

    # plot the validation loss and accuracy
    models_history.plot()
    plt.show()

    # printing the history of the model :loss and accuracy
    print(models_history)

    # plot only the accuracy
    models_history[['accuracy', 'val_accuracy']].plot()
    plt.show()

    # plot only the validation loss
    models_history[['loss', 'val_loss']].plot()
    plt.show()

    # get the predictions
    predictions = my_model.predict_classes(X_test)

    # some evaluation

    # validation loss and the validation accuracy
    print(my_model.evaluate(X_test, y_cat_test, verbose=0))

    # predict image
    my_image = X_test[0]
    print(y_cat_test[0])
    print(my_model.predict_classes(my_image.reshape(1, 128, 128, 3)))
