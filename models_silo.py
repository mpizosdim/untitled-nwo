from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd


class CNNsequantial(object):
    def __init__(self, file_to_save, dropout=0.2, learning_rate=0.001):
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.file_to_save = file_to_save
        self.gmodel = self.getModel()
        self.callbacks = self.get_callbacks()
        self.history = None

    def getModel(self):
        # Building the model
        gmodel = Sequential()
        # Conv Layer 1
        gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)))
        gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        gmodel.add(Dropout(self.dropout))

        # Conv Layer 2
        gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        gmodel.add(Dropout(self.dropout))

        # Conv Layer 3
        gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        gmodel.add(Dropout(self.dropout))

        # Conv Layer 4
        gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        gmodel.add(Dropout(self.dropout))

        # Flatten the data for upcoming dense layers
        gmodel.add(Flatten())

        # Dense Layers
        gmodel.add(Dense(512))
        gmodel.add(Activation('relu'))
        gmodel.add(Dropout(self.dropout))

        # Dense Layer 2
        gmodel.add(Dense(256))
        gmodel.add(Activation('relu'))
        gmodel.add(Dropout(self.dropout))

        # Sigmoid Layer
        gmodel.add(Dense(1))
        gmodel.add(Activation('sigmoid'))

        mypotim = Adam(lr=self.learning_rate)
        gmodel.compile(loss='binary_crossentropy', optimizer=mypotim, metrics=['accuracy'])
        gmodel.summary()
        return gmodel

    def get_callbacks(self, patience=5):
        es = EarlyStopping('val_loss', patience=patience, mode="min")
        msave = ModelCheckpoint(self.file_to_save, save_best_only=True)
        return [es, msave]

    def fit_model(self, x_train, y_train, x_valid, y_valid, batch_size=24, epochs=50):
        self.history = self.gmodel.fit(x_train, y_train,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       verbose=1,
                                       validation_data=(x_valid, y_valid),
                                       callbacks=self.callbacks)

    def evaluate_model(self, x, y):
        self.gmodel.load_weights(filepath=self.file_to_save)
        score = self.gmodel.evaluate(x, y, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        fig = plt.figure()
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

    def predict_from_model_and_write(self, x, ids=None, write_file=None):
        predicted_test = self.gmodel.predict_proba(x)
        if write_file:
            submission = pd.DataFrame()
            submission['id'] = ids
            submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))
            submission.to_csv(write_file, index=False)
        return predicted_test


