
import pickle

import numpy as np

# Import the sequential model from Keras
from keras.models import Sequential

# Import regular densely-connected NN layer, flatten layer, and dropout layer
from keras.layers import Dense, Flatten, Dropout

# Import 2D convolution layer and 2D max pooling layer
from keras.layers import Conv2D, MaxPooling2D

# Import Adam optimizer
from keras.optimizers import Adam

# Import the callbacks for Tensorboard and ModelCheckpoint
from keras.callbacks import TensorBoard, ModelCheckpoint


class SmallCNN(object):
    def __init__(self):
        """ Define constants """
        # Define the 2D kernel size
        self.kernel_size = (5, 5)
        self.pooling_size = (2, 2)

        # Define the batch size for training
        self.batch_size = 50

        # Define the number of training cycle
        self.epochs = 20

        # Define the input image shape, depth at the last
        self.img_rows = 224
        self.img_cols = 224
        self.input_shape = (self.img_rows, self.img_cols, 3)
        self.num_of_classes = 6

        self.filters = [64, 128, 256]

        """ Define the convolutional nerual network """
        # Initiate the convolutional nerual network as a sequential model in Keras
        self.model = Sequential()

        for i, filt in enumerate(self.filters):
            if i == 0:
                conv_layer = Conv2D(filters=filt,
                    kernel_size=self.kernel_size, 
                    padding='same',
                    activation='relu',
                    input_shape=self.input_shape)
            else:
                conv_layer = Conv2D(filters=filt,
                    kernel_size=self.kernel_size, 
                    padding='same',
                    activation='relu')

            self.model.add(conv_layer)
            self.model.add(MaxPooling2D(pool_size=self.pooling_size))

        # Add a flatten layer to flatten the intermediate nodes
        self.model.add(Flatten())

        # Add a regular densely-connected layer with ReLu
        self.model.add(Dense(2048, activation='relu'))

        # Add a dropout layer for 0.5 keep probability
        self.model.add(Dropout(0.5))

        # Add the output layer to the network with softmax
        self.model.add(Dense(self.num_of_classes, activation='softmax'))

        # Generate an optimizer with learning rate = 1e-4
        adam_opt = Adam(lr=1e-4)

        # Compile the cnn network for training
        # It uses the Adam optimizer, categorical crossentropy loss function,
        # and accuracy as validation metrics
        self.model.compile(loss='categorical_crossentropy',
                   optimizer=adam_opt,
                   metrics=['accuracy'])

    def to_onehot(self, y):
        m = len(y)
        y_array = np.zeros((m, self.num_of_classes))

        for i, label in enumerate(y):
            y_array[i, label - 1] = 1

        return y_array


    def load_data(self):
        pickle_dir = './pickle_files/'
        x_train_file = pickle_dir + 'tr_dataset.pickle'
        y_train_file = pickle_dir + 'tr_labels.pickle'

        self.X_train = pickle.load(open(x_train_file, 'rb'))
        self.y_train = self.to_onehot(pickle.load(open(y_train_file, 'rb')))
        
        x_validation_file = pickle_dir + 'v_dataset.pickle'
        y_validation_file = pickle_dir + 'v_labels.pickle'

        self.X_validation = pickle.load(open(x_validation_file, 'rb'))
        self.y_validation = self.to_onehot(pickle.load(open(y_validation_file, 'rb')))
        
        x_test_file = pickle_dir + 'te_dataset.pickle'
        y_test_file = pickle_dir + 'te_labels.pickle'

        self.X_test = pickle.load(open(x_test_file, 'rb'))
        self.y_test = self.to_onehot(pickle.load(open(y_test_file, 'rb')))

    def train(self):
        # Define the TensorBoard callback
        tb = TensorBoard()

        # Define the file name of the best model 
        best_model = './models/weights.best.hdf5'

        # Define the modelcheckpoint callback
        mc = ModelCheckpoint(best_model,
                            save_best_only=True, 
                            save_weights_only=True)

        # Training process
        history = self.model.fit(self.X_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        verbose=1,
                        callbacks=[tb, mc],
                        validation_data=(self.X_validation, self.y_validation))

    def predict(self, X):
        pass


    def evaluate(self, X_test):
        pass


if __name__ == '__main__':
    sc = SmallCNN()

    sc.load_data()

    sc.train()

    import gc
    gc.collect()

