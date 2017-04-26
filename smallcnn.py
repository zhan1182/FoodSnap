
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
        self.input_shape = (img_rows, img_cols, 3)
        self.num_of_classes = 5

        self.filters = [32, 64, 128]

        """ Define the convolutional nerual network """
        # Initiate the convolutional nerual network as a sequential model in Keras
        self.model = Sequential()

        for filt in self.filters:
            conv_layer = Conv2D(filters=filt,
                kernel_size=self.kernel_size, 
                padding='same',
                activation='relu',
                input_shape=self.input_shape)

            self.model.add(conv_layer)
            self.model.add(MaxPooling2D(pool_size=pooling_size))

        # Add a flatten layer to flatten the intermediate nodes
        self.model.add(Flatten())

        # Add a regular densely-connected layer with ReLu
        self.model.add(Dense(1024, activation='relu'))

        # Add a dropout layer for 0.5 keep probability
        self.model.add(Dropout(0.5))

        # Add the output layer to the network with softmax
        self.model.add(Dense(10, activation='softmax'))

        # Generate an optimizer with learning rate = 1e-4
        adam_opt = Adam(lr=1e-4)

        # Compile the cnn network for training
        # It uses the Adam optimizer, categorical crossentropy loss function,
        # and accuracy as validation metrics
        self.model.compile(loss='categorical_crossentropy',
                   optimizer=adam_opt,
                   metrics=['accuracy'])

        def load_data(self):
            pass

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
            history = self.model.fit(X_train, y_train, 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            verbose=0, 
                            callbacks=[tb, mc],
                            validation_data=(X_validation, y_validation))



        def predict(self, X):
            pass


