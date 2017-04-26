

import os
import h5py

import matplotlib.pyplot as plt
import pickle

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend
from keras import optimizers


def build_vgg16(framework='tf'):

    if framework == 'th':
        # build the VGG16 network in Theano weight ordering mode
        backend.set_image_dim_ordering('th')
    else:
        # build the VGG16 network in Tensorflow weight ordering mode
        backend.set_image_dim_ordering('tf')
 
    img_width = 224
    img_height = 224

    model = Sequential()
    if framework == 'th':
        model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    else:
        model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
        
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    return model


weights_path = 'vgg16_weights.h5'
th_model = build_vgg16('th')

assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(th_model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    
    # If the weights are for the convolutional layer
    if th_model.layers[k].__class__.__name__ == 'Conv2D':
        # Uppack the weights to W and b
        kernel, bias = weights
        # Reshape the W
        kernel = np.transpose(kernel, (2, 3, 1, 0))
        # Load the W and b
        th_model.layers[k].set_weights([kernel, bias])
    else:
        th_model.layers[k].set_weights(weights)
    
f.close()
print('Model loaded.')

tf_model = build_vgg16('tf')

# transfer weights from th_model to tf_model
for th_layer, tf_layer in zip(th_model.layers, tf_model.layers):
    if th_layer.__class__.__name__ == 'Conv2D':
        kernel, bias = th_layer.get_weights()
        tf_layer.set_weights([kernel, bias])
    else:
        tf_layer.set_weights(th_layer.get_weights())


top_model = Sequential()
print(Flatten(input_shape=tf_model.output_shape[1:]))
top_model.add(Flatten(input_shape=tf_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(6, activation='sigmoid'))


tf_model.add(top_model)

# Freeze the first 25 layers
for layer in tf_model.layers[:25]:
    layer.trainable = False

# Initialize an SGD optimizer with a small learning rate
sgd_optimizer = optimizers.SGD(lr=1e-4, momentum=0.9)    

# Compile the transferred VGG model
tf_model.compile(loss = 'binary_crossentropy',
              optimizer = sgd_optimizer,
              metrics=['accuracy'])

# Define the tensorboard callback
vgg_tensorboard_log_dir = './logs/vgg16_fine_tuning/'
vgg_tensorboard_callback = TensorBoard(log_dir=vgg_tensorboard_log_dir)

# Define the model checkpoint call back to save the best model
vgg16_fine_tuning_model_path = './models/vgg16_fine_tuning_weights.best.hdf5'
vgg_checkpoint_callback = ModelCheckpoint(vgg16_fine_tuning_model_path, 
                                      monitor='val_acc', 
                                      save_best_only=True, 
                                      save_weights_only=True)
# Define the batch size
batch_size = 32


sc = SmallCNN()

sc.load_data()


# # Train the top layer for 5 epochs
# freezed_vgg_nb_epoch = 5

# # Start to train
# hist_vgg16_freezed = tf_model.fit(
#         train_generator,
#         steps_per_epoch = nb_train_samples // batch_size,
#         epochs = freezed_vgg_nb_epoch,
#         validation_data = validation_generator,
#         validation_steps = nb_validation_samples // batch_size,
#         callbacks=[vgg_tensorboard_callback, vgg_checkpoint_callback]
# )


