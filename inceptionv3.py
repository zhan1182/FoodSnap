from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint

from smallcnn import SmallCNN

# this could also be the output a different Keras model or layer
# input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

base_model = InceptionV3(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(6, activation='softmax')(x)


# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# Define the batch size
batch_size = 32
epochs = 5

sc = SmallCNN()
sc.load_data()

# Define the tensorboard callback
inc_tensorboard_log_dir = './logs/inc_fine_tuning/'
inc_tensorboard_callback = TensorBoard(log_dir=inc_tensorboard_log_dir)

# Define the model checkpoint call back to save the best model
inc_fine_tuning_model_path = './models/inc_fine_tuning_weights.best.hdf5'
inc_checkpoint_callback = ModelCheckpoint(inc_fine_tuning_model_path, 
                                      monitor='val_acc', 
                                      save_best_only=True, 
                                      save_weights_only=True)

model.fit(sc.X_train, sc.y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[inc_tensorboard_callback, inc_checkpoint_callback],
        validation_data=(sc.X_validation, sc.y_validation))

for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

sgd_optimizer = SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 10
model.fit(sc.X_train, sc.y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[inc_tensorboard_callback, inc_checkpoint_callback],
        validation_data=(sc.X_validation, sc.y_validation))
