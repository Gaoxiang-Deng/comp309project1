# coding: utf-8

# In[1]:
# from IPython import get_ipython
# from keras import Sequential
# from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
#
# get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:

# input tensorflow and keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Convolution2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import regularizers

import numpy as np
import random
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import random

# Online code
# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


# Set seed
seed = 1990
random.seed(seed)
np.random.seed(seed)
tf.sets_random_seed(seed)

# In[4]:


# Set path

# online
data_path = 'C:/workspace/traindata/traindata'
# data_path = "drive/My Drive/Colab Notebooks/small_Train_data"

# local
# data_path = "Desktop/vu/2018 T2/COMP309/COMP309_2018_Project/Train_data"
# data_path = "Desktop/vu/2018 T2/COMP309/COMP309_2018_Project/small_Train_data"


# In[5]:


# load data
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# build the generator
# data preprocessing
train_datagen = ImageDataGenerator(

    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    validation_split=1 / 4,
    horizontal_flip=True,
    fill_mode='nearest',
    shear_range=0.2,
    zoom_range=0.2,

)

# normal genrator
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=1 / 4,

)

# Set batch size
batch_size = 30

train_data = train_datagen.flow_from_directory(
    data_path,
    target_size=(300, 300),
    subset='training',
    batch_size=batch_size
)

validation_data = train_datagen.flow_from_directory(
    data_path,
    target_size=(300, 300),
    subset='validation',
    batch_size=batch_size
)

train_size = train_data[0][0].shape[0]*len(train_data)
validation_size = validation_data[0][0].shape[0]*len(validation_data)
print(validation_size)


# In[6]:


# Set GPU allow growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# In[7]:

from keras import Sequential
#basic model
model = Sequential()
model. add(Flatten(input_shape=(300,300,3)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))
model.summary()


# In[8]:



# In[9]:


# input keras applications
from tensorflow.keras.applications import VGG16, InceptionResNetV2

vgg16 = VGG16(include_top=False, input_shape=(300, 300, 3), classes=3)
ICRNV2 = InceptionResNetV2(include_top=False,input_shape=(300, 300, 3), classes=3)

# Build the model (VCG-16)
model = Sequential()
for layer in vgg16.layers:
    model.add(layer)

# In[10]:


# add flatten layer
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax', activity_regularizer=regularizers.l1(0.001)))
model.summary()


# In[11]:


# Set optimizer
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=1e-4, rho=0.9, epsilon=None, decay=0.0)
adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=True)

# model compile
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[categorical_accuracy])

# In[12]:


# Call back
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

tbCallBack = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_images=False)
checkpoint = ModelCheckpoint(filepath='models/best_weights.hdf5', save_best_only='True', monitor='val_loss')
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

# In[ ]:


# Start training fit the model
model.fit_generator(

    train_data,
    steps_per_epoch=len(train_data),
    verbose=1,
    epochs=50,
    validation_data=validation_data,
    validation_steps=len(validation_data),
    callbacks=[tbCallBack, checkpoint])  # <-- add earlyStop to get the best model from call back

# In[ ]:


# Evaluate
model.load_weights('models/best_weights.hdf5')
loss_and_metrics = model.evaluate_generator(validation_data, len(validation_data))
print('Test loss:{}\nTest accuracy:{}'.format(loss_and_metrics[0], loss_and_metrics[1]))

# In[ ]:


# save model
import shutil

model.call
model.save('models/VGG16_sgd_prepor_FULL_loss 0.5362')

# In[ ]:


# Test model
data_path = "C:/workspace/traindata/testdata"
model_path = 'models/best_weights.hdf5'

test_datagen = ImageDataGenerator(

    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode='nearest',
    shear_range=0.2,
    zoom_range=0.2

)

test_data = test_datagen.flow_from_directory(
    data_path,
    target_size=(300, 300),
    batch_size=30
)

loss_and_metrics = model.evaluate_generator(test_data, steps=len(test_data))
print('Test loss:{}\nTest accuracy:{}'.format(loss_and_metrics[0], loss_and_metrics[1]))
