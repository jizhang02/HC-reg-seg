'''
-----------------------------------------------
File Name: model_reg$
Description:
Author: Jing$
Date: 6/29/2021$
-----------------------------------------------
'''
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout, Dense
from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D
from keras.layers import Concatenate as concatenate
from keras.layers import Input, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from efficientnet.keras import EfficientNetB0
from efficientnet.keras import EfficientNetB1
from efficientnet.keras import EfficientNetB2
from efficientnet.keras import EfficientNetB3
from efficientnet.keras import EfficientNetB4
from efficientnet.keras import EfficientNetB5
from efficientnet.keras import EfficientNetB6
from efficientnet.keras import EfficientNetB7
from keras.layers import GlobalMaxPooling2D, ZeroPadding2D,AveragePooling2D,GlobalAveragePooling2D
from keras import regularizers
from keras.optimizers import Adam
def efficientnets(input_shape):
    print('The model is EfficientNet.')
    base = EfficientNetB2(weights='imagenet',input_shape=input_shape,include_top=False)
    x = base.output
    x = Flatten()(x) # Flatten()(x) | GlobalAveragePooling2D | GlobalMaxPooling2D
    x = Dropout(0.5)(x)
    x = Dense(units=1000, activation="relu")(x)
    x = Dense(units=32, activation="relu")(x)
    x = Dense(units=16, activation="relu")(x)
    pred = Dense(1, activation='linear',name='HC')(x)
    model = Model(inputs=base.inputs, outputs=pred)
    #model.summary()
    return model

def create_cnn(inputShape=(128, 128, 1), filters=(16, 32, 64), regress='hc'):
    # initialize the input shape and channel dimension, assuming TensorFlow/channels-last ordering
    # inputShape = (height, width, depth)
    chanDim = -1

    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Dense(32)(x)
    x = Activation("relu")(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Dense(16)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress == 'hc':
        x = Dense(1, activation="linear")(x)  # 1 channel for the head circumference
    if regress == 'ellipse':
        x = Dense(5, activation="linear")(x)  # 5 channels for the 5 parameters of ellipse

    # construct the CNN and return
    model = Model(inputs, x)
    return model

def vgg16_fromscratch(inputShape=(128, 128, 1)):
    model = Sequential()
    model.add(Conv2D(input_shape=inputShape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=1, activation="linear"))
    return model

def vgg16(input_shape, top='flatten'):
    if top not in ('flatten', 'avg', 'max'):
        raise ValueError('unexpected top layer type: %s' % top)
    print('The model is VGG16.')
    #base = VGG16(input_shape=input_shape, include_top=False)#train from scratch
    base = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)
    x = base.output
    x = GlobalAveragePooling2D()(x) # Flatten | GlobalAveragePooling2D | GlobalMaxPooling2D
    x = Dropout(0.7)(x)
    x = Dense(512,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
    x = Dense(256, kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l1(0.01))(x)
    x = Dense(128, kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l1(0.01))(x)
    x = Dense(64, kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l1(0.01))(x)
    x = Dense(32, kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l1(0.01))(x)
    pred = Dense(1, activation='linear')(x)
    model = Model(inputs=base.inputs, outputs=pred)

    '''
    Classif 
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base.input, outputs = predictions)
    '''
    return model


def resnet50(input_shape, top='flatten'):
    if top not in ('flatten', 'avg', 'max'):
        raise ValueError('unexpected top layer type: %s' % top)
    print('The model is ResNet50.')
    #base = ResNet50(input_shape=input_shape, include_top=False)#train from scratch
    base = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)#pretrained with ImageNet

    x = base.output
    x = Flatten()(x) # Flatten | GlobalAveragePooling2D | GlobalMaxPooling2D
    x = Dropout(0.3)(x)
    pred = Dense(1, activation='linear')(x)
    model = Model(inputs=base.inputs, outputs=pred)

    '''
    Classif 
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base.input, outputs = predictions)
    '''
    return model

def densenet121(input_shape, top='flatten'):
    if top not in ('flatten', 'avg', 'max'):
        raise ValueError('unexpected top layer type: %s' % top)
    print('The model is DenseNet121.')
    base = DenseNet121(weights='imagenet', input_shape=input_shape, include_top=False)
    x = base.output
    x = Flatten()(x) # Flatten | GlobalAveragePooling2D | GlobalMaxPooling2D
    x = Dropout(0.5)(x)
    pred = Dense(1, activation='linear')(x)
    model = Model(inputs=base.inputs, outputs=pred)

    '''
    Classif 
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base.input, outputs = predictions)
    '''
    return model

def xception(input_shape, top='flatten'):
    if top not in ('flatten', 'avg', 'max'):
        raise ValueError('unexpected top layer type: %s' % top)
    print('The model is Xception.')
    base = Xception(weights='imagenet', input_shape=input_shape, include_top=False)
    x = base.output
    x = Flatten()(x) # Flatten | GlobalAveragePooling2D | GlobalMaxPooling2D
    x = Dropout(0.5)(x)
    pred = Dense(1, activation='linear')(x)
    model = Model(inputs=base.inputs, outputs=pred)

    '''
    Classif 
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base.input, outputs = predictions)
    '''
    return model

def mobilenet(input_shape, top='flatten'):
    if top not in ('flatten', 'avg', 'max'):
        raise ValueError('unexpected top layer type: %s' % top)
    print('The model is MobileNet.')
    base = MobileNet(weights='imagenet', input_shape=input_shape, include_top=False)
    x = base.output
    x = Flatten()(x) # Flatten | GlobalAveragePooling2D | GlobalMaxPooling2D
    x = Dropout(0.5)(x)
    pred = Dense(1, activation='linear')(x)
    model = Model(inputs=base.inputs, outputs=pred)

    '''
    Classif 
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base.input, outputs = predictions)
    '''
    return model
def inception(input_shape, top='flatten'):
    if top not in ('flatten', 'avg', 'max'):
        raise ValueError('unexpected top layer type: %s' % top)
    print('The model is InceptionV3.')
    base = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False)
    x = base.output
    x = Flatten()(x) # Flatten | GlobalAveragePooling2D | GlobalMaxPooling2D
    x = Dropout(0.5)(x)
    pred = Dense(1, activation='linear')(x)
    model = Model(inputs=base.inputs, outputs=pred)

    '''
    Classif 
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base.input, outputs = predictions)
    '''
    return model
