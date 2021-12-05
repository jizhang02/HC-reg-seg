'''
-----------------------------------------------
File Name: model_seg$
Description:
Author: Jing$
Date: 6/29/2021$
-----------------------------------------------
'''
from keras.layers import *
from keras.models import *


def unet(input_size=(256, 256), filters=32, pretrained_weights=None, batch_norm=True):
    if not batch_norm:
        # input layer
        inputs = Input(input_size)

        # Level 1
        conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # Level 2
        conv2 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Level 3
        conv3 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # Level 4
        conv4 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # Level 5:bottom
        conv5 = Conv2D(filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        # Level 1' up
        up6 = Conv2D(filters * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        # Level 2' up
        up7 = Conv2D(filters * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        # Level 3' up
        up8 = Conv2D(filters * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        # Level 4' up
        up9 = Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        # final layer
        output = Conv2D(1, 1, activation='sigmoid')(conv9)
        model = Model(input=inputs, output=output)

    else:
        # input layer
        inputs = Input(input_size)

        # Level 1
        conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # Level 2
        conv2 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Level 3
        conv3 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # Level 4
        conv4 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # Level 5:bottom
        conv5 = Conv2D(filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        # Level 1' up
        up6 = Conv2D(filters * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)

        # Level 2' up
        up7 = Conv2D(filters * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)

        # Level 3' up
        up8 = Conv2D(filters * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)

        # Level 4' up
        up9 = Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)

        # final layer
        output = Conv2D(1, 1, activation='sigmoid')(conv9)
        model = Model(input=inputs, output=output)
        model.summary()

    if (pretrained_weights):
        print("pretrained weights: yes")
        model.load_weights(pretrained_weights,
                           by_name=False)  # True is useful for fine-tuning or transfer-learning models where some of the layers have changed.

    return model

unet((224,224,3))