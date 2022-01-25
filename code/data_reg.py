'''
-----------------------------------------------
File Name: data$
Description:
Author: Jing$
Date: 6/29/2021$
-----------------------------------------------
'''
from __future__ import division

import warnings

warnings.filterwarnings('ignore')  # ignore warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # only show error
import glob, os
import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from model_reg import *

def Info(images_path):
    imagelist = sorted(glob.glob(os.path.join(images_path, '*.png')))
    for i in (range(len(imagelist))):
        print(imagelist[i][10:])


def load_data(img_path_aug, img_path_ori, csv_aug, csv_ori, H, W, channels):
    df_aug = pd.read_csv(csv_aug)
    df_ori = pd.read_csv(csv_ori)

    filename_list_aug = df_aug['filename'].values
    filename_list_ori = df_ori['filename'].values

    img_aug = []
    label_aug = []
    pixel_aug = []
    pixel_size_aug = df_aug['pixel size(mm)'].values
    y_aug = df_aug['head circumference(mm)'].values
    #y_aug = df_aug['area(mm2)'].values

    img_ori = []
    label_ori = []
    pixel_ori = []
    pixel_size_ori = df_aug['pixel size(mm)'].values
    y_ori = df_aug['head circumference(mm)'].values
    #y_ori = df_aug['area(mm2)'].values


    for (i, f) in enumerate(filename_list_aug):
        img = cv.imread(img_path_aug + f)
        img = cv.resize(img, (W, H), interpolation=cv.INTER_AREA)
        img_norm = (img - np.mean(img)) / np.std(img)  # normalize
        img_aug.append(img_norm.reshape(H, W, channels))
        label_aug.append(y_aug[i])
        pixel_aug.append(pixel_size_aug[i])
    for (i, f) in enumerate(filename_list_ori):
        img = cv.imread(img_path_ori + f)
        img = cv.resize(img, (W, H), interpolation=cv.INTER_AREA)
        img_norm = (img - np.mean(img)) / np.std(img)  # normalize
        img_ori.append(img_norm.reshape(H, W, channels))
        label_ori.append(y_ori[i])
        pixel_ori.append(pixel_size_ori[i])
    print("load data successfully!")
    return np.asarray(img_aug), np.asarray(label_aug), np.asarray(pixel_aug), \
           np.asarray(img_ori), np.asarray(label_ori), np.asarray(pixel_ori)


def predictions(X, y, ps, maxHC, model):
    # make predictions on data
    preds = model.predict(X)

    # Compute MAE
    preds = preds.flatten()
    print("predicted value in mm:", preds * maxHC * ps)
    diff = (preds * maxHC - y * maxHC) * ps
    absDiff = np.abs(diff)

    # compute mae in mm
    mean = round(np.mean(absDiff), 2)
    std = round(np.std(absDiff), 2)

    # compute mae in pixel
    absDiff_pix = absDiff / ps
    mean_pix = round(np.mean(absDiff_pix), 2)
    std_pix = round(np.std(absDiff_pix), 2)

    # compute percentage mae
    pmae_mean = np.mean(np.abs((y - preds) / y)) * 100
    pmae_std = np.std(np.abs((y - preds) / y)) * 100

    print('\t  Mean abs diff in mm :', round(mean, 2), 'mm (+-)', round(std, 2), 'mm')
    print('\t  Mean abs diff in pixel :', round(mean_pix, 2), 'pixels (+-)', round(std_pix, 2))
    print('\t  pmae :', round(pmae_mean, 2), '% (+-)', round(pmae_std, 2))

    return mean, std, mean_pix, std_pix, pmae_mean, pmae_std

def predictions_area(X, y, ps, maxHC, model):
    # make predictions on data
    preds = model.predict(X)

    # Compute MAE
    preds = preds.flatten()
    print("predicted value in mm:", preds * maxHC * ps* ps)
    diff = (preds * maxHC - y * maxHC) * ps* ps
    absDiff = np.abs(diff)

    # compute mae in mm
    mean = round(np.mean(absDiff), 2)
    std = round(np.std(absDiff), 2)

    # compute mae in pixel
    absDiff_pix = absDiff / ps
    mean_pix = round(np.mean(absDiff_pix), 2)
    std_pix = round(np.std(absDiff_pix), 2)

    # compute percentage mae
    pmae_mean = np.mean(np.abs((y - preds) / y)) * 100
    pmae_std = np.std(np.abs((y - preds) / y)) * 100

    print('\t  Mean abs diff in mm2 :', round(mean, 2), 'mm2 (+-)', round(std, 2), 'mm2')
    print('\t  Mean abs diff in pixel :', round(mean_pix, 2), 'pixels (+-)', round(std_pix, 2))
    print('\t  pmae :', round(pmae_mean, 2), '% (+-)', round(pmae_std, 2))

    return mean, std, mean_pix, std_pix, pmae_mean, pmae_std

def fold_cross_valid(root, x_aug, y_aug, ps_aug,x_ori, y_ori,
                     ps_ori, inputshape2D, loss='mse', nb_epoch=50, batch_size=8, learning_rate=1e-3, best_filename='best.h5'):
    test_mae_HC_px = []
    test_mae_HC_mm = []
    test_pmae_HC = []

    # set early stopping criteria
    pat = 90  # this is the number of epochs with no improvment after which the training will stop
    early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)
    model_checkpoint = ModelCheckpoint(best_filename, monitor='val_loss',verbose=0, save_best_only=True)
    log_path = [
         'logs/log-1.csv',
         'logs/log-2.csv',
         'logs/log-3.csv',
         'logs/log-4.csv',
         'logs/log-5.csv'
    ]

    for i in range(0, 5):
        # use same split for all model
        idx_train = np.load( 'cv_array/train' + str(i) + '.npy', allow_pickle=True)
        idx_valid = np.load('cv_array/valid' + str(i) + '.npy', allow_pickle=True)
        idx_test = np.load( 'cv_array/test' + str(i) + '.npy', allow_pickle=True)
        idx_train2 = [j + 999 for j in idx_train]
        x_train = np.concatenate((x_ori[idx_train], x_aug[idx_train]), axis=0) #first 600 data augmentation
        y_train = np.concatenate((y_ori[idx_train], y_aug[idx_train]), axis=0)
        ps_train = np.concatenate((ps_ori[idx_train],ps_aug[idx_train]),axis = 0)

        x_train = np.concatenate((x_train, x_aug[idx_train2]), axis=0)  #second 600 data augmentation
        y_train = np.concatenate((y_train, y_aug[idx_train2]), axis=0)
        ps_train = np.concatenate((ps_train, ps_aug[idx_train2]), axis=0)

        x_valid = x_ori[idx_valid]
        y_valid = y_ori[idx_valid]
        ps_valid = ps_ori[idx_valid]

        x_test = x_ori[idx_test]
        y_test = y_ori[idx_test]
        ps_test = ps_ori[idx_test]

        y_train = y_train / ps_train # compute in pixels
        y_valid = y_valid / ps_valid
        y_test = y_test / ps_test

        maxHC = np.max(y_train)
        y_train = y_train / maxHC # normalization
        y_valid = y_valid / maxHC
        y_test = y_test / maxHC
        
        # with MSE
        model = vgg16(inputshape2D)
        #model = resnet50(inputshape2D)
        #model = efficientnets(inputshape2D)
        #model = densenet121(inputshape2D)
        #model = xception(inputshape2D)
        #model = mobilenet(inputshape2D)
        #model = inception(inputshape2D)
        #model.summary()

        # compilation of the network
        model.compile(loss=loss, optimizer=Adam(lr=learning_rate), metrics=["mae"])

        # fit
        model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                  callbacks=[early_stopping, model_checkpoint, CSVLogger(log_path[i], append=True)],
                  batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=2)

        model.load_weights(best_filename)

        # -------------- Test Predictions ----------------------------
        mean, std, mean_px, std_px, pmae_mean, pmae_std = predictions(x_test, y_test, ps_test, maxHC, model)
        test_mae_HC_px.append([mean_px, std_px])
        test_mae_HC_mm.append([mean, std])
        test_pmae_HC.append([pmae_mean, pmae_std])
    # end of for loop
    CV_mean_mae_HC_px, CV_std_mae_HC_px = np.mean(test_mae_HC_px, axis=0)
    CV_mean_mae_HC_mm, CV_std_mae_HC_mm = np.mean(test_mae_HC_mm, axis=0)
    CV_pmae_mean_HC, CV_pmae_std_HC = np.mean(test_pmae_HC, axis=0)

    print('-' * 60)
    print('5CV Mean abs diff HC in pixels :', round(CV_mean_mae_HC_px, 3), 'px (+-)', round(CV_std_mae_HC_px, 3), 'px')
    print('5CV Mean abs diff HCRV in millimeter :', round(CV_mean_mae_HC_mm, 3), 'mm (+-)', round(CV_std_mae_HC_mm, 3),
          'mm')
    print('5CV pmae of HC :', round(CV_pmae_mean_HC, 3), '% (+-)', round(CV_pmae_std_HC, 3))


def huber_loss(y_true, y_pred, clip_delta=1):
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)
