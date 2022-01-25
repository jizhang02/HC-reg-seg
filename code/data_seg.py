'''
-----------------------------------------------
File Name: data_seg$
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
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
import skimage.io as io
from skimage import img_as_ubyte
import os
from medpy.metric.binary import dc, hd, assd
from keras import backend as K
from keras.optimizers import Adam
#from tensorflow.keras.optimizers import Adam # only for doubleunet
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import segmentation_models as sm
from model_seg import *
from doubleu_net import *

def load_data(img_path_aug, img_path_ori, gt_path_aug, gt_path_ori, csv_aug, csv_ori, H, W):
    df_ori = pd.read_csv(csv_ori)
    df_aug = pd.read_csv(csv_aug)
    filename_list_ori = df_ori['filename'].values
    filename_list_aug = df_aug['filename'].values
    pixel_size_ori = df_ori['pixel size(mm)'].values
    hcpx_ori = df_ori['head circumference(px)'].values
    img_ori = []
    label_ori = []
    img_aug = []
    label_aug = []
    pixel_ori = []
    label_hc = []

    for (i, f) in enumerate(filename_list_ori):
        img = Image.open(img_path_ori + f).convert('RGB')  # 3 channels
        img = img.resize((H,W))
        img = np.array(img)
        img_norm = (img - np.mean(img)) / np.std(img)  # normalize
        img_ori.append(img_norm)
        pixel_ori.append(pixel_size_ori[i])
        label_hc.append(hcpx_ori[i])

        gt = Image.open(gt_path_ori + f).convert('L')
        gt = gt.resize((H,W))
        gt = np.array(gt)
        gt[gt > 0.5] = 1  # normalize
        gt[gt <= 0.5] = 0
        gt = gt[:, :, np.newaxis]
        label_ori.append(gt)

    for (i, f) in enumerate(filename_list_aug):
        img = Image.open(img_path_aug + f).convert('RGB')
        img = img.resize((H,W))
        img = np.array(img)
        img_norm = (img - np.mean(img)) / np.std(img)  # normalize
        # img = img_norm[:, :, np.newaxis]
        img_aug.append(img_norm)

        gt = Image.open(gt_path_aug + f).convert('L')
        gt = gt.resize((H,W))
        gt = np.array(gt)
        gt[gt > 0.5] = 1  # normalize
        gt[gt <= 0.5] = 0
        gt = gt[:, :, np.newaxis]
        label_aug.append(gt)

    print("load data successfully!")
    return np.asarray(img_aug, dtype=np.float64), np.asarray(label_aug), np.asarray(label_hc), \
           np.asarray(img_ori, dtype=np.float64), np.asarray(label_ori), np.asarray(pixel_ori)


def save_data(save_path, segment_results, label, shape=(800, 540)):
    image_resize = []
    label_resize = []
    for i, item in enumerate(segment_results):
        img = item[:, :, 0]
        if np.isnan(np.sum(img)):
            img = img[~np.isnan(img)]  # just remove nan elements from vector

        img[img > 0.5] = 1
        img[img <= 0.5] = 0
        img_resize = cv.resize(img, shape, interpolation=cv.INTER_AREA)
        image_resize.append(img_resize)
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img_as_ubyte(img_resize))
    for i, item in enumerate(label):
        gt_resize = cv.resize(item, shape, interpolation=cv.INTER_AREA)
        label_resize.append(gt_resize)

    print("save data successfully!")
    return np.asarray(image_resize), np.asarray(label_resize)


def Dice(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def Dice_score(gt, seg):
    dice = []
    for item in range(len(gt)):
        dice.append(dc(gt[item], seg[item]))
    print("The mean and std dice score is: ", '%.2f' % np.mean(dice), '%.2f' % np.std(dice))
    return np.mean(dice), np.std(dice)


def HausdorffDistance_score(gt, seg, pixelsize):
    hausdorff = []
    for item in range(len(gt)):
        if np.sum(seg[item]) > 0:  # If the structure is predicted on at least one pixel
            hausdorff.append(hd(seg[item], gt[item], voxelspacing=[pixelsize[item], pixelsize[item]]))
    print("The mean and std Hausdorff Distance is: ", '%.2f' % np.mean(hausdorff), '%.2f' % np.std(hausdorff))
    return np.mean(hausdorff), np.std(hausdorff)


def ASSD_score(gt, seg, pixelsize):
    ASSD = []
    for item in range(len(gt)):
        if np.sum(seg[item]) > 0:
            ASSD.append(assd(seg[item], gt[item], voxelspacing=[pixelsize[item], pixelsize[item]]))
    print("The mean and std ASSD is: ", '%.2f' % np.mean(ASSD), '%.2f' % np.std(ASSD))
    return np.mean(ASSD), np.std(ASSD)

def EllipseCircumference(a,b):
    # Ramanujan approximation II
    #  HC = pi*(a+b)*(1+3h/(10+sqrt(4-3*h))),h = (a-b)**2/(a+b)**2
    h = ((a / 2 - b / 2) ** 2) / ((a / 2 + b / 2) ** 2)
    perimeter_ellipse = np.pi * (a / 2 + b / 2) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    return perimeter_ellipse

def HC_calculate(pred):
    '''
    3 ways to calculate HC:
    1. the number of contour points;
    2. the length of contour;
    3. the length of fitted ellipse
    '''
    num_points_pp = []
    len_contour_pp = []
    len_ellipse_pp = []
    num_points = []
    len_contour = []
    len_ellipse = []

    for item in range(len(pred)):
        image = np.uint8(pred[item])
        image[image > 0.5] = 255
        image[image <= 0.5] = 0
        contour = cv.Canny(image, 80, 160)
        contours, hierarchy = cv.findContours(contour, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

        #print("performing with post processing")
        max_contour = []
        for i in range(len(contours)):
            if len(contours[i]) > len(max_contour):
                max_contour = contours[i]
        if len(max_contour) != 0:
            perimeter_contour = cv.arcLength(max_contour, True)  # para2:indicating whether the curve is closed or not
            # fitting ellipse, return center points, axis
            (cx, cy), (a, b), angle = cv.fitEllipse(max_contour)
            perimeter_ellipse = EllipseCircumference(a,b)
        else:
            perimeter_contour = 0
            perimeter_ellipse = 0

        num_points_pp.append(len(max_contour))
        len_contour_pp.append(perimeter_contour)
        len_ellipse_pp.append(perimeter_ellipse)

        #print("performing without post processing")
        if len(contours) !=0:
            num_points_unit=0
            len_contour_unit=0
            len_ellipse_unit=0
            for i in range(len(contours)):
                num_points_unit +=len(contours[i])
                len_contour_unit +=cv.arcLength(contours[i], True)
                if len(contours[i])>5:#There should be at least 5 points to fit the ellipse in function 'cv::fitEllipse'
                    (cx, cy), (a, b), angle = cv.fitEllipse(contours[i])
                    len_ellipse_unit +=EllipseCircumference(a,b)
        else:
            num_points_unit = 0
            len_contour_unit = 0
            len_ellipse_unit = 0

        num_points.append(num_points_unit)
        len_contour.append(len_contour_unit)
        len_ellipse.append(len_ellipse_unit)


    return np.asarray(num_points), np.asarray(len_contour), np.asarray(len_ellipse),\
           np.asarray(num_points_pp), np.asarray(len_contour_pp), np.asarray(len_ellipse_pp)


def predictions(x_text, y_test, label_hc_px, pixelsize, model, save_path):
    score = model.evaluate(x_text, y_test, verbose=2)
    print("Loss,iou sore:", '%.2f' % score[0], '%.2f' % score[1])

    results = model.predict(x_text)  # return probability
    pred, y_test = save_data(save_path, results, y_test)
    # segmentation analysis
    mean_dice, std_dice = Dice_score(y_test, pred)
    mean_hd, std_hd = HausdorffDistance_score(y_test, pred, pixelsize)
    mean_assd, std_assd = ASSD_score(y_test, pred, pixelsize)
    # HC analysis
    HC_pred_points, HC_pred_contour, HC_pred_ellipse,\
    HC_pred_points_pp, HC_pred_contour_pp, HC_pred_ellipse_pp= HC_calculate(pred)
    print("predicted value in mm:", HC_pred_ellipse_pp * pixelsize)
    print("predicted value in mm wo pp:", HC_pred_ellipse * pixelsize)

    absDiff_points = np.abs((HC_pred_points - label_hc_px) * pixelsize)
    absDiff_contour = np.abs((HC_pred_contour - label_hc_px) * pixelsize)
    absDiff_ellipse = np.abs((HC_pred_ellipse - label_hc_px) * pixelsize)

    absDiff_points_pp = np.abs((HC_pred_points_pp - label_hc_px) * pixelsize)
    absDiff_contour_pp = np.abs((HC_pred_contour_pp - label_hc_px) * pixelsize)
    absDiff_ellipse_pp = np.abs((HC_pred_ellipse_pp - label_hc_px) * pixelsize)

    mean_mae_points = round(np.mean(absDiff_points), 2)  # compute mae in mm
    mean_mae_contour = round(np.mean(absDiff_contour), 2)  # compute mae in mm
    mean_mae_ellipse = round(np.mean(absDiff_ellipse), 2)  # compute mae in mm

    mean_mae_points_pp = round(np.mean(absDiff_points_pp), 2)  # compute mae in mm
    mean_mae_contour_pp = round(np.mean(absDiff_contour_pp), 2)  # compute mae in mm
    mean_mae_ellipse_pp = round(np.mean(absDiff_ellipse_pp), 2)  # compute mae in mm

    std_mae_points = round(np.std(absDiff_points), 2)
    std_mae_contour = round(np.std(absDiff_contour), 2)
    std_mae_ellipse = round(np.std(absDiff_ellipse), 2)

    std_mae_points_pp = round(np.std(absDiff_points_pp), 2)
    std_mae_contour_pp = round(np.std(absDiff_contour_pp), 2)
    std_mae_ellipse_pp = round(np.std(absDiff_ellipse_pp), 2)

    mean_mae_px_points = round(np.mean(absDiff_points / pixelsize), 2)  # compute mae in pixel
    mean_mae_px_contour = round(np.mean(absDiff_contour / pixelsize), 2)  # compute mae in pixel
    mean_mae_px_ellipse = round(np.mean(absDiff_ellipse / pixelsize), 2)  # compute mae in pixel

    mean_mae_px_points_pp = round(np.mean(absDiff_points_pp / pixelsize), 2)  # compute mae in pixel
    mean_mae_px_contour_pp = round(np.mean(absDiff_contour_pp / pixelsize), 2)  # compute mae in pixel
    mean_mae_px_ellipse_pp = round(np.mean(absDiff_ellipse_pp / pixelsize), 2)  # compute mae in pixel

    std_mae_px_points = round(np.std(absDiff_points / pixelsize), 2)
    std_mae_px_contour = round(np.std(absDiff_contour / pixelsize), 2)
    std_mae_px_ellipse = round(np.std(absDiff_ellipse / pixelsize), 2)

    std_mae_px_points_pp = round(np.std(absDiff_points_pp / pixelsize), 2)
    std_mae_px_contour_pp = round(np.std(absDiff_contour_pp / pixelsize), 2)
    std_mae_px_ellipse_pp = round(np.std(absDiff_ellipse_pp / pixelsize), 2)

    mean_pmae_points = np.mean(np.abs((label_hc_px - HC_pred_points) / label_hc_px)) * 100  # compute percentage mae
    mean_pmae_contour = np.mean(np.abs((label_hc_px - HC_pred_contour) / label_hc_px)) * 100  # compute percentage mae
    mean_pmae_ellipse = np.mean(np.abs((label_hc_px - HC_pred_ellipse) / label_hc_px)) * 100  # compute percentage mae

    mean_pmae_points_pp = np.mean(np.abs((label_hc_px - HC_pred_points_pp) / label_hc_px)) * 100  # compute percentage mae
    mean_pmae_contour_pp = np.mean(np.abs((label_hc_px - HC_pred_contour_pp) / label_hc_px)) * 100  # compute percentage mae
    mean_pmae_ellipse_pp = np.mean(np.abs((label_hc_px - HC_pred_ellipse_pp) / label_hc_px)) * 100  # compute percentage mae

    std_pmae_points = np.std(np.abs((label_hc_px - HC_pred_points) / label_hc_px)) * 100
    std_pmae_contour = np.std(np.abs((label_hc_px - HC_pred_contour) / label_hc_px)) * 100
    std_pmae_ellipse = np.std(np.abs((label_hc_px - HC_pred_ellipse) / label_hc_px)) * 100

    std_pmae_points_pp = np.std(np.abs((label_hc_px - HC_pred_points_pp) / label_hc_px)) * 100
    std_pmae_contour_pp = np.std(np.abs((label_hc_px - HC_pred_contour_pp) / label_hc_px)) * 100
    std_pmae_ellipse_pp = np.std(np.abs((label_hc_px - HC_pred_ellipse_pp) / label_hc_px)) * 100

    print('\t  HC_mae in mm(points) w/o pp:', round(mean_mae_points, 2), 'mm (+-)', round(std_mae_points, 2), 'mm')
    print('\t  HC_mae in mm(contour)w/o pp:', round(mean_mae_contour, 2), 'mm (+-)', round(std_mae_contour, 2), 'mm')
    print('\t  HC_mae in mm(ellipse)w/o pp:', round(mean_mae_ellipse, 2), 'mm (+-)', round(std_mae_ellipse, 2), 'mm')
    print('\t  HC_mae in mm(points)  w pp:', round(mean_mae_points_pp, 2), 'mm (+-)', round(std_mae_points_pp, 2), 'mm')
    print('\t  HC_mae in mm(contour) w pp:', round(mean_mae_contour_pp, 2), 'mm (+-)', round(std_mae_contour_pp, 2), 'mm')
    print('\t  HC_mae in mm(ellipse) w pp:', round(mean_mae_ellipse_pp, 2), 'mm (+-)', round(std_mae_ellipse_pp, 2), 'mm')

    print('\t  HC_mae in px(points) w/o pp:', round(mean_mae_px_points, 2), 'pixels (+-)', round(std_mae_px_points, 2))
    print('\t  HC_mae in px(contour)w/o pp:', round(mean_mae_px_contour, 2), 'pixels (+-)', round(std_mae_px_contour, 2))
    print('\t  HC_mae in px(ellipse)w/o pp:', round(mean_mae_px_ellipse, 2), 'pixels (+-)', round(std_mae_px_ellipse, 2))
    print('\t  HC_mae in px(points)  w pp:', round(mean_mae_px_points_pp, 2), 'pixels (+-)', round(std_mae_px_points_pp, 2))
    print('\t  HC_mae in px(contour) w pp:', round(mean_mae_px_contour_pp, 2), 'pixels (+-)', round(std_mae_px_contour_pp, 2))
    print('\t  HC_mae in px(ellipse) w pp:', round(mean_mae_px_ellipse_pp, 2), 'pixels (+-)', round(std_mae_px_ellipse_pp, 2))

    print('\t  pmae(points) w/o pp:', round(mean_pmae_points, 2), '% (+-)', round(std_pmae_points, 2))
    print('\t  pmae(contour)w/o pp:', round(mean_pmae_contour, 2), '% (+-)', round(std_pmae_contour, 2))
    print('\t  pmae(ellipse)w/o pp:', round(mean_pmae_ellipse, 2), '% (+-)', round(std_pmae_ellipse, 2))
    print('\t  pmae(points)  w pp:', round(mean_pmae_points_pp, 2), '% (+-)', round(std_pmae_points_pp, 2))
    print('\t  pmae(contour) w pp:', round(mean_pmae_contour_pp, 2), '% (+-)', round(std_pmae_contour_pp, 2))
    print('\t  pmae(ellipse) w pp:', round(mean_pmae_ellipse_pp, 2), '% (+-)', round(std_pmae_ellipse_pp, 2))

    return mean_dice, std_dice, mean_hd, std_hd, mean_assd, std_assd, \
           mean_mae_points, mean_mae_contour, mean_mae_ellipse, \
           std_mae_points, std_mae_contour, std_mae_ellipse, \
           mean_mae_px_points, mean_mae_px_contour, mean_mae_px_ellipse, \
           std_mae_px_points, std_mae_px_contour, std_mae_px_ellipse, \
           mean_pmae_points, mean_pmae_contour, mean_pmae_ellipse, \
           std_pmae_points, std_pmae_contour, std_pmae_ellipse,\
           mean_mae_points_pp, mean_mae_contour_pp, mean_mae_ellipse_pp, \
           std_mae_points_pp, std_mae_contour_pp, std_mae_ellipse_pp, \
           mean_mae_px_points_pp, mean_mae_px_contour_pp, mean_mae_px_ellipse_pp, \
           std_mae_px_points_pp, std_mae_px_contour_pp, std_mae_px_ellipse_pp, \
           mean_pmae_points_pp, mean_pmae_contour_pp, mean_pmae_ellipse_pp, \
           std_pmae_points_pp, std_pmae_contour_pp, std_pmae_ellipse_pp


def fold_cross_valid(root,x_aug, y_aug, x_ori, y_ori, label_hc, ps_ori, inputshape2D, loss, save_path,
                     nb_epoch=50, batch_size=8, learning_rate=1e-3, best_filename='best.h5'):
    test_dice = []
    test_hd = []
    test_assd = []

    test_mae_HC_px_point = []
    test_mae_HC_px_contour = []
    test_mae_HC_px_ellipse = []
    test_mae_HC_px_point_pp = []
    test_mae_HC_px_contour_pp = []
    test_mae_HC_px_ellipse_pp = []

    test_mae_HC_mm_point = []
    test_mae_HC_mm_contour = []
    test_mae_HC_mm_ellipse = []
    test_mae_HC_mm_point_pp = []
    test_mae_HC_mm_contour_pp = []
    test_mae_HC_mm_ellipse_pp = []

    test_pmae_HC_point = []
    test_pmae_HC_contour = []
    test_pmae_HC_ellipse = []
    test_pmae_HC_point_pp = []
    test_pmae_HC_contour_pp = []
    test_pmae_HC_ellipse_pp = []



    early_stopping = EarlyStopping(monitor='val_loss', patience=90, verbose=1)
    model_checkpoint = ModelCheckpoint(best_filename, verbose=0, save_best_only=True)
    log_path = [
        'logs/log-1.csv',
        'logs/log-2.csv',
        'logs/log-3.csv',
        'logs/log-4.csv',
        'logs/log-5.csv'
    ]

    for i in range(0, 5):
        idx_train = np.load('cv_array/train' + str(i) + '.npy', allow_pickle=True)
        idx_test = np.load('cv_array/test' + str(i) + '.npy', allow_pickle=True)
        idx_valid = np.load('cv_array/valid' + str(i) + '.npy', allow_pickle=True)

        # idx_train2 = [i + 999 for i in idx_train]
        # x_train = np.concatenate((x_ori[idx_train], x_aug[idx_train]), axis=0)  # first 600 data augmentation
        # y_train = np.concatenate((y_ori[idx_train], y_aug[idx_train]), axis=0)
        #
        # x_train = np.concatenate((x_train, x_aug[idx_train2]), axis=0)  # second 600 data augmentation
        # y_train = np.concatenate((y_train, y_aug[idx_train2]), axis=0)

        x_train = x_ori[idx_train]
        y_train = y_ori[idx_train]

        x_valid = x_ori[idx_valid]
        y_valid = y_ori[idx_valid]

        x_test = x_ori[idx_test]
        y_test = y_ori[idx_test]
        ps_test = ps_ori[idx_test]
        hc_test = label_hc[idx_test]
        metric = sm.metrics.iou_score
        #model = unet(inputshape2D)
        model = sm.Unet(backbone,input_shape=inputshape2D,encoder_weights = None)
        #model = smx.Xnet(backbone,input_shape=inputshape2D)
        #model = sm.FPN(backbone,input_shape=inputshape2D) #
        #model = sm.Linknet(backbone,input_shape=inputshape2D)
        #model = sm.PSPNet(backbone,input_shape=inputshape2D) # require 480,480!!!
        #model = doubleUNet(inputshape2D) # require two keras frames!!!
        #model = sm.PSPNet(backbone, encoder_weights = 'imagenet', classes = 1,
        #encoder_freeze=False, activation='sigmoid', downsample_factor=16, input_shape=(480,480,3),
        #psp_conv_filters=1024, psp_pooling_type='avg')
        #model.summary()
        model.compile(loss=loss, optimizer=Adam(lr=learning_rate), metrics=[metric])

        model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                  callbacks=[early_stopping, model_checkpoint, CSVLogger(log_path[i], append=True)],
                  batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1)

        model.load_weights(best_filename)

        # -------------- Test Predictions ----------------------------
        mean_dice, std_dice, mean_hd, std_hd, mean_assd, std_assd, \
        mean_mae_points, mean_mae_contour, mean_mae_ellipse, \
        std_mae_points, std_mae_contour, std_mae_ellipse, \
        mean_mae_px_points, mean_mae_px_contour, mean_mae_px_ellipse, \
        std_mae_px_points, std_mae_px_contour, std_mae_px_ellipse, \
        mean_pmae_points, mean_pmae_contour, mean_pmae_ellipse, \
        std_pmae_points, std_pmae_contour, std_pmae_ellipse, \
        mean_mae_points_pp, mean_mae_contour_pp, mean_mae_ellipse_pp, \
        std_mae_points_pp, std_mae_contour_pp, std_mae_ellipse_pp, \
        mean_mae_px_points_pp, mean_mae_px_contour_pp, mean_mae_px_ellipse_pp, \
        std_mae_px_points_pp, std_mae_px_contour_pp, std_mae_px_ellipse_pp, \
        mean_pmae_points_pp, mean_pmae_contour_pp, mean_pmae_ellipse_pp, \
        std_pmae_points_pp, std_pmae_contour_pp, std_pmae_ellipse_pp = \
        predictions(x_test, y_test, hc_test, ps_test, model, save_path[i])

        test_dice.append([mean_dice, std_dice])
        test_hd.append([mean_hd, std_hd])
        test_assd.append([mean_assd, std_assd])

        test_mae_HC_px_point.append([mean_mae_px_points, std_mae_px_points])
        test_mae_HC_px_contour.append([mean_mae_px_contour, std_mae_px_contour])
        test_mae_HC_px_ellipse.append([mean_mae_px_ellipse, std_mae_px_ellipse])
        test_mae_HC_px_point_pp.append([mean_mae_px_points_pp, std_mae_px_points_pp])
        test_mae_HC_px_contour_pp.append([mean_mae_px_contour_pp, std_mae_px_contour_pp])
        test_mae_HC_px_ellipse_pp.append([mean_mae_px_ellipse_pp, std_mae_px_ellipse_pp])

        test_mae_HC_mm_point.append([mean_mae_points, std_mae_points])
        test_mae_HC_mm_contour.append([mean_mae_contour, std_mae_contour])
        test_mae_HC_mm_ellipse.append([mean_mae_ellipse, std_mae_ellipse])
        test_mae_HC_mm_point_pp.append([mean_mae_points_pp, std_mae_points_pp])
        test_mae_HC_mm_contour_pp.append([mean_mae_contour_pp, std_mae_contour_pp])
        test_mae_HC_mm_ellipse_pp.append([mean_mae_ellipse_pp, std_mae_ellipse_pp])

        test_pmae_HC_point.append([mean_pmae_points, std_pmae_points])
        test_pmae_HC_contour.append([mean_pmae_contour, std_pmae_contour])
        test_pmae_HC_ellipse.append([mean_pmae_ellipse, std_pmae_ellipse])
        test_pmae_HC_point_pp.append([mean_pmae_points_pp, std_pmae_points_pp])
        test_pmae_HC_contour_pp.append([mean_pmae_contour_pp, std_pmae_contour_pp])
        test_pmae_HC_ellipse_pp.append([mean_pmae_ellipse_pp, std_pmae_ellipse_pp])

    # end of for loop

    CV_mean_dice, CV_std_dice = np.mean(test_dice, axis=0)
    CV_mean_hd, CV_std_hd = np.mean(test_hd, axis=0)
    CV_mean_assd, CV_std_assd = np.mean(test_assd, axis=0)

    CV_mean_mae_HC_px_point, CV_std_mae_HC_px_point = np.mean(test_mae_HC_px_point, axis=0)
    CV_mean_mae_HC_px_contour, CV_std_mae_HC_px_contour = np.mean(test_mae_HC_px_contour, axis=0)
    CV_mean_mae_HC_px_ellipse, CV_std_mae_HC_px_ellipse = np.mean(test_mae_HC_px_ellipse, axis=0)
    CV_mean_mae_HC_px_point_pp, CV_std_mae_HC_px_point_pp = np.mean(test_mae_HC_px_point_pp, axis=0)
    CV_mean_mae_HC_px_contour_pp, CV_std_mae_HC_px_contour_pp = np.mean(test_mae_HC_px_contour_pp, axis=0)
    CV_mean_mae_HC_px_ellipse_pp, CV_std_mae_HC_px_ellipse_pp = np.mean(test_mae_HC_px_ellipse_pp, axis=0)

    CV_mean_mae_HC_mm_point, CV_std_mae_HC_mm_point = np.mean(test_mae_HC_mm_point, axis=0)
    CV_mean_mae_HC_mm_contour, CV_std_mae_HC_mm_contour = np.mean(test_mae_HC_mm_contour, axis=0)
    CV_mean_mae_HC_mm_ellipse, CV_std_mae_HC_mm_ellipse = np.mean(test_mae_HC_mm_ellipse, axis=0)
    CV_mean_mae_HC_mm_point_pp, CV_std_mae_HC_mm_point_pp = np.mean(test_mae_HC_mm_point_pp, axis=0)
    CV_mean_mae_HC_mm_contour_pp, CV_std_mae_HC_mm_contour_pp = np.mean(test_mae_HC_mm_contour_pp, axis=0)
    CV_mean_mae_HC_mm_ellipse_pp, CV_std_mae_HC_mm_ellipse_pp = np.mean(test_mae_HC_mm_ellipse_pp, axis=0)

    CV_pmae_mean_HC_point, CV_pmae_std_HC_point = np.mean(test_pmae_HC_point, axis=0)
    CV_pmae_mean_HC_contour, CV_pmae_std_HC_contour = np.mean(test_pmae_HC_contour, axis=0)
    CV_pmae_mean_HC_ellipse, CV_pmae_std_HC_ellipse = np.mean(test_pmae_HC_ellipse, axis=0)
    CV_pmae_mean_HC_point_pp, CV_pmae_std_HC_point_pp = np.mean(test_pmae_HC_point_pp, axis=0)
    CV_pmae_mean_HC_contour_pp, CV_pmae_std_HC_contour_pp = np.mean(test_pmae_HC_contour_pp, axis=0)
    CV_pmae_mean_HC_ellipse_pp, CV_pmae_std_HC_ellipse_pp = np.mean(test_pmae_HC_ellipse_pp, axis=0)

    print('-' * 60)
    print('5CV Mean dice score :', round(CV_mean_dice, 3), '(+-)', round(CV_std_dice, 3))
    print('5CV Mean hd score :', round(CV_mean_hd, 3), 'mm (+-)', round(CV_std_hd, 3), 'mm')
    print('5CV Mean assd :', round(CV_mean_assd, 3), 'mm (+-)', round(CV_std_assd, 3), 'mm')
    print('-' * 60)
    print('5CV mae HC(px) in points  w/o pp:', round(CV_mean_mae_HC_px_point, 3), 'px (+-)', round(CV_std_mae_HC_px_point, 3), 'px')
    print('5CV mae HC(px) in contour w/o pp:', round(CV_mean_mae_HC_px_contour, 3), 'px (+-)', round(CV_std_mae_HC_px_contour, 3), 'px')
    print('5CV mae HC(px) in ellipse w/o pp:', round(CV_mean_mae_HC_px_ellipse, 3), 'px (+-)', round(CV_std_mae_HC_px_ellipse, 3), 'px')
    print('5CV mae HC(px) in points w pp:', round(CV_mean_mae_HC_px_point_pp, 3), 'px (+-)',
          round(CV_std_mae_HC_px_point_pp, 3), 'px')
    print('5CV mae HC(px) in contour w pp:', round(CV_mean_mae_HC_px_contour_pp, 3), 'px (+-)',
          round(CV_std_mae_HC_px_contour_pp, 3), 'px')
    print('5CV mae HC(px) in ellipse w pp:', round(CV_mean_mae_HC_px_ellipse_pp, 3), 'px (+-)',
          round(CV_std_mae_HC_px_ellipse_pp, 3), 'px')

    print('5CV mae HC(mm) in points  w/o pp:', round(CV_mean_mae_HC_mm_point, 3), 'mm (+-)', round(CV_std_mae_HC_mm_point, 3), 'mm')
    print('5CV mae HC(mm) in contour w/o pp:', round(CV_mean_mae_HC_mm_contour, 3), 'mm (+-)', round(CV_std_mae_HC_mm_contour, 3), 'mm')
    print('5CV mae HC(mm) in ellipse w/o pp:', round(CV_mean_mae_HC_mm_ellipse, 3), 'mm (+-)', round(CV_std_mae_HC_mm_ellipse, 3), 'mm')
    print('5CV mae HC(mm) in points w pp:', round(CV_mean_mae_HC_mm_point_pp, 3), 'mm (+-)',
          round(CV_std_mae_HC_mm_point_pp, 3), 'mm')
    print('5CV mae HC(mm) in contour w pp:', round(CV_mean_mae_HC_mm_contour_pp, 3), 'mm (+-)',
          round(CV_std_mae_HC_mm_contour_pp, 3), 'mm')
    print('5CV mae HC(mm) in ellipse w pp:', round(CV_mean_mae_HC_mm_ellipse_pp, 3), 'mm (+-)',
          round(CV_std_mae_HC_mm_ellipse_pp, 3), 'mm')

    print('5CV pmae HC in points  w/o pp:', round(CV_pmae_mean_HC_point, 3), '% (+-)', round(CV_pmae_std_HC_point, 3))
    print('5CV pmae HC in contour w/o pp:', round(CV_pmae_mean_HC_contour, 3), '% (+-)', round(CV_pmae_std_HC_contour, 3))
    print('5CV pmae HC in ellipse w/o pp:', round(CV_pmae_mean_HC_ellipse, 3), '% (+-)', round(CV_pmae_std_HC_ellipse, 3))
    print('5CV pmae HC in points  w pp:', round(CV_pmae_mean_HC_point_pp, 3), '% (+-)', round(CV_pmae_std_HC_point_pp, 3))
    print('5CV pmae HC in contour w pp:', round(CV_pmae_mean_HC_contour_pp, 3), '% (+-)', round(CV_pmae_std_HC_contour_pp, 3))
    print('5CV pmae HC in ellipse w pp:', round(CV_pmae_mean_HC_ellipse_pp, 3), '% (+-)', round(CV_pmae_std_HC_ellipse_pp, 3))



def Dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def Kappa_loss(y_true, y_pred, N=224 * 224):
    Gi = K.flatten(y_true)
    Pi = K.flatten(y_pred)
    numerator = 2 * K.sum(Pi * Gi) - K.sum(Pi) * K.sum(Gi) / N
    denominator = K.sum(Pi * Pi) + K.sum(Gi * Gi) - 2 * K.sum(Pi * Gi) / N
    Kappa = 1 - numerator / denominator
    return Kappa
