'''
-----------------------------------------------
File Name: predict-seg$
Description:
Author: Jing$
Date: 7/17/2021$
-----------------------------------------------
'''

import cv2 as cv
import numpy as np
import math
import os
from keras.models import load_model
#from tensorflow.keras.models import load_model # only for doubleunet

from data_seg import *
#from model_seg import *
from doubleu_net import *
import time
def Ellipse_Circumference(a,b):
    h = ((a / 2 - b / 2) ** 2) / ((a / 2 + b / 2) ** 2)
    len_ellipse = np.pi * (a / 2 + b / 2) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    return len_ellipse

def HC_calculate(img): # with post processing
    print(img)
    image = cv.imread(img)
    contour = cv.Canny(image, 80, 160)
    #cv.imshow("canny_output", contour)
    #cv.waitKey(0)
    contours, hierarchy = cv.findContours(contour, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    max_contour = [1]
    for i in range(len(contours)):
        if len(contours[i])>len(max_contour):
            max_contour = contours[i]
    # draw detected contour # -1:all the contours are drawn; Color; Thickness of lines
    cv.drawContours(image,max_contour, -1, (0, 255, 0), 4)
    # fitting ellipse, return center points, axis
    (cx, cy), (a, b), angle = cv.fitEllipse(max_contour)
    # generate ellipse # 0:start_angle; 360:end_angle; color; -1 filled ellipse; thickness, linetype
    newimg = np.zeros((540, 800, 3), np.uint8)  # 生成一个空灰度图像
    cv.ellipse(newimg, (np.int32(cx), np.int32(cy)), (np.int32(a / 2), np.int32(b / 2)),
               angle, 0, 360, (0, 0, 255), 0, 3, 0)
    save_path = 'fitted_results/'+img[-15:]
    len_ellipse = Ellipse_Circumference(a, b)
    cv.imwrite(save_path,newimg)
    cv.imshow("canny_output", newimg)
    #cv.waitKey(0)


    #print(len_ellipse)
    #print(b)


def HC_calculate_multi(img): # without post processing
    image = cv.imread(img)
    contour = cv.Canny(image, 80, 160)
    contours, hierarchy = cv.findContours(contour, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    points_contours = []
    lenth_ellipse = []
    peri_contours = []
    for i in range(len(contours)):
       points_contours.append(len(contours[i]))
       cv.drawContours(image,contours[i], -1, (0, 255, 0), 4)
       (cx, cy), (a, b), angle = cv.fitEllipse(contours[i])
       cv.ellipse(image, (np.int32(cx), np.int32(cy)), (np.int32(a / 2), np.int32(b / 2)),
               angle, 0, 360, (0, 0, 255), -1, 4, 0)
       len_ellipse = Ellipse_Circumference(a,b)
       peri_contours.append(cv.arcLength(contours[i], True))
       lenth_ellipse.append(len_ellipse)

    print(sum(lenth_ellipse))


#HC_calculate_multi()

# batch compute HC, compute time of post processing
path = 'HCdata/test/cv5-unet-original/'
img_list = os.listdir(path)
img_list.sort(key=lambda x: int(x[:-12]))  ##文件名按数字排序,屏蔽除数字以外的字符
print(img_list)
time_start = time.time()

for i in range(len(img_list)):
    img_name = path + img_list[i]
    HC_calculate_multi(img_name)# without post processing
    #HC_calculate(img_name) # with post processing
time_end = time.time()
print('time cost', time_end - time_start, 's')





from memory_profiler import profile

# H = 224 # 480 for psp net
# W = 224
# test_path = 'HCdata/test/image/'
# test_gt = 'HCdata/test/HC_ori_test.csv'
# X_aug, Y_aug, label_hc, \
# X_ori, Y_ori, ps_ori = load_data(test_path, test_path, test_path, test_path, test_gt, test_gt, H, W)
#
# custom_objects = {'dice_loss': sm.losses.dice_loss,
#                       'iou_score': sm.metrics.iou_score
#                       }
# save_path = 'HCdata/Challenge/test_result_version2/'
#     # Test stage
# model = load_model('segmodels/best_unet.h5', custom_objects)
# print('predicting')
# @profile #Decorator
# def pred():
#     time_start = time.time()
#     preds = model.predict(X_ori, batch_size=16)  # The output value is between (0,1) due to normalization.
#     time_end = time.time()
#     print('time cost', time_end - time_start, 's')
def predict():
    H = 224 # 480 for psp net
    W = 224
    test_path = 'HCdata/test/image/'
    test_gt = 'HCdata/test/HC_ori_test.csv'
    X_aug, Y_aug, label_hc, \
    X_ori, Y_ori, ps_ori = load_data(test_path, test_path, test_path, test_path, test_gt, test_gt, H, W)

    custom_objects = {'dice_loss': sm.losses.dice_loss,
                      'iou_score': sm.metrics.iou_score
                      }
    save_path = 'HCdata/test/cv5-unet-resnet-none/'
    # Test stage
    model = load_model('segmodels/u-net-resnet-none.h5', custom_objects)
    print('predicting')
    time_start = time.time()
    preds = model.predict(X_ori, batch_size=16)  # The output value is between (0,1) due to normalization.
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    save_data(save_path, preds, Y_ori)

#predict()
#if __name__ == "__main__":
#    pred()