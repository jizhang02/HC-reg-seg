'''
-----------------------------------------------
File Name: main_seg$
Description:
Author: Jing$
Date: 6/29/2021$
-----------------------------------------------
'''
import warnings
warnings.filterwarnings('ignore')  # ignore warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # only show error
from data_seg import *
from memory_usage import *
import segmentation_models as sm
import segmentation_models_xnet as smx

root = 'HCRegSeg/'
img_path_aug = 'HCdata/image_aug/'
img_path_ori = 'HCdata/image_ori/'
gt_path_aug = 'HCdata/gt_aug/'
gt_path_ori = 'HCdata/gt_ori/'
csv_aug = 'HCdata/HC_aug.csv'
csv_ori = 'HCdata/HC_ori.csv'
save_path = ['HCdata/predictions/predict_result_1/', 'HCdata/predictions/predict_result_2/',
             'HCdata/predictions/predict_result_3/', 'HCdata/predictions/predict_result_4/',
             'HCdata/predictions/predict_result_5/']
H = 224
W = 224
slice = 3
learning_rate = 1e-3
batch_size = 16
nb_epoch = 1
backbone = 'resnet50' #'vgg16','resnet50', 'efficientnetb2'
inputshape2D = (W, H, slice)
#X_aug, Y_aug, label_hc, \
#X_ori, Y_ori, ps_ori = load_data(img_path_aug, img_path_ori, gt_path_aug, gt_path_ori, csv_aug, csv_ori,H,W)

########## training and predicting###########

#gigabytes = get_model_memory_usage(batch_size,model)
#print("Memory usage:",gigabytes)
loss = sm.losses.dice_loss
#loss = sm.losses.binary_crossentropy
# fold_cross_valid(root, X_aug, Y_aug, X_ori, Y_ori, label_hc, ps_ori, inputshape2D, loss, save_path,
#                  nb_epoch=nb_epoch, batch_size=batch_size, learning_rate=learning_rate,
#                  best_filename='best_unet.h5')
