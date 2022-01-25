'''
-----------------------------------------------
File Name: main_reg$
Description:
Author: Jing$
Date: 6/29/2021$
-----------------------------------------------
'''
'''
scp -r jzhang01@myria.criann.fr:/home/2017011/jzhang01/project/project/874* D:/Experiments/HCRegSeg/

'''
import warnings

warnings.filterwarnings('ignore')  # ignore warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # only show error
from data_reg import *
from memory_usage import *

root = 'HCRegSeg/'
img_aug = 'HCdata/image_aug/'
img_ori = 'HCdata/image_ori/'
csv_aug = 'HCdata/HC_aug.csv'
csv_ori = 'HCdata/HC_ori.csv'
H = 224
W = 224
slice = 3
learning_rate = 1e-4
batch_size = 16
nb_epoch = 1
inputshape2D = (W, H, slice)
loss = [huber_loss]  # "MAE" ,huber_loss
# load_data_single_slice,  load_data_multislice
#X_aug, Y_aug, ps_aug, \
#X_ori, Y_ori, ps_ori = load_data(img_aug, img_ori, csv_aug, csv_ori, H, W, slice)

#print("lr:",str(learning_rate),"bs:",str(batch_size),'Epoch:',str(nb_epoch),'X_aug:',str(X_aug.shape),'loss:',str(loss))
########## training and predicting###########

gigabytes = get_model_memory_usage(batch_size,model)
print("Memory usage:",gigabytes)
# fold_cross_valid(root, X_aug, Y_aug, ps_aug,X_ori, Y_ori, ps_ori,
#                  inputshape2D, loss, nb_epoch=nb_epoch, batch_size=batch_size, learning_rate=learning_rate,
#                  best_filename=root + 'best_MSE.h5')
