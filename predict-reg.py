'''
-----------------------------------------------
File Name: predict$
Description:
Author: Jing$
Date: 7/17/2021$
-----------------------------------------------
'''
import time
from keras.models import load_model
from data_reg import *
from model_reg import *
from memory_profiler import profile

H = 224
W = 224
slice = 3
test_path = 'HCdata/test/image/'
test_gt = 'HCdata/test/HC_ori_test.csv'
X_aug, Y_aug, ps_aug, \
X_ori, Y_ori, ps_ori = load_data(test_path, test_path, test_gt, test_gt, H, W, slice)

absdff_mm = []
absdff_px = []
abspmae = []

inputshape = (W, H, slice)
y_test = Y_ori / ps_ori  # length in pixels
#HC_max = np.max(y_test)
HC_max = 1786.50024241547 # the largest value in training set.
y_test = y_test / HC_max  # Normalized HC in pixels
custom_objects = {'huber_loss': huber_loss}
    # Test stage
model = load_model('regmodels/best_efnMAE.h5', custom_objects)
print('predicting')

@profile #Decorator
def pred():
    time_start=time.time()
    preds = model.predict(X_ori,batch_size=16)  # The output value is between (0,1) due to normalization.
    preds = preds.flatten()
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
def predict():

    H = 224
    W = 224
    slice = 3
    test_path = 'HCdata/test/image/'
    test_gt = 'HCdata/test/HC_ori_test.csv'
    X_aug, Y_aug, ps_aug, \
    X_ori, Y_ori, ps_ori = load_data(test_path, test_path, test_gt, test_gt, H, W, slice)

    absdff_mm = []
    absdff_px = []
    abspmae = []

    inputshape = (W, H, slice)
    y_test = Y_ori / ps_ori  # length in pixels
    #HC_max = np.max(y_test)
    HC_max = 1786.50024241547 # the largest value in training set.
    y_test = y_test / HC_max  # Normalized HC in pixels
    custom_objects = {'huber_loss': huber_loss}
    # Test stage
    model = load_model('regmodels/best_efnMAE.h5', custom_objects)
    print('predicting')


    time_start=time.time()
    preds = model.predict(X_ori,batch_size=16)  # The output value is between (0,1) due to normalization.
    preds = preds.flatten()
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
    #predict_results = preds * HC_max * ps_ori
    #print(predict_results.shape)
    #print("The predicted HC in mm:")

#for i in (predict_results):
#    print(i)
if __name__ == "__main__":
    pred()