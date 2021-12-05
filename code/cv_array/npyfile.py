'''
Title: Randomly generate numbers in some ranges for train, valid and test.
Date: 2021-03-24
Author: Jing Zhang
'''

import numpy as np
import random
import math

def randomNums(a, b, n):# From a to b, n numbers
    all = list(range(a, b))
    res = []

    while n:
        index = math.floor(random.random() * len(all))
        res.append(all[index])
        del all[index]
        n -= 1
    return res

# generate train sequence, run the selected code
# fold 0:
train0 = sorted(randomNums(0,600,600))
valid0 = sorted(randomNums(600,799,199))
test0 = sorted(randomNums(799,999,200))
print("train is:",train0)
print("valid is:",valid0)
print("test is:",test0)

np.save('train0.npy',train0)
np.save('valid0.npy',valid0)
np.save('test0.npy',test0)
# fold 1:
train1 = sorted(randomNums(100,700,600))
valid1 = sorted(randomNums(700,899,199))
test1b = sorted(randomNums(899,999,100))
test1a = sorted(randomNums(0,100,100))
test1 = test1a + test1b

print("train is:",train1)
print("valid is:",valid1)
print("test is:",test1)
np.save('train1.npy',train1)
np.save('valid1.npy',valid1)
np.save('test1.npy',test1)
# fold 2:
train2 = sorted(randomNums(200,800,600))
valid2 = sorted(randomNums(800,999,199))
test2 = sorted(randomNums(0,200,200))

print("train is:",train2)
print("valid is:",valid2)
print("test is:",test2)
np.save('train2.npy',train2)
np.save('valid2.npy',valid2)
np.save('test2.npy',test2)
# fold 3:
train3 = sorted(randomNums(300,900,600))
valid3b = sorted(randomNums(900,999,99))
valid3a = sorted(randomNums(0,100,100))
valid3 = valid3a+valid3b
test3 = sorted(randomNums(100,300,200))
print("train is:",train3)
print("valid is:",valid3)
print("test is:",test3)
np.save('train3.npy',train3)
np.save('valid3.npy',valid3)
np.save('test3.npy',test3)
# fold 4:
train4 = sorted(randomNums(399,999,600))
valid4 = sorted(randomNums(0,199,199))
test4 = sorted(randomNums(199,399,200))

print("train is:",train4)
print("valid is:",valid4)
print("test is:",test4)
np.save('train4.npy',train4)
np.save('valid4.npy',valid4)
np.save('test4.npy',test4)


# for a test
array=np.load('cv_array/test4.npy')
print(len(array))
print(array)