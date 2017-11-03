# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/10/16'
__version__ = ''

import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = u'./data.mat'
mat = sio.loadmat(path)
dataset = mat['y']
params = {
    'w': np.random.normal(size=(2, 1)),
    'b': np.random.normal(size=(1))
}


def predict(batch):
    logits = np.dot(batch, params['w'])+params['b']
    return 1*(logits > 0)+(-1)*(logits < 0)

def regression(x):
    w1 = params['w'][0]
    w2 = params['w'][1]
    return -w1/w2*x-params['b']/w2

learning_rate = 0.01
# 采用错误驱动的方式去更新参数，因为感知器的阶跃函数不能传递梯度。
def random_get_data(dataset, batch_size):
    batch_id = random.sample(range(dataset.shape[0]), batch_size)
    ret_batch = dataset[batch_id, 0:2]
    ret_label = dataset[batch_id, 2]
    return ret_batch, ret_label


batch_size = 128
for each_epoch in range(1000):
    ret_batch, ret_label = random_get_data(dataset, batch_size)
    result = predict(ret_batch)
    ret_label = np.reshape(ret_label, (batch_size, 1))
    err = (ret_label != result)*1.0  # 为1的是分类错误的，需要驱动更新
    err_batch = np.concatenate((ret_batch, err), axis=1)
    err_id = np.where(err_batch[:, 2] == 1)
    if len(err_id[0]) != 0:
        id = err_id[0][0]
        batch = np.reshape(np.transpose(err_batch[id, 0:2]), (2, 1))
        params['w'] = params['w']+learning_rate*ret_label[id]*batch
        params['b'] = params['b']+learning_rate*ret_label[id]
    else:
        break
print(each_epoch)
fig = plt.figure()
x1 = dataset[0:201, 0]
y1 = dataset[0:201, 1]
x2 = dataset[201:402, 0]
y2 = dataset[201:402, 1]
plt.scatter(x1, y1, c='r')
plt.scatter(x2, y2, c='b')
x = np.arange(-10, 10, 0.1)
y = regression(x)
plt.plot(x, y, c='g')

print(params['w'][0])
print(params['w'][1])
print(params['b'][0])
plt.show()






