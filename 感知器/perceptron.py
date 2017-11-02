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

path = u'./samples.mat'
mat = sio.loadmat(path)
dataset = mat['samples']
params = {
    'w': np.random.normal(size=(3, 1)),
    'b': np.random.normal(size=(1))
}


def predict(batch):
    logits = np.dot(batch, params['w'])+params['b']
    return 1*(logits > 0)+(-1)*(logits < 0)

def regression(x, y):
    w1 = params['w'][0]
    w2 = params['w'][1]
    w3 = params['w'][2]
    return -w1/w3*x-w2/w3*y-params['b']/w3

learning_rate = 0.001
# 采用错误驱动的方式去更新参数，因为感知器的阶跃函数不能传递梯度。
def random_get_data(dataset, batch_size):
    batch_id = random.sample(range(dataset.shape[0]), batch_size)
    ret_batch = dataset[batch_id, 0:3]
    ret_label = dataset[batch_id, 3]
    return ret_batch, ret_label


batch_size = 128
for each_epoch in range(10000):
    ret_batch, ret_label = random_get_data(dataset, batch_size)
    result = predict(ret_batch)
    ret_label = np.reshape(ret_label, (batch_size, 1))
    ret_label = (ret_label == 0)*(-1.0)+(ret_label == 1)*1.0
    err = (ret_label != result)*1.0  # 为1的是分类错误的，需要驱动更新
    err_batch = np.concatenate((ret_batch, err), axis=1)
    err_id = np.where(err_batch[:, 3] == 1)
    if len(err_id[0]) != 0:
        id = err_id[0][0]
        batch = np.reshape(np.transpose(err_batch[id, 0:3]), (3, 1))
        params['w'] = params['w']+learning_rate*ret_label[id]*batch
        params['b'] = params['b']+learning_rate*ret_label[id]
    else:
        break
fig = plt.figure()
ax = Axes3D(fig)
x = dataset[0:100, 0]
y = dataset[0:100, 1]
z = dataset[0:100, 2]
ax.scatter(x, y, z, c='b')
x = dataset[100:200, 0]
y = dataset[100:200, 1]
z = dataset[100:200, 2]
ax.scatter(x, y, z, c='r')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
# 网格化数据
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, regression(X, Y), cmap='rainbow')
print(params['w'][0])
print(params['w'][1])
print(params['w'][2])
print(params['b'][0])
plt.show()






