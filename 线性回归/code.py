# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/10/15'
__version__ = ''

import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = u'./samples.mat'

mat = sio.loadmat(path)
dataset = mat['samples']

batch_size = 128

def random_get_samples(mat, batch_size):
    batch_id = random.sample(range(mat.shape[0]), batch_size)
    ret_batch = mat[batch_id, 0:2]
    ret_line = mat[batch_id, 2:3]
    return ret_batch, ret_line

params = {
    'w': np.random.normal(size=(2, 1)),
    'b': np.random.normal(size=(1))
}

def predict(batch):
    return np.dot(batch, params['w'])+params['b']

learning_rate = 0.0003

for i in range(10000):
    batch, true_z = random_get_samples(dataset, batch_size)
    z_pred = predict(batch)
    delta = true_z-z_pred
    params['w'] = params['w']+learning_rate*np.reshape(np.transpose(np.sum(delta*batch, axis=0)), (2, 1))/batch_size
    params['b'] = params['b']+learning_rate*np.sum(delta)/batch_size
    if i % 100 == 0:
        print(np.sum(np.abs(delta))/batch_size)

fig = plt.figure()
ax = Axes3D(fig)
x = dataset[:, 0]
y = dataset[:, 1]
z = dataset[:, 2]
z_pred = predict(dataset[:, 0:2])
print(params['w'])
print(params['b'])
ax.scatter(x, y, z, c='b')
ax.scatter(x, y, z_pred, c='r')
plt.show()
