# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/10/15'
__version__ = ''

import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt

path = u'./samples.mat'
mat = sio.loadmat(path)
dataset = mat['samples']

batch_size = 1 # ͨ������batch_size���Էֱ�ʵ��SGD��BGD��MBGD
# ����Ϊ1ʱΪSGD������Ϊdataset.shape[0]ʱΪBGD������Ϊ�����̶�����ʱ��128��ΪMBGD

def random_get_samples(mat, batch_size):
    batch_id = random.sample(range(mat.shape[0]), batch_size)
    ret_batch = mat[batch_id, 0]
    ret_line = mat[batch_id, 1]
    return ret_batch, ret_line


params = {
    'w1': np.random.normal(size=(1)),
    'b': np.random.normal(size=(1))
}

def predict(x):
    return params['w1']*x+params['b']

learning_rate = 0.001

for i in range(3000):
    batch, line = random_get_samples(dataset, batch_size)
    y_pred = predict(batch)
    y_pred = np.reshape(y_pred, (batch_size, 1))
    line = np.reshape(line, (batch_size, 1))
    batch = np.reshape(batch, (batch_size, 1))
    delta = line-y_pred
    params['w1'] = params['w1']+learning_rate*np.sum(delta*batch)/batch_size
    params['b'] = params['b']+learning_rate*np.sum(delta)/batch_size
    if i % 100 == 0:
        print(np.sum(np.abs(line-y_pred))/batch_size)

print(params['w1'])
print(params['b'])
x = dataset[:, 0]
line = dataset[:, 1]
y = params['w1']*x+params['b']
plt.figure(1)
plt.plot(x, line, 'b--')
plt.plot(x, y, 'r--')
plt.show()