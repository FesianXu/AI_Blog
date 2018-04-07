<div align=center>
<font size="6"><b>《Keras使用系列1》Keras介绍与使用</b></font> 
</div>

# 前言
**Keras是一个基于TensorFlow，CNTK或者Theano作为后端的高级深度学习API，可以实现快速敏捷地开发深度学习模型原型，在这里纪录学习和使用Keras的过程和心得。**

**如有谬误，请联系指正。转载请注明出处。**

*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`


*****

# 为什么要使用Keras
我们使用keras之前，需要明白为什么要用keras。目前市面上流行很多深度学习的框架，如目前大火的**TensorFlow**，经典的计算机视觉相关的**Caffe**，和MATLAB相似的**PyTorch**，和**Theano**，**CNTK**，**MXNET**等等。诸多的框架中，如TensorFlow，在搭建一个模型的时候，很多时候都需要从原子操作开始搭建起来，如矩阵加法，乘法等等。这些诸多的原子操作掩盖了模型的大局观，使得开发者疲于细节而难以专注于模型本身，在需要调整模型的一些子单元的时候更是触一发而动全身，给研究人员带来了不变。因此，为了研究人员和开发者对于模型的研究和开发，**Keras团队对TensorFlow，CNTK，Theano等框架的API进行了封装，并且称这些底层框架为后端（backend），对于开发者来说，只需要调用Keras提供的封装好的模块即可以实现大部分模型研究工作，即是出现了需要新定制的新模块的问题，也可以通过后端API自行定制，可谓是既方便又灵活。**

| ![layer_1][layer_1] | ![layer_2][layer_2] |  
    | :--------:   | :-----:   | 
    | **很多时候我们只关心模型之间层与层之间的关系**        | **而不是其实现该层的原子操作细节**      |   

用代码说明，既是：
在TensorFlow中，实现一个基本的LSTM，大概是：
```python
from tensorflow.python.ops import rnn_cell_impl as rnn

def __get_lstm_cell(hidden_layer):
  return rnn.BasicLSTMCell(hidden_layer)
cell = __get_lstm_cell(10)
state = cell.zero_state(batch_size, tf.float32)
outputs, fin_state = tf.nn.dynamic_rnn(cell, inputs_v, initial_state=state)
```
而在Keras中，只需要一句：
```python
outputs = LSTM(units=10, return_sequences=True, dropout=0.50, recurrent_dropout=0.50)(inputs)
```
即可

# Keras使用入门
Keras的编程范式中，模型大致分为两种，一种是**序贯式模型**， 一种是**函数式模型**。这两种模型都有着广泛的应用，而前者可以看成是后者的一种特殊情况。

## 序贯式模型
序贯式，既是整个模型的层从输入到输出都是一条路走到底，类似于VGG模型，如下图：

![vgg][vgg]

其中不能有岔路，也不能有反馈，回环等，其使用方法如：

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
		Dense(32, units=784),
		Activation('relu'),
		Dense(10),
		Activation('softmax'),
])
```
这样就实现了简单的FC全连接分类器网络。这样在初始化模型的时候就添加层的方法不适合于大型网络，因此可以采用`add()`一层层添加层。

```python
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))
```

## 小插曲：编译模型
定义好了你的模型，别忘了编译你的模型，只有编译后了的模型才能够训练和评估测试。在编译模型的时候可以指定很多参数，如最优化器，学习率，损失函数，指标列表(metrics)等。其方法如下：

```python
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
其中需要说明的是metrics这个参数，指标metrics是衡量一个模型优劣和有效与否的直接指标，而模型是通过loss函数的最优化进行学习的，但是我们要注意到，loss函数下降，不一定其指标就一定是提高的，这个指标可以是识别率，mAP，ROC曲线，AUC面积，或者是用户自己定义的任何一个函数都可以。如果选用keras自带的metrics，则可以通过字符串如'accuracy'指定，如果是用户自定义的指标，则需要传入一个定义函数的句柄，如：
```python
import keras.backend as K
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```
这里需要导入keras的后端，这里的后端可以是tensorflow也可以是Theano，这个可以设置。我们要做的就是写一个含有`y_true`和`y_pred`的函数，并且对其进行一定的运算，得到你需要的指标。通过这种方法，我们也可以自定义loss函数（当然，你这个loss函数的每一个操作是需要确保可以求梯度的。）

## 函数式模型
序贯式模型只能建立一条路走到底的模型，而很多时候，我们需要用到的模型会有分岔，回环等，如ResNet下图所示：

![resnet][resnet]

这个时候，我们就需要用到函数式模型去构建我们的模型了，其使用方法也很简单，我们首先先定义出一个或多个输入：
```python
from keras.layers import Input
input_1 = Input(shape=(784,))
input_2 = Input(shape=(784,))
```
随后，开始定义模型实体中的层，单元等
```python
x = Dense(64, activation='relu')(input_1)
x = Dense(64, activation='relu')(x)
pred_1 = Dense(10, activation='softmax')(x)

x = Dense(64, activation='tanh')(input_2)
x = Dense(64, activation='tanh')(x)
pred_2 = Dense(10, activation='softmax')(x)
```
其中，如果查阅Keras的源码，可以发现，这里的用法如：
```python
x = Dense(64, activation='relu')(input_1)
```
其实是由`Dense()`先定义出一个Dense实例后，通过调用`call`函数，接受其输入，输出这个层的结果`x`，`Dense`的`call()`函数源码如：
```python
def call(self, inputs):
	output = K.dot(inputs, self.kernel)
	if self.use_bias:
		output = K.bias_add(output, self.bias)
    if self.activation is not None:
        output = self.activation(output)
    return output
```
可以看出就是普通的线性操作，符合我们的理论知识。其他层操作如`LSTM`也是如此。
于是最后，我们得到了两个输出`pred_1`和`pred_2`，我们的loss函数，就需要对这两个函数和label进行操作，从而实现训练过程。
因此，用函数式模型编写resnet的单元，代码如下：
```python

```


# Reference
1. [Keras中文文档](http://keras-cn.readthedocs.io/en/latest/)


[layer_1]: ./imgs/layer_1.png
[layer_2]: ./imgs/layer_2.png
[vgg]: ./imgs/vgg.jpg
[resnet]: ./imgs/resnet.jpg

