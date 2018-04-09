<div align=center>
<font size="6"><b>TensorFlow中的LSTM源码理解与二次开发</b></font> 
</div>

# 前言
**学习TensorFlow其官方实现是一个很好的参考资料，而LSTM是在深度学习中经常使用的，我们开始探讨下其在TensorFlow中的实现方式，从而学习定制自己的RNN单元。**
**如有谬误，请联系指正。转载请注明出处。**

*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`
**有关代码开源**: [click][click]

*****

在TensorFlow中，已经实现了基本的循环神经网络结构，如Basic RNN, Basic LSTM, GRU, 双向LSTM等。其与RNN相关的python实现源码，路径位于：
**\tensorflow\python\ops\rnn_cell_impl.py**， github链接： [rnn_cell_impl.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py)

其实现直观明了，我们这里简单介绍下。
首先，需要大概了解下LSTM的单元结构，其公式描述如下：

$$\left(\begin{array}{}
		i \\ 
		f \\
        o \\
        g
	\end{array}\right) =
    \left(\begin{array}{}
		sigmoid \\ 
		sigmoid \\
        sigmoid \\
        tanh
	\end{array}\right) (W\left(\begin{array}{}
		x_{t} \\ 
		h_{t-1} 
	\end{array}\right))
\tag{1.1}
$$

$$
c_t = f \odot c_{t-1}+ i \odot g
\tag{1.2}
$$

$$
h_t =  o \odot tanh(c_t)
\tag{1.3}
$$

而其细胞结构图，如下图所示：

<div align=center>![lstm_cell][lstm_cell]</div>

具体原理这里不介绍，请参考： [[译] 理解 LSTM 网络](https://www.jianshu.com/p/9dc9f41f0b29)

于是，我们的任务就是实现出一个具有这种结构的单元，输入$c_{t-1}$和$h_{t-1}$，输出$c_{t}$和$h_{t}$，让我们看看TensorFlow官方是怎么写的。

*****
以下是LSTM的Basic实现，其主要初始化参数为`num_unit`（隐藏层的输入神经元数），`activation`（激活函数，默认采用`tanh`），`forget_bias`是给遗忘门加的偏置，可以减少过拟合，`state_is_tuple`是state格式控制的，一般用True即可。为了简洁，将源码中的一些文档注释去掉了，添加了自己的中文注释。（**在TensorFlow version 1.4.1下**）

```python
class BasicLSTMCell(RNNCell):
  ## 这里继承了RNNCell父类，所有RNN相关的单元都建议继承RNNCell父类。 comment_1
  def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None):
    super(BasicLSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

  # 这里有两个属性state_size和output_size分别是RNNCell里定义的属性，根据需求重写与否。
  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  # LSTM Cell的实体，定义了公式(1.1)-(1.3)中的结构。
  def call(self, inputs, state):
    sigmoid = math_ops.sigmoid
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
    ## 这里体现了参数state_is_tuple的作用了，如果为True，则传入的状态(c,h)需要为一个元组传入，如果False，则需要传入一个Tensor，其中分别是c和h层叠而成，建议采用第一种为True的方案，减少split带来的开销。

    concat = _linear([inputs, h], 4 * self._num_units, True)
    ## 这里将inputs和上一个输出状态拼接起来，尔后进行线性映射，见公式(1.1)。输出为4倍的隐藏层神经元数，是为了后面直接分割得到i,j,f,o(其中的j为公式中的g，代表gate)
    ## 其中的_linear()是rnn_cell_impl.py中的一个函数，作用就是线性映射，有兴趣各位可以移步去看看，其实很简单的。

    i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
    # 分割
    new_c = (
        c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
    new_h = self._activation(new_c) * sigmoid(o)
    # 核心计算，更新状态得到new_c和new_h

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state

```
我们可以看出，其实定义一个循环神经网络的细胞单元很简单，首先继承父类RNNCell，按照你的需求重写`state_size`, `output_size`和`call`，有时候还需要重写`zeros_state`，其中`state_size`, `output_size`重写这两个属性才能在`MultiRNNCell`,`DropoutWrapper`,`dynamic_rnn`等方法中正常使用。至于`call`就是其核心的方法了，如果自定义自己的单元，肯定需要重写的，`zeros_state`是进行细胞状态初始化的，一般初始化为全0张量即可。

而其中的核心运算将会涉及到张量的算术运算和张量其他运算如`split`,`concat`等，这个时候就不能直接使用`tf.add`,`tf.concat`等了，我们需要调用`array_ops.py`和`math_ops.py`里面的对array的操作（如在numpy中的相似）和一些算术运算，并且我们还需要调用`ops`，其中也包含了很多张量操作。这三个文件的位置在：
1. `array_ops.py`, `math_ops` -> **\tensorflow\python\ops\**
2. `ops` -> **\tensorflow\python\framework**

******
我这里实现了 [《NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis》](https://arxiv.org/abs/1604.02808)中的P-LSTM，细胞结构图如：

![plstm_cell][plstm_cell]

待我整理好后给大家分享一下，探讨下具体将如何更改RNN单元。



[plstm_cell]: ./imgs/plstm_cell.png
[lstm_cell]: ./imgs/lstm_cell.png
[click]: https://github.com/FesianXu
