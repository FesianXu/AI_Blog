# TensorFlow中的LSTM源码

TensorFlow中，与RNN相关的python实现源码，路径位于：
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

而其结构图，如下

![lstm_cell][lstm_cell]






[lstm_cell]: ./imgs/lstm_cell.png
