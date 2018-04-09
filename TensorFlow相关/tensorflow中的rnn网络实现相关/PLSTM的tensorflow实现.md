<h1 align = "center">PLSTM的TensorFlow实现与解释</h1>

## 前言
** 我们在上篇文章[《TensorFlow中的LSTM源码理解与二次开发》](https://blog.csdn.net/loseinvain/article/details/79642721)中已经提到了lstm cell在tensorflow中的实现。这篇博文主要介绍了如何在TensorFlow中通过修改lstm cell来定制自己的lstm网络，并且以 [《NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis》](https://arxiv.org/abs/1604.02808)文章中提到的PLSTM为例子，对lstm进行修改。**

**如有谬误，请联系指正。转载请注明出处。**

*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`
**code**: [click](https://github.com/FesianXu/PLSTM)

*******************************************************

# PLSTM
PLSTM全称Part-Aware LSTM，其对普通的lstm进行了简单的修改，应用于基于3D骨骼点的动作识别，可以对身体部分（头，胳膊，腿等）进行上下文分析，从而提高性能等。其公式表达为：


$$\left(\begin{array}{}

		i^p \\ 
		f^p \\
        g^p
	\end{array}\right) =
    \left(\begin{array}{}
		sigmoid \\ 
		sigmoid \\
        tanh
	\end{array}\right) (W^p\left(\begin{array}{}
		x_{t}^p \\ 
		h_{t-1} 
	\end{array}\right))
\tag{1.1}
$$

$$
c^p_t = f^p \odot c^p_{t-1}+i^p \odot g^p
\tag{1.2}
$$

$$
o = sigmoid \left(\begin{array}{}
	W_o \left(\begin{array}{}
	x_t^1 \\
    \vdots \\
    x_t^p \\
    h_{t-1}	
	\end{array}\right)
	\end{array}\right)
\tag{1.3}
$$

$$
h_t = o \odot tanh\left(\begin{array}{}
	c^1_t \\
    \vdots \\
    c^p_t
	\end{array}\right)
\tag{1.4}
$$

其中$P$表示不同的身体部分，具体划分为：
```python
    divide_config = {
      'head':  ( 3, 4,              1,2,21), # head
      'r_arm': ( 5, 6, 7, 8,22,23,  1,2,21), # right arm
      'l_arm': ( 9,10,11,12,24,25,  1,2,21), # left arm
      'r_leg': ( 13,14,15,16,       1,2,21), # right leg
      'l_leg': ( 17,18,19,20,       1,2,21) # left leg
    }
```

其结构图如：

![plstm][plstm]

# 实现

现在我们可以开始我们的修改了，这里给出主要的代码

```python
class PartAwareLSTMCell(RNNCell):
  '''
  The implement of paper <NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis>
  Part-Aware LSTM Cell
  '''
  def __init__(self,num_units,forget_bias=1.0,state_is_tuple=True,activation=None,reuse=None):
    # here num_units has to be times of 5
    assert num_units % 5 == 0

    super(PartAwareLSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    self._num_units = int(num_units/5)
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    cs_size = self._num_units * 5
    return (LSTMStateTuple(cs_size, 5*self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units * 5


  def call(self, skel_inputs, state):
    '''
    here inputs with the shape of (batch_size, feat_dim)
    in kinect 2.0, feat_dim is 25*3 = 75
    for five part of a skeleton body.
    (head, r_arm, l_arm, r_leg, l_leg)
    divide config:
    head:  [ 3, 4,              1,2,21]
    r_arm: [ 5, 6, 7, 8,22,23,  1,2,21]
    l_arm: [ 9,10,11,12,24,25,  1,2,21]
    r_leg: [13,14,15,16,        1,2,21]
    l_leg: [17,18,19,20,        1,2,21]

    state: LSTMStateTuple with the format of (Tensor(c1, c2, ..., c5), Tensor(h))
    '''

    sigmoid = math_ops.sigmoid
    tanh = math_ops.tanh

    if self._state_is_tuple:
      cs, h = state
    else:
      cs, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
    # split he state into c and h
    # here cs mean c1 to c5, where each part means a part of body, cs is also a list or turple

    # split the cs into 5 parts
    cs = array_ops.split(cs, num_or_size_splits=5, axis=1)

    divide_config = {
      'head':  ( 3, 4,              1,2,21),
      'r_arm': ( 5, 6, 7, 8,22,23,  1,2,21),
      'l_arm': ( 9,10,11,12,24,25,  1,2,21),
      'r_leg': ( 13,14,15,16,       1,2,21),
      'l_leg': ( 17,18,19,20,       1,2,21)
    }
    # assert skel_inputs.shape[1] == 75

    reshaped_input = array_ops.reshape(skel_inputs, shape=(-1, 25, 3))
    head_joints = [reshaped_input[:, each-1, :] for each in divide_config['head']]
    r_arm_joints = [reshaped_input[:, each-1, :] for each in divide_config['r_arm']]
    l_arm_joints = [reshaped_input[:, each-1, :] for each in divide_config['l_arm']]
    r_leg_joints = [reshaped_input[:, each-1, :] for each in divide_config['r_leg']]
    l_leg_joints = [reshaped_input[:, each-1, :] for each in divide_config['l_leg']]

    body_list = [head_joints, r_arm_joints, l_arm_joints, r_leg_joints, l_leg_joints]

    body_list = ops.convert_n_to_tensor(body_list)

    for ind, each in enumerate(body_list):
      tmp = array_ops.transpose(each, perm=(1,0,2))
      batch_size = int(tmp.shape[0])
      body_list[ind] = array_ops.reshape(tmp, shape=(batch_size, -1))

    o_all_skel = _linear([body_list[0],
                          body_list[1],
                          body_list[2],
                          body_list[3],
                          body_list[4],
                          h], # here 111 + h_size
                         5 * self._num_units, True)
    o_all_skel = sigmoid(o_all_skel)
    new_c_list = []
    for ind, each_part in enumerate(body_list):
      concat_p = _linear([each_part, h],
                         3 * self._num_units,
                         weight_name='weight_%d' % ind,
                         bias_name='bias_%d' % ind,
                         bias=True)
      ip, fp, gp = array_ops.split(value=concat_p, num_or_size_splits=3, axis=1)
      ip, fp, gp = sigmoid(ip), sigmoid(fp), tanh(gp)
      new_c = cs[ind] * (fp+self._forget_bias) + ip * gp
      new_c_list.append(new_c)


    new_c_tensors = array_ops.concat(new_c_list, axis=1)
    new_h = o_all_skel * tanh(array_ops.concat(new_c_list, 1))

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c_tensors, new_h)
    else:
      new_state = array_ops.concat([new_c_tensors, new_h], 1)

    return new_h, new_state
```

这里的修改思路和上一篇文章是一模一样的，但是我们要注意，我们需要将一个输入划分为五个body part，而不是直接传入一个五个part组成的列表，因为这样做可以使得你新定制的lstm单元可以直接应用于`dynamic_rnn`和`static_rnn`等函数而不需要改变其他东西。所以我们的原则就是传入和传出参数和普通lstm完全相同，至于需要分割合并等操作都放到cell里面完成。至于其他的也没有什么可说的，和上一篇文章都相似。

这里的`_linear()`为线性连接，直接用tf提供的即可，代码为：

```python
def _linear(args,
            output_size,
            bias,
            weight_name=_WEIGHTS_VARIABLE_NAME,
            bias_name=_BIAS_VARIABLE_NAME,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        weight_name, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)

    # if the args is a single tensor then matmul it with weight
    # if the args is a list of tensors then concat them in axis of 1 and matmul
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          bias_name, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)
```

整个训练代码和数据加载代码等见github，地址 https://github.com/FesianXu/PLSTM


[plstm]: ./imgs/plstm_cell.png