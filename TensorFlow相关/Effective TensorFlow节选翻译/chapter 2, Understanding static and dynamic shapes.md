# 理解静态和动态的Tensor类型的形状

**本文翻译自： [《Understanding static and dynamic shapes》](http://usyiyi.cn/translate/effective-tf/1.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

*************************************************************************

在**TensorFlow**中，`tensor`有一个在图构建过程中就被决定的**静态形状属性**， 这个静态形状可以是**没有明确加以说明的(underspecified)**，比如，我们可以定一个具有形状**[None, 128]**大小的`tensor`。

```python
import tensorflow as tf
a = tf.placeholder(tf.float32, [None, 128])
```

这意味着`tensor`的第一个维度可以是任何尺寸，这个将会在`Session.run()`中被动态定义。当然，你可以查询一个`tensor`的静态形状，如：

```python
static_shape = a.shape.as_list()  # returns [None, 128]
```

为了得到一个`tensor`的动态形状，你可以调用`tf.shape`操作，这将会返回指定tensor的形状，如：

```python
dynamic_shape = tf.shape(a)
```

tensor的静态形状可以通过方法`Tensor_name.set_shape()`设定，如：

```python
a.set_shape([32, 128])  # static shape of a is [32, 128]
a.set_shape([None, 128])  # first dimension of a is determined dynamically
```

调用`tf.reshape()`方法，你可以动态地重塑一个`tensor`的形状，如：

```python
a =  tf.reshape(a, [32, 128])
```

可以定义一个函数，当静态形状的时候返回其静态形状，当静态形状不存在时，返回其动态形状，如：
```python
def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims
``` 
现在，如果我们需要将一个三阶的`tensor`转变为2阶的`tensor`，通过折叠（collapse）第二维和第三维成一个维度，我们可以通过我们刚才定义的`get_shape()`方法进行，如：

```python
b = tf.placeholder(tf.float32, [None, 10, 32])
shape = get_shape(b)
b = tf.reshape(b, [shape[0], shape[1] * shape[2]])
```

注意到无论这个`tensor`的形状是静态指定的还是动态指定的，这个代码都是有效的。事实上，我们可以写出一个通用的reshape函数，用于折叠任意在列表中的维度（any list of dimensions）:

```python
import tensorflow as tf
import numpy as np

def reshape(tensor, dims_list):
  shape = get_shape(tensor)
  dims_prod = []
  for dims in dims_list:
    if isinstance(dims, int):
      dims_prod.append(shape[dims])
    elif all([isinstance(shape[d], int) for d in dims]):
      dims_prod.append(np.prod([shape[d] for d in dims]))
    else:
      dims_prod.append(tf.prod([shape[d] for d in dims]))
  tensor = tf.reshape(tensor, dims_prod)
  return tensor
```

然后折叠第二个维度就变得特别简单了。
```python
b = tf.placeholder(tf.float32, [None, 10, 32])
b = reshape(b, [0, [1, 2]])
```