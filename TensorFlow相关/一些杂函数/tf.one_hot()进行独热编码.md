# tf.one_hot()进行独热编码
首先肯定需要解释下什么叫做**独热编码（one-hot encoding）**，独热编码一般是在有监督学习中对数据集进行标注时候使用的，**指的是在分类问题中，将存在数据类别的那一类用X表示，不存在的用Y表示，这里的X常常是1， Y常常是0。**，举个例子：
比如我们有一个5类分类问题，我们有数据$(x_i, y_i)$，其中类别$y_i$有五种取值（因为是五类分类问题），所以如果$y_j$为第一类那么其独热编码为：
$[1, 0, 0, 0, 0]$，如果是第二类那么独热编码为：$[0, 1, 0, 0, 0]$，也就是说只对存在有该类别的数的位置上进行标记为1，其他皆为0。这个编码方式经常用于多分类问题，特别是损失函数为交叉熵函数的时候。接下来我们再介绍下**TensorFlow**中自带的对数据进行独热编码的函数`tf.one_hot()`，首先先贴出其API手册：
```python
one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)
```
需要指定`indices`，和`depth`，其中`depth`是编码深度，`on_value`和`off_value`相当于是编码后的开闭值，如同我们刚才描述的X值和Y值，需要和`dtype`相同类型（指定了`dtype`的情况下），`axis`指定编码的轴。这里给个小的实例：


```python
var = tf.one_hot(indices=[1, 2, 3], depth=4, axis=0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(var)
    print(a)
```
输出
```python
[[ 0.  0.  0.]
 [ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]]
```

因为向`axis=0`轴进行编码，`depth`为4，相当于是朝着列方向扩展的。
将`axis`改为1之后，为：
```python
[[ 0.  1.  0.  0.]
 [ 0.  0.  1.  0.]
 [ 0.  0.  0.  1.]]
```






