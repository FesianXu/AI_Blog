# tf.concat, tf.stack和tf.unstack的用法

`tf.concat`相当于`numpy`中的`np.concatenate`函数，用于将两个张量在某一个维度(axis)合并起来，例如：
```python
a = tf.constant([[1,2,3],[3,4,5]]) # shape (2,3)
b = tf.constant([[7,8,9],[10,11,12]]) # shape (2,3)
ab1 = tf.concat([a,b], axis=0) # shape(4,3)
ab2 = tf.concat([a,b], axis=1) # shape(2,6)
```

`tf.stack`其作用类似于`tf.concat`，都是拼接两个张量，而不同之处在于，`tf.concat`拼接的是两个shape完全相同的张量，并且产生的张量的阶数不会发生变化，而`tf.stack`则会在新的张量阶上拼接，产生的张量的阶数将会增加，例如：
```python
a = tf.constant([[1,2,3],[3,4,5]]) # shape (2,3)
b = tf.constant([[7,8,9],[10,11,12]]) # shape (2,3)
ab = tf.stack([a,b], axis=0) # shape (2,2,3)
```

改变参数axis为2，有：
```python
import tensorflow as tf
a = tf.constant([[1,2,3],[3,4,5]]) # shape (2,3)
b = tf.constant([[7,8,9],[10,11,12]]) # shape (2,3)
ab = tf.stack([a,b], axis=2) # shape (2,2,3)
```

所以axis是决定其层叠(stack)张量的维度方向的。

而`tf.unstack`与`tf.stack`的操作相反，是将一个高阶数的张量在某个axis上分解为低阶数的张量，例如：
```python
a = tf.constant([[1,2,3],[3,4,5]]) # shape (2,3)
b = tf.constant([[7,8,9],[10,11,12]]) # shape (2,3)
ab = tf.stack([a,b], axis=0) # shape (2,2,3)

a1 = tf.unstack(ab, axis=0)
```
其a1的输出为
```python
[<tf.Tensor 'unstack_1:0' shape=(2, 3) dtype=int32>,
 <tf.Tensor 'unstack_1:1' shape=(2, 3) dtype=int32>]
```
