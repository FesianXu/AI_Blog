## tf.squeeze()用于压缩张量中为1的轴

```python
squeeze(
    input,
    axis=None,
    name=None,
    squeeze_dims=None
)
```
该函数会除去张量中形状为1的轴。

例子：
```python
import tensorflow as tf

raw = tf.Variable(tf.random_normal(shape=(1, 3, 2)))
squeezed = tf.squeeze(raw)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(raw.shape)
    print('-----------------------------')
    print(sess.run(squeezed).shape)
```
输出如：
```python
(1, 3, 2)
-----------------------------
(3, 2)
```