<h1 align = "center">sigmoid与softmax</h1>

## 前言
**sigmoid函数和softmax函数常用于神经网络最后的判别层的输出，其使用非常普遍，在这里简述其使用方法。**
**如有谬误，请联系指正。转载请注明出处。**
*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`

****

# sigmoid函数
sigmoid函数形式如：
$$
\sigma(x) = \frac{ 1 }{ 1+e^{-x} }
$$
sigmoid函数将输入归一化到(0,1)之间，例子如下：
```python
var1 = tf.Variable([1.0, 0.0, 6.0, 4.0], tf.float32)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(tf.sigmoid(var1)))
```
对于输入，sigmoid对所有的值进行sigmoid函数操作，最后输出
```python
output: [ 0.7310586   0.5         0.99752742  0.98201376]
```
sigmoid函数常常用于交叉熵损失函数的前端进行处理。


# softmax函数
softmax函数的目的是将输入转变成概率分布，可以参考softmax网络或者logistic回归中的例子，其公式为：
$$
softmax(x_i) = \frac{ e^{x_i} }{ \sum_{i=1}^{N} e^{x_i}}
$$
其中N为输出的数量，不难得出：
$$
\sum_{i=1}^{N} softmax(x_i) = 1
$$
softmax函数的例子如下：
```python

```








