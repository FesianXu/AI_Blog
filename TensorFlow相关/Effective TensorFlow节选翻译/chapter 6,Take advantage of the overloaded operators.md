# Effective TensorFlow Chapter 6: 在TensorFlow中的运算符重载

**本文翻译自： [《Take advantage of the overloaded operators》](http://usyiyi.cn/translate/effective-tf/6.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

****************************************************************************************

和Numpy一样，TensorFlow重载了很多python中的运算符，使得构建计算图更加地简单，并且使得代码具有可读性。
**切片（slice）**操作是重载的诸多运算符中的一个，它可以使得索引张量变得很容易：

```python
z = x[begin:end]  # z = tf.slice(x, [begin], [end-begin])
```

但是在使用它的过程中，你还是需要非常地小心。切片操作非常低效，因此最好避免使用，特别是在切片的数量很大的时候。为了更好地理解这个操作符有多么地低效，我们先观察一个例子。我们想要人工实现一个对矩阵的行进行reduce操作的代码：

```python
import tensorflow as tf
import time

x = tf.random_uniform([500, 10])

z = tf.zeros([10])
for i in range(500):
    z += x[i]

sess = tf.Session()
start = time.time()
sess.run(z)
print("Took %f seconds." % (time.time() - start))
```

在笔者的MacBook Pro上，这个代码花费了2.67秒！那么耗时的原因是我们调用了切片操作500次，这个运行起来超级慢的！一个更好的选择是使用`tf.unstack()`操作去将一个矩阵切成一个向量的列表，而这只需要一次就行！

```python
z = tf.zeros([10])
for x_i in tf.unstack(x):
    z += x_i
```

这个操作花费了0.18秒，当然，最正确的方式去实现这个需求是使用`tf.reduce_sum()`操作：
```python
z = tf.reduce_sum(x, axis=0)
```

这个仅仅使用了0.008秒，是原始实现的300倍！
TensorFlow除了切片操作，也重载了一系列的数学逻辑运算，如：
```python
z = -x  # z = tf.negative(x)
z = x + y  # z = tf.add(x, y)
z = x - y  # z = tf.subtract(x, y)
z = x * y  # z = tf.mul(x, y)
z = x / y  # z = tf.div(x, y)
z = x // y  # z = tf.floordiv(x, y)
z = x % y  # z = tf.mod(x, y)
z = x ** y  # z = tf.pow(x, y)
z = x @ y  # z = tf.matmul(x, y)
z = x > y  # z = tf.greater(x, y)
z = x >= y  # z = tf.greater_equal(x, y)
z = x < y  # z = tf.less(x, y)
z = x <= y  # z = tf.less_equal(x, y)
z = abs(x)  # z = tf.abs(x)
z = x & y  # z = tf.logical_and(x, y)
z = x | y  # z = tf.logical_or(x, y)
z = x ^ y  # z = tf.logical_xor(x, y)
z = ~x  # z = tf.logical_not(x)
```
你也可以使用这些操作符的增广版本，如 `x += y`和`x **=2`同样是合法的。
注意到python不允许重载`and`,`or`和`not`等关键字。
TensorFlow也不允许把张量当成`boolean`类型使用，因为这个很容易出错：

```python
x = tf.constant(1.)
if x:  # 这个将会抛出TypeError错误
    ...
```

如果你想要检查这个张量的值的话，你也可以使用`tf.cond(x,...)`，或者使用`if x is None`去检查这个变量的值。
有些操作是不支持的，比如说等于判断`==`和不等于判断`!=`运算符，这些在numpy中得到了重载，但在TF中没有重载。如果需要使用，请使用这些功能的函数版本`tf.equal()`和`tf.not_equal()`。

