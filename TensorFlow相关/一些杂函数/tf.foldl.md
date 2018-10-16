# TensorFlow高阶函数之 tf.foldl()和tf.foldr()

在TensorFlow中有着若干高阶函数，如之前已经介绍过了的`tf.map_fn()`，见博文[TensorFlow中的高阶函数：tf.map_fn()](https://blog.csdn.net/LoseInVain/article/details/78815130)，此外，还有几个常用的高阶函数，分别是`tf.foldl()`，`tf.foldr()`，我们简要介绍下。

*****

`tf.foldl()`类似于python中的`reduce()`函数，假设`elems`是一个大于等于一阶的张量或者列表，形状如`[n,...]`，那么该函数将会重复地调用`fn`与这个列表上，从左到右进行处理，这里讲得不清楚，我们看看官方的API手册和一些例子理解一下，地址：https://www.tensorflow.org/api_docs/python/tf/foldl
```python
tf.foldl(
    fn,
    elems,
    initializer=None,
    parallel_iterations=10,
    back_prop=True,
    swap_memory=False,
    name=None
)
```
其中`fn`是一个可调用函数，也可以用lambda表达式；`elems`是需要处理的列表；`initializer`是一个可选参数，可以作为`fn`的初始累加值（accumulated value）；至于`parallel_iterations`是并行数，其他的函数可以查询官网，就不累述了。
最重要的参数无非是`fn`,`elems`和`initializer`，其中`fn`是一个具有两个输入参数的函数，需要返回一个值作为累计结果，`elems`是一个列表或者张量，用于被`fn`累计处理。形象的来说，就是
```python
tf.foldl(fn,elems=[x1,x2,x3,x4]) = fn(fn(fn(x1,x2),x3),x4)
```
如果给定了`initializer`，那么初始的累计参数（也就是`fn`的第一个参数）就是他了，如果没有给定，也即是`initializer=None`那么`elems`中必须至少有一个值，第一个值将会被作为初始值。例子：
```python
import tensorflow as tf
elems = [1, 2, 3, 4, 5, 6]
sum = tf.foldl(lambda a, x: a + x, elems)
with tf.Session() as sess:
  print(sess.run(sum))
```
将会输出21，也即是(((((1+2)+3)+4)+5)+6)，如果给定了一个初始化值，就变为：
```python
import tensorflow as tf
elems = [1, 2, 3, 4, 5, 6]
sum = tf.foldl(lambda a, x: a + x, elems,initializer=10)
with tf.Session() as sess:
  print(sess.run(sum))
```
输出变为31。此处的累加是最简单的应用，还可以有更多复杂的应用，就看应用场景了。至于`tf.foldr()`和这个函数是基本上一样的，无非就是从右边开始计算到左边而已。










