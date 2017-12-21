# Effective TensorFlow Chapter 8: 在TensorFlow中的控制流：条件语句和循环

**本文翻译自： [《Control flow operations: conditionals and loops》](http://usyiyi.cn/translate/effective-tf/8.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

****************************************************************************************

当我们在构建一个复杂模型如RNN（循环神经网络）的时候，我们可能需要使用操作的控制流，比如条件控制和循环等。在这一节，我们介绍一些在TensorFlow中常用的控制流。
让我们假想，我们现在需要通过一个条件判断来决定我们是否**相加**还是**相乘**两个变量。这个可以通过调用`tf.cond()`简单实现，它表现出像python中`if...else...`相似的功能。

```python
a = tf.constant(1)
b = tf.constant(2)

p = tf.constant(True)

x = tf.cond(p, lambda: a + b, lambda: a * b)

print(tf.Session().run(x))
```

因为这个条件判断为True，所以这个输出应该是加法输出，也就是输出3。
在使用TensorFlow的大部分时间中，我们都会使用大型的张量，而且想要在一个批次（a batch）中进行操作。一个与之相关的条件操作符是`tf.where()`，它需要提供一个条件判断，就和`tf.cond()`一样，但是`tf.where()`将会根据这个条件判断，在一个批次中选择输出，如：

```python
a = tf.constant([1, 1])
b = tf.constant([2, 2])

p = tf.constant([True, False])

x = tf.where(p, a + b, a * b)

print(tf.Session().run(x))
```

这个代码将会输出`[3, 2]`，另一个广泛使用的控制流操作是`tf.while_loop()`。它允许在TensorFlow中构建动态的循环，这个可以实现对一个序列的变量进行操作（**译者：如在RNN中**）。让我们看看我们可以如何通过该函数生成一个斐波那契数列吧：

```python
n = tf.constant(5)

def cond(i, a, b):
    return i < n

def body(i, a, b):
    return i + 1, b, a + b

i, a, b = tf.while_loop(cond, body, (2, 1, 1))

print(tf.Session().run(b))
```



This will print 5. tf.while_loops takes a condition function, and a loop body function, in addition to initial values for loop variables. These loop variables are then updated by multiple calls to the body function until the condition returns false.

Now imagine we want to keep the whole series of Fibonacci sequence. We may update our body to keep a record of the history of current values:
这个将会输出5， `tf.while_loop()`需要一个条件函数和一个循环体函数。

```python
n = tf.constant(5)

def cond(i, a, b, c):
    return i < n

def body(i, a, b, c):
    return i + 1, b, a + b, tf.concat([c, [a + b]], 0)

i, a, b, c = tf.while_loop(cond, body, (2, 1, 1, tf.constant([1, 1])))

print(tf.Session().run(c))
```

Now if you try running this, TensorFlow will complain that the shape of the the fourth loop variable is changing. So you must make that explicit that it's intentional:

```python
i, a, b, c = tf.while_loop(
    cond, body, (2, 1, 1, tf.constant([1, 1])),
    shape_invariants=(tf.TensorShape([]),
                      tf.TensorShape([]),
                      tf.TensorShape([]),
                      tf.TensorShape([None])))
```

This is not only getting ugly, but is also somewhat inefficient. Note that we are building a lot of intermediary tensors that we don't use. TensorFlow has a better solution for this kind of growing arrays. Meet tf.TensorArray. Let's do the same thing this time with tensor arrays:

```python
n = tf.constant(5)

c = tf.TensorArray(tf.int32, n)
c = c.write(0, 1)
c = c.write(1, 1)

def cond(i, a, b, c):
    return i < n

def body(i, a, b, c):
    c = c.write(i, a + b)
    return i + 1, b, a + b, c

i, a, b, c = tf.while_loop(cond, body, (2, 1, 1, c))

c = c.stack()

print(tf.Session().run(c))
```
TensorFlow while loops and tensor arrays are essential tools for building complex recurrent neural networks. As an exercise try implementing beam search using tf.while_loops. Can you make it more efficient with tensor arrays?










