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

这个将会输出5， `tf.while_loop()`需要一个条件函数和一个循环体函数，还需对循环变量的初始值。这些循环变量在每一次循环体函数调用完之后都会被更新一次，直到这个条件返回False为止。现在想象我们想要保存这个斐波那契序列，我们可能更新我们的循环体函数以至于可以纪录当前值的历史（**译者：也就是说需要输出一个序列而不仅仅是最后一个值**）。

```python
n = tf.constant(5)

def cond(i, a, b, c):
    return i < n

def body(i, a, b, c):
    return i + 1, b, a + b, tf.concat([c, [a + b]], 0)

i, a, b, c = tf.while_loop(cond, body, (2, 1, 1, tf.constant([1, 1])))

print(tf.Session().run(c))
```

当你尝试运行这个程序的时候，TensorFlow将会“抱怨”说**第四个循环变量的形状正在改变**。所以你必须要把“改变它的形状”这个事儿变得是有意为之的（**显式地定义**），而不是意外的编码错误，这样系统才会认可你的行为（**So you must make that explicit that it's intentional**）。

```python
i, a, b, c = tf.while_loop(
    cond, body, (2, 1, 1, tf.constant([1, 1])),
    shape_invariants=(tf.TensorShape([]),
                      tf.TensorShape([]),
                      tf.TensorShape([]),
                      tf.TensorShape([None])))
```


这个使得代码变得丑陋不堪，而且变得有些低效率。注意到我们这个操作在代码中够早了一大堆我们并不需要使用的中间张量。TF对于这种增长式的数组，其实有一个更好的解决方案，让我们会会`tf.TensorArray`吧，仅仅是通过了张量数组的形式，它能使得一切变得简单有效：

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
（**译者：TensorArray是TF动态尺度的数组，刻意为动态迭代运算设计的，其中的write方法指的是在某个`index`位置上写入特定值，stack方法是将TensorArray返回为一个层叠的张量。**）








