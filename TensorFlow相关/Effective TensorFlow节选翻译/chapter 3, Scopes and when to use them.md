# Effective TensorFlow Chapter 3: 理解变量域Scope和何时应该使用它

**本文翻译自： [《Scopes and when to use them》](http://usyiyi.cn/translate/effective-tf/1.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

*************************************************************************

在TensorFlow中，变量(**Variables**)和张量(**tensors**)有一个名字属性，用于作为他们在图中的标识。如果你在创造变量或者张量的时候，不给他们显式地指定一个名字，那么TF将会自动地，隐式地给他们分配名字，如：

```python
a = tf.constant(1)
print(a.name)  # prints "Const:0"

b = tf.Variable(1)
print(b.name)  # prints "Variable:0"
```

你也可以在定义的时候，通过显式地给变量或者张量命名，这样将会重写（**overwrite**）他们的默认名，如：

```python
a = tf.constant(1, name="a")
print(a.name)  # prints "b:0"

b = tf.Variable(1, name="b")
print(b.name)  # prints "b:0"
```

TF引进了两个不同的上下文管理器，用于更改张量或者变量的名字，第一个就是`tf.name_scope`，如：

```python
with tf.name_scope("scope"):
  a = tf.constant(1, name="a")
  print(a.name)  # prints "scope/a:0"

  b = tf.Variable(1, name="b")
  print(b.name)  # prints "scope/b:0"

  c = tf.get_variable(name="c", shape=[])
  print(c.name)  # prints "c:0"
```

我们注意到，在TF中，我们有两种方式去定义一个新的变量，通过`tf.Variable()`或者调用`tf.get_variable()`。在调用`tf.get_variable()`的时候，给予一个新的名字，将会创建一个新的变量，但是如果这个名字并不是一个新的名字，而是已经存在过这个变量空间（variable scope）中的，那么就会抛出一个ValueError异常，意味着重复声明一个变量是不被允许的。


`tf.name_scope()`只会影响到**通过调用`tf.Variable`创建的**张量和变量的名字，而**不会影响到通过调用`tf.get_variable()`创建**的变量和张量。
和`tf.name_scope()`不同，`tf.variable_scope()`也会修改，影响通过`tf.get_variable()`创建的变量和张量，如：

```python
with tf.variable_scope("scope"):
  a = tf.constant(1, name="a")
  print(a.name)  # prints "scope/a:0"

  b = tf.Variable(1, name="b")
  print(b.name)  # prints "scope/b:0"

  c = tf.get_variable(name="c", shape=[])
  print(c.name)  # prints "scope/c:0"
with tf.variable_scope("scope"):
  a1 = tf.get_variable(name="a", shape=[])
  a2 = tf.get_variable(name="a", shape=[])  # Disallowed
```

但是如果我们真的想要重复使用一个先前声明过了变量怎么办呢？变量管理器同样提供了一套机制去实现这个需求：

```python
with tf.variable_scope("scope"):
  a1 = tf.get_variable(name="a", shape=[])
with tf.variable_scope("scope", reuse=True):
  a2 = tf.get_variable(name="a", shape=[])  # OK
This becomes handy for example when using built-in neural network layers:

features1 = tf.layers.conv2d(image1, filters=32, kernel_size=3)
# Use the same convolution weights to process the second image:
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
  features2 = tf.layers.conv2d(image2, filters=32, kernel_size=3)
```  

这个语法可能看起来并不是特别的清晰明了。特别是，如果你在模型中想要实现一大堆的变量共享，你需要追踪各个变量，比如说什么时候定义新的变量，什么时候要复用他们，这些将会变得特别麻烦而且容易出错，因此TF提供了TF模版（**TensorFlow templates**）自动解决变量共享的问题：

```python
conv3x32 = tf.make_template("conv3x32", lambda x: tf.layers.conv2d(x, 32, 3))
features1 = conv3x32(image1)
features2 = conv3x32(image2)  # Will reuse the convolution weights.
```

你可以将任何函数都转换为TF模版。当第一次调用这个模版的时候，在这个函数内声明的变量将会被定义，同时在接下来的连续调用中，这些变量都将自动地复用。