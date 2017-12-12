# Effective TensorFlow Chapter 7: TensorFlow中的执行顺序和控制依赖

**本文翻译自： [《Understanding order of execution and control dependencies》](http://usyiyi.cn/translate/effective-tf/7.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

****************************************************************************************

我们知道，TensorFlow是属于符号式编程的，它不会直接运行定义了的操作，而是在计算图中创造一个相关的节点，这个节点可以用`Session.run()`进行执行。这个使得TF可以在优化过程中(**do optimization**)决定优化的顺序(**the optimal order**)，并且在运算中剔除一些不需要使用的节点，而这一切都发生在运行中(**run time**)。如果你只是在计算图中使用`tf.Tensors`，你就不需要担心依赖问题（**dependencies**），但是你更可能会使用`tf.Variable()`，这个操作使得问题变得更加困难。笔者的建议是如果张量不能满足这个工作需求，那么仅仅使用`Variables`就足够了。这个可能不够直观，我们不妨先观察一个例子：

```python
import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)
a = a + b

tf.Session().run(a)
```

计算`a`将会返回3，就像期望中的一样。注意到我们现在有3个张量，两个常数张量和一个储存加法结果的张量。注意到我们不能重写一个张量的值（**译者：这个很重要，张量在TF中表示操作单元，是一个操作而不是一个值，不能进行赋值操作等。**），如果我们想要改变张量的值，我们就必须要创建一个新的张量，就像我们刚才做的那样。


> **小提示：**如果你没有显式地定义一个新的计算图，TF将会自动地为你构建一个默认的计算图。你可以使用`tf.get_default_graph()`去获得一个计算图的句柄（handle），然后，你就可以查看这个计算图了。比如，可以打印属于这个计算图的所有张量之类的的操作都是可以的。如：

```python
print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
```

不像张量，变量Variables可以更新，所以让我们用变量去实现我们刚才的需求：

```python
a = tf.Variable(1)
b = tf.constant(2)
assign = tf.assign(a, a + b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(assign))
```

同样，我们得到了3，正如预期一样。注意到`tf.assign()`返回的代表这个赋值操作的张量。目前为止，所有事情都显得很棒，但是让我们观察一个稍微有点复杂的例子吧：

```python
a = tf.Variable(1)
b = tf.constant(2)
c = a + b

assign = tf.assign(a, 5)

sess = tf.Session()
for i in range(10):
    sess.run(tf.global_variables_initializer())
    print(sess.run([assign, c]))
```
（**译者：这个会输出[7, 5]**）

注意到，张量`c`并没有一个确定性的值。这个值可能是3或者7，取决于加法和赋值操作谁先运行。
你应该也注意到了，你在代码中定义操作(**ops**)的顺序是不会影响到在TF运行时的执行顺序的，唯一会影响到执行顺序的是**控制依赖**。控制依赖对于张量来说是直接的。每一次你在操作中使用一个张量时，操作将会定义一个对于这个张量来说的隐式的依赖。但是如果你同时也使用了变量，事情就变得更糟糕了，因为变量可以取很多值。
当处理这些变量时，你可能需要显式地去通过使用`tf.control_dependencies()`去控制依赖，如：

```python
a = tf.Variable(1)
b = tf.constant(2)
c = a + b

with tf.control_dependencies([c]):
    assign = tf.assign(a, 5)

sess = tf.Session()
for i in range(10):
    sess.run(tf.global_variables_initializer())
    print(sess.run([assign, c]))
```
（**译者：这个会输出[5, 3]**）

这会确保赋值操作在加法操作之后被调用。

**译者：**
这里贴出`tf.control_dependencies()`的API文档，希望有所帮助：

**tf.control_dependencies(control_inputs)**

**control_inputs**：一个操作或者张量的列表，这个列表里面的东西必须在运行定义在下文中的操作执行之前执行。当然也可以为None，这样会消除控制依赖的作用。