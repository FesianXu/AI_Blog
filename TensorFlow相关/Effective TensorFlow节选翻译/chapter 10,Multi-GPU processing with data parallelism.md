# Effective TensorFlow Chapter 10: TensorFlow中的多GPU数据并行处理

**本文翻译自： [《Multi-GPU processing with data parallelism》](http://usyiyi.cn/translate/effective-tf/10.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

****************************************************************************************

如果你曾经用C++在一个单核CPU上编写程序，那么你就会知道若是要让这个程序能在多GPU上并行运行，就需要重头开始重写代码。但是在TensorFlow里并不需要这样，因为TF是符号式编程的系统，TF可以隐藏所有的复杂性，使得代码迁移到多CPU系统或多GPU系统这个过程变得简单。
让我们观察一个简单的，在CPU上将两个矢量相加的例子：
```python
import tensorflow as tf

with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
    a = tf.random_uniform([1000, 100])
    b = tf.random_uniform([1000, 100])
    c = a + b

tf.Session().run(c)
```

相同的过程可以简单的在GPU上实现：

```python
with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
    a = tf.random_uniform([1000, 100])
    b = tf.random_uniform([1000, 100])
    c = a + b
```

但是如果你又两块GPU而且你想要同时利用两块GPU的话，你可以先分离数据，然后用每一个GPU处理各一半数据，这样就可以提高并行性了：

```python
split_a = tf.split(a, 2)
split_b = tf.split(b, 2)

split_c = []
for i in range(2):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
        split_c.append(split_a[i] + split_b[i])

c = tf.concat(split_c, axis=0)
```

让我们用更加通用的格式重写这一段代码，这样我们就可以并行地实现其他操作而不仅仅是加法了：

```python
def make_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                out_split.append(fn(**{k : v[i] for k, v in in_splits.items()}))

    return tf.concat(out_split, axis=0)


def model(a, b):
    return a + b

c = make_parallel(model, 2, a=a, b=b)
```

You can replace the model with any function that takes a set of tensors as input and returns a tensor as result with the condition that both the input and output are in batch. Note that we also added a variable scope and set the reuse to true. This makes sure that we use the same variables for processing both splits. This is something that will become handy in our next example.

Let's look at a slightly more practical example. We want to train a neural network on multiple GPUs. During training we not only need to compute the forward pass but also need to compute the backward pass (the gradients). But how can we parallelize the gradient computation? This turns out to be pretty easy.

Recall from the first item that we wanted to fit a second degree polynomial to a set of samples. We reorganized the code a bit to have the bulk of the operations in the model function:

你可以用这个函数替换任何一个接受一个张量输入，返回一个张量的函数。注意到我们需要添加一个名字空间(**variable scope**)并且设置`reuse`为true。这个可以确保我们对于每一个片段，我们都是使用相同的变量。

```python
import numpy as np
import tensorflow as tf

def model(x, y):
    w = tf.get_variable("w", shape=[3, 1])

    f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
    yhat = tf.squeeze(tf.matmul(f, w), 1)

    loss = tf.square(yhat - y)
    return loss

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

loss = model(x, y)

train_op = tf.train.AdamOptimizer(0.1).minimize(
    tf.reduce_mean(loss))

def generate_data():
    x_val = np.random.uniform(-10.0, 10.0, size=100)
    y_val = 5 * np.square(x_val) + 3
    return x_val, y_val

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for _ in range(1000):
    x_val, y_val = generate_data()
    _, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})

_, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})
print(sess.run(tf.contrib.framework.get_variables_by_name("w")))
```

Now let's use make_parallel that we just wrote to parallelize this. We only need to change two lines of code from the above code:
```python
loss = make_parallel(model, 2, x=x, y=y)

train_op = tf.train.AdamOptimizer(0.1).minimize(
    tf.reduce_mean(loss),
    colocate_gradients_with_ops=True)
```

The only thing that we need to change to parallelize backpropagation of gradients is to set the colocate_gradients_with_ops flag to true. This ensures that gradient ops run on the same device as the original op.