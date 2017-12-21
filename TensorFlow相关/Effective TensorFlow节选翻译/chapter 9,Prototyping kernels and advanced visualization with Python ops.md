# Effective TensorFlow Chapter 9: TensorFlow模型原型的设计和利用python ops的高级可视化
**本文翻译自： [《Prototyping kernels and advanced visualization with Python ops》](http://usyiyi.cn/translate/effective-tf/9.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

****************************************************************************************

在TensorFlow中，操作的内核都是完全由C++写成的，这样做更具有效率。但是在C++中编写TensorFlow的内核操作是一个苦差事，所以在花掉好多时间实现属于自己的内核之前，你也许需要先实现一个操作的原型，这样开发更快捷简单，虽然说运行效率会远远不如用C++编写的内核代码。通过`tf.py_func()`你可以将任何一个python源代码转换为TensorFlow的操作。
举个例子而言，这里有一个用python自己实现的ReLU非线性激活函数，通过`tf.py_func()`转换为TensorFlow操作的例子：

```python
import numpy as np
import tensorflow as tf
import uuid

def relu(inputs):
    # Define the op in python
    def _relu(x):
        return np.maximum(x, 0.)

    # Define the op's gradient in python
    def _relu_grad(x):
        return np.float32(x > 0)

    # An adapter that defines a gradient op compatible with TensorFlow
    def _relu_grad_op(op, grad):
        x = op.inputs[0]
        x_grad = grad * tf.py_func(_relu_grad, [x], tf.float32)
        return x_grad

    # Register the gradient with a unique id
    grad_name = "MyReluGrad_" + str(uuid.uuid4())
    tf.RegisterGradient(grad_name)(_relu_grad_op)

    # Override the gradient of the custom op
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        output = tf.py_func(_relu, [inputs], tf.float32)
    return output
```

通过TensorFlow的**gradient checker**，你可以确认这些梯度是否计算正确：

```python
x = tf.random_normal([10])
y = relu(x * x)

with tf.Session():
    diff = tf.test.compute_gradient_error(x, [10], y, [10])
    print(diff)
```

compute_gradient_error() computes the gradient numerically and returns the difference with the provided gradient. What we want is a very low difference.


`compute_gradient_error()`数值化地计算梯度，返回与理论上的梯度的差别，我们所期望的是一个非常小的差别。（**译者：这里我们引用[ref_1][ref_1], 这里有梯度检查gradient check API的解释，见附录**）
注意到我们的这种实现是非常低效率的，这仅仅在实现模型原型的时候起作用，因为python代码并不能并行化而且不能在GPU上运算（导致速度很慢）。一旦你确定了你的idea，你就需要用C++重写其内核。
在实践中，我们一般在Tensorboard中用python操作进行可视化。如果你是在构建一个图片分类模型，而且想要在训练过程中可视化你的模型预测，那么TF允许你通过`tf.summary.image()`函数进行图片的可视化。

```python
image = tf.placeholder(tf.float32)
tf.summary.image("image", image)
```


但是这仅仅是可视化了输入的图片，为了可视化其预测结果，你还必须找一个法儿添加预测标识在图片上，当然这在现有的tensorflow操作中是不存在的。一个最简单的方法就是通过python将预测标志绘制到图片上，然后再封装它。

```python
import io
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

def visualize_labeled_images(images, labels, max_outputs=3, name="image"):
    def _visualize_image(image, label):
        # Do the actual drawing in python
        fig = plt.figure(figsize=(3, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.imshow(image[::-1,...])
        ax.text(0, 0, str(label),
          horizontalalignment="left",
          verticalalignment="top")
        fig.canvas.draw()

        # Write the plot as a memory file.
        buf = io.BytesIO()
        data = fig.savefig(buf, format="png")
        buf.seek(0)

        # Read the image and convert to numpy array
        img = PIL.Image.open(buf)
        return np.array(img.getdata()).reshape(img.size[0], img.size[1], -1)

    def _visualize_images(images, labels):
        # Only display the given number of examples in the batch
        outputs = []
        for i in range(max_outputs):
            output = _visualize_image(images[i], labels[i])
            outputs.append(output)
        return np.array(outputs, dtype=np.uint8)

    # Run the python op.
    figs = tf.py_func(_visualize_images, [images, labels], tf.uint8)
    return tf.summary.image(name, figs)
```

# 附录
**梯度检查(Gradient checking)**

可对比`compute_gradient`和`compute_gradient_error`函数的用法

操作 | 描述 
:-: | :-: 
tf.test.compute_gradient(x, x_shape, y, y_shape, x_init_value=None, delta=0.001, init_targets=None) | 计算并返回理论的和数值的Jacobian矩阵
tf.test.compute_gradient_error(x, x_shape, y, y_shape, x_init_value=None, delta=0.001, init_targets=None) | 计算梯度的error。在计算所得的与数值估计的Jacobian中 为dy/dx计算最大的error


[ref_1]: http://blog.csdn.net/lenbow/article/details/52218551#reply

