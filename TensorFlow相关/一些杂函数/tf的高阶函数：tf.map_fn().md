# TensorFlow中的高阶函数：tf.map_fn()
在**TensorFlow**中，有一些函数被称为高阶函数（**high-level function**），和在python中的高阶函数意义相似，其也是将函数当成参数传入，以实现一些有趣的，有用的操作。其中`tf.map_fn()`就是其中一个。我们这里介绍一下这个函数。

首先引入一个TF在应用上的问题：一般我们处理图片的时候，常常用到卷积，也就是`tf.nn.conv2d()`，但是这个函数的输入格式如：`(batch_size, image_height, image_width, image_channel)`，其中`batch_size`为一个批次的大小，我们注意到，如果按照这种输入的话，我们只能对一张图片进行卷积操作。在需要对视频进行卷积操作的时候，因为视频的输入格式一般如：`(batch_size, video_size, frame_height, frame_width, frame_channel)`，其中多出了一个视频的长度`video_size`，这样我们就不能简单对视频进行卷积了。我们用一张图描述我们面临的问题和map_fn是如何帮我们解决的：
![video_conv2d][video_conv2d]
其中，我们的数据就是从`clip #1`到`clip #n`，我们需要对每一个clip应用相同的方法function，比如是卷积函数`tf.nn.conv2d()`，从而得到batch_size个结果`result #1`到`result #n`。这个就是map_fn()的效果，而且，因为各个batch之间没有关联，所以可以并行快速高效地处理。我们再看看这个函数的用法，先贴出其API手册：

**tf.map_fn**
```python
map_fn(
    fn,
    elems,
    dtype=None,
    parallel_iterations=10,
    back_prop=True,
    swap_memory=False,
    infer_shape=True,
    name=None
)
```
其中的`fn`是一个可调用的(callable)函数，就是我们图中的function，一般会使用lambda表达式表示。`elems`是需要做处理的Tensors，TF将会将`elems`从第一维展开，进行map处理。主要就是那么两个，其中`dtype`为可选项，但是比较重要，他表示的是`fn`函数的输出类型，如果`fn`返回的类型和`elems`中的不同，那么就必须显式指定为和`fn`返回类型相同的类型。下面给出一个视频卷积的例子：
```python
batch = data.get_batch()  # batch with the shape of (batch_size, video_size, frame_height, frame_width, frame_channel)
cnn = tf.map_fn(fn=lambda inp: tf.nn.conv2d(inp, kernel, stride, padding='SAME'),
                elems=batch,
                dtype=tf.float32)
```
这样我们就对每一个batch的（batch-wise）进行了卷积处理了，大家不妨试试吧！



[video_conv2d]: ./imgs/video_conv2d.png