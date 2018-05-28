# TensorFlow和Keras中的Crop函数

在计算机视觉算法中，有些需要对图像进行裁剪（crop）操作的，如下图所示：

![crop][crop]

而在TensorFlow和Keras中(**针对于TensorFlow版本1.8**)，提供了一系列的函数用于crop操作，分别是:

* Keras中的
1. `tf.keras.layers.Cropping1D`（用于一维信号如语音信号的裁剪）
2. `tf.keras.layers.Cropping2D`（二维信号如图像的裁剪，在空间域上进行裁剪，也就是会影响width和height）
3. `tf.keras.layers.Cropping3D`（三维数据，也就是时空数据如视频上的裁剪）
 
* TensorFlowsharp中的
1. `tf.image.crop_and_resize` （从原图中提取出多个crop后，用双线性插值进行图片的resize到一个`crop_size`）
2. `tf.image.crop_to_bounding_box`（从原图中裁剪出一个大小为`[target_width, target_height]`的bounding box）
3. `tf.random_crop`（在图中随机地裁剪，每个crop大小为`size`）
4. `tf.image.central_crop`（在图片的中心按照原图比例的`central_fraction`进行裁剪）
5. `tf.image.decode_and_crop_jpeg`（传入一个jpeg图片的地址后进行解码并且裁剪）
6. `tf.image.resize_image_with_crop_or_pad`（用裁剪或者均匀填充0的方式进行resize图片）

这些方法大同小异，主要想说说的是一开始并没有看懂`tf.keras.layers.Cropping2D`中的参数的意义，这里贴出来先：

![crop2d][crop2d]

其中的主要参数是`cropping`，可以为一个int；也可以是一个有两个int构成的元组，如`(0,0)`；也可以是一个由两个元组构成的元组，其中每一个元组都是由两个int构成的，如`((1,2),(3,4))`

需要注意的是，如最后一个`((1,2),(3,4))`为例，指的是从图片中，除去原图的从上面看的第1索引之前的所有像素，从底下看的第2索引之前的所有像素，从左边算起来的3排像素，从右边算起来的4排像素。其参数的意义是这样的。

其实就相当于
```python
return x[:, :, 1:-2, 3:4]
#  x with the shape of (data_sample_idx, channel, height, width).
```

[crop2d]: ./imgs/crop2d.png
[crop]: ./imgs/crop.png