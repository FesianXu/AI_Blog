# TensorFlow中的GPU使用相关

## 控制程序使用的GPU
```python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 这里的1指定使用GPU 1号
```

## 控制程序使用的GPU的显存
TensorFlow程序一旦运行默认情况下会占用显卡的所有显存，因此需要设定使用的显存大小
```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	pass
```