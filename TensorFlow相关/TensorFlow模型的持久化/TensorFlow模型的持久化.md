<h1 align = "center">TensorFlow模型的持久化</h1>

## 前言
**在TensorFlow中，一旦模型训练完成，就需要对其进行持久化操作，也就是将其保存起来，在需要进行对新样本进行测试时，程序加载已经持久化后的模型。在这个过程中就涉及到了模型的持久化操作，在这里简单分享下自己的所见所学。**
**如有谬误，请联系指正。转载请注明出处。**
*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`

****

# 模型的持久化，保存为ckpt文件
TensorFlow可以将模型保存为ckpt文件，此时会将模型分开计算图和图上参数取值分开储存。其使用方法如下所示：
```python
import tensorflow as tf
var1 = tf.Variable([1.0], dtype=tf.float32, name='v1')
var2 = tf.Variable([2.0], dtype=tf.float32, name='v2')
addOp = var1+var2
init = tf.global_variables_initializer()
saver = tf.train.Saver()
path_model = 'C://model.ckpt'
with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, path_model)
```

对其进行加载时，如下所示：
```python
import tensorflow as tf
var1 = tf.Variable([1.0], dtype=tf.float32, name='v1')
var2 = tf.Variable([2.0], dtype=tf.float32, name='v2')
addOp = var1+var2

saver = tf.train.Saver()
path_model = 'C://model.ckpt'
with tf.Session() as sess:
    saver.restore(sess, path_model)
```
可以发现，在对模型进行加载时候，需要定义出与原来的计算图结构完全相同的计算图，然后才能进行加载，并且不需要对定义出来的计算图进行初始化操作。
这样保存下来的模型，会在其文件夹下生成三个文件，分别是：
* .ckpt.meta文件，保存tensorflow模型的计算图结构。
* .ckpt文件，保存计算图下所有变量的取值。
* checkpoint文件，保存目录下所有模型文件列表。

**如果不希望重复定义图上的结构**，可以直接加载已经持久化后的图：
```python
import tensorflow as tf
saver = tf.train.import_meta_graph("C://model.ckpt/model.ckpt.meta")
with tf.Session() as sess:
	saver.restore(sess, "C://model.ckpt")
```





















