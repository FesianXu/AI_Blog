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
* **.ckpt.meta**文件，保存tensorflow模型的计算图结构。
* **.ckpt**文件，保存计算图下所有变量的取值。
* **checkpoint**文件，保存目录下所有模型文件列表。

**如果不希望重复定义图上的结构**，可以直接加载已经持久化后的图：
```python
import tensorflow as tf
saver = tf.train.import_meta_graph("C://model.ckpt/model.ckpt.meta")
with tf.Session() as sess:
	saver.restore(sess, "C://model.ckpt")
```


-----


**有的时候只希望加载或者保存部分变量**， 这个时候在什么tf.train.Saver()的时候，可以提供一个列表指定需要保存或者加载的变量名，比如在加载模型中使用
```python
saver = tf.train.Saver([v1])
```
那么只有变量v1会被加载，同样的，也支持在加载的时候进行变量改名，有利于加载滑动平均变量（既是将持久化了的滑动平均变量加载进模型，并且替代程序中的原普通变量）。
```python
var1 = tf.Variable([1.0], name='other-v1')
var2 = tf.Variable([2.0], name='other-v2')

# 如果直接使用tf.train.Saver()进行加载模型，因为原来保存的模型中的变量名是v1和v2而不是
# other-v1和other-v2所以会报出错误，所以需要在加载模型的时候对变量进行改名。

# 使用一个dict就可以重命名变量了，这里将原来名称为v1的变量加载到变量v1中(名字在此为other-v1)，同样对v2也是
# 如此。
saver = tf.train.Saver({
    'v1': v1,
    'v2': v2
})
```



*****


# 模型的持久化，保存为pb文件
使用tf.train.Saver会保存运行程序需要的全部信息，然而有时候并不需要某些信息，比如在测试或者离线预测的时候，只需要知道**如何从神经网络的输入层经过前向传播计算得到输出层**即可，而不需要类似变量初始化，模型保存等辅助节点的信息。可以利用tensorflow提供的convert_variables_to_constant函数，可以将所有计算图中的变量和取值通过常量的形式保存，这样整个计算图就可以统一放在一个pb文件中。（在C#中的TensorFlowSharp可以用于加载模型，后面会谈到如何在TensorFlowSharp中加载TensorFlow模型进行预测）。





















