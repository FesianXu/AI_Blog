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

## 保存pb文件
```python
import tensorflow as tf
var1 = tf.Variable([1.0], dtype=tf.float32, name='v1')
var2 = tf.Variable([2.0], dtype=tf.float32, name='v2')
addop = tf.add(var1, var2, name='add')

initop = tf.global_variables_initializer()
model_path = 'C://model.pb'

with tf.Session() as sess:
    sess.run(initop)
    
    # 导出当前图的GraphDef部分，单靠这一部分就可以完成从输入层到输出层的计算过程。
    graph_def = tf.get_default_graph().as_graph_def()
    
    # 将图上的变量以及取值转换成常量，同时将图上不必要的节点去掉。因为一些系统运行也会转变成计算图中的节点，比如变量初始化操作
    # 如果只是关心某一些操作，和此无关的节点就可以不用保存了。
    # 这里的output_node_name给出了需要保存的输出节点名称。
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['add'])
    
    # 将导出模型存入文件
    with tf.gfile.GFile(model_path, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
```

这里给出关键的convert_variables_to_constants()函数的声明：
> convert_variables_to_constants(
    **sess**,
    **input_graph_def**,
    **output_node_names**,
    **variable_names_whitelist=None**,
    **variable_names_blacklist=None**
)

> **Args:**
> * **sess**: Active TensorFlow session containing the variables.
* **input_graph_def**: GraphDef object holding the network.
* **output_node_names**: List of name strings for the result nodes of the graph.
* **variable_names_whitelist**: The set of variable names to convert (by default, all * variables are converted).
* **variable_names_blacklist**: The set of variable names to omit converting to constants.

> **Returns:**
> * GraphDef containing a simplified version of the original.


## 加载pb文件
在python中加载pb文件如下所示：
```python
from tensorflow as tf
with tf.Session() as sess:
    model_path = "C://model.pb"
    # 读取保存的pb文件，并且将其加载进计算图中。
    with tf.gfile.FastGFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 将graph_def中的保存的计算图加载到当前计算图中，return_elements给出了返回的张量名称add
    # 已经:0表示了是add节点的第1个输出，在加载的时候需要指定输出节点的第几个输出。所以是add:0
    # 详见文章[节点的表示]
    result = tf.import_graph_def(graph_def, return_elements="[add:0]")
    print(sess.run(result))
```
也可以通过`get_tensor_by_name()`获得tensor的句柄，然后run进行运算，如：
```python
from tensorflow as tf
wb_saver_path = u'C://model.pb'
with tf.gfile.FastGFile(wb_saver_path, 'rb') as f:
	graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
with tf.Session() as sess:
    result = tf.import_graph_def(graph_def, name='')
    op = sess.graph.get_tensor_by_name('Net/add:0')
    sess.run(op, feed_dict={
    	'Net/feed:0': batch_feed
    })
```

*****

在TensorFlowSharp中加载pb文件，其类似如下所示：
```cs
string modelFile = "C://model.pb"
var graph = new TFGraph();
// 从文件加载序列化的GraphDef
var model = File.ReadAllBytes(modelFile);
//导入GraphDef
graph.Import(model, "");
using (var session = new TFSession (graph))
{
	var runner = session.GetRunner ();
    // 其中的graph["input"][0], graph["output"][0]指的是，input节点的第1个输出，和   output节点的第1个输出，等同于python中的input:0 output:0
    // 其中Fetch()用于取得输出变量。
	runner.AddInput (graph ["input"] [0], tensor).Fetch (graph ["output"] [0]);
	var output = runner.Run ();
	var result = output [0];
    var val = (float [,])result.GetValue (jagged: false);
}
```

















