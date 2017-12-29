<div align=center>
<font size="6"><b>利用numpy数组保存TensorFlow模型的参数</b></font> 
</div>

我们在前文[《TensorFlow模型的保存和持久化》][ref_1]中，讨论了如何利用TensorFlow自带的**Saver**类进行模型参数的保存和持久化。利用原生的API，这种方法好处就是非常的简单方便，但是也存在一点不灵活的地方，就是这样进行保存模型参数，加载模型参数的时候需要保证每个参数的名字空间(**variable_scope**)是完全一样的。也就是说，如果你的模型中修改了名字空间，或者不存在名字空间，只要在需要读取的ckpt文件中找不到这个名字空间，就会发生读取错误。在命名空间经常变的情况下，这样会导致已经预先训练好的模型没法加载进去，就只是因为命名空间不合！所以，这里介绍一种利用numpy的数组保存TensorFlow模型的参数的方法，这个方法是不考虑命名空间的，也就是只要参数的类型和形状一致，就可以正常加载。

我们在定义TensorFlow的graph的时候，维护一个parameter列表，用于储存Tensor，如下：

```python
class test(object):
    params = []  # 维护parameter

    def __init__(self):
        with tf.variable_scope('scope_1', initializer=tf.zeros_initializer()):
            var1 = tf.get_variable('var1', shape=(10,10))
            ops1 = tf.get_variable('ops1', shape=(10))
            self.params += [var1, ops1]
        
        with tf.variable_scope('scope_2', initializer=tf.ones_initializer()):
            var2 = tf.get_variable('var2', shape=(5, 5))
            ops2 = tf.get_variable('ops2', shape=(5))
            self.params += [var2, ops2]
    def save(self):
            param = []
            for each in self.params:
                param.append(np.array(each.eval()))
            param = np.array(param)
            np.save('./a.npy', param)

	def load(self, sess, path='./a.npy'):
		mat = np.load(path)
        for ind, each in enumerate(self.params):
        	sess.run(self.params[ind].assign(mat[ind]))
```

在定义完相对应的图后，将参数张量添加到parameter列表中。在完全定义完graph后，在save方法里，将参数`eval()`成矩阵并且保存为一个npy文件，其形状为(4,)。然后在读取load方法中，只需要按序读取并且`assign()`参数值即可！这样读取参数就不用考虑命名空间的差异性了！




[ref_1]: http://blog.csdn.net/loseinvain/article/details/78241000