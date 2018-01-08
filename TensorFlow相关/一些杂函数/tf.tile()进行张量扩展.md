# tf.tile()进行张量扩展

`tf.tile()`应用于需要张量扩展的场景，具体说来就是：
如果现有一个形状如[`width`, `height`]的张量，需要得到一个基于原张量的，形状如[`batch_size`,`width`,`height`]的张量，其中每一个batch的内容都和原张量一模一样。`tf.tile`使用方法如：
```python
tile(
    input,
    multiples,
    name=None
)
```
其中输出将会重复input输入multiples次。例子如：
```python
import tensorflow as tf

raw = tf.Variable(tf.random_normal(shape=(1, 3, 2)))
multi = tf.tile(raw, multiples=[2, 1, 1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(raw.eval())
    print('-----------------------------')
    print(sess.run(multi))
    
```
输出如：
```python
[[[-0.50027871 -0.48475555]
  [-0.52617502 -0.2396145 ]
  [ 1.74173343 -0.20627949]]]
-----------------------------
[[[-0.50027871 -0.48475555]
  [-0.52617502 -0.2396145 ]
  [ 1.74173343 -0.20627949]]

 [[-0.50027871 -0.48475555]
  [-0.52617502 -0.2396145 ]
  [ 1.74173343 -0.20627949]]]
```

可见，multi重复了raw的0 axes两次，1和2 axes不变。




