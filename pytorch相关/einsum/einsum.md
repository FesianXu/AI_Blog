**einsum**全称为Einstein summation convention，是一种求和的范式，在很多基于多维张量的张量运算库，如**numpy**,**tensorflow**,**pytorch**中都有所应用。einsum可以用一种很简单的，统一的方式去表示很多多维张量的运算。让我们以numpy中的einsum为例子，理解这种运算表达方式。

这里贴出numpy中的einsum的API：
```python
numpy.einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', optimize=False)
```
其中关键的参数有`subscripts`用于指定计算模式,`operands`用于指定操作数，我们给个例子，如果现在我们有两个矩阵
```python
A = np.array([[1,2,3],[1,3,4],[2,3,4]])
B = np.array([[9,2,4],[1,1,7],[5,2,4]])
'''
A -> array([[1, 2, 3],
           [1, 3, 4],
           [2, 3, 4]])
B -> array([[9, 2, 4],
           [1, 1, 7],
           [5, 2, 4]])
'''
```
如果我们现在想实现一个运算，如下公式所述:
$$
s(j) = \sum_{i=0}^{2} A[i,j]*B[i,j]
$$
我们利用einsum这种形式就能够很好的表达，如:
```python
s = np.einsum('ij,ij->j',A,B)
```
其输出结果为
```python
array([20, 13, 56])
```
其中的`subscripts`参数就很好地描述了上述公式描述的运算过程，我们这里可以细究下这个参数。这个参数由三大部分构成，`a,b->c`其中`a`和`b`是描述的输入张量的索引，如上面的`ij`表示A和B张量的`i`行`j`列。`c`表示的是输出的索引，如上文中的`j`。当你指定了输出的索引之后，就可以把这个索引看成是固定的值了，因为他将会是作为一个自变量参数存在的，而可以把其他的索引变量（输入的索引变量）看成是循环变量。这个方式可以实现很多复杂的矩阵运算，如
```python
a = np.arange(60.).reshape(3,4,5)
b = np.arange(24.).reshape(4,3,2)
np.einsum('ijk,jil->kl', a, b)
```












