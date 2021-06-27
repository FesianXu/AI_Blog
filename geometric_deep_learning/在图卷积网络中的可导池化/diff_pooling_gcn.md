<div align='center'>
    在图卷积网络中的可导池化操作
</div>

<div align='right'>
    FesianXu 20210627 at Baidu search team 
</div>

# 前言

我们在之前的博文[1,2,3]中初步讨论过图卷积网络的推导和信息传递的本质等，本文继续讨论在图卷积网络中的可导池化操作。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。
$\nabla$ 联系方式：
**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

----

这一篇鸽了很久了，今天突然想到就顺手写完了吧。之前我们在[1,2,3]中曾经讨论过图卷积网络的推导，以及其和消息传递（message passing）之间的关系，但是我们还没有讨论一个重要问题，那就是在图卷积网络中的池化（pooling）操作。池化操作对于一个卷积网络来说是很重要的，特别是对于节点众多的大规模图卷积网络，池化可以使得网络的参数大幅度减少，增强泛化性能并且提高模型的层次性结构化特征性能等。如何在图卷积网络中定义出如同在卷积网络中的可导的池化操作呢？单纯的聚类操作因为缺乏梯度流，不能实现端到端的训练而不能直接使用，在文章[4]中提出了DiffPool算子，该算子可以实现图卷积网络的可导池化。

![pool][pool]

<div align='center'>
    <b>
        Fig 1. 对于卷积网络中的池化操作，要怎么才能在图卷积网络中找到其合适的替代品呢?
    </b>
</div>

# DiffPool

DiffPool的思路很简单，可以用Fig 2表示，其中的$\mathbf{X}^{l} \in \mathbb{R}^{n^{l} \times d}$是上一层的输出特征，而$n^{l}$表示第$l$层的节点数。其中的DiffPool操作其实很简单，就是用一个分配矩阵（assign matrix）去进行自动聚类，有：
$$
\begin{aligned}
\mathbf{X}^{(l+1)} &= (S^{l})^{\mathrm{T}} Z^{l} \in \mathbb{R}^{n^{(l+1)} \times d} \\
A^{(l+1)} &= (S^{l})^{\mathrm{T}} A^{l} S^{l} \in \mathbb{R}^{n^{(l+1)} \times n^{(l+1)}}
\end{aligned}
\tag{1}
$$
其中的$S^{l} \in \mathbb{R}^{n^{l} \times n^{(l+1)}}$就是第$l$层的分配矩阵，注意到其是一个实矩阵。

![diffpool][diffpool]

<div align='center'>
    <b>
        Fig 2. DiffPool的示意简图。
    </b>
</div>

现在的问题在于分配矩阵如何学习得到，可以认为DiffPool是一个自动端到端聚类的过程，其中分配矩阵代表了该层聚类的结果。如Fig 2所示，我们发现第$l$层的分配矩阵和特征都是由共同输入$\mathbf{X}^{l}$学习得到的，我们有：
$$
\begin{aligned}
Z^{l} &= \mathrm{GNN_{l,emb}} (A^l, \mathbf{X}^l) \\
S^l &= \mathrm{softmax}(\mathrm{GNN_{l,pool}}(A^l, \mathbf{X}^l))
\end{aligned}
\tag{2}
$$
其中的$\mathrm{GNN_{xxx}}()$表示的是由图卷积单元层叠若干次而成的卷积模块，其中每一层可以表示为
$$
H^{(k)} = M(A, H^{(k-1)}; W^{(k)}) = \mathrm{ReLU}(\tilde{D}^{\frac{1}{2}} \tilde{A} \tilde{D}^{\frac{1}{2}} H^{(k-1)} W^{(k-1)})
\tag{3}
$$
其中的$\mathrm{ReLU}(...)$表示的是经典的消息传递过程，具体见[3]。注意到$S^l$的形状决定了下一层的节点数$n^{(n+1)}$，参考公式(3)，这个超参数由$W^{(k-1)} \in \mathbb{R}^{d \times n^{(l+1)}}$指定，而显然有$\tilde{D}^{\frac{1}{2}} \tilde{A} \tilde{D}^{\frac{1}{2}} \in \mathbb{R}^{n^{l} \times n^{l}}, H^{(k-1)} \in \mathbb{R}^{n^{l} \times d}$。为了约束分配矩阵的值的范围，对其进行了概率分布化，也即是$\mathrm{softmax}(\cdot)$，按论文的说法，是逐行（row-wise）生效的。

在$\mathrm{GNN_{l,emb}}(\cdot)$中则负责特征$Z^l$的生成，再与分配矩阵$S^l$进行DiffPool，见式子(1)，即完成了整个操作。



# 辅助训练目标

然而据文章说，在实践中，单纯依靠梯度流去训练可导池化版本的GNN难以收敛，需要加些辅助约束条件。作者加了几个先验约束，第一作者认为 **一个节点邻居的节点应该尽可能地池化到一起** (nearby nodes should be pooled together)，通过Frobenius 范数进行约束，有式子(4)
$$
L_{LP} = ||A^l -S^{l}(S^{l})^{\mathrm{T}}||_F
\tag{4}
$$
另一个约束是，分配矩阵的应该每一行尽可能是一个one-hot向量，这样每个聚类结果才能更清晰地被定义出来。通过最小化熵可以对其进行约束，有：
$$
L_E = \dfrac{1}{n} \sum_{i=1}^n H(S_i)
\tag{5}
$$
其中$H(S_i)$表示对$S^l$的第$i$行求熵（entropy）。作者声称在图分类损失中添加(4)和(5)约束可以有着更好的性能，即便训练收敛需要更长的时间才能达到。从结果Fig 3中可以发现的确是添加了约束的效果要好些。其中在GraphSAGE的基线上，和其他池化方法（SET2SET，SORTPOOL）的对比说明了DiffPool的有效性和先进性。

![result][result]

<div align='center'>
    <b>
        Fig 3. 实验结果图。
    </b>
</div>

# More

那么DiffPool得到的分配矩阵结果是否可靠呢？是否可以看成是聚类的结果呢？作者在原文中也提及了这件事儿，并且对池化结果进行了可视化，如Fig 4所示。可以发现DiffPool其的确是对节点进行了合理的聚类。

![vis][vis]

<div align='center'>
    <b>
        Fig 4. DiffPool结果的可视化，可以形成合理的聚类结果。
    </b>
</div>

就笔者个人的读后感而已，DiffPool的操作类似于现在流行的自注意学习机制，分配矩阵不妨可以看成是自注意力矩阵对节点进行聚类，也可以认为自注意力机制在图网络中也是生效的。

# Reference

[1]. https://fesian.blog.csdn.net/article/details/88373506

[2]. https://blog.csdn.net/LoseInVain/article/details/90171863

[3]. https://blog.csdn.net/LoseInVain/article/details/90348807

[4]. Ying, Rex, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, and Jure Leskovec. "Hierarchical graph representation learning with differentiable pooling." *arXiv preprint arXiv:1806.08804* (2018).



[qrcode]: ./imgs/qrcode.jpg
[pool]: ./imgs/pool.png
[diffpool]: ./imgs/diffpool.png
[result]: ./imgs/result.png
[vis]: ./imgs/vis.png



