∇ 联系方式：

**e-mail**:  FesianXu@gmail.com

**github**:  https://github.com/FesianXu

**知乎专栏**:  计算机视觉/计算机图形理论与应用

**微信公众号**：
![qrcode][qrcode]


----

在Transformer [1]中作者提出了用自注意力取代CNN，RNN在序列建模中的作用，并且取得了显著的实验效果，对整个NLP，CV领域有着深远影响。然而自注意力机制的时间复杂度是$\mathcal{O}(n^2)$的，如式子(1)所示
$$
\mathrm{Attention}(QW^Q,KW^K,VW^V) = \mathrm{softmax}(\dfrac{QW^Q(KW^K)^{\mathrm{T}}}{\sqrt{d_k}}) VW^V
\tag{1}
$$
显然有$QW^Q \in \mathbb{R}^{n \times d_{model}}$，$KW^K \in \mathbb{R}^{n \times d_{model}}$， $VW^V \in \mathbb{R}^{n \times d_{model}}$，其中$n$是序列长度，$d_{model}$是隐层维度，那么显然这里的`softmax()`内的计算将会是$\mathcal{O}(n^2)$时间复杂度的。 这个称之为 **密集自注意力** (dense self-attention)。这个复杂度对于长文本来说很不友好。

在论文[2]中，作者证明了密集自注意力是所谓低秩（low-rank）的，意味着可以用更小的矩阵去表征这个$n \times n$大小的自注意力矩阵，从而达到减少复杂度的目的。作者的方法很简单，如Fig 1所示，在Q和K的后续添加两个`Projection`单元，将序列长度`n`映射到低维的`k`，作者将这种单元称之为`Linformer`。
![linformer][linformer]

<div align='center'>
<b>
 Fig 1. Linformer模型的自注意力单元只是比传统的Transformer多了俩Projection单元。
</b>
</div>

公式也很简单，如式子(2)所示
$$
\mathrm{Attention}(QW^Q,EKW^K,FVW^V) = \\
\mathrm{softmax}(\dfrac{QW^Q(EKW^K)^{\mathrm{T}}}{\sqrt{d_k}}) FVW^V
\tag{2}
$$
其中的$E \in \mathbb{R}^{k \times n}, F \in \mathbb{R}^{k \times n}$，理论上，因为$k$是常数，那么时间复杂度是$\mathcal{O}(n)$，降秩的映射也如下图所示。
![lowrank_project][lowrank_project]

这种做法在序列长度非常长的时候，会有很大的提速效果，如下图实验所示。因此适合于长文本序列。
![result][result]
从本质来说，我们可以从式子(2)得到式子(3)，我们发现$QW^Q(KW^K)^{\mathrm{T}}$和传统的Transformer没有一点区别，而只是后面多了个$E^{\mathrm{T}} \in \mathbb{R}^{n \times k}$，这个映射操作相当于对前面的$n \times n$自注意力矩阵进行池化(pooling)到$n \times k$。因此个人感觉，这个网络也许并不是很必要，如果我们能考虑对长文本进行分句，然后计算特征后用某种方式将所有分句的特征融合起来，理论上也不需要用`Linformer`。
$$
\mathrm{Attention}(QW^Q,EKW^K,FVW^V) = \\
\mathrm{softmax}(\dfrac{QW^Q(KW^K)^{\mathrm{T}} E^{\mathrm{T}}}{\sqrt{d_k}}) FVW^V
\tag{3}
$$






# Reference
[1]. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017

[2]. Wang, Sinong, Belinda Li, Madian Khabsa, Han Fang, and Hao Ma. "Linformer: Self-attention with linear complexity." arXiv preprint arXiv:2006.04768 (2020).


[qrcode]: ./imgs/qrcode.png

[linformer]: ./imgs/linformer.png
[lowrank_project]: ./imgs/lowrank_project.png
[result]: ./imgs/result.png

