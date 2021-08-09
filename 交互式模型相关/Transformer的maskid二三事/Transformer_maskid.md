<div align='center'>
    Transformer的mask id两三事
</div>

<div align='right'>
    FesianXu 20210808 at Baidu Search Team
</div>

# 前言

在Transformer中有着诸多的id，比如token id，position id，segment id，mask id等等，本文简单纪录下笔者在使用mask id时候的一些问题。**如有谬误请联系指出，本文遵守[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

----

有一段时间笔者利用Transformer实现一个功能，输入有三种模态，分别是$A,B,C$，其中预训练过程中需要对$A$或者$B$进行动态地掩膜，这个过程中，笔者想到通过mask id进行实现。我们从之前的博文[1]中知道，自注意力机制的计算公式可表示为(1)所示
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\dfrac{QK^{\mathrm{T}}}{\sqrt{d_k}}+\mathrm{attn\_bias}) V
\tag{1}
$$
其中的$\mathrm{attn\_bias}=(1-\mathbf{m}\mathbf{m}^{\mathrm{T}})*(-10000)$，$\mathbf{m}\in\mathbb{R}^{M \times 1}$是注意力掩膜（mask id），其作用用于屏蔽`[PAD]`字符的影响，具体细节可见博文[1]。如果把mask id也考虑进来，那么如Fig 1所示，一共有四种类型的id。

![input_decom][input_decom]

<div align='center'>
    <b>
        Fig 1. Transformer的输入通常由Token id，Segment id和Position id，mask id组成。
    </b>
</div>

笔者当时的推导是，如果将指定模态输入的mask id置为0，比如B模态的，那么Transformer计算过程中就应该会不考虑这一段的输入了吧。推导过程很简单，在$m=0$的时候，$\mathrm{attn\_bias=-10000}$，那么如式子(1)所示，$\dfrac{QK^{\mathrm{T}}}{\sqrt{d_k}}+\mathrm{attn\_bias}$的计算的值就会非常接近于$-10000$，因此$\mathrm{Attention(Q,K,V)}$输出接近为0，因此该部分就不会被计算到。笔者在实验中也进行了实验验证，做法是这样的：固定预训练模型参数，将模型设置为`eval`模式，固定A和C模态输入，并且输入不同的B，发现最后的运算输出没有差别。

# Reference

[1]. https://fesian.blog.csdn.net/article/details/116137177



[input_decom]: ./imgs/input_decom.png

