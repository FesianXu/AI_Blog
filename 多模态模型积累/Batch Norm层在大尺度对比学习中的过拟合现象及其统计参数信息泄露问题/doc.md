<div align='center'>
    Batch Norm层在大尺度对比学习中的过拟合现象及其统计参数信息泄露问题
</div>


<div align='right'>
    FesianXu 20210830 at Baidu Search Team
</div>

# 前言

在之前的博文[1,2]中已经说明了在对比学习中提高batch size的巨大作用，然而在大尺度对比学习的训练过程中，被广泛实践证明有效的Batch Norm层则很容易出现过拟合的现象。笔者在本文对该现象进行笔记，并且纪录其解决方案。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：
**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]



----

在大尺度的对比学习训练过程中，Batch Norm层所造成的过拟合现象被众多公开论文所报道过，比如MoCo[3],  SimCLR[4]和其他一些工作[5]。之前笔者在MoCo的笔记中也简单谈到过这个问题[1]，然而当时尚未深究，现在在工作中实际遇到了这个问题，就权当笔记将其纪录。

在大尺度的对比学习中，一种常见的实践是：设置一个较大的batch size，比如4096，显然一张GPU卡很难塞下，特别是在多模态模型中，因此通过数据并行将大batch size均分到不同卡上，比如16张卡。在双塔模型中，我们需要对两个塔输出的特征进行计算得到打分矩阵，如Fig 1所示。然而分布在不同卡上的双塔特征$\mathbf{f}_i \in \mathbb{R}^{256 \times D}$​的批次大小为256，如果对每个卡上的双塔特征进行打分，那么得到的打分矩阵$\mathbf{S} \in \mathbb{R}^{256 \times 256}$​。可以看到此时多GPU并没有真正意义上增大batch size，只是通过数据并行的方式更快地遍历了训练数据集，提高了训练速度而已，我们期望中的打分矩阵应该是$\mathbf{S} \in \mathbb{R}^{4096 \times 4096}$​，在这种大尺度的打分矩阵上能更高效地获得足够好的负样本进行对比学习。

![batchneg][batchneg]

<div align='center'>
    <b>
        Fig 1. Batch Negative的方式从一个batch中构造负样本。
    </b>
</div>

一种实现这种方式的实践是通过各个深度学习框架提供的`all_gather`机制，比如`pytorch`中的`torch.distributed.all_gather()` [6]或者`paddle`中的`paddle.distributed.all_gather()` [7]。这些函数可以从所有的GPU中汇聚某个矩阵，比如特征矩阵$\mathbf{f}_i \in \mathbb{R}^{256 \times D}$，将$i=0,\cdots,15$所有的GPU中的特征矩阵进行汇聚，可以得到$\mathbf{f}_{gather} \in \mathbb{R}^{4096 \times D}$的汇聚特征矩阵。我们可以对这个$\mathbf{f}_{gather}$进行打分，然后进行对比学习的训练。

然而，这只是大尺度对比学习的第一步，我们发现在汇聚特征之前，特征的计算都是在各自的GPU中进行计算的，假如模型中具有Batch Norm层，那么其统计参数$\mu, \sigma^2$​​​​都是在各自的GPU中计算的（假设是异步Batch Norm机制），和Batch Norm有关的知识可见之前博文[8]。而由于BN层的统计参数和`all_gather`机制，会导致在大尺度对比学习训练过程中的严重过拟合现象。然而BN的统计参数导致的过拟合问题并不只在存在`all_gather`机制的对比学习模型中存在，注意到MoCo看成是维护了一个负样本队列[1]，因此可以视为不采用`all_gather`机制，也可在单卡上进行超大batch size的训练。然而MoCo也会遇到BN层的统计参数泄露信息的问题。且让笔者慢慢道来。

一般来说，提高负样本数量的方法有以下几种：

1. 端到端，此时通过`all_gather` 机制可以扩大batch size，进而扩大负样本数量。 **在这种方式下，负样本数量和batch size耦合**。
2. MoCo，这种方法通过负样本队列和动量更新保证了Query-Key编码器的状态一致性和足够大的负样本词表。 **在这种方式下，负样本数量和batch size解耦**。
3. Memory Bank [10]，此时通过维护一个负样本数据库，称之为memory bank进行，然而此时的Query-Key编码器不是一致状态的，Key编码器永远落后于Query编码器。

|             | 提高batch size的方式   | 提高负样本数量的方式 | batch size和负样本数量是否耦合 | Query-Key编码器状态一致性  | 正样本对中QK编码器是否状态一致 | 是否会遇到BN层统计参数泄露 |
| ----------- | ---------------------- | -------------------- | ------------------------------ | -------------------------- | ------------------------------ | -------------------------- |
| 端到端      | `all_gather`           | 通过提高batch size   | 是                             | 一致更新                   | 一致更新                       | 是                         |
| MoCo        | 一般无需提高batch size | 通过维护负样本队列   | 否                             | 一致更新                   | 一致更新                       | 否                         |
| Memory Bank | 一般无需提高batch size | 通过维护负样本队列   | 否                             | 不一致，Key永远落后于Query | 不一致，                       | 否                         |

我们留意到，并不是所有方法都会收到BN层信息泄露问题的，只有在（正样本对）Query-Key编码器**一致更新**的模型中才会遇到，而在Memory Bank中就不会遇到。其中，我们先讨论端到端的形式中的BN层信息泄露。注意，我们这里说的状态一致，或者一致更新，并不是指的数值上的一致，而是假使存在一个训练状态，这两者是同步的。

# 端到端模式的对比学习

![end2end][end2end]

<div align='center'>
  <b>
    Fig 2. 端到端模式更新的对比学习过程示意图。
  </b>
</div>

在端到端模式的对比学习过程中，Query-Key编码器是一致更新的，简单来说就是两个塔的参数在同个step中进行更新。此时如果采用了多卡进行`all_gather`，并且采用的BN层是异步BN（也就是每张卡的统计参数$\mu, \sigma^2$和学习参数$\gamma, \beta$​​​​​​​​​​​都是不同的，分别是独立计算的），那么可想而知，通过`all_gather`之后，其形成的打分矩阵如Fig 3所示。Fig 3中的不同颜色块表示来自于不同GPU上的正样本Query-Key对，省略号表示的是通过不同卡汇聚得到的特征进行打分。我们注意到由于异步BN的原因，不同颜色块上的统计参数是不相同的，而正样本显然又位于打分矩阵的对角线上，正样本都由同一个GPU进行计算，此时由`gather`得到的诸多负样本的统计参数会和同一个GPU下的正样本的统计参数存在明显差别。由BN的计算公式(1)可知，不仅通过学习表征，通过『学习』统计参数也可以『等价』于学习表征，让模型『预测』正样本的位置。然而，这种『等价』并不是真正的等价，而『预测』也不是通过真正学习表征得到的，因此表现为过拟合，严重影响模型的表征性能，这个情况笔者称之为BN层统计参数泄露。在这种情况下，由于统计参数泄露了『正样本所在于对角线』这个秘密，导致表征学习以失败告终。
$$
\begin{aligned}
\hat{x} &= \dfrac{x-\mu}{\sqrt{\sigma^2+\epsilon}} \\
y &= \gamma \hat{x} + \beta
\end{aligned}
\tag{1}
$$
![all_gather_stat_leak][all_gather_stat_leak]

<div align='center'>
  <b>
    Fig 3. 端到端模式下的all_gather将会导致BN层统计参数的泄露。
  </b>
</div>

从以上的分析来看，在端到端模式下导致统计参数泄露的本质还是在于**统计参数都是在各自的GPU中进行计算**的，那么解决方案自然要从这里着手。在simCLR[4]中，作者提出的方案是采用所谓的Global BN，其方法就是同样`gather`不同GPU上的统计参数，然后计算出一个新的统计参数后分发到所有GPU上，此时所有GPU的统计参数都是相同，也就谈不上泄露了。当然你还可以用更简单的方法，比如在[5]中，作者采用Layer Norm取代了Batch Norm。从Fig 4.中可以看出，Layer Norm进行统计参数计算的维度是`[Feature, Channel]`，而不涉及`Batch`维度，统计参数不会跨Batch使得统计参数不会泄露样本之间的信息。

这个方法相当地直观，因为最理想的情况下，我们应该对所有的正负样本一个个地送到编码器中，以达到完全隔离不同样本之间的目的，通过将BN替换成LN，达到了这个目的。

![layer_norm][layer_norm]

<div align='center'>
  <b>
    Fig 4. Layer Norm的统计参数不会跨batch内的样本，因此不会泄露统计参数。
  </b>
</div>



# MoCo

还有一种非常火的实践是何凯明大佬的MoCo[3,1]，这种方式不仅需要维护一个大尺度的负样本队列，还需要用动量更新的方式去一致更新Query-Key编码器，如Fig 5.所示。此时的负样本数量的提升不是由于`all_gather`机制导致的，并且负样本数量和batch size也是解耦了。因此这种情况下，我们认为即便是单GPU也可以跑很大的负样本数量的对比学习。那么此时在上文所说的BN层统计参数泄露问题在MoCo中存在吗？

![moco][moco]

<div align='center'>
  <b>
    Fig 5. MoCo的QK编码器更新由于采用了动量更新，因此是状态一致的。
  </b>
</div>

很遗憾，即便在MoCo中，BN层的统计参数泄露还是存在的，但是原因显然不是由于多GPU的异步BN统计参数导致的，因为即便只有一张卡也可以理论上跑MoCo。如Fig 6.所示，此时正样本打分的计算如Code 1.所示，是通过对某个样本进行数据增广后，将其视为正样本，再进行打分。这个部分也就是Fig 6.的蓝色部分。而负样本是直接采用Key-负样本队列中的特征，直接和Queryt特征进行打分，如Fig 6.的绿色部分所示。注意到代码中的`k = k.detach()`, 这意味着构造出来的正样本的梯度流只会更新Query编码器，而Key编码器是通过动量更新的。

<div align='center'>
  <b>
    Code 1. MoCo中的正样本打分计算。
  </b>
</div>

```python
f_k.params = f_q.params # initialize
for x in loader: # load a minibatch x with N samples
    x_q = aug(x) # a randomly augmented version
    x_k = aug(x) # another randomly augmented version
    q = f_q.forward(x_q) # queries: NxC
    k = f_k.forward(x_k) # keys: NxC
    k = k.detach() # no gradient to keys
    # positive logits: Nx1
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1))
    # 正样本打分计算
    l_neg = mm(q.view(N,C), queue.view(C,K))
    # 负样本打分计算
    # logits: Nx(1+K)
    logits = cat([l_pos, l_neg], dim=1)
    labels = zeros(N) # positives are the 0-th
    loss = CrossEntropyLoss(logits/t, labels)
    loss.backward()
    update(f_q.params)
    enqueue(queue, k) # enqueue the current minibatch
    dequeue(queue) # dequeue the earliest minibatch
```

![moco_samples_organization][moco_samples_organization]

<div align='center'>
  <b>
    Fig 6. 在MoCo中采用组织超大规模负样本队列的方式进行负样本组织，因此batch size可以不需要很大，并且理论上可在单卡中完成计算。
  </b>
</div>

即便在MoCo中不通过正样本构造的`k`进行key编码器的更新，但是同个step中，将会通过式子(2)进行动量更新，因此QK编码器的更新也是一致的。
$$
\theta_k \leftarrow m\theta_k + (1-m)\theta_q
\tag{2}
$$
注意到此时的正样本是在同一状态下的Query-Key编码器中被编码的，如Fig 7.所示，其中的正样本Query和正样本Key的统计特征是整个batch内共享的，分别是$\mu_{Q}, \sigma_{Q}^2$和$\mu_{K+}, \sigma_{K+}^2$​​​​​​。这意味着整个batch内的所有正样本都共享着相同的统计特征，而且两者的状态保持一致，此时的统计特征将会很容易泄露正样本的『位置』。笔者理解是，此时因为统计特征都一样，而且正样本的位置都在第一列，那么模型会尝试『欺骗』整个训练任务，通过更简单地『学习』统计特征的分布进而『预测』正样本的位置。毕竟统计特征的维度更小，而表征的维度高得多，因此在某些数据集上可能就会出现BN层的统计参数泄露的问题。

![score][score]



<div align='center'>
  <b>
    Fig 7. MoCo中的正样本对打分。
  </b>
</div>

因此，虽然MoCo和端到端的方式中数据组成方式大相径庭，但是BN层的统计特征在QK编码器一致更新的过程中，都存在泄露正样本的位置的可能。为了解决这个问题，在MoCo中采用的是Shuffling BN，分为以下几个步骤，注意Shuffling BN必须运行在多卡环境下：

1. 将输入进行`all_gather`，并且进行随机打乱，此时需要记下打乱后的索引`unshuffle_idx`，因为最后需要『反打乱』回一开始的样本顺序。
2. 将打乱后的样本平均分发到$M$​​​个GPU上，每个GPU上的batch size大小为$N/M$。
3. 通过Key编码器计算特征$\mathbf{f}_i$，注意到Key编码器中存在BN层，比如经典的`resnet`结构。
4. 将所有GPU上计算得到的特征$\mathbf{f}_i$进行`all_gather`，并且通过`unshuffle_idx`进行反打乱回原来的样本顺序。

整个逻辑的示意图如Fig 8所示。整个过程其实就是通过打乱和分发到多个GPU，实现统计参数的打乱，解耦统计参数和正样本位置的关联。Shuffling BN通过打乱实现随机，而Global BN通过`gather->分发`的方式实现统计参数的全局统一。一个是『打乱』一个是统一，这些手段都保证了BN的统计参数不会带有正样本的位置信息。

![shuffle_bn][shuffle_bn]

<div align='center'>
  <b>
    Fig 8. MoCo中的Shuffling BN逻辑示意图。
  </b>
</div>



# Memory Bank



![memory_bank][memory_bank] 

<div align='center'>
  <b>
    Fig 9. Memory Bank的示意图。
  </b>
</div>

然而，在memory bank [10, 12]中却并不会出现BN层统计参数泄露的问题，那是因为memory bank是通过『异步』的方式取正样本key的。具体而言，如Fig 9.所示memory bank维护了一个负样本队列$\mathcal{V}$​，假如负样本词表大小为10000，而输出的embedding特征维度为128，那么模型一开始将会初始化一个$10000\times128$​的随机矩阵作为负样本的初始值，记作 $\mathcal{V} = \{\mathbf{v}_i\}, \mathbf{v}_i \in \mathbb{R}^{128}, i=1,\cdots,10000$​。假如编码器为$\mathbf{v}_i = f_{\theta}(\mathbf{x}_i)$​将对第$i$​​个样本$\mathbf{x}_i$​进行编码，输出$\mathbf{v}_i$​​。在[10]这篇文章中，作者将参数化的交叉熵损失(3)替换成了非参数化形式的交叉熵损失(4)，原因很简单，在这种对比学习场景中，一个样本$x_i$除了和自己可以看成是正样本外，其他的样本$x_j, j \neq i$​都视为负样本。这意味着当训练数据足够大时（通常对比学习的数据集都可以非常大），将有着数不胜数的负样本，此时如果交叉熵损失是参数化的形式，那么权值矩阵$\mathbf{w}$​将变得大到无法计算。
$$
P(i|\mathbf{v}) = \dfrac{\exp(\mathbf{w}_i^{\mathrm{T}}\mathbf{v} / \tau)}{\sum_{j=1}^{n} \exp(\mathbf{w}_{j}^{\mathrm{T}}\mathbf{v} / \tau)}
\tag{3}
$$

$$
P(i|\mathbf{v}) = \dfrac{\exp(\mathbf{v}_{i}^{\mathrm{T}} \mathbf{v} / \tau)}{\sum_{j=1}^{n} \exp(\mathbf{v}_{j}^{\mathrm{T}} \mathbf{v} / \tau)}
\tag{4}
$$

而我们如果采用相关性计算的方式计算样本之间的距离，那么就可以形成如式子(4)所示的非参数化的交叉熵损失（其中的$\tau$是温度系数，通常可以设置为常数），通过NCE（Noise Contrastive Estimation）可以进一步减少负样本太多造成的计算负担。

因此，按照这个逻辑，此时我们计算正样本的相关性时候，采用式子(5)
$$
P(s_{+}|\mathbf{v}) = \dfrac{\exp(\mathbf{v}_{i+}^{\mathrm{T}} \mathbf{v} / \tau)}{\exp(\mathbf{v}_{i+}^{\mathrm{T}} \mathbf{v} / \tau)+\sum_{j\in \{z|z=1,\cdots,n, z \neq i+\} } \exp(\mathbf{v}_{z}^{\mathrm{T}}\mathbf{v} / \tau)} 
\tag{5}
$$
此时的$i+$表示输入样本$\mathbf{x}_i$的索引$i$在$\mathcal{V}$ 中取到的第$i$​个特征​。是的，在memory bank中，除了当前输入的特征$\mathbf{v}$​​​之外，其他的正样本Key亦或是负样本Key都是从memory bank中取得的。而memory bank中的每一个样本表征都是从之前的编码器计算后插入得到的，这意味着无论是正样本Key还是负样本Key的更新状态都是落后于当前的编码器的。因此正样本Key的统计特征早已落后于当前编码器中的BN层统计特征了，此时如果模型还想通过统计特征去『欺骗』训练过程将变得很困难。



![memory_bank_non_param_ce][memory_bank_non_param_ce]

<div align='center'>
  <b>
    Fig 10. 非参数化后的交叉熵损失的正样本Key和负样本Key都来自于memory bank，而memory bank的状态是落后于当前的编码器的。
  </b>
</div>




# 说在最后

对于对比学习的研究，在学校中，我们没有足够的计算资源和超大型的数据量，因此比较难发现在大batch size甚至是超大batch size下才会出现的BN层统计参数泄露问题。目前在学术界有报道这个问题的文章据笔者了解也就[3,4,5]这几篇。然而在公司的实践中，在面对数以亿计的大量数据时，简单粗暴地提高batch size将导致意料外的结果，因此笔者将其进行笔记，希望对读者有所帮助。此外，以上结论并不一定在所有数据集上都成立，我们发现数据集的特性也很重要，如果读者在相同实践中也遇到了对比学习的过拟合问题，不妨也可以往着BN层统计参数泄露这方面考虑。



# Reference

[1]. https://fesian.blog.csdn.net/article/details/119515146

[2]. https://fesian.blog.csdn.net/article/details/119516894

[3]. He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9729-9738).

[4]. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020, November). A simple framework for contrastive learning of visual representations. In *International conference on machine learning* (pp. 1597-1607). PMLR.

[5]. Hénaff, O. J., Razavi, A., Doersch, C., Eslami, S., and Oord, A.v. d. Data-efficient image recognition with contrastive predictive coding. arXiv preprint arXiv:1905.09272, 2019.  

[6]. https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather

[7]. https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/all_gather_cn.html

[8]. https://blog.csdn.net/LoseInVain/article/details/86476010

[9]. [Memory Bank Code: https://github.com/zhirongw/lemniscate.pytorch](https://github.com/zhirongw/lemniscate.pytorch)

[10]. Wu, Z., Xiong, Y., Yu, S. X., & Lin, D. (2018). Unsupervised feature learning via non-parametric instance discrimination. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3733-3742).

[11]. https://github.com/facebookresearch/moco/blob/master/moco/builder.py

[12]. https://github.com/zhirongw/lemniscate.pytorch









[qrcode]: ./imgs/qrcode.jpg
[batchneg]: ./imgs/batchneg.png
[end2end]: ./imgs/end2end.png
[all_gather_stat_leak]: ./imgs/all_gather_stat_leak.png
[layer_norm]: ./imgs/layer_norm.png
[moco]: ./imgs/moco.png
[moco_samples_organization]: ./imgs/moco_samples_organization.png

[score]: ./imgs/score.png
[shuffle_bn]: ./imgs/shuffle_bn.png

[memory_bank_non_param_ce]: ./imgs/memory_bank_non_param_ce.png
[memory_bank]: ./imgs/memory_bank.png

