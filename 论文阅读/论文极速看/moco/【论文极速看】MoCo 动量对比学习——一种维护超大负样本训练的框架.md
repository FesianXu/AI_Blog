<div align='center'>
MoCo 动量对比学习——一种维护超大负样本训练的框架
</div>
<div align='right'>
FesianXu  20210803 at Baidu Search Team
</div>

# 前言

在拥有着海量数据的大型互联网公司中，对比学习变得逐渐流行起来，大家都拿它进行表征学习的探索。本文对MoCo这篇论文进行笔记，希望对读者有所帮助。

**如有谬误请联系指出，本文遵守[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

e-mail: FesianXu@gmail.com

github: https://github.com/FesianXu

知乎专栏: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

微信公众号：

![qrcode][qrcode]

----

对比学习(Contrastive Learning)是最近在表征学习(Representation Learning)中非常流行的学习范式。对比学习不同于一般的分类模型，考虑对数据进行多个类别的分类，在对比学习中，我们通常需要考虑的只有两类：正样本和负样本。举个实际应用系统的例子，在搜索系统中的QU相关性中[2]，我们通常需要判断一个Query和一个Doc之间的相关性，如果相关的我们认为是正样本，而不相关的我们认为是负样本。目前很多模型都可以基于对比学习设计预训练任务，并且得到非常可观的性能提升，比如CLIP模型[3]。对于存有着海量用户行为数据（比如用户搜索的点击数据，浏览数据，电商的购买数据，停留时间等）的大型互联网公司而言，在对比学习这个方向有着更大的探索需求。

“正样本很多是正得相似，但是负样本却负得各有不同”，这句话虽然不太准确，但是描述了对比学习中的本质困难：如何去寻找合适的，更多的负样本去进行训练，从而让模型学习出足够能区分正负样本的特征？在Kaiming大佬的MoCo[1]这篇工作中，大佬给出了他的解决方案。

# 对比学习中的海量负样本需求

在对比学习中，对负样本有着海量的需求。在搜索场景中，我们会想尽办法去构造尽可能多的负样本。比如将展现了但是并没有被用户点击的样本当作是负样本，比如将同为点击了的样本混搭组成batch negative负样本。如Fig 1.1所示，通过对Query的特征和Doc的特征进行组合，如式子(1.1)所示，可以构建出打分矩阵$\mathbf{S} \in \mathbb{R}^{N \times N}$，其中$N$为batch size，$D$为特征大小。
$$
\mathbf{S} = \mathbf{Q} \times \mathbf{U} \in \mathbb{R}^{N \times N} \\
\mathbf{Q} \in \mathbb{R}^{N \times D} \\
\mathbf{U} \in \mathbb{R}^{N \times D}
\tag{1.1}
$$
那么可知道只有对角线的打分才是正样本，而其他绿色部分的打分$Q_i \cdot U_j, i \neq j$则是负样本打分，如果负样本的打分越高，那么就可以认为是难负样本。通过batch negative从一个batch内构造的负样本，因为这些Doc都是被用户点击过的，而且Query都是被搜索过的，通过对他们进行混搭构成的负样本足够“真实”。而通过无点数据构成的负样本则可能因为是展现的Doc过多，已经满足了用户的需求而没有被点击，因此可能出现将正样本误分为负样本的情况。当然，这个和具体的业务有关。

![batchneg][batchneg]

<div align='center'>
    <b>
        Fig 1.1 Batch Negative的方式从一个batch中构造负样本。
    </b>
</div>

不管怎么说，通过以上构造负样本的方式，我们有能力构造足够多的负样本。特别是通过batch negative的方式，在batch size为$N$的情况下我们能构造出$N^2-N$的负样本，其中很可能就会出现足够“好”的负样本让模型学习的好的表征。但是问题在于，我们受限于显存的大小，我们无法无限增大batch size，即便通过多机多卡分布式的方法，每张卡上的batch size大小也是有极限的，即便通过某些机制去汇总多卡之间的负样本打分，在汇总阶段也会遇到瓶颈。因此目前能看到的公开工作中，据笔者了解，batch size最大的就是CLIP[3]，开到了32,768。然而如果继续增大呢？可能就会遇到无法逾越的硬件瓶颈了。

Kaiming大佬在MoCo中，认为对比学习可以从某种形式看成是“字典查询”（Dictionary look-up），这一点不难理解，我们的正样本可以认为是分布在诸多的负样本之中，而输入一个Query，那么我们期望能匹配到正确的Doc（也是字典中的Key）。一次理想的匹配应该是：query尽可能相似于正样本的Key，而和其他负样本的Key尽可能不相似。从这个角度上看，我们能发现对比学习的一个根本难题在于：

1. 如何提高这个字典的词表大小。
2. 如何提高字典的连续性（consistency）。

在NLP任务中，特别是word2vec这种在所有有限词表中选取负样本的方法，因为语言的词表显然是个离散有限集合，可能也就两万来个Key，因此对比学习在NLP中可能更好地进行。而对于视觉任务而言，因为视觉信息是连续的，很难将其划分为离散的有限Key集合将其覆盖。因此视觉任务的对比学习会比NLP更难。

让我们再总结下，对比学习的两个难题，第一个其实就可以任务是如何去增大负样本的数量；第二个其实可以认为是如何稳定Query和Key的编码器的一致性 （为了表述的通用性，我们将会用Key表示匹配任务中的Doc，Item端）。要知道，Query端和Key端的编码器可以是一样的，也可以是不一样的；可以是共享参数的，也可以是独立参数的。因此在参数更新过程中，要保证两端编码器状态的一致性。 （本文考虑的是双塔模型，因此交互式模型暂时不考虑。）

我们下面对已有的一些解决负样本数量的方法进行概述，以便于了解提出MoCo的历史渊源。

# 端到端方式

最为传统的提高负样本（也就是词典的词表）的方式就是提高Batch size大小，然后在更新参数阶段同时对Query和Key的编码器进行更新。这种方式可以最大程度上保证Query和Key编码器的一致性，但是显然会遇到我们之前讨论过的硬件瓶颈。

![end2end][end2end]

<div align='center'>
    <b>
        Fig 1.2 通过端到端的方式进行对比学习。
    </b>
</div>

# Memory Bank

一种直接的想法是将词典词表大小和batch大小进行解耦，从而使得词表不受限于batch大小。在[4]中，作者提出Memory Bank框架，该框架的主要特点在于维护一个队列作为负样本集合。在Memory Bank框架中，首先需要对数据集所有数据进行刷特征向量，然后在每个mini batch中从中随机挑选负样本作为词典，但是并不进行梯度回传，也就是不会对Key编码器进行参数更新。当然，为了优化特征，会对Query编码器进行参数更新，然后将当前Query编码器的特征入到负样本队列，并且将负样本队列中的旧特征出队，从而实现对负样本队列的更新。这个负样本队列称之为Memory Bank。

然而这个过程中，虽然这个方法满足了“词典词表足够大”的条件，但是“QK编码器一致性”没有保证，因为Key编码总是落后于Query编码的，这个过程中无法保证Query-Key匹配的一致性。可以认为Memory Bank提出的编码器更新方式是“异步”逐渐更新的。

![memory_bank][memory_bank]

<div align='center'>
    <b>
        Fig 1.3 Memory Bank框架示意图，主要通过维护一个负样本队列实现大尺度范围的负样本训练。
    </b>
</div>

# MoCo

kaiming大佬提出MoCo解决了以上的问题。首先进行符号化表达，考虑一个Query $q$和一系列编码后的样本$\{k_0, k_1, \cdots,k_n\}$，这些样本是词典的key，假设这个词典中只有唯一的$k_{+}$和$q$匹配，那么在对比学习中，只有$q$和$k_{+}$匹配的时候，损失函数才是最低的。利用infoNCE (info Noise Contrastive Estimate)损失，我们可以对这个过程建模为(1.2)所示。
$$
\mathcal{L}_{q} = -\log{\dfrac{\exp(q \cdot k_{+} / \tau)}{\sum_{i=0}^K \exp(q \cdot k_i / \tau)}}
\tag{1.2}
$$
其中的$\tau$是温度系数，而$K$是负样本个数。我们用$f_q(\cdot)$表示query编码器，$f_{k}(\cdot)$表示key编码器，那么有$q=f_{q}(x^q), k=f_k(x^k)$，用$\theta_q, \theta_k$分别表示这两个编码器的参数。

![moco][moco]

<div align='center'>
    <b>
        Fig 1.4 MoCo框架示意图。
    </b>
</div>

为了保持大规模的词典词表大小，和Memory Bank一样需要保持一个负样本队列，不同点在于对Key编码器的更新方式不再采用“异步”的方式，而是采用了“同步”的方式。一种最为朴素的同步更新方式是：直接将当前step更新好的Query编码器的参数拷贝到Key编码器中（当然前提是QK编码器结构一致），如式子(1.3)所示。然而这种方式将会导致Key编码器训练状态的完全紊乱，因此无法训练将会一直震荡，那么通过引入动量进行QK状态协调就是一个自然而然的方案，如式子(1.4)所示。
$$
\theta_k \leftarrow \theta_q
\tag{1.3}
$$

$$
\theta_k \leftarrow m\theta_k +(1-m)\theta_q 
\tag{1.4}
$$

引入动量，将Key编码器的参数更新，以一种合适地方式引入了Query编码器的状态。此处的$m$通常是一个比较大的值，比如0.99。从Fig 1.5中，我们可以看到不同大小的动量对结果的影响，其中0.999是最为合适的。这也告诉我们，保持QK一致性很重要，同时保持Key编码器训练的一致性也很重要。

![momentum_ablation][momentum_ablation]

<div align='center'>
    <b>
        Fig 1.5 不同大小的动量大小对结果的影响。
    </b>
</div>

作者在文章中还以伪代码的形式举了一个应用例子，如Alg 1所示。

<div align='center'>
    <b>
        Alg 1. 以PyTorch的方式写的MoCo流程伪代码。
    </b>
</div>

```python
# f_q, f_k: encoder networks for query and key
# queue: dictionary as a queue of K keys (CxK)
# m: momentum
# t: temperature
f_k.params = f_q.params # initialize
for x in loader: # load a minibatch x with N samples
    x_q = aug(x) # a randomly augmented version
    x_k = aug(x) # another randomly augmented version
    q = f_q.forward(x_q) # queries: NxC
    k = f_k.forward(x_k) # keys: NxC
    k = k.detach() # no gradient to keys
    # positive logits: Nx1
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1))
    # negative logits: NxK
    l_neg = mm(q.view(N,C), queue.view(C,K))
    # logits: Nx(1+K)
    logits = cat([l_pos, l_neg], dim=1)
    # contrastive loss, Eqn.(1)
    labels = zeros(N) # positives are the 0-th
    loss = CrossEntropyLoss(logits/t, labels)
    # SGD update: query network
    loss.backward()
    update(f_q.params)
    # momentum update: key network
    f_k.params = m*f_k.params+(1-m)*f_q.params
    # update dictionary
    enqueue(queue, k) # enqueue the current minibatch
    dequeue(queue) # dequeue the earliest minibatch
```

在这个例子中，作者将同一个图片的不同数据增广样本当成是正样本，而其他所有的图片都是负样本。在代码最后，保持了对队列的更新（通过将新的minibatch入队列和将旧的出队列实现），并且通过动量更新实现了对QK编码器的更新。

作者对比了端到端，Memory Bank和MoCo框架的效果，如Fig 1.6所示，我们可以看到几个值得注意的现象：

1. 端到端的方式无法一直提高负样本大小，可以发现MoCo和端到端在256-1024的负样本大小阶段的结果可以匹配（track）上的。
2. MoCo和Memory Bank都可以将负样本大小提高到很大。
3. 但是MoCo的效果一直都比Memory Bank高约2%点。

这个现象也和之前的分析保持一致。

![three_compare][three_compare]

<div align='center'>
    <b>
        Fig 1.6 三种框架的实验结果对比。
    </b>
</div>

# Shuffling BN

作者在文中提到了一嘴“Shuffling BN”，而这似乎是在本文才引出来的概念，我们在这儿讨论一下。在实践中，研究者发现在对比学习中的编码器使用Batch Normalization可能会使得模型学习不到好的表征[5]，之前的研究会采取不采用BN层的方法。在本工作中，作者认为编码器中的传统BN层可能会影响预训练任务，进而让预训练很容易就找到了类似于平凡解的“低损失解”，而这将会使得预训练失效。

这个过程在“同步”更新的对比学习框架中容易出现，比如如Fig 1.7所示，其中BN层将会在N-F面上进行标准化，我们知道一个batch中的Query和Key是一一匹配的，都是正样本，也即是$q_i, k_j, i=j$是匹配的。但是$q_i, k_j, i \neq j$时却不是匹配的，因此通过BN可能会将本是负样本的信息“泄漏”到正样本中去。而在Memory Bank中不会出现这个问题，因为Memory Bank的正样本Key是来自于过去的mini batch的，因为是“异步”过程所以不会影响当前结果。

![bn_shuffle][bn_shuffle]

<div align='center'>
    <b>
        Fig 1.7 在对比学习中，BN层导致预训练任务失效的原因猜想。
    </b>
</div>

为了解决这个问题，作者提出了Shuffling BN的概念，而这个层必须在多GPU卡上才能进行。首先作者将不同GPU的sub mini batch进行汇聚，然后对汇聚后的batch进行打乱，并且记下打乱后的索引（便于恢复shuffle结果），在提取了Key编码特征后，通过之前记下的索引，恢复shuffle结果。注意到这个过程只对Key编码器进行，而不考虑Query编码器。正如作者提供的代码code 1.1所示。

```python
 # compute key features
    with torch.no_grad():  # no gradient to keys
        self._momentum_update_key_encoder()  # update the key encoder

        # shuffle for making use of BN
        im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

        k = self.encoder_k(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)

        # undo shuffle
        k = self._batch_unshuffle_ddp(k, idx_unshuffle)
```

# 总结

本文简单对MoCo进行了讨论，从实验上看，的确MoCo能有很大程度的提高，但是也正如作者文章最后讨论的，并没有完全利用到数据。如Fig 1.8所示，MoCo在100万的数据和在十亿级别的数据上性能差别似乎不是很大。而且，目前的MoCo动量更新框架似乎要求QK编码器的结构需要一致？这一点不知道是否笔者理解正确了。

![result_1][result_1]

<div align='center'>
    <b>
        Fig 1.8 MoCo在IN-1M和IG-1B上的结果并没有差别很大。
    </b>
</div>



# Reference

[1]. He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9729-9738).

[2]. https://fesian.blog.csdn.net/article/details/116377189

[3]. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *arXiv preprint arXiv:2103.00020*.

[4]. Wu, Z., Xiong, Y., Yu, S. X., & Lin, D. (2018). Unsupervised feature learning via non-parametric instance discrimination. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 3733-3742).

[5]. Henaff, O. (2020, November). Data-efficient image recognition with contrastive predictive coding. In *International Conference on Machine Learning* (pp. 4182-4192). PMLR.



[qrcode]: ./imgs/qrcode.jpg

[batchneg]:  ./imgs/batchneg.png
[end2end]: ./imgs/end2end.png
[memory_bank]: ./imgs/memory_bank.png

[moco]: ./imgs/moco.png

[momentum_ablation]: ./imgs/momentum_ablation.png

[three_compare]: ./imgs/three_compare.png
[bn_shuffle]: ./imgs/bn_shuffle.png
[result_1]: ./imgs/result_1.png

