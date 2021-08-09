∇ 联系方式：

**e-mail**:  FesianXu@gmail.com

**github**:  https://github.com/FesianXu

**知乎专栏**: 计算机视觉/计算机图形理论与应用

**微信公众号**：
![qrcode][qrcode]

---


之前写过『清华ERNIE』 与 『百度ERNIE』 的对比，也着重介绍了下百度的ERNIE系列模型，主要是ERNIE 1.0 [1]和ERNIE 2.0 [2]。就笔者的读后感而言，百度的工作和学术界有着较大不同，其没有对网络模型本身进行结构上的大改动，而是着重于如何构造合理的预训练任务，以及如何更好地利用数据构造无监督的训练集。合理构造预训练任务这个特点在ERNIE 2.0上体现的淋漓尽致，在ERNIE 2.0中有着三个层次的预训练层级，每个层级还有着不同粒度的子训练任务，具体细节可以参考以前的博文 [3]

而最近百度对ERNIE又有了后续的更新工作，推出了ERNIE 3.0，笔者今天也粗略去看了下文章，在此纪录下个人的看法。ERNIE 3.0 [5] 看起来是接着ERNIE 2.0进行的工作，同样采用了多个语义层级的预训练任务，不同的是：其考虑到不同的预训练任务具有不同的高层语义，而共享着底层的语义（比如语法，词法等），为了充分地利用数据并且实现高效预训练，ERNIE 3.0中对采用了多任务训练中的常见做法，将不同的特征层分为了通用语义层和任务相关层，也即是Fig 1中的Universal Representation和Task-specific Representation，其中的通用语义层一旦预训练完成，就不再更新了（即便在fine-tune时也不再更新），而Task-specific Representation层则会在fine-tune下游任务时候更新，这样保证了fine-tune的高效。同时作者认为不同的预训练任务可以分为两大类，分别是Language Understanding和Language Generation，因此也就是设置了两个Task-specific Representation模块，如Fig 1所示。
![ERNIE3][ERNIE3]

<div align='center'>
<b>
 Fig 1. ERNIE 3.0的框图示意。
</b>
</div>

其中的持续学习和多任务学习文中没有太多介绍，应该是继承了ERNIE 2.0的做法。在3.0中也采用了很多层级的预训练任务，大部分和2.0中的一样，不过其中最后一个提及用知识图谱进行知识加强的预训练任务值得提一下。这个预训练任务笔者看起来，有点像是ERNIE-VIL [4]的预训练设计，不同点在于ERNIE-VIL是针对多模态数据的，比如图片和文本以及其文本对应的解析知识图谱；而ERNIE 3.0采用的是文本以及其解析得到的知识图谱三元组。然而这两个的基本思想极其相似，都是通过知识图谱的三元组作为桥梁去预测其他数据。在ERNIE 3.0中，如Fig 2所示，作者利用知识图谱挖掘算法，对一句话进行三元组挖掘，比如：
> The Nightingale is written by Danish author Hans Christian Andersen.

经过对其中的实体检测，关系检测后，得到一个三元组`<Andersen, Write, Nightingale>`，此时可以将三元组和元语句拼接在一起作为模型输入，有：
> Andersen Write Nightingale [SEP] The Nightingale is written by Danish author Hans Christian Andersen [SEP]

那么三元组（用A段表示）可以代表了一对实体以及其关系，这个关系具有一定的语义信息，比如逻辑关系，这个我们一般认为是知识（Knowledge），而元语句（用B段表示）则代表着原始的文本信息（Plain Text）。为了让模型能学习到这个知识图谱的关系，可以采用以下两种方法：
1. 将三元组中的某个实体或者关系去掉，然后通过B段去预测A段的masked部分。
2. 将B段的某个实体去掉，通过A段去预测B段被masked的部分。

我们发现这个设计其实和ERNIE-VIL的做法极其相似，因为A，B段包含的信息是冗余的，那么通过A理应可以预测B的缺失部分，而通过B也应该可以预测A的缺失部分。如果缺失部分是和语义（知识）有关的，那么模型应该可以学习到一定的知识才对。因此在ERNIE 3.0中作者称之为“知识图谱加强”。
![kg-enhanced][kg-enhanced]
<div align='center'>
<b>
 Fig 2. 通过知识图谱进行知识强化的预训练任务。
</b>
</div>







# Reference
[1]. Sun, Yu, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, and Hua Wu. “Ernie: Enhanced representation through knowledge integration.” arXiv preprint arXiv:1904.09223 (2019).

[2]. Sun, Y., Wang, S., Li, Y., Feng, S., Tian, H., Wu, H., & Wang, H. (2020, April). Ernie 2.0: A continual pre-training framework for language understanding. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 05, pp. 8968-8975).

[3]. https://fesian.blog.csdn.net/article/details/113859683

[4]. https://fesian.blog.csdn.net/article/details/116275484

[5]. Yu Sun and Shuohuan Wang and Shikun Feng etc. (2021. July). ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation. arXiv preprint 	arXiv:2107.02137 (2021)



[kg-enhanced]: ./imgs/kg-enhanced.png
[ERNIE3]: ./imgs/ERNIE3.png
[qrcode]: ./imgs/qrcode.png

