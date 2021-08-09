∇ 联系方式：

**e-mail:** FesianXu@gmail.com

**github:** https://github.com/FesianXu

**知乎专栏:** 计算机视觉/计算机图形理论与应用

**微信公众号：**
![qrcode][qrcode]

----

ERNIE [1,2,3]是百度提出用于对文本进行建模的模型，为了对图文信息进行多模态建模，百度在后续还提出了ERNIE-VIL模型[4]。ERNIE-VIL模型的建模思路很直接，作者认为图片信息主要由以下几种类型，如Fig 1所示。分别是图片中有哪些物体（**Objects**）， 图片的物体有哪些属性（**Attributes**）， 图片中物体之间的关系（**Relationships**）。这些图片信息代表着视觉上最基础，最直观的信息。而更深层的语义信息，比如图像的内涵，历史背景，人物动作，因果关系推理等，本文作者则暂时没有考虑这些。我们不妨把这些称之为视觉元素（Visual Cues，VC）。
![types][types]

<div align='center'>
<b>
Fig 1. 图片信息可以分为以上几种类型。
</b>
</div>

从图中也不难发现，图中的视觉元素在对应的文本描述中也有对应的体现，也就是说视觉元素能找到对应的文本元素。我们知道文本具有一定的结构化信息，因为文本具有一定的词法，句法语法等，如果能够将这些结构化信息在多模态信息融合过程中考虑进来，将会是一个很好的idea，将有助于模型理解图中物体之间的结构关系，属性，类别等。

这正是ERNIE-VIL考虑的，如Fig 2所示，这是ERNIE-VIL的网络结构图。左图的输入分别是对一张图进行ROI检测之后的ROI区域图片特征，另一个输入是文本的embedding特征。
![ernie_vil][ernie_vil]

<div align='center'>
<b>
Fig 2. ERNIE-VIL模型的结构图，左边是网络输入输出示例，而右图是对文本进行场景图解析后的各个文本元素。
</b>
</div>

此处的文本$\mathbf{w}$需要利用场景图解析（Scene Graph Parsing）解析出各种文本元素，解析结果可以表示为：
$$
G(\mathbf{w}) = <O(\mathbf{w}), E(\mathbf{w}), K(\mathbf{w})>
\tag{1}
$$
其中的$O(\mathbf{w})$表示文本中提到的物体的集合，$E(\mathbf{w}) \subseteq  O(\mathbf{w}) \times R(\mathbf{w}) \times O(\mathbf{w})$表示由物体和物体根据关系$R(\mathbf{w})$构成的边关系（edge），而$K(\mathbf{w}) \subseteq O(\mathbf{w}) \times A(\mathbf{w})$表示由属性修饰的物体。论文中给出了一个场景图解析的例子，如下图所示。

![example][example]
那么此时我们的视觉元素输入是一堆显著性ROI区域的特征（以及其检测框），文本元素是经过场景图解析后的诸多文本结构信息，那么对Masked Language Model（MLM）进行扩展，我们可以mask住某个模态的信息，然后期望模型可以根据另一个模态的信息『恢复』出被mask的信息，这个恢复过程既考虑了同一个模态的上下文信息，也考虑到了跨模态的建模。文中举了几个这种例子。比如物体预测，可以表示为：
$$
\mathcal{L}_{obj}(\theta) = -E_{(\mathbf{w}, \mathbf{v}) \sim D}(\log(P(\mathbf{w}_{O_i}|\mathbf{
w}_{/O_i}, \mathbf{v})))
\tag{2}
$$
而属性预测可以表示为：
$$
\mathcal{L}_{attr}(\theta) = -E_{(\mathbf{w}, \mathbf{v}) \sim D} (\log(P(\alpha_i | \mathbf{w}_{O_i}, \mathbf{w}_{/\alpha_i}, \mathbf{v})))
\tag{3}
$$
而关系预测则可以表示为：
$$
\mathcal{L}_{rel} = -E_{(\mathbf{w}, \mathbf{v}) \sim D} (\log(P(\mathbf{w}_{r_{i}}|\mathbf{w}_{O_{i1}}, \mathbf{w}_{O_{i2}}, \mathbf{w}_{/\mathbf{w}_{r_i}} ,\mathbf{v})))
\tag{4}
$$



# Reference
[1]. Sun, Yu, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, and Hua Wu. “Ernie: Enhanced representation through knowledge integration.” arXiv preprint arXiv:1904.09223 (2019).

[2]. Sun, Y., Wang, S., Li, Y., Feng, S., Tian, H., Wu, H., & Wang, H. (2020, April). Ernie 2.0: A continual pre-training framework for language understanding. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 05, pp. 8968-8975).

[3]. https://fesian.blog.csdn.net/article/details/113859683

[4]. Yu, Fei, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. "Ernie-vil: Knowledge enhanced vision-language representations through scene graph." arXiv preprint arXiv:2006.16934 (2020).



[qrcode]: ./imgs/qrcode.png
[types]: ./imgs/types.png
[ernie_vil]: ./imgs/ernie_vil.png
[example]: ./imgs/example.png

