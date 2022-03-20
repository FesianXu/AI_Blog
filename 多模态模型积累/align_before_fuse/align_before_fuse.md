<div align='center'>
  图文多模态语义融合前的语义对齐——一种单双混合塔多模态模型
</div>

<div align='right'>
  FesianXu 20220127 at Baidu Search Team
</div>

# 前言

之前在博文[2-4]中介绍了一些图文多模态语义对齐相关的模型，分别是WenLan 1.0， WenLan 2.0和CLIP等，这些模型都是双塔结构模型，然而在实际的应用场景中，我们会有使用单塔模型的需求，笔者在本文将介绍一篇论文[1]的思路，将单塔模型和双塔模型结合在一起进行图文多模态语义融合和对齐。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：
**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

-----



# 双塔多模态模型的优势与缺陷

之前在博文[2-4]中曾经简单介绍过一些图文多模态模型，分别是WenLan 1.0 [5]和WenLan 2.0 [6]以及CLIP [7]，这些多模态模型在不同的模态上，都有着各自模态各自的编码器。如Fig 1.1所示，CLIP中的图片编码器和文本编码器共同组成了一个双塔结构模型进行损失计算。双塔模型在很多业务场景有着广泛应用，比如在图文信息检索场景中，我们要衡量用户Query和图片之间的图文相关性。假如图片编码器是$y_v =f_v(I), y_v \in \mathbb{R}^{D}$​​​​​​​​，文本编码器是$y_w = f_w(T), y_w \in \mathbb{R}^{D}$​​​，​​​​​​而待检索库中所有图片的集合记为$\mathcal{D}=\{I_i|i=1,\cdots,M\}$​​​，那么可以预先对所有图片进行特征提取，形成图片的正排（Forward Index）特征并且建库，记为$\mathcal{D}_{FI}=\{f_v(I_i),i=1,\cdots,N\}$​，在用户输入检索词$Q$的时候，只需要对Query进行文本编码器的在线计算，得到文本特征$f_w=f_w(Q)$​​​​，然后对待排序的样本进行图片正排库取特征，进行相关性计算（在线计算余弦距离）就可以判断候选图片与Query之间的图文相关程度了。利用双塔模型可以预先建库并且在线计算相关性的特性，可以很大程度上用空间换时间节省很多计算资源，而这也是双塔模型在搜索系统（不仅仅是图文搜索）中被广泛应用的原因之一。

![clip_training_model][clip_training_model]

<div align='center'>
  <b>
    Fig 1.1 CLIP中的图片编码器和文本编码器一起组成了双塔结构。
  </b>
</div>
这个世界没有银弹，双塔模型中的图片和文本信息不能在线交互的特性决定了其对一些细致的图文匹配需求无法满足。举个例子，比如去搜索『黑色上衣白色裤子』，那么百度返回的结果如图Fig 1.2所示，除去开始三个广告不计，用红色框框出来的Top3结果中有俩结果都是『白色上衣黑色裤子』，显然搜索结果并没有理解到『黑色上衣』和『白色裤子』这两个概念，而是单独对『黑色』『白色』和『上衣』『裤子』这两个属性进行了组合，因此才会得到『白色上衣黑色裤子』被排到Top20结果的情况。

![baidu_example][baidu_example]

<div align='center'>
  <b>
    Fig 1.2 百度图搜对于『黑色上衣白色裤子』的搜索结果。
  </b>
</div>

当然读者可能觉得百度搜索可能不够可靠，笔者在google图搜上也进行了测试，如Fig 1.3所示，的确大部分结果都是正确的（如蓝色框所示），但是也有少量的『白色上衣黑色裤子』被排上了Top20结果（如红框所示），即便只有几个误排，也说明业界中对于这种细粒度的多模态搜索仍然是需要继续探索的（读者可以试试『红色杯子』这类型的Query，排到Top20的都是很准确的）。

![google_example][google_example]

<div align='center'>
  <b>
    Fig 1.3 Google图搜对于『黑色上衣白色裤子』的搜索结果。
  </b>
</div>

这种多模态匹配细粒度结果不尽人意的原因，很大程度上是双塔模型中的图片编码器和文本编码器无法在线进行交互导致的。可以想象到，我们的图片编码器由于是预先对所有图片提特征进行建库的，那么就无法对所有属性的组合都进行考虑，必然的就会对一些稀疏的组合进行忽略，而倾向于高频的属性组合，因此长尾的属性组合就无法很好地建模。双塔模型这种特点，不仅仅会使得多属性的Query的检索结果倾向于高频组合，而且还会倾向于图片中的一些大尺寸物体，比如Fig 1.4中的小黄人尺寸较小，在进行特征提取的时候，其在整张图片中的重要性就有可能被其他大尺寸物体（比如键盘和显示屏等）掩盖。

![tiny_feature_missing][tiny_feature_missing]

<div align='center'>
  <b>
    Fig 1.4 图片中的小黄人尺寸较小，特征提取结果可能会被同图片中其他大尺寸物体给掩盖。
  </b>
</div>



# 单塔模型进行在线交互

双塔模型有以上的一些天然的劣势，此时就需要用单塔交互模型对用户Query和图片进行在线的交互（当然此时模型计算由于是在线的，受限于计算资源就只能对粗排的Top20/40等结果进行打分精排了），通过在线交互，细粒度的图文匹配能取得更好的结果，稀疏的属性组合也能通过在线交互得到合理的打分，而不至于被高频组合给『吃掉』。双塔模型一般可以通过大规模对比学习，从诸多负例中挑选出最难的负例，通过将正例和最难负例进行对比损失优化，从而学习出表征。但是单塔模型无法像双塔模型一般进行对比学习去挑选难负样本，因为双塔模型可以通过打分矩阵将$N^2-N$​个负样本打分和$N$​​个正样本打分同时得到，而单塔模型由于需要在线交互，则需要对$N$个Query和$N$个图片进行$\mathcal{O}(N^2)$​​​​​​​​​​次模型计算，才能得到和双塔模型一次计算同样量级的打分，这个计算时间代价太大以至于实际中无法这样进行训练。对于单塔模型，如Fig 2.1所示，我们一般只能通过平移样本得到若干个负样本，进行匹配损失计算，这样得到的负样本数量通常都很小，远远无法达到双塔模型的量级，由此构造出的负样本也往往不够『难』，导致这样训练出来的单塔模型语义对齐（Semantic Alignment）能力弱于用大规模对比学习训练出来的双塔模型。

![shift_single_tower_neg][shift_single_tower_neg]

<div align='center'>
  <b>
    Fig 2.1 在单塔模型训练时，通过平移样本构造负样本。
  </b>
</div>



# 多模态语义融合前的语义对齐

由此来看，单塔模型擅长的往往是语义融合（Semantic Fusion），而非语义对齐（Semantic Alignment），我们可以考虑用大规模对比学习去进行语义对齐，而基于良好的语义对齐用单塔模型去进行语义融合。如Fig 3.1所示，语义对齐尝试找到不同文本实体（Query Entity）与视觉实体（Vision Entity）之间的关联关系，而语义融合尝试找到复合实体的组合关系。

![align_and_fuse][align_and_fuse]

<div align='center'>
  <b>
    Fig 3.1 语义对齐去尝试找到文本实体与视觉实体之间的关联关系；语义融合尝试找到复合实体之间的组合关系。
  </b>
</div>

文章[1]提出了ALBEF模型（ALign BEfore Fuse，ALBEF），尝试通过将双塔模型和单塔模型结合在一起，通过用双塔模型去进行语义对齐，并且通过双塔模型进行难负样本挑选，以备送给单塔模型进行更好的语义融合，这个思路理论上可以融合单塔模型和双塔模型的优点，而不至于带来太多的计算负担。如Fig 3.1所示，ALBEF整个模型主要由BERT组成，其编码器分为单模态（Unimodal）编码器和多模态（multimodal）编码器，单模态编码器主要由图像编码器和文本编码器组成，其图像编码器采用了12层的ViT-B/16模型，而文本编码器和多模态编码器都采用的是6层的$\mathrm{BERT_{base}}$模型。通过图片编码器，将图片输入$\mathbf{I}$编码成embedding序列$\{\mathbf{v}_{CLS},\mathbf{v}_1,\cdots,\mathbf{v}_N\}$，同样对于文本输入$\mathbf{T}$而言，其embedding序列输出为$\{\mathbf{w}_{CLS},\mathbf{w}_1,\cdots,\mathbf{w}_N\}$​​。其预训练目标有两大类：

1. 语义对齐： 通过单模态编码器（其实就是双塔模型）进行图文对比学习（Image-Text Contrastive Learning）进行图文语义对齐
2. 语义融合：将语义对齐后的图/文特征在多模态编码器中进行跨模态交互，通过Masked Language Model（MLM）和图文匹配（Image-Text Matching）任务进行图文语义融合。

![albef][albef]

<div align='center'>
  <b>
    Fig 3.1 语义对齐去尝试找到文本实体与视觉实体之间的关联关系；语义融合尝试找到复合实体之间的组合关系。
  </b>
</div>

## 语义对齐

语义对齐可以通过双塔模型的大规模对比学习进行，其目标是让图片-文本对的相似度尽可能的高，也就是$s=g_v(\mathbf{v}_{cls})^{\mathrm{T}}g_w(\mathbf{w}_{cls})$，其中的$g_v(\cdot)$和$g_w(\cdot)$是对`[CLS]`的线性映射，其将`[CLS]`特征维度映射到了多模态共同特征子空间。类似于MoCo [8,9]，在ALBEF模型中，作者同样采用了两个图片/文本样本队列和动量图片/文本编码器，这两个队列维护了最近的动量编码器的$M$个表征（具体维护过程见博文[8]），这些来自于动量编码器的特征表示为$g_{v}^{\prime}(\mathbf{v}^{\prime}_{cls})$和$g_{w}^{\prime}(\mathbf{w}^{\prime}_{cls})$​​​​ 。那么类似于MoCo中的做法进行多模态打分计算，如式子(3-1)所示
$$
\begin{aligned}
s(I,T) &= g_v(\mathbf{v}_{cls})^{\mathrm{T}}g_{w}^{\prime}(\mathbf{w}^{\prime}_{cls}) \\
s(T,I) &= g_w(\mathbf{w}_{cls})^{\mathrm{T}}g_{v}^{\prime}(\mathbf{v}^{\prime}_{cls})
\end{aligned}
\tag{3-1}
$$
那么可以定义出图-文/文-图相关性，如式子(3-2)所示，其中的$N$​​是`batch size`（这一点是代码实现，和论文有些偏差[10]）
$$
\begin{aligned}
p^{i2t}_{m}(I) &= \dfrac{\exp(s(I, T_m)/\tau)}{\sum_{m=1}^{M+N}\exp(s(I,T_m)\tau)} \\
p^{t2i}_{m}(T) &= \dfrac{\exp(s(T, I_m)/\tau)}{\sum_{m=1}^{M+N}\exp(s(T, I_m)\tau)} 
\end{aligned}
\tag{3-2}
$$
令$\mathbf{y}^{i2t}(I)$和$\mathbf{y}^{t2i}(T)$​表示真实的标签，通过交叉熵损失定义出图文对比损失（Image-Text Contrastive Loss， ITC）:
$$
\mathcal{L}_{itc} = \dfrac{1}{2} \mathbb{E}_{(I,T) \sim D} [H(\mathbf{y}^{i2t}(I), \mathbf{p}^{i2t}(I))+H(\mathbf{y}^{t2i}(T), \mathbf{p}^{t2i}(T))]
\tag{3-3}
$$

## 语义融合

ALBEF模型的底层是双塔语义对齐，其上层是单塔语义融合，为了实现语义融合，论文中采用了Masked Language Model（MLM）损失进行建模。作者以$15\%$​​概率将输入的Token进行替代，将其替代为特殊令牌`[MASK]`，令$\hat{T}$​表示被掩膜后的文本，$\mathbf{p}^{msk}(I,\hat{T})$​​表示对掩膜后的令牌的预测结果，而$\mathbf{y}^{msk}$表示被掩膜令牌的真实标签，那么MLM目的在于最小化以下交叉熵损失：
$$
\mathcal{L}_{mlm} = \mathbb{E}_{(I, \hat{T})\sim D} H(\mathbf{y}^{msk}, \mathbf{p}^{msk}(I,\hat{T}))
\tag{3-4}
$$
通过MLM损失建模，可以让多模态实体之间不仅语义对齐，而且能找到各个实体之间的复合语义关系，如Fig 3.2所示，MLM损失约束模型去融合不同实体，挖掘他们之间的多模态关系，从而对被掩膜后的实体做出预测。

![mlm][mlm]

<div align='center'>
  <b>
    Fig 3.2 MLM损失约束模型去融合不同实体的语义关系，从而对被掩膜后的实体做出预测。
  </b>
</div>

除了MLM损失，文章中还通过图文匹配损失（Image-Text Matching，ITM）对难负样本进行匹配学习，从而期望模型能够对难负样本有着更好的区分能力，从而弥补单塔模型无法进行难负样本选取的缺点，以提升多模态模型的语义对齐和语义融合能力。作者挑选难负样本的依据是根据双塔模型的打分，从式子(3-2)中可以挑选出同一个Query下面最为难的Image（打分最高，但却是预测错误的），也可以挑选出同个Image下最难的Query（论文中是根据打分大小设置概率进行采样得到的）。由此可以得到$N$​个正例和$2N$个难负例构成了ITM任务的输入，其损失如式子(3-5)所示。
$$
\mathcal{L}_{itm} = \mathbb{E}_{(I,T)\sim D} H(\mathbf{y}^{itm}, \mathbf{p}^{itm}(I,T))
\tag{3-5}
$$

最后的预训练阶段损失由以上三个损失构成，如式子(3-6)所示：
$$
\mathcal{L} = \mathcal{L}_{itc}+\mathcal{L}_{mlm}+\mathcal{L}_{itm}
\tag{3-6}
$$



## 动量蒸馏（Momentum Distillation， MoD）

用于预训练的图文数据大多来自于互联网 [3]，通常都是所谓的弱标注数据集，文本中可能有些词语和图片的实体是毫无关系的，图片也可能包含有文本中完全没提到的东西。对于ITC损失而言，一个图片的负样本文本也有可能能够匹配上这个图片（特别是如果该图文对数据来自于用户点击数据）；对于MLM损失而言，被掩膜掉的令牌也许被其他令牌替代也能对图像进行描述（甚至可能更合适）。作者认为，在ITC和MLM任务中采用`one-hot`标签进行训练会对所有的负例进行打压，而不考虑这些负例倒底是不是真正的『负例』。为了解决这个问题，作者提出动量编码器可以看成是单模态/多模态编码器的一种指数滑动平均版本（exponential-moving-average），可以通过动量编码器去生成ITC和MLM任务的『伪标签』，笔者并没有特别理解为什么可以通过动量编码器去生成伪标签，可能这样做能使得标签更为平滑，而不像`one-hot`标签一样吧。总而言之，通过动量编码器，我们有动量编码器打分:
$$
\begin{aligned}
s^{\prime}(I,T) &= g^{\prime}_{v}(\mathbf{v}_{cls}^{\prime})^{\mathrm{T}} g^{\prime}_{w}(\mathbf{w}_{cls}^{\prime}) \\
s^{\prime}(T,I) &= g^{\prime}_{w}(\mathbf{w}_{cls}^{\prime})^{\mathrm{T}} g^{\prime}_{v}(\mathbf{v}_{cls}^{\prime})
\end{aligned}
\tag{3-7}
$$
将(3-7)中的$s^{\prime}$替代式子(3-2)中的$s$​，我们得到伪标签$\mathbf{q}^{i2t}, \mathbf{q}^{t2i}$，那么$ITC_{MoD}$损失定义为：（实际代码实现有些差别，可能要另一篇博文里面去写了）
$$
\mathcal{L}_{itc}^{mod} = (1-\alpha)\mathcal{L}_{itc}+\dfrac{\alpha}{2}\mathbb{E}_{(I,T) \sim D} [KL(\mathbf{q}^{i2t}(I) || \mathbf{p}^{i2t}(I)) + KL(\mathbf{q}^{t2i}(T) || \mathbf{p}^{t2i}(T))]
\tag{3-8}
$$
类似的，$MLM_{MoD}$损失可以定义为：
$$
\mathcal{L}_{mlm}^{mod} = (1-\alpha)\mathcal{L}_{mlm} + \alpha\mathbb{E}_{(I,\hat{T}) \sim D} KL(\mathbf{q}^{msk}(I, \hat{T}) ||\mathbf{p}^{msk}(I, \hat{T}))
\tag{3-9}
$$


# 读后感

这篇文章比较复杂，最近笔者比较忙看了好久才大致看懂些，有些细节猜不透的去看了下代码，发现代码实现好像有些和论文有差别，后续有空再补充下代码实现的阅读笔记可能会更好些。总体来看，这篇文章结合了双塔模型可以进行大规模对比学习，和单塔模型可以进行细粒度交互的优势，提出了ALBEF模型对多模态数据进行语义对齐+语义融合，其思路是值得在业界进行尝试的。


# Reference

[1]. Li, Junnan, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. "Align before fuse: Vision and language representation learning with momentum distillation." *Advances in Neural Information Processing Systems* 34 (2021).

[2]. https://blog.csdn.net/LoseInVain/article/details/121699533

[3]. https://blog.csdn.net/LoseInVain/article/details/120364242

[4]. https://fesian.blog.csdn.net/article/details/119516894

[5]. Huo, Yuqi, Manli Zhang, Guangzhen Liu, Haoyu Lu, Yizhao Gao, Guoxing Yang, Jingyuan Wen et al. "WenLan: Bridging vision and language by large-scale multi-modal pre-training." *arXiv preprint arXiv:2103.06561* (2021).

[6]. Fei, Nanyi, Zhiwu Lu, Yizhao Gao, Guoxing Yang, Yuqi Huo, Jingyuan Wen, Haoyu Lu et al. "WenLan 2.0: Make AI Imagine via a Multimodal Foundation Model." *arXiv preprint arXiv:2110.14378* (2021).

[7]. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *arXiv preprint arXiv:2103.00020*.

[8]. https://fesian.blog.csdn.net/article/details/119515146

[9].  He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9729-9738).

[10]. https://github.com/salesforce/ALBEF/issues/22



[qrcode]: ./imgs/qrcode.jpg
[clip_training_model]: ./imgs/clip_training_model.png

[baidu_example]: ./imgs/baidu_example.png
[google_example]: ./imgs/google_example.png
[tiny_feature_missing]: ./imgs/tiny_feature_missing.png
[align_and_fuse]: ./imgs/align_and_fuse.png
[shift_single_tower_neg]: ./imgs/shift_single_tower_neg.png
[albef]: ./imgs/albef.png
[mlm]: ./imgs/mlm.png
[batch_neg]: ./imgs/batch_neg.png









