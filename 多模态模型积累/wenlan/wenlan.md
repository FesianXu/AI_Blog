<div align='center'>
  图文搜索系统中的多模态模型：将MoCo应用在多模态对比学习上
</div>

<div align='right'>
  FesianXu 20210917 at Baidu Search Team
</div>

# 前言

之前我们在[1]中介绍过超大负样本对于对比学习训练的重要意义，并且在[2,3]中介绍了MoCo，Memory Bank等方法去突破硬件限制地去进一步增大负样本数量。然而，之前这些方法都尝试在单模态数据上进行对比学习[4]，在文章[5]中，作者团队提出了WenLan项目，尝试在多模态模型中采用MoCo的形式进行大尺度负样本对比学习。本文是对WenLan的简单读后感，并且尝试对其进行分析。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：
**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

-----

之前已经对MoCo进行过比较详细的介绍了[2,3]，建议读者无相关知识储备的先提前去了解MoCo的机制与原理。总得来说，在MoCo中需要维护一个负样本队列，同时利用动量更新对`key encoder`进行参数更新，而对于`query encoder`则采用传统的梯度更新。在编码器学习过程中，将生成的特征入队负样本队列，从而实现负样本队列的更新。对于多模态模型来说，则需要维护两套编码器，不过我们先卖个关子，且让我按照WenLan的逻辑进行分析，最后再讨论WenLan的代码实现。



# 图文搜索场景

图文搜索场景是多模态模型应用的一个主要阵地，相信大家对此不会有太多异议。图文搜索要求模型对图片语义和文本语义有很强的理解。对于图片语义而言，包括了：图片里面有什么实体，他们之间存在什么关系，图片处在什么背景中，图片的情绪色彩...这些都可以是图片语义的一部分，然而对于图片搜索而言，大多数Query还是集中在存在客观实体的检索上。比如你可能会去百度图片里面搜索『月亮』『太阳』『哈士奇』，但是你不太可能会去里面搜索『正义是什么』『人类的思想本质』这类型的抽象内容，显然地用户也对他们的检索有所认知，当检索无明显的视觉需求时候，用户会倾向于在百度主搜里面进行搜索。我们把这类检索称之为视觉无关检索。

然而，很多时候一个检索的视觉需求游离在『有』和『无』中。从Query字面上看，很难判断一个检索有无视觉需求，用户搜索『生日快乐！』可能期望出现的是：在生日中可能会出现的东西，比如蛋糕，蜡烛，生日帽等。这类型的Query文本语义和视觉语义之间的差异巨大，而目前流行的数据集都是具有非常明显视觉相关的文本描述，如Fig 1.1所示，其中的狗，男人和沙发都是非常明显的视觉概念，在图片中也很容易找到一一对应。

![perfect_visual_text_pair][perfect_visual_text_pair]

<div align='center'>
  <b>
    Fig 1.1 具有强视觉关联的图文数据对，其中文本描述中的视觉实体和图片视觉实体可以一一对应。
  </b>
</div>

我们把这种类型的样本称之为**强关联（strong correlation）样本**，反之则是**弱关联（weak correlation）样本** ，Fig 1.2对这两类型的样本进行了图示。然而在图文搜索系统中，通过用户的点击行为可以构建大规模的数据集，同时用户的Query千变万化，足以构建出超大规模的多模态数据集。我们认为用户某个Query下的点击过的图片，属于一对正样本`<Query, Image>`，然而由于图文搜索场景中通常可以检索出很多相关的高质量图片，当某个图片已经满足用户需求时候，用户就很可能不再点击其他图片了，如Fig 1.3所示，因此未点击的图片也不能当成是负样本。通常会采用Batch Negative的方法构建负样本，见[2]中关于Batch Negative采样的描述。

![strong_weak_correlation][strong_weak_correlation]

<div align='center'>
  <b>
    Fig 1.2 强关联样本中Query的视觉概念容易识别，弱关联样本中Query更为口语化和抽象，难以直接抽离相关视觉概念。
  </b>
</div>

![nonclick_neg_invalid][nonclick_neg_invalid]

<div align='center'>
  <b>
    Fig 1.3 在图片搜索中，大多数情况下因为能排上很多的高质量图片，用户未点击的图片并不意味着是负样本。
  </b>
</div>

总而言之，目前在图文搜索中我们能得到很多和视觉强关联的样本，也能通过用户点击得到很多视觉弱关联的样本，如何用好这些样本呢？采用超大负样本的多模态对比学习是一个可行的方法，我们正式地开始介绍WenLan。



# 多模态模型中的MoCo

图文多模态模型具有图片和文本两个模态，两个模态都需要独立维护一个负样本队列，因此有两个负样本队列。并且，由于在MoCo中，只有Query编码器是进行**梯度更新**的，而Key编码器是进行**动量更新**的，那么在多模态模型中，我们现在有`Image Encoder`和`Text Encoder`两种模态的编码器，让谁充当Query编码器进行梯度更新，让谁充当Key编码器进行动量更新呢？答案就是在WenLan中同时存在两套Query-Key编码器，在第一套编码器中，由Image编码器充当Query编码器，Text编码器充当Key编码器；在第二套编码器中，由Text编码器充当Query编码器，由Image编码器充当Key编码器。

我们用框图详细解释下整个流程，我们以其中一套编码器$f^{I}, f^{T}_m$​​为例子，如Fig 2.1的上半部分所示，假设图片和文本的batch size为$M$​​。其中的$Q^T\in\mathbb{R}^{D \times K}$​​是负样本队列，$K$​​是队列大小，$D$​​是特征维度。$z^{I}\in\mathbb{R}^{M\times D}$​​是图片经过$f^I$​​编码器（Query编码器）后的特征输出，$z^T\in\mathbb{R}^{M \times D}$​​是对应的文本经过$f^T_{m}$​​编码器（此处是Key编码器）后的特征输出。定义算子$\bigotimes$​​​为：
$$
a \bigotimes b = \sum_{j}(a\cdot b^{T})_{ij} \in\mathbb{R}^{M\times 1}\\
a \in\mathbb{R}^{M\times D}, b\in\mathbb{R}^{M\times D}
\tag{2.1}
$$
可以发现，其实$z^I \bigotimes z^T$​​​在进行正样本打分，最终对​当前输入的$M$​​​个样本进行了正样本打分，计算代码见Code 2.1；而$z^I \cdot Q^T$​​​是对负样本进行负样本打分。最后在最后一个`axis`进行拼接后，得到了正负样本打分$\mathbf{S}_{I2T}\in\mathbb{R}^{M \times (1+K)}$​​​ ，其中第一个为正样本打分，其余的$K$​​​​个为负样本打分。随后即可通过交叉熵损失进行计算损失，得到$\mathcal{L}_{I2T}$​​​。完成损失计算后，对Key编码器计算得到的特征进行负样本队列入队，以达到更新负样本队列的目的。注意，此处在具体实现过程中，需要对所有GPU中的$z^T$​​​​​进行汇聚(all gather)，代码可以参考MoCo的实现[6]。

<div align='center'>
  <b>
    Code 2.1 正样本打分和负样本打分计算代码。
  </b>
</div>

```python
l_pos = torch.einsum('nc,nc->n', [zI, zT]).unsqueeze(-1) # 一个batch中的M个正样本打分计算，大小为M x 1
l_neg = torch.einsum('nc,ck->nk', [zI, QT.clone().detach()]) # 一个batch中的所有样本和负样本队列进行负样本打分，大小为M x K
```

![brivl_imp_frame_whole][brivl_imp_frame_whole]



<div align='center'>
  <b>
    Fig 2.1 以其中一套编码器为例子开始理解多模态MoCo。
  </b>
</div>
当然，此处只是一套编码器，如果考虑另一套编码器，那么整体框图如Fig 2.1整体所示，通过另一套编码器我们可以得到损失$\mathcal{L}_{T2I}$，将两套编码器的损失相加得到最终的损失：
$$
\mathcal{L} = \mathcal{L}_{I2T}+\mathcal{L}_{T2I}
\tag{2.2}
$$

用下标$j$表示对向量进行索引，那么这两部分损失可以细化表示为：
$$
\begin{aligned}
\mathcal{L}_{I2T} &= -\sum_{j} \log\dfrac{\exp(z^{I}_{j} \cdot z^{T}_j / \tau)}{\exp(z^{I}_{j} \cdot z^{T}_j / \tau)+\sum_{n^T\in Q^T} \exp(z^{I}_{j} \cdot n^T / \tau)} \\
\mathcal{L}_{T2I} &= -\sum_{j} \log\dfrac{\exp(z^{T}_{j} \cdot z^{I}_j / \tau)}{\exp(z^{T}_{j} \cdot z^{I}_j / \tau)+\sum_{n^I \in Q^I} \exp(z^{T}_{j} \cdot n^I / \tau)}
\end{aligned}
\tag{2.3}
$$
从框图Fig 2.1来看，对于一个batch的输入$\{B^{T}, B^I\}$​​​，需要分别喂入两套编码器进行计算，因此计算量和对GPU的显存有比较高的要求，再加上在WenLan里面采用了2560大小的隐层大小，导致其即便在A100中都只能开到每张卡batch size=16的程度。同时，这也约束了负样本队列的大小，也许这是以后可以进一步改进的点。

我们再来说下其对负样本队列的更新策略。对于每个模态，我们会在GPU中手动维护一个队列矩阵$Q\in \mathbb{R}^{D \times K}$​​​​，以及一个队列指针`queue_ptr`用于指示在队列的何处更新队列。​如Fig 2.2（以其中一套编码器为例子）所示，假设我们现在有两张GPU同时进行数据并行计算[7]，那么在Key编码器计算完后产生的特征$z \in \mathbb{R}^{M \times D}$​需要入队负样本队列$Q$​​​，此时为了更快速地更新负样本队列，我们会将所有GPU上的特征进行汇聚(`gather`)， 并且计算汇聚后的`batch size=G*M`，其中的G是卡数。此时根据`queue_ptr`，在`Q[:, queue_ptr:queue_ptr+batch_size]`处将汇聚后的特征赋值。整个过程也可见Code 2.2所示。

<div align='center'>
  <b>
    Code 2.2 负样本队列更新策略代码示意。
  </b>
</div>

```python
feature_z_gathered = concat_all_gather(feature_z) # 此处汇聚所有GPU上的相同张量。
batch_size = feature_z_gathered.shape[0] 
Q[:, queue_ptr:queue_ptr + batch_size] = feature_z_gathered.transpose()
```

![queue_update][queue_update]

<div align='center'>
  <b>
    Fig 2.2 负样本队列更新图示。
  </b>
</div>



# 一些结果分析

刷数据集的数值指标就暂时不讨论了，我们在这个章节主要看一些WenLan的可视化结果，并且尝试对其讨论。在image caption任务中，对输入图片进行文本描述推理，如Fig 3.1所示，我们发现WenLan能对图片中的视觉语义进行很好的捕捉，比如『微笑』『清朗天空』『戏服』『红绿灯』等等。考虑到我们实际的应用场景，也即是商业图文搜索场景，我们用户的检索可能会出现视觉语义弱相关的情况，比如检索『生日快乐~』，此时虽然Query并没有明显的视觉实体，但是可以推测出用户想要检索的是与『生日快乐』有关的视觉实体，比如蛋糕，生日帽，蜡烛等。

多模态模型并不能很好地解决这类型的问题，多模态模型能做到把图片的视觉概念挖掘出来就达到了设计目的，至于深入挖掘图片的更为深层次的人文背景，人物关系，作品等等，则需要采用知识图谱（Knowledge Graph）进行概念之间的关联。比如Fig 3.1中的第三个case，我们都知道这个是电影『大话西游』中的一个名场景，但是从视觉中模型只能知道是『一个穿着戏服的男人和一个
穿着戏服的女孩在一起』，显然没有用户会用如此不符合检索习惯的语句进行搜索，更可能的检索是『大话西游 假如上天再给我一个机会』『大话西游 名场面』之类的。显然这些概念多模态模型无法捕捉，这也许也就是多模态模型的局限了吧。

![vis_1][vis_1]

<div align='center'>
  <b>
    Fig 3.1 WenLan对图片进行描述，其能挖掘出较好的视觉语义信息。
  </b>
</div>

我们回到WenLan，它对于自然图像可以进行不错的多模态关联，而对其他绘画作品也有着不错的表现，如Fig 3.2所示，模型可以挖掘出漫画，动画中的视觉概念，这对于图文检索是一个不错的功能，用户经常也会去检索一些动画漫画素材，此时要求模型具有对绘画作品等非自然图像的视觉语义理解能力。但是，显然还是无法理解更为深层次的人文背景，比如最后一个case是[东方Project](https://baike.baidu.com/item/东方Project)同人作品中的角色雾雨魔理沙（きりさめ まりさ），但是模型只能知道她是一个女巫，带着一个巫师帽，骑着扫帚等。如果用户去检索『魔理沙 东方』那么就无法通过视觉概念进行检索，只能通过图片的上下文文本信息进行文-文匹配了。这些都有望通过知识图谱结合多模态模型进一步解决。

![vis_2][vis_2]

<div align='center'>
  <b>
    Fig 3.2 WenLan对于非自然图像的绘画作品也有着不错的表现。
  </b>
</div>





# Reference

[1]. https://fesian.blog.csdn.net/article/details/119516894

[2]. https://fesian.blog.csdn.net/article/details/119515146

[3]. https://fesian.blog.csdn.net/article/details/120039316

[4]. https://github.com/facebookresearch/moco

[5]. Huo Y, Zhang M, Liu G, et al. WenLan: Bridging vision and language by large-scale multi-modal pre-training[J]. arXiv preprint arXiv:2103.0656

[6]. https://github.com/facebookresearch/moco/blob/78b69cafae80bc74cd1a89ac3fb365dc20d157d3/moco/builder.py#L53

[7]. https://blog.csdn.net/LoseInVain/article/details/105808818







[qrcode]: ./imgs/qrcode.jpg
[perfect_visual_text_pair]: ./imgs/perfect_visual_text_pair.png
[strong_weak_correlation]: ./imgs/strong_weak_correlation.png
[nonclick_neg_invalid]: ./imgs/nonclick_neg_invalid.png
[brivl_frame]: ./imgs/brivl_frame.png

[brivl_imp_frame]: ./imgs/brivl_imp_frame.png

[brivl_imp_frame_whole]: ./imgs/brivl_imp_frame_whole.png

[queue_update]: ./imgs/queue_update.png
[vis_1]: ./imgs/vis_1.png
[vis_2]: ./imgs/vis_2.png

