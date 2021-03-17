<div align='center'>
    语义标签(Semantic label)
</div>

<div align='right'>
    FesianXu 20210317 at Baidu intern
</div>

# 前言

语义标签指的是通过特殊方式使得样本的标签具有一定的语义信息，从而实现更好的泛化，是解开放集问题（open set）和zero-shot问题中的常见思路。**如有谬误请联系指出，本文遵守[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

----

在分类任务中，我们的标签通常是“硬标签（hard label）”，指的是对于某个样本，要不是类别A，那么就是类别B，或者类别C等等，可以简单用one-hot编码表示，比如`[0,1,0], [1,0,0]`等，相信做过分类任务的朋友都不陌生。以ImageNet图片分类为例子，人工进行图片类别标注的过程并不是完全准确的，人也会犯错，而且犯错几率不小。那么很可能某些图片会被标注错误，而且图片信息量巨大，其中可能出现多个物体。此时one-hot编码的类别表示就难以进行完整的样本描述。我们这个时候就会认识到，原来标注是对样本进行描述，而描述存在粒度的粗细问题。one-hot编码的标签可以认为是粒度最为粗糙的一种，如果图片中出现多个物体，而我们都对其进行标注，形成multi-hot编码的标签，如`[0,1,1]`等，那么此时粒度无疑更为精细了，如果我们对物体在图片中的位置进行标注，形成包围盒（bounding box，bbox），那么无疑粒度又进一步精细了。

也就是说，对于标注，我们会考虑两个维度：1）标注信息量是否足够，2）标注粒度是否足够精细。然而，对于一般的xxx-hot标签而言，除了标注其类别，是不具有其他语义（semantic）信息的，也就是说，我们很难知道类别A和类别B之间的区别，类别C与类别B之间的区别。因为人类压根没有告诉他，如Fig 1所示，基于one-hot标签的类别分类任务，每个标签可以视为是笛卡尔坐标系中彼此正交的轴上的基底，这意味着每个类别之间的欧式距离是一致的，也就是说，模型认为猫，狗，香蕉都是等价的类别，但是显然，猫和狗都属于动物，而香蕉属于植物。基于one-hot标注，模型无法告诉我们这一点。

![non_semantic_labels][non_semantic_labels]

<div align='center'>
    <b>
        Fig 1. 在one-hot场景中，每个类别标签之间的距离是一致的，但是显然，猫和狗属于动物类别，而香蕉属于植物类别，这种标签无法提供足够的语义信息。
    </b>
</div>

也就是说，猫和狗，相比于香蕉，有着更为接近的语义，也许Fig 2会是个更好的选择。如果我们的标签不再是one-hot的，而是所谓的语义标签，或者在NLP领域称之为分布式标签（Distributing label, Distributing vector）或者嵌入标签（embedding label, embedding vector），那么类别标签之间的欧式距离就可以描述类别之间的相似程度，这个可视为是简单的语义信息，然而很多高层语义信息都依赖于此。

![semantic_classification_label][semantic_classification_label]

<div align='center'>
    <b>
        Fig 2. 如果我们的标签是语义标签，那么此时类别标签之间的欧式距离可以衡量类别之间的相似程度，这点可视为是简单的语义信息。
    </b>
</div>

获取语义标签不能依靠于人工标注，因为人无法很好很客观地描述每个类别之间的相似程度，而且人工精细地标注这个做法在很多高级任务中，无法实现。因此，更为可行的方法是利用多模态信息融合，比如结合NLP和CV，我们知道一个类别称之为“狗”，另一个类别称之为“猫”，还有一个类别是“香蕉”，我们通过word embedding的方法，可以得到每个类别描述的词向量，因为词向量是基于共现矩阵或者上下文局部性原理得到的，因此大概率语义相关的类别会具有类似的词向量，从而实现语义标签的生成。

当然，这种语义标签只能表达粗糙的，低层次的语义信息，比如类别之间的相似程度。如果涉及到更高层的语义呢？比如VQA，给定一个图片，我们基于图片给出一个问题，然后期望模型回答问题；比如Image Caption，给定图片，然后模型需要尝试用语言对图片进行描述。这些任务都是需要很高层次的语义标注才能实现的。通常来说，此时人工标注能做到的就是给定一个图片，让多个人以相近的标准去进行描述，然后形成图文对`<image, text#1, text#2...text#n>`，让模型进行学习。当然这种需要大量人力进行标注的工作量惊人，因此更好的方式是在互联网挖掘海量的无标签带噪信息，比如同一个网页的图文我们认为是同一个主题的，比如朋友圈，微博的图文评论等，这些带有噪声，但是又具有相关性的海量数据也是可以挖掘的。

当然，高层语义信息也依赖于底层语义的可靠，诸如目前很多transformer在多模态的应用，如ViLBERT [1]，ERNIE-ViL [2]等，都依赖与词向量的可靠，然后才谈得上高层语义的可靠。从这个角度来看，其实从底层语义，底层CV&NLP任务到高层语义多模态任务，其实是有一脉相承的逻辑在的。我们将在以后的博文里面继续探讨多模态的一些想法。



# Reference

[1]. Lu, Jiasen, Dhruv Batra, Devi Parikh, and Stefan Lee. "Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks." *arXiv preprint arXiv:1908.02265* (2019).

[2]. Yu, F., Tang, J., Yin, W., Sun, Y., Tian, H., Wu, H., & Wang, H. (2020). Ernie-vil: Knowledge enhanced vision-language representations through scene graph. *arXiv preprint arXiv:2006.16934*.





[qrcode]: ./imgs/qrcode.jpg
[non_semantic_labels]: ./imgs/non_semantic_labels.png
[semantic_classification_label]: ./imgs/semantic_classification_label.png

