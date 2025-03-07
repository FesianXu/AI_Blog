<div align='center'>
    视频分析与多模态融合，为什么需要多模态融合
</div>



<div align='right'>
    FesianXu 20210130 at Baidu search team
</div>


# 前言

在前文《万字长文漫谈视频理解》[1]中，笔者曾经对视频理解中常用的一些技术进行了简单介绍，然而限于篇幅，意犹未尽。在实习工作中，笔者进一步接触了更多视频分析在视频搜索中的一些应用，深感之前对视频分析在业界中应用的理解过于狭隘。本文作为笔者对前文的一个补充，进一步讨论一下视频分析以及其在搜索推荐系统中的一些应用。**如有谬误请联系指出，本文遵守[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

----

注：本文同样是属于漫谈性质的博客，因此行文较为随意，可能逻辑脉路不是特别清晰，各处充盈笔者的随笔漫想，如有谬误，请各位读者谅解并指出讨论。阅读本文之前，可以先阅读笔者之前的一篇博文[1]，以确保叙事的承前启后以及完整性。

# 视频分析不仅有动作识别

之前笔者写过一篇长篇博客[1]，主要介绍了一些视频理解的技术，其中偏重于基于动作识别技术的视频分析，包括数据集和模型设计方法都是偏向于动作识别的。其中学术界中，动作分析数据集最常用的包括：`HMDB-51`，`ucf-101`, `sports-1M`，`Kinectics`等，这些数据集的标签基本上都是一些动作类型，大部分是为了动作识别任务而特别筛选过的YouTube视频。视频动作识别技术在安防领域有着广泛地使用场景，特别地，安防领域对于**多视角**视频动作识别技术有着更为急切的需求，因为在这种场景中，摄像头需要部署在各种可能的地方，因此摄像机姿态各异，需要利用多视角的方法挖掘不同视角下的共同表征，以减少对视角不同场景中重新收集数据的需求。同时，多视角动作识别也会和行人重识别（ReID）技术有着一些交叠，不过那是后话了。

然而，在互联网场景中，以视频搜索/推荐系统为例子，我们需要面对的是用户上传的各种各样的视频，这些视频中语义复杂，不单单是人体动作识别能够简单解决的，甚至很多用户视频并没有包括人的出场。笔者因为在研究生阶段研究的是基于人体的视频分析，因此在前文[1]中没能对互联网中用户视频类型进行准确判断，进而有以下判断

> 视频动作分析可以视为视频理解的核心

这个论述不能说完全错误，以人为主要出场的视频的视频理解的**核心思路之一**的确是动作分析，但是，首先考虑到线上很多视频是用户拍摄周围的景色，或者是动漫片段，或者是其他类型的视频，如Fig 1.1所示，有很多视频并没有人的出场，自然动作分析也就失效了。其次，视频语义和视频中的动作语义并不是完全对齐的，举个例子，一个视频中两个人在碰杯喝酒，我们通过动作识别模型只能知道这两个人在碰杯喝酒，仅此而已。但是其实视频中的两个人是为了庆祝某件重大的事情才碰杯喝酒的，这个“重大的事情”才是整个视频强调的语义，在目前的搜索系统中大多只能通过Query-Title匹配/语义分析的方法才能对此进行召回和排序，如果搜索模型把关注点集中到了视频的动作语义，就难以对该视频进行召回和排序了。

![video_no_human][video_no_human]

<div align='center'>
    <b>
        Fig 1.1 有很多视频并没有人的出场，此时基于人体动作识别的方法不能准确捕捉视频语义信息。
    </b>
</div>


总结来说，视频理解是一个非常宏大的话题，笔者在[1]中只是对视频动作识别进行了简单的介绍，应用场景比较受限于智能监控分析场景，远远还达不到能在互联网线上系统使用的程度。通用的视频理解，面临着数据集缺少，视频语义复杂多变，视频多模态语义融合，非线性流的视频理解等诸多问题，笔者很难在这些领域都有全面的了解，笔者尽量在本文介绍一些自己知道的工作，希望对读者有所帮助。

# 数据集的补充介绍

在[1]中我们介绍过一些常用的动作识别数据集，这些数据集在通用视频分析中并不太够用，目前存在有一些多模态数据集，也存在有一些现实场景中通用的视频数据集，还有一类型称之为`HowTo`类型视频的数据集。

## 多模态数据集

在搜索场景中，用户给出检索词Query，需要返回合适的视频给用户，在这个过程中，涉及到了模型对Query与视频内容，视频标题等诸多元素之间相关性的度量问题。因此这并不是一个简单的对视频进行特征提取的过程，而且涉及到了文本-视觉多模态之间的特征融合，交互过程。据笔者了解，目前存在有若干多模态相关的数据集，很多都用于Image caption或者video caption，Video QA，Image QA等任务，通过迁移也可以应用在Query与视频之间的特征交互。以下列举几种常用于Visual+Language任务中用于预训练的数据集：

**COCO Caption** [4]： 从COCO数据集中进行采样，然后让人工进行一句话描述图片内容的样本对 `<image, text-description>`，可用于V+L任务的预训练。

**Visual Genome Dense Captions**  [5]: 类似于COCO Caption，从Visual Genome数据中采集而成。

**Conceptual Captions** [6]: 类似于COCO Caption

**SBU Caption** [7]:   类似于COCO Caption

这些数据如Fig 2.1所示，一般是图文对的形式出现的，一幅图片搭配有起码一句人工描述的文本，有些数据集可能会有`alt-text`等，如Conceptual Caption数据集。

![image_language_pair][image_language_pair]

<div align='center'>
    <b>
        Fig 2.1 图片-文本对的形式的多模态数据集，常用于进行预训练。
    </b>
</div>

这里谈到的四种数据集的数据集大小以及一些小细节在[10]中的`in-domain和out-of-domain`一节有过介绍，这里不再累述。

## YouTube数据集

YouTube有着海量的视频资源，有很多数据集也是从YouTube中进行采样得到的，其中包括Kinetics系列数据集，YouTube 8M数据集等，其中YouTube 8M数据集具有6.1M的视频量，视频时长总计大约为350K小时，一共有3862个标签，平均每个视频有3个标签，其标签的语义包括了诸多日常场景，如Fig2.2所示，可见其实一个明显的长尾分布。

![label_semantics][label_semantics]

<div align='center'>
    <b>
        Fig 2.2 YouTube 8M数据集中的标签的语义范围。
    </b>
</div>
YouTube 8M因为数据量庞大，没有提供每个视频的原始帧，而是提供了用CNN模型处理过后的特征，该特征是用CNN模型对数据集中每个视频的每帧进行特征提取后得到的。基于这种帧特征化后的数据集，之前谈到的一些光流方法，3D卷积方法将无法使用。然而在线上的实践中，这种方法还是最为常见的。

除了YouTube 8M之外，还有MSR-VTT [11]也是采集于YouTube的通用视频数据集。

## Instructional数据集

Instructional视频，是一类视频的总称，意在教导人们如何完成某件事情，因此也称之为HowTo视频，如Fig 2.3所示，这类型视频的特点就是会出现一个人，以语音解说伴随着动作指导观众完成某个事情。这一类视频在网络视频中具有一定的比重，因此对其的文本语义-视觉信息的语义对齐是很值得研究的一个问题。目前公开用于预训练或者模型验证的数据集有以下几个。

![howto_combine][howto_combine]

<div align='center'>
    <b>
        Fig 2.3 HowTo视频的示例。
    </b>
</div>
**HowTo100M** [11]：该数据集通过在WikiHow [13]中挑选了23,611个howto任务，然后依次为检索词query在YouTube上进行搜索，然后将前200个结果进行筛选，得到了最后的数据集，一共有136.6M个视频。因为这类型的视频普遍存在语音解说，作者用ASR（Automatic Speech Recognition）去提取每个视频每一帧的解说语音（如果该视频本身由作者上传有字幕，则使用原生的字幕信息），将其转换成文本形式的叙述（narrations），也即是此时存在`<文本叙述，视频帧>`的样本对，通过这种手段，作者生成了大规模的带噪声的文本-视频样本对用于多模态任务的预训练。将howto任务分为12个大类，如Fig 2.4所示，我们发现howto视频也是呈现一个典型的长尾分布。

![Howto100m_categories][Howto100m_categories]

<div align='center'>
    <b>
        Fig 2.4 从WikiHow中挑选的12个大类的howto类别。
    </b>
</div>


# 多模态语义融合

什么叫做多模态呢？我们之前已经谈到过了，无非是对于同一个概念，同一个事物通过不同的模态的媒体进行描述，比如用图片，用视频，用语言，用语音对某一个场景进行描述，这就是多模态的一个例子。多模态目前是一个很火的研究方向，我们在上文讨论中发现，目前视频语义复杂，特别是在搜索推荐系统中，可能包含有各种种类的视频，光从动作语义上很难进行描述。如果扩充到其他更广阔的语义，则需要更加精细的标注才能实现。通常而言，动作分类的类别标注就过于粗糙了。

考虑到搜索推荐系统中广泛存在的长尾现象，进行事无巨细的样本标注工作显然是不可取的，再回想到我们之前在[15]中谈到的“语义标签”的概念，如Fig 3.1所示，即便有足够的人力进行标注，**如何进行合适的样本标注设计也是一件复杂的问题**。对于一张图（亦或是一个视频），单纯给予一个动作标签不足以描述整个样本的语义，额外对样本中的每个物体的位置，种类进行标注，对每个样本发生的事情进行文本描述，对样本的场景，环境进行描述，这些都是可以采取的进一步的标注方式。

就目前而言，据笔者了解，在预训练阶段，为了保证预训练结果能够在下游任务中有效地泛化，不能对预训练的语义进行狭义的约束，比如 **仅** 用动作类别语义进行约束就是一个狭义约束。为了使得标注有着更为通用的语义信息，目前很多采用的是多模态融合的方法。

![semantic_classification_label][semantic_classification_label]

<div align='center'>
    <b>
        Fig 3.1 语义标签可以使得具有相同语义的物体在特征空间更为地接近。如何对样本进行语义标注是一件复杂的事情。
    </b>
</div>
在多模态融合方法中，以图片为例子，可以考虑用一句话去描述一张图中的元素和内容（此处的描述语句可以是人工标注的，也可以是通过网络的海量资源中自动收集得到的，比如用户对自己上传图片的评论，描述甚至是弹幕等），比如在ERNIE-VIL [16]中采用的预训练数据集Conceptual Captions (CC) dataset [6]，其标注后的样本如Fig 3.2所示。其中的虚线框是笔者添加的，我们注意到左上角的样本，其标注信息是"Trees in a winter snowstorm"，通过这简单一个文本，伴随配对的图片，我们可以知道很多信息：

1. 在暴风雪下，天气以白色为主。
2. 树的形状和模样，一般是直立在土地上的。
3. 暴风雪时候，能见度很低。

如果数据集中还有些关于树木的场景的描述文本，比如“Two boys play on the tree”， 那么模型就很有可能联合这些样本，学习到“树（tree）”这个概念，即便没有人类标注的包围盒标签（bounding box label）都可以学习出来，除此之外因为语义标签的通用性，还提供了学习到其他关于暴风雪概念的可能性。通过这种手段，可以一定程度上缓解长尾问题导致的标签标注压力。并且因为文本的嵌入特征具有语义属性，意味着文本标签可以对相近语义的表述（近义词，同义词等等）进行兼容，进一步提高了模型的通用性，这些都是多模态融合模型的特点。

![concept_label][concept_label]

<div align='center'>
    <b>
        Fig 3.2 用具有更为通用语义的标注方式，进行图片的描述可以大大减少标注压力。
    </b>
</div>

综上所述，笔者认为，多模态融合模型具有以下优点：

1. 对标注的精准性要求更低，可以通过人类直观的看图说话进行标注。
2. 可以通过互联网大量收集弱标注的图片描述数据，如图片评论，弹幕，用户的自我描述等。
3. 语义更为通用，可作为预训练模型供多种下游任务使用。
4. 可以缓解长尾问题，对语义相近的场景更为友好。

因此，笔者认为，在当前互联网弱标注数据海量存在的时代，算力大大增强的时代，在跨模态的视频，图片搜索推荐这些应用中，采用多模态融合的方法是势在必行的，是现在和未来的方向。我们后面的系列文章将会对这些方法进行简单的介绍。










## 双塔模型

## 交互式模型



## Image-Language模型





## Video-Language模型





# 非线性流的视频理解

## 视频帧特征的聚合

## 图神经网络的引入








# 长视频非均匀抽帧



## TSN



# 短/小视频均匀抽帧



# Reference

[1]. https://fesian.blog.csdn.net/article/details/105545703

[2]. https://fesian.blog.csdn.net/article/details/108212429

[3]. https://fesian.blog.csdn.net/article/details/87901764

[4]. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: ECCV (2014)  

[5]. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., et al.: Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV (2017)  

[6]. Sharma, P., Ding, N., Goodman, S., Soricut, R.: Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In: ACL (2018)  

[7]. Ordonez, V., Kulkarni, G., Berg, T.L.: Im2text: Describing images using 1 million captioned photographs. In: NeurIPS (2011)  

[8]. https://github.com/rohit497/Recent-Advances-in-Vision-and-Language-Research

[9]. https://github.com/lichengunc/pretrain-vl-data

[10]. https://blog.csdn.net/LoseInVain/article/details/103870157

[11]. Miech, A., Zhukov, D., Alayrac, J. B., Tapaswi, M., Laptev, I., & Sivic, J. (2019). Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 2630-2640).

[12]. Zhou, L., Xu, C., & Corso, J. (2018, April). Towards automatic learning of procedures from web instructional videos. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 32, No. 1).

[13]. https://www.wikihow.com/  

[14]. Rohrbach, Anna, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville, and Bernt Schiele. "Movie description." *International Journal of Computer Vision* 123, no. 1 (2017): 94-120.

[15]. https://blog.csdn.net/LoseInVain/article/details/114958239

[16]. Yu, Fei, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. “Ernie-vil: Knowledge enhanced vision-language representations through scene graph.” arXiv preprint arXiv:2006.16934 (2020).

[17]. 







[qrcode]: ./imgs/qrcode.jpg

[video_no_human]: ./imgs/video_no_human.png

[image_language_pair]: ./imgs/image_language_pair.png

[label_semantics]: ./imgs/label_semantics.png

[howto_combine]: ./imgs/howto_combine.png
[Howto100m_categories]: ./imgs/Howto100m_categories.png

[semantic_classification_label]: ./imgs/semantic_classification_label.png

[concept_label]: ./imgs/concept_label.png

