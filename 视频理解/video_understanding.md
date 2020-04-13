<div align='center'>
    漫谈视频理解
</div>

<div align='right'>
    2020/4/12 FesianXu
</div>

# 前言

AI算法已经渗入到了我们生活的方方面面，无论是购物推荐，广告推送，搜索引擎还是多媒体影音娱乐，都有AI算法的影子。作为多媒体中重要的信息载体，视频的地位可以说是数一数二的，然而目前对于AI算法在视频上的应用还不够成熟，理解视频内容仍然是一个重要的问题亟待解决攻克。本文对视频理解进行一些讨论，虽然只是笔者对互联网的一些意见的汇总和漫谈，有些内容是笔者自己的学习所得，希望还是能对诸位读者有所帮助。**如有谬误，请联系指出，转载请注明出处。**

$\nabla$联系方式：
**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**: 973926198
github: https://github.com/FesianXu

------



# 为什么是视频

**以视频为代表的动态多媒体，结合了音频，视频，是当前的，更是未来的互联网流量之王**。 根据来自于国家互联网信息办公室的[中国互联网络发展状况统计报告](http://www.cac.gov.cn/wxb_pdf/0228043.pdf)[1]

> 截至 2018 年 12 月，网络视频、网络音乐和网络游戏的用户规模分别为 6.12 亿、5.76 亿和 4.84 亿，使用率分别为 73.9%、69.5%和 58.4%。短视频用户规模达 6.48 亿，网民使用比例为 78.2%。截至 2018 年 12 月，网络视频用户规模达 6.12 亿，较 2017 年底增加 3309 万，占网民 整体的 73.9%；手机网络视频用户规模达 5.90 亿，较 2017 年底增加 4101 万，占手机 网民的 72.2%。

其中各类应用使用时长占比如图Fig 1.1所示：

![apps_pros][apps_pros]

<div align='center'>
    <b>
    Fig 1.1 2018年各类应用使用时长占比。
    </b>
</div>

我们很容易发现，包括短视频在内的视频用户时长占据了约20%的用户时长，占据了绝大多数的流量，同时网络视频用户的规模也在逐年增加。

互联网早已不是我们20年前滴滴答答拨号上网时期的互联网了，互联网的接入速率与日俱增，如Fig 1.2所示。视频作为与人类日常感知最为接近的方式，比起单独的图片和音频，文本，能在单位时间内传递更多的信息，从而有着更广阔的用户黏性。在单位流量日渐便宜，并且速度逐渐提升的时代，视频，将会撑起未来媒体的大旗。

![download_rate][download_rate]

<div align='center'>
    <b>
        Fig 1.2 固定带宽/4G平均下载速率变化曲线
    </b>
</div>

的确，我们现在是不缺视频的时代，我们有的是数据，大型公司有着广大的用户基础，每天产生着海量的数据，这些海量的数据，然而是可以产生非常巨大的价值的。以色列历史学家尤瓦尔·赫拉利在其畅销书《未来简史》和《今日简史》中，描述过一种未来的社会，在那个社会中，数据是虚拟的黄金，垄断着数据的公司成为未来的umbrella公司，控制着人们的一言一行，人们最终成为了数据和算法的奴隶。尽管这个描述过于骇人和科幻，然而这些都并不是空穴来风，我们能知道的是，从数据中，我们的确可以做到很多事情，我们可以通过数据进行用户画像描写，知道某个用户的各个属性信息，知道他或她喜爱什么，憎恶什么，去过何处，欲往何方。我们根据用户画像进行精确的广告推送，让基于大数据的算法在无形中控制你的购物习惯也不是不可能的事情。数据的确非常重要，然而可惜的是，目前AI算法在视频——这个未来媒体之王上的表现尚不是非常理想（当前在sports-1M上的R(2+1)D模型[7]的表现不足80%，不到可以实用的精度。），仍有很多问题亟待解决，然而未来可期，我们可以预想到，在视频上，我们终能成就一番事业。



# 理解视频——嗯，很复杂

利用视频数据的最终目标是让算法理解视频。**理解视频(understanding the video)**是一件非常抽象的事情，在神经科学尚没有完全清晰的现在，如果按照人类感知去理解这个词，我们终将陷入泥淖。我们得具体点，在理解视频这个任务中，我们到底在做什么？首先，我们要知道对比于文本，图片和音频，视频有什么特点。视频它是动态的按照时间排序的图片序列，然而图片帧间有着密切的联系，存在上下文联系；视频它有音频信息。因此进行视频理解，我们势必需要进行时间序列上的建模，同时还需要空间上的关系组织。

就目前来说，理解视频有着诸多具体的子任务：

1. 视频动作分类：对视频中的动作进行分类

2. 视频动作定位：识别原始视频中某个动作的开始帧和结束帧

3. 视频场景识别：对视频中的场景进行分类

4. 原子动作提取

5. 视频文字说明（Video Caption）：给给定视频配上文字说明，常用于视频简介自动生成和跨媒体检索

6. 集群动作理解：对某个集体活动进行动作分类，常见的包括排球，篮球场景等，可用于集体动作中关键动作，高亮动作的捕获。

7. 视频编辑。

8. 视频问答系统（Video QA）：给定一个问题，系统根据给定的视频片段自动回答

9. 视频跟踪：跟踪视频中的某个物体运动轨迹

10. 视频事件理解：不同于动作，动作是一个更为短时间的活动，而事件可能会涉及到更长的时间依赖

    ...

当然理解视频不仅仅是以上列出的几种，这些任务在我们生活中都能方方面面有所体现，就目前而言，理解视频可以看成是解决以上提到的种种问题。

通常来说，目前的理解视频主要集中在以人为中心的角度进行的，又因为视频本身是动态的，因此描述视频中的物体随着时间变化，在进行什么动作是一个很重要的任务，可以认为动作识别在视频理解中占据了一个很重要的地位。因此本文的理解视频将会和视频动作理解大致地等价起来，这样可能未免过于粗略，不过还是能提供一些讨论的地方的。

视频分析的主要难点集中在：

1. 需要大量的算力，视频的大小远大于图片数据，需要更大的算力进行计算。
2. 低质量，很多真实视频拍摄时有着较大的运动模糊，遮挡，分辨率低下，或者光照不良等问题，容易对模型造成较大的干扰。
3. 需要大量的数据标签！特别是在深度学习中，对视频的时序信息建模需要海量的训练数据才能进行。时间轴不仅仅是添加了一个维度那么简单，其对比图片数据带来了时序分析，因果分析等问题。



# 视频动作理解——新手村

## 视频数据模态

然而视频动作理解也是一个非常广阔的研究领域，我们输入的视频形式也不一定是我们常见的RGB视频，还可能是depth深度图序列，Skeleton关节点信息，IR红外光谱等。

![multiple_video_modality][multiple_video_modality]

<div align='center'>
    <b>
        Fig 3.1 多种模态的视频形式
    </b>
</div>

就目前而言，RGB视频是最为易得的模态，然而随着很多深度摄像头的流行，深度图序列和骨骼点序列的获得也变得容易起来[2]。深度图和骨骼点序列对比RGB视频来说，其对光照的敏感性较低，数据冗余较低，有着许多优点。

关于骨骼点序列的采集可以参考以前的博文[2]。我们在本文讨论的比较多的还是基于RGB视频模态的算法。

-----

## 视频动作分类数据集

现在公开的视频动作分类数据集有很多，比较流行的in-wild数据集主要是在YouTube上采集到的，包括以下的几个。

- HMDB-51，该数据集在YouTube和Google视频上采集，共有6849个视频片段，共有51个动作类别。

- UCF101，有着101个动作类别，13320个视频片段，大尺度的摄像头姿态变化，光照变化，视角变化和背景变化。

  ![ucf101][ucf101]

- sport-1M，也是在YouTube上采集的，有着1,133,157 个视频，487个运动标签。

  ![sports1m][sports1m]

- YouTube-8M, 有着6.1M个视频，3862个机器自动生成的视频标签，平均一个视频有着三个标签。

- YouTube-8M Segments[3]，是YouTube-8M的扩展，其任务可以用在视频动作定位，分段（Segment，寻找某个动作的发生点和终止点），其中有237K个人工确认过的分段标签，共有1000个动作类别，平均每个视频有5个分段。该数据集鼓励研究者利用大量的带噪音的视频级别的标签的训练集数据去训练模型，以进行动作时间段定位。

  ![youtube8M][youtube8M]

- Kinectics 700，这个系列的数据集同样是个巨无霸，有着接近650,000个样本，覆盖着700个动作类别。每个动作类别至少有着600个视频片段样本。

以上的数据集模态都是RGB视频，还有些数据集是多模态的：

- NTU RGB+D 60： 包含有60个动作，多个视角，共有约50k个样本片段，视频模态有RGB视频，深度图序列，骨骼点信息，红外图序列等。
- NTU RGB+D 120：是NTU RGB+D 60的扩展，共有120个动作，包含有多个人-人交互，人-物交互动作，共有约110k个样本，同样是多模态的数据集。

----

## 在深度学习之前

在深度学习之前，AI算法工程师是特征工程师，我们手动设计特征，而这是一个非常困难的事情。手动设计特征并且应用在视频分类的主要套路有：

**特征设计**：挑选合适的特征描述视频

1. 局部特征（Local features）：比如HOG（梯度直方图 ）+ HOF（光流直方图）
2. 基于轨迹的（Trajectory-based）：Motion Boundary Histograms（MBH）[4]，improved Dense Trajectories （iDT） ——有着良好的表现，不过计算复杂度过高。

**集成挑选好的局部特征：** 光是局部特征或者基于轨迹的特征不足以描述视频的全局信息，通常需要用某种方法集成这些特征。

1. 视觉词袋（Bag of Visual Words，BoVW），BoVW提供了一种通用的通过局部特征来构造全局特征的框架，其受到了文本处理中的词袋（Bag of Word，BoW）的启发，主要在于构造词袋（也就是字典，码表）等。

   ![BoVW][BoVW]

2. Fisher Vector，FV同样是通过集成局部特征构造全局特征表征。具体详细内容见[5]

   ![FV][FV]

要表征视频的时序信息，我们主要需要表征的是动作的运动（motion）信息，这个信息通过帧间在时间轴上的变化体现出来，通常我们可以用光流（optical flow）进行描述，如TVL1和DeepFlow。

![optical_flow][optical_flow]

在深度学习来临之前，这些传统的CV算法在视频动作理解中占了主要地位，即便是如今在深度学习大行其道的时代，这些传统的算子也没有完全退出舞台，很多算法比如Two Stream Network等还是会显式地去使用其中的一些算子，比如光流，比如C3D也会使用iDT作为辅助的特征。了解，学习研究这些算子对于视频分析来说，还是必要的。

---

## 深度学习时代

在深度学习时代，视频动作理解的主要工作量在于如何设计合适的深度网络，而不是手动设计特征。我们在设计这样的深度网络的过程中，需要考虑两个方面内容：

1. 模型方面：什么模型可以最好的从现有的数据中捕获时序和空间信息。
2. 计算量方面：如何在不牺牲过多的精度的情况下，减少模型的计算量。

组织时序信息是构建视频理解模型的一个关键点，Fig 3.2展示了若干可能的对多帧信息的组织方法。[6]

1. Single Frame，只是考虑了当前帧的特征，只在最后阶段融合所有的帧的信息。
2. Late Fusion，晚融合使用了两个共享参数的特征提取网络（通常是CNN）进行相隔15帧的两个视频帧的特征提取，同样也是在最后阶段才结合这两帧的预测结果。
3. Early Fusion，早融合在第一层就对连续的10帧进行特征融合。
4. Slow Fusion，慢融合的时序感知野更大，同时在多个阶段都包含了帧间的信息融合，伴有层次（hierarchy）般的信息。这是对早融合和晚融合的一种平衡。

在最终的预测阶段，我们从整个视频中采样若各个片段，我们对这采样的片段进行动作类别预测，其平均或者投票将作为最终的视频预测结果。

![fusions][fusions]

<div align='center'>
    <b>
        Fig 3.2 融合多帧信息的不同方式。
    </b>
</div>

最终若干个帧间信息融合的方法在sport-1M测试集上的结果如Fig 3.3所示：

![fusion_exp][fusion_exp]

<div align='center'>
    <b>
        Fig 3.3 不同帧间融合方法在sport-1M数据集上的表现。
    </b>
</div>
另外说句，[6]的作者从实验结果中发现即便是时序信息建模很弱的Single-Frame方式其准确率也很高，即便是在很需要motion信息的sports体育动作类别上，这个说明不仅仅是motion信息，单帧的appearance信息也是非常重要的。

回到主题，这些融合方法，都是手动设计的融合帧间特征的方式，而深度学习网络基本上只在提取单帧特征上发挥了作用。这样可能不够合理，我们期望设计一个深度网络可以进行端到端的学习，无论是时序信息还是空间信息。于是我们想到，既然视频序列和文本序列，语音序列一样，都是序列，为什么我们不尝试用RNN去处理呢？

的确是可以的，我们可以结合CNN和RNN，直接把视频序列作为端到端的方式进行模型学习。设计这类模型，我们有几种选择可以挑选：

1. 考虑输入的数据模态：a> RGB；   b> 光流； c> 光流+RGB
2. 特征： a> 人工设计； b> 通过CNN进行特征提取
3. 时序特征集成：a> 时序池化； b> 用RNN系列网络进行组织

时序池化如Fig 3.4所示，类似于我们之前讨论的时序融合，不过在细节上不太一样，这里不展开讨论了，具体见文章[8]。

![temporal_pooling][temporal_pooling]

<div align='center'>
    <b>
        Fig 3.4 不同方式的时序池化。
    </b>
</div>

然而，[8]的作者得出的结论是时序池化比LSTM进行时序信息组织的效果好，这个结论然而并不是准确的，因为[8]的作者并不是端到端去训练整个网络。

如果单纯考虑CNN+RNN的端到端训练的方式，那么我们就有了LRCN网络[9]，如Fig 3.5所示，我们可以发现其和[8]的不同在于其是完全的端到端网络，无论是时序和空间信息都是可以端到端训练的。同样的，[9]的作者的输入同样进行了若干种结合，有单纯输入RGB视频，单纯输入光流，结合输入光流和RGB的，结论发现结合输入光流和RGB的效果最为优越。这点其实值得细品，我们知道光流信息是传统CV中对运动motion信息的手工设计的特征，需要额外补充光流信息，说明光靠这种朴素的LSTM的结构去学习视频的时序信息，motion信息是不足够的，这点也在侧面反映了视频的时序组织的困难性。

![LRCNactrec_high][LRCNactrec_high]

<div align='center'>
    <b>
        Fig 3.5 LRCN网络应用在动作识别问题。
    </b>
</div>

LRCN当然不可避免存在缺点，采用了光流信息作为输入意味着需要大量的预先计算用于计算视频的光流；而视频序列的长时间依赖，motion信息可能很难被LSTM捕获；同时，因为需要把整个视频分成若干个片段，对片段进行预测，在最后平均输出得到最终的视频级别的预测结果，因此如果标注的动作只占视频的很小一段，那么模型很难捕获到需要的信息。

结合光流信息并不是LRCN系列网络的专利，Two Stream Network双流网络[10]也是结合视频的光流信息的好手。在双流网络中，我们同样需要对整个视频序列进行采样，得到若干个片段，然后我们从每个片段中计算得到光流信息作为motion信息描述这个动作的运动，然后从这个片段中采样得到一帧图像作为代表，表征整个片段的appearance信息。最终融合motion和appearance信息，分类得到预测结果。这种显式地利用光流来组织时序信息，把motion流和appearance流显式地分割开进行模型组织的，也是一大思路。

![2stream_high][2stream_high]

<div align='center'>
    <b>
        Fig 3.6 双流网络的网络示意图，需要输入视频的光流信息作为motion信息，和其中某个采样得到的单帧信息作为appearance信息。
    </b>
</div>

当然，LRCN具有的问题，双流网络也有，包括计算光流的计算复杂度麻烦，采样片段中可能存在的错误标签问题（也就是采样的片段可能并不是和视频级别有着相同的标签，可能和视频级别的标注相符合的动作只占整个视频的很小一段。）对长时间依赖的动作信息组织也是一个大问题。

到目前为止，我们都是尝试对视频的单帧应用2D卷积操作进行特征提取，然后在时间轴上进行堆叠得到最终的含有时间序列信息的特征。

![2Dconv][2Dconv]

我们自然就会像，如果有一种卷积，能在提取空间信息的同时又能够提取时序信息，那岂不是不需要手工去堆叠时序特征了？一步到位就行了。的确的，我们把这种卷积称之为3D卷积，3D卷积正如其名，其每个卷积核有三个维度，两个在空间域上平移，而另一个在时间轴上滑动卷积。

![3Dconv][3Dconv]

这样的工作可以追溯到2012年的文章[11]，那该文章中，作者提出的网络不是端到端可训练的，同样设计了手工的特征，称之为`input-hardwired`，作者把原视频的灰度图，沿着x方向的梯度图，沿着y方向的梯度图，沿着x方向的光流图，沿着y方向的光流图堆叠层H1层，然后进行3D卷积得到最终的分类结果。如果我们仔细观察Fig 3.7中的3D卷积核的尺寸，我们发现其不是我们现在常见的$3\times3\times3$的尺寸。这个网络开创了3D卷积在视频上应用的先河，然而其也有不少缺点，第一就是其不是端到端可训练的，还是涉及到了手工设计的特征，其二就是其设计的3D卷积核尺寸并不是最为合适的，启发自VGG的网络设计原则，我们希望把单层的卷积核尽可能的小，尽量把网络设计得深一些。

![raw_3dconv][raw_3dconv]

<div align='center'>
    <b>
        Fig 3.7 3D卷积网络的最初尝试。
    </b>
</div>

这些缺点带来了C3D[12]网络，与[11]最大的不同就是，其使用的卷积核都是相同的尺寸大小，为$3\times3\times3$，并且其不涉及到任何手工设计特征输入，因此是完全的端到端可训练的，作者尝试把网络设计得更深一些，最终达到了当时的SOTA(state-of-the-art)结果。作者发现结合了iDT特征，其结果能有5%的大幅度提高（在ufc101-split1数据上从85.2%到90.4%）。

![c3d][c3d]
<div align='center'>
    <b>
        Fig 3.8 C3D网络框图示意。
    <b/>
</div>


![c3d_gif][c3d_gif]
<div align='center'>
    <b>
        Fig 3.9 3D卷积动图示意。
    </b>
</div>
尽管在当时C3D达到了SOTA结果，其还是有很多可以改进的地方的，比如其对长时间的依赖仍然不能很好地建模，但是最大的问题是，C3D的参数量很大，导致整个模型的容量很大，需要大量的标签数据用于训练，并且，我们发现3D卷积很难在现有的大规模图片数据集比如ImageNet上进行预训练，这样导致我们经常需要从头训练C3D，如果业务数据集很小，那么经常C3D会产生严重的过拟合。随着大规模视频动作理解数据集的陆续推出，比如Kinectics[13]的推出，提供了很好的3D卷积网络pre-train的场景，因此这个问题得到了一些缓解。我们知道很多医疗影像都可以看成是类似于视频一样的媒体，比如MRI核磁共振，断层扫描等，我们同样可以用3D卷积网络对医学图像进行特征提取，不过笔者查阅了资料之后，仍然不清楚是否在大规模动作识别数据集上进行预训练对医学图像的训练有所帮助，毕竟医学图像的时序语义和一般视频的时序语义差别很大，个人感觉可能在大规模动作识别数据集上的预训练对于医学图像帮助不大。

C3D系列的网络是完全的3D卷积网络，缺点在于其参数量巨大，呈现的是3次方级别的增长，即便是在预训练场景中，也需要巨大的数据才能hold住。为了缓解这个问题，有一系列工作尝试把3D卷积分解为2D卷积和1D卷积，其中的2D卷积对空间信息进行提取，1D卷积对时序信息进行提取。典型的3D分解网络有$F_{ST}CN$ [14], 其网络示意图如Fig 3.10所示。

![fstcn_high][fstcn_high]

<div align='center'>
    <b>
        Fig 3.10 FstCN 网络的结构框图。
    </b>
</div>

Pseudo-3D ResNet(P3D ResNet)网络[15]则在学习ResNet的残差设计的路上越走越远，其基本的结构块如Fig 3.11所示。这个策略使得网络可以设计得更深，参数量却更少，然而性能表现却能达到SOTA结果，如Fig 3.12和Fig 3.13所示。

![p3d][p3d]

<div align='center'>
    <b>
        Fig 3.11 组成P3D的3种基本单元，分别是P3D-A，P3D-B，P3D-C。
    </b>
</div>

![comparison_c3d_p3d][comparison_c3d_p3d]

<div align='center'>
    <b>
        Fig 3.12 P3D ResNet比C3D的模型更小，深度更深，性能却更加。
    </b>
</div>

![p3d_performance][p3d_performance]

<div align='center'>
    <b>
        Fig 3.13 在UCF101上，众多模型的表现，P3D ResNet有着出色的表现。
    </b>
</div>

不仅如此，将3D分解成2D+1D的操作使得其在图像数据集上预训练成为了可能。

![p3d_pretrain_imagenet][p3d_pretrain_imagenet]

在文章[7]中，作者对比了一系列不同的2D+1D的分解操作，包括一系列2D+3D的操作，如Fig 3.14所示。与P3D ResNet[15]不同的是，R(2+1)D采用了结构相同的单元，如Fig 3.15所示，而不像P3D中有3种不同的残差块设计。这种设计简化了设计，同时达到了SOTA效果。

![R2+1D][R2+1D]

<div align='center'>
    <b>
        Fig 3.14 众多结合2D卷积和3D卷积的方法，其中实验发现R(2+1)D效果最佳。
    </b>
</div>


![2+1D_block][2+1D_block]

<div align='center'>
    <b>
        Fig 3.15 （2+1）D conv单元示意图，把3D卷积进行分解成了空间域卷积和时间域卷积。
    </b>
</div>



至此，我们讨论了视频动作识别中的若干基础思路：1.通过CNN+RNN；2.通过双流，显式地分割motion信息流和appearance信息流；3.通过3D卷积进行直接空间时间信息提取。 

我们的旅途就到此为止了吗？不，我们刚出新手村呐，我们的冒险才刚刚开始，我们后续讨论的网络，或多或少受到了以上几种模型的启发，或者将以上几种模型融合起来进行改造，或添加了新的亮点，比如加入了attention注意力机制，self-attention自注意力机制等。

不过我们暂且在本站结尾做个小总结：视频分析难，难在其特征不仅仅是2D图像中的二维特征了，二维特征图像现在有着多种大规模的图像数据集可以提供预训练，并且对图像进行人工标注，在很多任务中都比对视频标注工作量要小。正因为超大规模的图像标注数据集的推出，使得很多图像问题在深度学习方法加持下得到了快速的发展，在某些领域甚至已经超过了人类。

然而视频分析不同，视频是有一系列语义上有联系的单帧二维图像在时间轴上叠加而成的，而提取时序语义信息是一件不容易的事情，种种实验证明，现存的深度网络在提取时序语义特征上并没有表现得那么好，否则就不需要人工设计的光流特征进行加持了。深度网络在时序特征提取上的缺失，笔者认为大致有几种原因：

1. 标注了的视频数据量不足。
2. 时序信息的分布变化比二维图像分布更为多样，对于图像，我们可以进行插值，采样进行图像的缩放，只要不是缩放的非常过分，人类通常还是能正常辨认图像的内容。而视频帧间插帧却是一件更为困难的事情。因此不同长度之间的视频之间要进行适配本身就是比较困难的事情。当然你可以进行视频时序下采样，但是如果关键帧没有被采样出来，那么就会造成有效信息的丢失，相反，图像的缩放比较少会出现这种问题。说回到时序信息的分布的多样性就体现在这里，同一个动作，发生的长度可能截然不同，所造成的时序是非常复杂的，需要组织不同长度的时序之间的对齐，使得组织动作的motion变得很不容易，更别说不同人的同一个动作的motion可能千差万别，涉及到了原子动作的分解。

标注的视频数据量不足并不一定体现在视频级别的标注少，带有动作标签的视频级别的数据可能并不少，但是这些视频可能并没有进行过裁剪，中间有着太多非标注动作类别相关的内容。对视频进行动作发生时段的准确定位需要非常多的人工，因此标注视频变得比较困难。同时，一个视频中可能出现多个人物，而如果我们只关注某个人物的动作，对其进行动作标注，如果在样本不足的情况下，便很难让模型学习到真正的动作执行人，因此对视频的标注，单纯是视频级别的动作标注是非常弱的一种标注（weak-supervision）。我们可能还需要对一个视频进行多种标注，比如定位，动作类别，执行人bounding-box等等。

同时，给视频标注的信息也不一定准确，标签本身可能是带有噪声的。有很多标签可能来自于视频分类的tag，这些tag分类信息大多数来自于视频本身的上传者，是上传者自己指定的，可能存在有较大的噪声，然而这类型的数据量非常巨大，不利用却又过于可惜。类似于这种tag标签，现在弹幕网站的弹幕也是一种潜在的可以利用的带噪声的标签。

在视频数据上，弱监督学习，带噪声的标签的监督学习，自监督学习，半监督学习将有广阔的空间。

# 视频动作理解——更进一步





# 其他模态的视频序列动作分析





# 在视频动作理解中应用自监督，无监督学习





# 视频动作分析为什么可以视为视频理解的核心







# Reference

[1]. http://www.cac.gov.cn/wxb_pdf/0228043.pdf

[2]. https://blog.csdn.net/LoseInVain/article/details/87901764

[3]. https://research.google.com/youtube8m/

[4]. Wang H, Kläser A, Schmid C, et al. Dense trajectories and motion boundary descriptors for action recognition[J]. International journal of computer vision, 2013, 103(1): 60-79.

[5]. https://hal.inria.fr/hal-00830491v2/document

[6]. Karpathy A, Toderici G, Shetty S, et al. Large-scale video classification with convolutional neural networks[C]//Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2014: 1725-1732.

[7]. Tran D, Wang H, Torresani L, et al. A closer look at spatiotemporal convolutions for action recognition[C]//Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2018: 6450-6459.

[8]. Yue-Hei Ng J, Hausknecht M, Vijayanarasimhan S, et al. Beyond short snippets: Deep networks for video classification[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 4694-4702.

[9]. Donahue J, Anne Hendricks L, Guadarrama S, et al. Long-term recurrent convolutional networks for visual recognition and description[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 2625-2634.

[10]. Simonyan K, Zisserman A. Two-stream convolutional networks for action recognition in videos[C]//Advances in neural information processing systems. 2014: 568-576.

[11]. Ji S, Xu W, Yang M, et al. 3D convolutional neural networks for human action recognition[J]. IEEE transactions on pattern analysis and machine intelligence, 2012, 35(1): 221-231.

[12]. Tran D, Bourdev L, Fergus R, et al. Learning spatiotemporal features with 3d convolutional networks[C]//Proceedings of the IEEE international conference on computer vision. 2015: 4489-4497.

[13]. Carreira J, Zisserman A. Quo vadis, action recognition? a new model and the kinetics dataset[C]//proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 6299-6308.

[14]. Sun L, Jia K, Yeung D Y, et al. Human action recognition using factorized spatio-temporal convolutional networks[C]//Proceedings of the IEEE international conference on computer vision. 2015: 4597-4605.

[15]. Qiu Z, Yao T, Mei T. Learning spatio-temporal representation with pseudo-3d residual networks[C]//proceedings of the IEEE International Conference on Computer Vision. 2017: 5533-5541.

[16]. 













[apps_pros]: ./imgs/apps_pros.jpg
[download_rate]: ./imgs/download_rate.jpg
[multiple_video_modality]: ./imgs/multiple_video_modality.jpg
[ucf101]: ./imgs/ucf101.jpg

[sports1m]: ./imgs/sports1m.jpg

[youtube8M]: ./imgs/youtube8M.jpg
[BoVW]: ./imgs/BoVW.png

[FV]: ./imgs/FV.png
[optical_flow]: ./imgs/optical_flow.png
[fusions]: ./imgs/fusions.png
[fusion_exp]: ./imgs/fusion_exp.png

[rgb+of]: ./imgs/rgb+of.png
[temporal_pooling]: ./imgs/temporal_pooling.png
[LRCNactrec_high]: ./imgs/LRCNactrec_high.png
[2stream_high]: ./imgs/2stream_high.png
[2Dconv]: ./imgs/2Dconv.png
[3Dconv]: ./imgs/3Dconv.png
[raw_3dconv]: ./imgs/raw_3dconv.png
[c3d]: ./imgs/c3d.png
[c3d_gif]: ./imgs/c3d_gif.gif

[R2+1D]: ./imgs/R2+1D.png
[2+1D_block]: ./imgs/2+1D_block.png

[fstcn_high]: ./imgs/fstcn_high.png

[p3d]: ./imgs/p3d.png
[comparison_c3d_p3d]: ./imgs/comparison_c3d_p3d.png
[p3d_performance]: ./imgs/p3d_performance.png
[p3d_pretrain_imagenet]: ./imgs/p3d_pretrain_imagenet.png











