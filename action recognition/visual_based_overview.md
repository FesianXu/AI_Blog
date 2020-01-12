



医学图像，如MRI核磁共振图像，CT图像等都是一些3D图像，其和视频非常的类似，它们都在时间维度上编码了二维的空间信息。就像从3D图像中进行医学异常诊断一般，从视频中进行动作识别需要从整个视频中捕获上下文信息，而不仅仅是单纯的从每个帧里面提取信息，却不考虑其帧之间的上下文关系。

在本文中，作者总结了一些关于在视频中进行动作识别的文章，并且按照以下的顺序进行组织：

1. **什么是动作识别以及为什么这个问题那么难？**
2. **方法综述**

# 什么是动作识别以及为什么这个问题那么难

动作识别任务涉及到从视频片段（即是二维图像序列）中识别出不同的动作类别，注意到这个动作不一定是贯彻于整个视频片段中的，也就是说，有可能这个动作只发生在视频中的中间，开头或者结尾等。这个问题看起来是图像分类任务的一个自然扩展，无非是要对许多帧进行分类，并且将每帧的分类结果进行汇聚而已。尽管在图像分类领域，以深度学习为代表的方法取得了巨大成功，但是在视频分类和表征学习的领域上，却进展缓慢。

是什么使得这个问题如此困难呢？可以分为以下几点挑战：

1. **巨大的计算能力需求**，一个用以分类101个类的简单的二维卷积网络只需要大概5M个参数，然而当扩展到三维（加入了时间维）的情况下，则相似的网络架构需要大概33M的参数量。这使得在`UCF101`训练集上训练一个`3DConvNet`需要花费大概4天，而在`Sports-1M`数据集上则夸张的需要大概2个月才能训练出来，这使得对该网络的扩展变得困难，并且很容易出现过拟合。
2. **如何提取长时间的上下文信息**，动作识别涉及到了捕获帧间的时空信息。另外，在获取空间信息的时候，还不得不考虑因为摄像头移动而导致的视角多变性，因此需要对此进行补偿。即便是有着很好的空间物体检测模型的辅助，却仍然是不足够的，因为物体帧间的动作信息同样提供了更好的动作细节。对于帧间的动作信息提取（这个能对动作的鲁棒估计有所帮助），现在有关于局部和全局的上下文信息的研究。比如，考虑到如Fig 2所示的视频表征，一个强大的图片分类器可以识别这两张图中的人类，水体，但是却难以区分自由泳和蛙泳，这两个动作都是时序周期的动作。
3. **如何设计分类结构**，设计可以捕获时空信息的网络架构涉及到了许多可供选择的方向（比如层数，超参数，卷积核大小等等），这个并不容易设计而且验证效果需要很大的成本。举个例子，一些可能的策略包括：
   - 只用一个网络进行时空信息捕获 VS 两个网络，一个用于时间信息，一个用于空间信息。
   - 如何融合不同视频片段的预测结果。
   - 端到端训练 VS 分别进行 特征提取+分类

<table>
    <tr>
        <td width="20%" height="100%">
            <img src="./imgs/breaststroke.gif",height="400" width="400" />
        </td>
        <td width="20%" height="100%">
            <img src="./imgs/fronststroke.gif",height="400" width="400" />
        </td>
    </tr>
</table>

<div align='center'>
    <b>
        Fig 2, 左图是蛙泳，右图是自由泳。捕获动作的时序信息是区分这两种相似情况的重点。并且需要注意的是，在自由泳这个例子中，相机视角的突然变化也是需要考虑的。
    </b>
</div>

4. **没有一个标准的基准**，曾经有段时间，最流行的基准数据集是`UCF101`和`Sports 1M`。在Sports 1M上去搜索合理有效的网络结果是非常昂贵的，需要大量的财力物力人力。对于UCF 101而言，尽管就帧数的总数量而言可以匹敌ImageNet，但是视频间的高空间相关性使得训练中的真实密度相等的低下（译者注：意思就是说很多动作的图像是很相似的）。而且，虽然这两个数据集都给定了相似的主题（比如运动），但是模型的性能在这两个不同基准数据集上的泛化（从一个数据集泛化到另一个迁移），仍然是一个大问题。这些慢慢地被Kinetics数据集的引入而缓解。

![banner][banner]

<div align='center'>
    <b>
        UCF101的样本例子展示。
    </b>
</div>

需要强调的是，3D医学图像的异常检测并不一定涉及到了这里谈到的所有挑战。在动作识别和医学图像之间的主要区别是：

1. 在医学图像中，时序上下文不一定和动作识别中的上下文一样重要。例如，检测头部CT扫描中的出血现象涉及到的切片之间的时序上下文信息并不多。颅内出血可以仅仅通过一张切片就可以检测出来。而相对的，在胸部CT中检测肺部小结会涉及到更多的时序信息的捕获，因为在二维扫描中，小结和支气管，血管看起来一样，都是圆形的。只有在三维上下文信息被引入之后，小结才能够观察到是否为球体，而不是和血管一样的圆柱体。（译者注：这里说到的时序，在医学图像中指的是立体信息，而不是和动作识别中一样的时间轴，当然你也可以认为如同CT般的设备是分时地扫描身体不同切片的，因此看成是时间轴信息，这样也是未尝不可的。）
2. 在动作识别中，大部分研究都是基于预训练好的2D卷积网络，然后在此基础上训练，作为起始点以求更好的效果。然而在医学图像中，这样的预训练网络通常是不可得的。

----



# 方法综述

在深度学习到来之前，大部分传统的用于动作识别的计算机视觉算法可以被分解为以下三个步骤：

1. 局部高维度视觉特征提取，这个特征描述了视频区域，可以分为密集特征(Dense)[3]或者是只描述了感兴趣特征点的稀疏特征(sparse)[4,5].
2. 提取出的特征进行拼接，成为一个固定长度的视频特征描述子。在这步有一个流行的变种就是进行`bag of visual words`（翻译成视觉词袋，来源之NLP中的BoW，通过层次聚类或者k-means聚类产生），利用此对视频特征进行编码。
3. 动作分类，比如说是SVM或者RF，在视觉词袋上进行训练用以最后的预测。

在这些用了浅层次的人工设计特征的步骤1的算法中，其中**improved Dense Trajectories**(iDT) [6]效果是最好的，iDT利用了密集采样的轨迹特征。同时，3D卷积[7]也在2013年被用以动作识别分类，其不需要额外很多改进便可直接应用在这个领域上。在2014年之后的不久，两个突破性的研究论文构成了我们在这个博文中讨论的所有论文的骨架。他们之间的主要区别围绕在如何组合时空特征的设计选择上。



## 方法一：单流网络(single stream network)

![Karpathy_fusion][Karpathy_fusion]

<div align='center'>
    <b>
        Fig 3, 不同时序融合的方法。
    </b>
</div>

在文献[8]中，基于2D预训练的卷积，作者探索了多种手段进行时序信息的融合。如Fig 3所示，在所有的融合方式中，视频中的连续帧都作为输入。`Single Frame`单帧方法在最后的阶段，使用一个单独的网络结构融合来自于所有帧的信息。`late fusion`晚融合方法使用两个共享参数的网络，分别隔开15帧进行特征提取，同样在最后阶段进行预测结果的融合。`Early fusion`早融合在第一层对连续的10帧进行卷积后进行融合。`Slow Fusion`慢融合是用了多个阶段进行融合，这是一个结合了早融合和晚融合的折中的方法。为了得到最后的预测结果，算法从整个视频中采样多个片段并且得到不同片段的预测得分，最后的预测结果是对这些不同片段得分的平均。

然而在扩展实验中，作者发现这些算法的结果远比领先的手工设计特征算法差，原因有多种：

1. 学习到的时空特征并没有真正地捕获到动作特征信息。（译者：对时空特征并没有很好地组织）
2. 数据集是趋同的，学习到细节特征并不容易。



## 方法二：双流网络(Two streams network)

在这篇开创新的工作[13]中，作者在之前工作的失败教训下，搭建了一个深度学习网络去学习动作特征。作者通过层叠光流向量的方法，对动作特征进行了显式地建模。和只考虑空间上下文的单独网络不同，这个架构有两个独立的网络，其中一个为了空间上下文设计（当然是经过了预训练的），另一个为了提取时序动作信息设计。空间网络的输入是视频中的其中一帧（译者：注意只有一帧，不是整个视频）。作者在时序动作信息网络上，对输入的种类进行了实验，并且发现双向的，层叠了连续10帧的光流特征表现是最好的。双流网络的两个网络是独立训练的，并且最后通过SVM分类器进行组合。最后的预测结果的策略和之前的工作一样，也就是在多个视频采样片段中进行得分平均。

![2stream_high][2stream_high]

<div align='center'>
    <b>
        Fig 4，双流网络结构。
    </b>
</div>

尽管这个方法通过显式地进行局部时序信息的建模，提高了单流网络的性能，其仍然存在一些缺点：

1. 因为视频层面上的预测是从多个采样片段中的得分进行平均得到的，长时间的时序信息建模仍然会在学习过程中遗失。
2. 因为训练片段是在视频中均匀采样的，它们都将碰到一个问题就是**错误标签**(false label)分配。每个片段的真实标签被假设为和整个视频的标签是一致的，然而这个假设不一定成立，当这个动作只发生在视频中的某个片段时，这个假设将会显然地失效。
3. 需要预先计算光流向量并且对其进行储存是一个不够理想的方法。同样的，将这些流分开独立地进行训练，意味着不能很好地进行端到端的训练。



## 在这之后

在这篇博文，接下来我们将基于之前的两篇论文（单流网络和双流网络），介绍一些其他工作，分别是：

1. [LRCN](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review#lrcn)
2. [C3D](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review#c3d)
3. [Conv3D & Attention](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review#attentionandconv3d)
4. [TwoStreamFusion](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review#2streamfusion)
5. [TSN](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review#tsn)
6. [ActionVlad](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review#actionvlad)
7. [HiddenTwoStream](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review#hidden2stream)
8. [I3D](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review#i3d)
9. [T3D](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review#t3d)

这些文章之中的重复的思想可以用下图Fig 5进行总结，这些列出的所有论文都是基于刚才提到的两篇文章的启发而得到的。

![recurrent_theme_high][recurrent_theme_high]

<div align='center'>
    <b>
        Fig 5. 一系列动作识别网络之中的相似的思想总结。
    </b>
</div>

在每一篇文章中，我都列出了他们的主要贡献点，并且对此进行了解释。我同样把它们在基准数据集UCF 101-split1上的表现列举了出来作为基准比较。

### 1. LRCN

> Long-term Recurrent Convolutional Networks for Visual Recognition and Description Donahue et al.
> Submitted on 17 November 2014
> [Arxiv Link](https://arxiv.org/abs/1411.4389)

**主要贡献点**：

- 与之前基于流的网络相对的，在之前采用RNN网络工作的基础上进行搭建。
- 是编码器-解码器架构在图像表征上的扩展延伸。
- 在动作识别应用上的端到端可训练网络架构。

**解释**：

在之前的工作中[9]，作者已经探索了 “利用LSTM网络在可训练的特征图上分别进行训练” 的实验，以确定是否其可以捕获视频片段中的时序信息。可惜的是，他们的结论是：对于被训练的模型提取出来的特征图而言，在卷积特征上进行时序池化被证实是比使用层叠LSTM更为有效的一种做法。在这篇论文中，作者采用相同的思路，在卷积模块（编码器）之后使用了LSTM单元（解码器），但不同的是，作者采用了一种可以端到端训练的网络架构。他们同样比较了RGB模态和光流模态作为输入选择时对于性能的影响，发现在同时输入这两种模态的同时，对预测得分进行加权后的最终结果是最佳结果。（译者：也就是说同时考虑RGB和光流的结果是最佳的，但是要对预测得分进行分别加权。）

![GenericLRCN_high][GenericLRCN_high]

<div align='center'>
    <b>
        Fig 6. 左边：LRCN在动作识别上的应用；右边：在所有任务上的通用的LRCN网络架构。
    </b>
</div>

**算法**：

在训练阶段，先从整个视频中采样出16帧的视频片段。无论对于RGB输入还是光流输入，这个网络架构都是可以端到端训练的。对于每个视频片段而言，其预测结果是每个时间步，也就是每帧的预测得分平均。对于整个视频的最终预测结果而言，其是每个视频片段的预测结果得分平均。

**基准**（UCF101-split1）：

| 得分（score） | 备注（comment）                   |
| ------------- | --------------------------------- |
| 82.92         | 对光流输入和RGB输入的得分加权结果 |
| 71.1          | 只用RGB的结果                     |

**作者评价**：

尽管作者提出了这样一个端到端可训练的网络框架，这个工作仍然是存在一些缺陷的：

- 在视频片段采样的过程中存在有错误标签的问题。
- 难以捕获长时间依赖的时序信息。
- 需要采用光流信息意味着需要分别预先计算出光流特征。

在工作[10]中，他们尝试通过采用更低的视频空间解析度，对这个时间序列建模问题进行补偿。因为视频空间解析度更低了，便可以采用更长的视频片段，比如说60帧而不是16帧，以提高整个模型的性能。（译者注：减少空间解析度是因为太大了目前的硬件条件很难支持，比如说显存大小等。）

-----

### 2. C3D

> Learning Spatiotemporal Features with 3D Convolutional Networks Du Tran et al.
> Submitted on 02 December 2014
> [Arxiv Link](https://arxiv.org/pdf/1412.0767)

**主要贡献点**：

- 提出用3D卷积网络作为特征提取器（以前也有人这样做过）
- 对最佳的3D卷积核的设置和结构进行了扩展性探索
- 采用转置卷积层对模型的决策进行了解释

**解释**：

在这个工作中，作者是在Karpathy的基础上搭建了整个网络。然而，与之不同的是，其没有采用在帧间应用2D卷积的策略，而是使用3D卷积在视频上进行特征提取。这个想法大致是在Sports1M这个大数据集上进行网络的预训练之后，然后采用这个预训练模型（可能有不同的时间帧数选择）作为特征提取器去在其他数据集上进行特征提取（译者注：3D卷积网络参数量巨大，在小数据集上直接应用很容易导致过拟合的现象，因此需要采用迁移学习的方法，在大数据集上进行预训练。）。他们发现一个简单的线性分类器比如说SVM，应用在提取出来的特征之后进行分类，其表现就可以超过以往的最好的模型。这个模型在一些人工提取的特征比如iDT等额外引入的情况下，性能更佳。

![c3d_high][c3d_high]

<div align='center'>
    <b>
        Fig 7. C3D网络和其他单流网络的区别。
    </b>
</div>

这个工作其他值得注意的地方是使用了转置卷积层去尝试解释模型作出的决策。他们发现这个网络首先注意到了前几帧的空间特征，并且对后续帧的动作（motion）进行着跟踪。

**算法**：

在训练中，首先从每个视频中采样五组随机的时长2秒钟的视频片段，这些视频片段的类别标签和该完整视频的标签一致。在测试阶段，从视频中随机采样10个视频片段，并且对其进行预测，最终预测结果同样是这些片段的预测结果得分的平均。

![trial][trial]

<div align='center'>
    <b>
        在时空维度上应用3D卷积提取特征。
    </b>
</div>

**基准**（UCF101-split1）：

| 得分（score） | 备注（comment）                 |
| ------------- | ------------------------------- |
| 82.3          | C3D (1 net) + linear SVM        |
| 85.2          | C3D (3 nets) + linear SVM       |
| 90.4          | C3D (3 nets) + iDT + linear SVM |

**评价**：

这个网络同样没能解决长时间的信息依赖问题。同时，训练这样一个大网络是需要很多计算资源的，并且对训练集的大小要求很高，特别是对于医学图像而言，因为医学图像并没法从经过自然图像预训练的模型的帮助下得到性能提高。

**笔记**：

与此同时，在工作[11]中，作者引入了分解3D卷积网络的概念（$F_{ST}CN$），也就是将3D卷积分解为空间2D卷积之后紧接着时间1D卷积的操作。这个分解3D卷积网络同样在UCF101 split上取得了不错的表现。

![fstcn_high][fstcn_high]

<div align='center'>
    <b>
        3D卷积的分解。
    </b>
</div>

----

### 3. C3D & 注意力机制

> Describing Videos by Exploiting Temporal Structure Yao et al.
> Submitted on 25 April 2015
> [Arxiv Link](https://arxiv.org/abs/1502.08029)

**关键贡献点**：

- 新颖的3D CNN-RNN编码器-解码器结构，这个结构可以提取局部的时空信息。
- 在CNN-RNN编码器-解码器框架中使用了注意力机制去提取全局的上下文信息。

**解释**：

尽管这个工作并不是和动作识别直接相关的，它却是视频表征中的一个代表性工作。在这个工作中，作者使用了一个3D CNN+LSTM网络结构作为基础架构去进行视频描述任务。在整个基础之上，作者使用了一个预训练后的3D CNN网络以提升性能。

**算法**：

整个工作的设置几乎和在LRCN中描述的编码器-解码器结构一样，除了存在两点区别：

1. 视频片段的**3D CNN特征图**和同样的视频片段的**经过层叠的2D特征图**将会拼接在一起以提高对视频帧的表征能力，而并不是简单地把3D CNN的特征图传递到LSTM后进行时序建模。注意：在这里使用的2D和3D CNN是经过预训练的，因此整个过程并不是像在LRCN中一样端到端训练的。
2. 与之前的工作简单的对所有片段的预测得分进行平均不同，在这个工作中，作者在时序特征中采用了在时间方向加权平均的方法。这个注意力加权取决于LSTM在每个时间步的输出结果。

![Larochelle_paper_high][Larochelle_paper_high]

<div align='center'>
    <b>
    	动作识别中的注意力机制。
    </b>
</div>

**基准**：

无，这个网络是用在视频描述而不是动作识别之中的。

**备注**：

这个2015年的工作是第一次在视频表征中引入注意力机制的标志性工作。

----

### 4. 双流融合

> Convolutional Two-Stream Network Fusion for Video Action Recognition Feichtenhofer et al.
> Submitted on 22 April 2016
> [Arxiv Link](https://arxiv.org/abs/1604.06573)

**主要贡献点**：

- 通过更好的长时间损失函数对长时间时序依赖问题进行了建模
- 新颖的多层次融合框架

**解释**：

在这个工作中，作者使用了双流网络框架（不过用了两种新颖的方法进行的，对比之前的工作而言），并且在没有任何模型参数量增加的情况下，提高了模型性能。作者主要探讨了以下两种主要想法的有效性：

1. 融合时间流和空间流。对于一个区分梳头和刷牙动作类别的任务而言，空间流网络可以捕获其在视频中的空间依赖（是否是头发或者是牙齿），而时间流网络可以捕获视频中的每个空间位置上的物体的周期性动作（motion）。 因此，将一个特定的区域映射到对应区域的时间特征图是非常重要的， 为了达到同样的目的，需要在早期阶段对网络进行融合，以便将相同像素位置的响应进行对应，而不是在最后进行融合（就像在基础的双流网络一般）。
2. 在时间帧的方向上组合时间流网络，使得其可以对长时间时序依赖进行建模。

**算法**：

这个工作大部分和之前的双流网络一致，除了：

1. 就像下图所示，来自于两个流的`conv_5`层的输出通过`conv+pooling`的手段进行了融合。在某端的一层同样有另外一种融合。最后的融合结果被用来计算时空损失函数。

![fusion_strategies_high][fusion_strategies_high]

<div align='center'>
    <b>
    	可供选择的融合时间流和空间流的策略。右边的策略表现更佳。
    </b>
</div> 

2. 对于时间流融合来说，时间流网络的输出，在时间轴上进行了层叠，通过`conv+pooling`层融合的结果同样被用于了另外一个时序损失函数的计算。

![2streamfusion][2streamfusion]

 

| 得分（score） | 备注（comment）  |
| ------------- | ---------------- |
| 92.5          | 双流融合网络     |
| 94.2          | 双流融合网络+iDT |

**备注**：

这个工作的作者用这个工作建立双流网络方法的“霸权”地位，因为其将模型性能提高到了C3D的程度，但是却没有C3D模型中的额外的参数量。

----

### 5. TSN

> Temporal Segment Networks: Towards Good Practices for Deep Action Recognition Wang et al.
> Submitted on 02 August 2016
> [Arxiv Link](https://arxiv.org/abs/1608.00859)

**主要贡献点**：

- 针对长时间时序依赖问题的有效的解决方案
- 使用了batch norm层，dropout层和预训练，并且取得了不错的效果，开启了后续使用这些“套路”的习惯。

**解释**：

在这个工作中，作者在双流架构的基础上进行扩展，达到了领先的水平。这个工作和之前的工作主要有两点不同：

1. 他们提出在视频中稀疏地进行视频片段采样，而不是对整个视频进行随机采样，以更好地对长时间时序信号进行建模。
2. 对于视频最后的类别进行预测，作者探索了多种策略，最后结果证实最好的策略是：
   - **分别**融合时间流和空间流（如果还有其他模态的数据，那么这些模态的数据也作为一个流需要包含入内）的预测结果，这里的融合指的是在不同的片段之间进行类别得分的平均操作。
   - 通过权值加权平均的方法融合最终时间和空间上的得分，并且在所有类别上应用softmax函数，得到最终的预测概率分布。

这个工作的其他重要部分是缓解了过拟合问题（因数据集大小过小导致的），并且证实了现在流行的很多技术手段的有效性，比如Batch Normalization，Dropout和预训练等



 The other important part of the work was establishing the problem of overfitting (due to small dataset sizes) and demonstrating usage of now-prevalent techniques like batch normalization, dropout and pre-trainign to counter the same. The authors also evaluated two new input modalities as alternate to optical flow - namely warped optical flow and RGB difference. 



# Reference

[1].  [ConvNet Architecture Search for Spatiotemporal Feature Learning](https://arxiv.org/abs/1708.05038) by Du Tran et al.

[2].  [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)

[3].  [Action recognition by dense trajectories](https://hal.inria.fr/inria-00583818/document) by Wang et. al.

[4].  [On space-time interest points](http://www.irisa.fr/vista/Papers/2005_ijcv_laptev.pdf) by Laptev

[5].  [Behavior recognition via sparse spatio-temporal features](http://webee.technion.ac.il/control/info/Projects/Students/2012/Itay Hubara and Amit Nishri/Book/Papers-STIP/DollarVSPETS05cuboids.pdf) by Dollar et al

[6].  [Action Recognition with Improved Trajectories](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Wang_Action_Recognition_with_2013_ICCV_paper.pdf) by Wang et al.

[7].  [3D Convolutional Neural Networks for Human Action Recognition](https://pdfs.semanticscholar.org/52df/a20f6fdfcda8c11034e3d819f4bd47e6207d.pdf) by Ji et al.

[8].  [Large-scale Video Classification with Convolutional Neural Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf) by Karpathy et al.

[9].  [Beyond Short Snippets: Deep Networks for Video Classification](https://arxiv.org/abs/1503.08909) by Ng et al.

[10].  [Long-term Temporal Convolutions for Action Recognition](https://arxiv.org/abs/1604.04494) by Varol et al.

[11].  [Human Action Recognition using Factorized Spatio-Temporal Convolutional Networks](https://arxiv.org/abs/1510.00562) by Sun et al.

[12].  [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) by Huang et al.

[13].  Simonyan K, Zisserman A. Two-stream convolutional networks for action recognition in videos[C]//Advances in neural information processing systems. 2014: 568-576. 





[2stream_high]: ./imgs/2stream_high.png
[Karpathy_fusion]: ./imgs/Karpathy_fusion.jpg
[banner]: ./imgs/banner.gif
[recurrent_theme_high]: ./imgs/recurrent_theme_high.png

[GenericLRCN_high]: ./imgs/GenericLRCN_high.jpg

[c3d_high]: ./imgs/c3d_high.png
[trial]: ./imgs/trial.gif
[fstcn_high]: ./imgs/fstcn_high.png

[Larochelle_paper_high]: ./imgs/Larochelle_paper_high.png

[fusion_strategies_high]: ./imgs/fusion_strategies_high.png
[2streamfusion]: ./imgs/2streamfusion.png





