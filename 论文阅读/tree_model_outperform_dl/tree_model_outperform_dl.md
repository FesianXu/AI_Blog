<div align='center'>
    为何基于树的模型在表格型数据中能优于深度学习？
</div>


<div align='right'>
    FesianXu 20220908 at Baidu Search Team
</div>

# 前言

基于树的模型（Tree-based model），比如GBDT，XGBoost，Random Forest等仍然是Kaggle，天池等数据比赛中最为常用的算法，在遇到表格型数据（Tabular data）的时候，这些树模型在大多数场景中甚至表现优于深度学习，要知道后者已经在诸多领域（CV，NLP，语音处理等）已经占据了绝对的优势地位。那么为何如此呢？论文[1]给出了一些可能的答案，本文对此进行笔记。**如有谬误请联系指出，本文遵守[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**github**: https://[github](https://so.csdn.net/so/search?q=github&spm=1001.2101.3001.7020).com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**： 机器学习杂货铺3号店



----

表格型数据如Fig 1.所示，其每一行是一个观测（observation），或者说样本（sample），每一列是一维特征（feature），或者说属性（attribution）。这种数据在现实生活中经常遇到，比如对银行客户进行放贷风险评估就有类似的表格型数据。表格型数据的特征一般有两类，数值型特征（numeric feature）和类别型特征（categories feature）。在天池和kaggle等数据比赛中，经常会遇到类似的表格型数据，而常常称霸榜单的算法通常都是基于树模型的方法，而不是深度学习模型，即使后者已经在诸多领域隐约一统江湖。那么为何深度学习在表格型数据上会遭到如此奇耻大辱呢？论文[1]从三个角度进行了分析。

![tabular_data.jpg][tabular_data.jpg]

<div align='center'>
    <b>
        Fig 1. 表格型数据示例，每一行是一个观测（observation），或者说样本（sample），每一列是一维特征（feature），或者说属性（attribution）。
    </b>
</div>

为了让实验结果更为置信可比，作者收集了45个用于对比试验的表格型数据集，这些数据集的采集和收集过程请参考原文，这里就不介绍了。由于原生的树模型无法对类别型特征进行处理（LightGBM除外，其采用了Fisher[2]的方法进行类别特征分组。），因此本文对类别型数据进行了one-hot编码处理。从分类和回归任务上看，如Fig 2.所示，无论从只有数值型特征的数据集，还是数值型和类别型数据集共存的数据集看，的确都是树模型（XGBoost, RandomForest, GBT）效果要更好。

![tree_model_outperform_dl][tree_model_outperform_dl]

<div align='center'>
    <b>
        Fig 2. 无论是在分类还是回归任务中，树模型在表格型数据上的表现都显著优于深度学习模型。
    </b>
</div>

那么表格型数据为什么那么神奇，能让树模型在各种真实场景的表格数据中都战胜深度学习呢？作者认为有以下三种可能：

1. 神经网络倾向于得到过于平滑的解
2. 冗余无信息的特征更容易影响神经网络
3. 表格型数据并不是旋转不变的

我们分别分析下作者给这三个观点带来的论据。



# 神经网络倾向于得到过于平滑的解

首先我们假设表格型数据的标注是具有噪声的，并且假设其是高斯噪声，那么可以通过高斯平滑（Gaussian Smooth）进行标注平滑，高斯平滑采用高斯核，可见博文[3]所示。高斯核公式如(1-1)所示
$$
\mathcal{K}(\mathbf{x}_i, \mathbf{x}_j) = \exp{(-\dfrac{1}{2}(\mathbf{x}_i-\mathbf{x}_j)^{\mathrm{T}} \Sigma^{-1}(\mathbf{x}_i-\mathbf{x}_j))} 
\tag{1-1}
$$
其中的$\mathbf{x}_i \in \mathbb{R}^{D}$为第$i$个样本的特征，一共有$D$维特征，$\Sigma \in \mathbb{R}^{D \times D}$为$\mathbf{x}_i$和$\mathbf{x}_j$的协方差矩阵。通过博文[1]的介绍，我们可知协方差矩阵$\Sigma = (\mathbf{A}^{\mathrm{T}}\mathbf{A})^{-1}$其实描述了高斯分布在特征不同维度的线性拉伸情况，为了人工对这个拉伸情况进行控制，可以在其基础上乘上一个尺度系数$\sigma^{2}$，也即是可以将(1-1)公式中的协方差矩阵改为$\tilde{\Sigma} = \sigma^2 \Sigma$，那么$\tilde{\Sigma}^{-1} = \dfrac{1}{\sigma^2} \Sigma^{-1}$，也即是$\sigma^2$越大，其拉伸扩大的更多，平滑效果也就更大。高斯核描述了两个样本之间在高斯分布上的相关程度$\mathcal{K}(\mathbf{x}_i, \mathbf{x}_j)$，可以根据这个相关程度对样本的标签进行加权平滑，如式子(1-2)所示。
$$
\tilde{Y}_i = \dfrac{\sum_{j=1}^{N} \mathcal{K}(\mathbf{x}_i, \mathbf{x}_j) Y_j}{\sum_{j=1}^{N} \mathcal{K}(\mathbf{x}_i, \mathbf{x}_j)}
\tag{1-2}
$$
其中的$Y_i$为第$i$个样本的真实标签，而$\tilde{Y}_i$为第$i$个样本的平滑后标签，可见到是根据高斯分布中的相关程度进行$N$个样本的加权平滑得到最终的样本标签。在本文中，作者分别将$\sigma^2$设为0.05, 0.1, 0.25，当$\sigma^2=0$的时候，认为是采用原始标签。如Fig 3. (a)所示，可以发现进行了标签的高斯平滑后，基于树的模型（GBT, 随机森林）的测试表现下降明显，而基于神经网络的模型（FT Transformer和Resnet）则下降不明显，并且可以观察到树模型下降后的性能和神经网络的性能差距，随着平滑系数的增大而减少。这说明了神经网络对于表格型数据，在某些程度上是进行了标签的高斯平滑处理的，而树模型则不会进行这个操作，因此神经网络的结果会更为的平滑（**笔者：虽然笔者认为这个结论很可能是成立的，但是从目前试验看，笔者认为这个只能证明是更加的高斯平滑，不能证明是更加平滑**）。同时，笔者对树模型和神经网络模型的决策边界进行了可视化，如Fig 3. (b)所示，作者通过树模型的权重大小，挑选了两维最为重要的特征，然后进行可视化。我们可以看到，树模型明显决策边界更为跳动，而神经网络模型则明显更为平滑，有部分边缘样本点将被神经网络漏检。**这里笔者主要有一点质疑，就是这里为了可视化方便而挑选了两位最为重要的特征作为横轴纵轴，但是挑选的依据是树模型的权重，而我们知道树模型的权重其实是根据分裂增益进行计算得到的，这样挑选出来的特征进行可视化决策边界，会不会天然对树模型有优势呢？比如会产生更多的分裂点，导致能对更多边缘样本进行检出？**

![finding_1_smooth_result][finding_1_smooth_result]

<div align='center'>
    <b>
        Fig 3. (a) 分别设置不同的平滑系数后，不同模型的测试集表现； (b) 随机森林和MLP算法对样本的测试集决策边界情况。
    </b>
</div>

总的来说，作者通过以上的试验，证实了我们的表格型数据的目标（也即是标签）大多数不是平滑的（至少不是高斯平滑的），对比于树模型，神经网络会倾向于去拟合那些不规则的样本，导致在某些程度对这些样本进行了平滑。



# 冗余无信息的特征更容易影响神经网络

作者认为表格型数据中含有更多无信息量（uninformative）的冗余特征，而树模型对这些无信息特征更为鲁棒。作者将表格型数据的特征按照重要性降序排序（此处的重要性同样是由树模型的分裂增益进行判断），然后按照百分比将不重要的特征依次剔除后进行试验。如Fig 4. (a)所示，其中的绿线是树模型对移除后的特征（也就是更为重要的特征）进行拟合得到测试曲线，我们发现移除大部分不重要特征对结果的影响并不大（去除了50%的特征后仍有80%左右的准确率），这也意味着其实只有少部分特征是具有高信息量的。而红线是树模型对移除的特征（也就是更为不重要的特征）进行拟合得到的测试曲线，我们能发现即便用了一半的（少信息量）特征，其测试结果也仅有50%，同样验证了我们之前得到的结论——少部分特征carry了全场。对比神经网络的结果，如Fig 4. (b)所示，左图表示树模型和神经网络模型在去除不同比例的不重要特征后的测试曲线变化，我们发现当去除更多的不重要特征后，神经网络和树模型的测试表现差别逐渐减少到相等，这意味着神经网络其实对于这种冗余无信息的特征更为不鲁棒。而Fig 4. (b)的右图则是通过高斯分布产出了一些伪特征，这些伪特征无信息量，通过把这些无信息量特征拼接到原有的样本上，我们模拟了引入无信息特征的过程。我们发现引入更多的无信息特征，神经网络和树模型的测试效果差距将会明显增大。

![finding_2_rm_add_uninformative_feat][finding_2_rm_add_uninformative_feat]

<div align='center'>
    <b>
        Fig 4. (a) 树模型在去除不同比例的无信息特征的表现；(b) 树模型和神经网络在无信息量特征上的表现差别，将会随着无信息量特征的减少而减少。
    </b>
</div>

至于笔者的看法，**笔者认为这里挑选重要性特征的依据，同样是根据树模型的权重进行判断的，用树模型权重挑选出所谓不重要的特征，然后进行测试验证去说明树模型对不重要特征更为鲁棒，是否会对神经网络不公平呢？** 当然Fig 4. (b)的试验由于不依赖与特征的权重，而是由高斯分布产出一些无信息量特征，笔者认为还是更可靠，更有说服力的。



# 表格型数据并不是旋转不变的

作者在文章中认为表格型数据并不是旋转不变的，而神经网络会对数据进行旋转不变的处理，因此效果更差。首先笔者要说明什么是旋转不变性（rotation invariant），对于函数$f(x)$，如果有：
$$
f(\mathbf{x}^{\prime}) = f(\mathbf{x} \mathbf{H}) = f(\mathbf{x})
\tag{3-1}
$$
则称之为该函数具有旋转不变性。其中的$\mathbf{x} \in \mathbb{R}^{1 \times n}, \mathbf{H} \in \mathbb{R}^{n \times n}$。不难发现，神经网络MLP天然具有这种旋转不变性，神经网络MLP每层由全连接层组成，数学形式正是如式子(3-1)描述的矩阵乘法。因此神经网络从原理上看，在保证基础网络（绿色节点）不变的情况下，只需要增加一层全连接层$\mathbf{H}^{\mathrm{T}} \in \mathbb{R}^{n \times n}$，只要蓝色节点的网络学到$\mathbf{H}\mathbf{H}^{\mathrm{T}} = \mathbf{H}^{\mathrm{T}} \mathbf{H} = \mathbf{I}$即可（也称为旋转不变矩阵）。这对于神经网络而言并不是一件难事。 

![rotation_invariance_of_MLP][rotation_invariance_of_MLP]

<div align='center'>
    <b>
        Fig 5. 神经网络MLP天然具有旋转不变性，最简单的例子就是添加一层全连接层即可实现。
    </b>
</div>

这种旋转不变性对于表格型数据而言并不是一件好事。不同于图片数据，图片数据训练过程中，经常会考虑采用对图片进行一定的旋转，以增强模型的旋转不变性。这个是因为图片像素作为一种各向同性的原始特征，每个像素并没有各自的物理含义，因此旋转也不会改变其物理含义。相反地，由于图片实体在不同角度下大部分都保持同一语义（当然也有例外，比如数字9和6的图片，进行180度旋转后可能导致语义错误），因此期望模型具有旋转不变性。但是表格型数据的每一列通常都是具有显著物理含义的，比如性别，年龄，收入，工作类型等等，对这些进行数据旋转，那么产出的特征将不具有任何物理含义了。

如Fig 5. (a)所示，在实验中将数据集的特征进行随机旋转，观察树模型和神经网络模型的测试结果。我们可以发现，基于Resnet的测试结果基本上没有任何变化，这证实了Resnet具有函数上的旋转不变性。而树模型GBT和随机森林均有大幅度的性能下降（~20%），由此我们可以得出结论，神经网络模型在处理特征的过程中，已经对特征进行了一定程度的旋转，因此在人工加入旋转干扰的情况下，神经网络的测试结果几乎不下降。而树模型无旋转不变性，当引入人工旋转干扰后，由于数据特征的物理含义完全被打乱了，因此性能大幅度下降。注意到一点，表格型数据中含有大量无信息量的特征，对数据进行的旋转操作，会直接导致有信息特征中混入无信息特征，从而影响特征效果。如Fig 5. (b)所示，当去掉不重要的特征后（即是按重要性排序的后50%特征），同样进行人工旋转干扰，我们发现树模型的结果下降得没有那么厉害了（~15%），这是因为无关特征被大量去除后，人工旋转干扰导致的无信息特征引入减少了。

![finding_3_rotation_invariance][finding_3_rotation_invariance]

<div align='center'>
    <b>
        Fig 5. (a) 进行数据旋转 VS 不进行数据旋转的试验结果； (b) 去除了50%不重要的特征后，重新进行人工特征旋转干扰试验。
    </b>
</div>



----



# Reference

[1]. Grinsztajn, Léo, Edouard Oyallon, and Gaël Varoquaux. "Why do tree-based models still outperform deep learning on tabular data?." *arXiv preprint arXiv:2207.08815* (2022).

[2]. Fisher, Walter D. "On grouping for maximum homogeneity." *Journal of the American statistical Association* 53, no. 284 (1958): 789-798.

[3]. https://blog.csdn.net/LoseInVain/article/details/80339201, 《理解多维高斯分布》

[4]. Andrew Y. Ng. Feature selection, L 1 vs. L 2 regularization, and rotational invariance. In Twenty-First International Conference on Machine Learning - ICML ’04, page 78, Banff, Alberta, Canada, 2004. ACM Press. doi: 10.1145/1015330.1015435.









[tabular_data.jpg]: ./imgs/tabular_data.jpg
[tree_model_outperform_dl]: ./imgs/tree_model_outperform_dl.png
[finding_1_smooth_result]: ./imgs/finding_1_smooth_result.png
[finding_2_rm_add_uninformative_feat]: ./imgs/finding_2_rm_add_uninformative_feat.png
[finding_3_rotation_invariance]: ./imgs/finding_3_rotation_invariance.png
[rotation_invariance_of_MLP]: ./imgs/rotation_invariance_of_MLP.png









