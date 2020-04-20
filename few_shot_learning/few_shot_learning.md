<div align='center'>
    数据，模型，算法共同决定深度学习模型效果
</div>

<div align='right'>
    2020/4/20 FesianXu
</div>



在文献[1]中对few-shot learning进行了很好地总结，其中提到了一个比较有意思的观点，这里和大家分享下。先抛开few-shot learning的概念，我们先从几个基本的机器学习的概念进行分析。



**期望风险最小化（expected risk minimization）：** 假设数据分布$p(\mathbf{x},y)$已知，其中$\mathbf{x}$是特征，$y$是标签，在给定了特定损失函数$\mathcal{L}(\cdot)$的情况下，对于某个模型假设$h \in \mathcal{H}$，我们期望机器学习算法能够最小化其期望风险，期望风险定义为：
$$
R(h) = \int \mathcal{L}(h(\mathbf{x}), y) dp(\mathbf{x}, y) = \mathbb{E}[\mathcal{L}(h(\mathbf{x}), y)]
\tag{1}
$$
假如模型的参数集合为$\theta$，那么我们的目标是：
$$
\theta = \arg \min _{\theta} R(h)
\tag{2}
$$


**经验风险最小化（empirical risk minimization）：** 实际上，数据分布$p(\mathbf{x},y)$通常不可知，那么我们就不能对其进行积分了，我们一般对该分布进行采样，得到若干个具有标签的样本，我们将其数量记为$I$，那么我们用采样结果对这个分布进行近似，因此，我们追求最小化经验风险，这里的经验（experience）的意思也就是指的是采样得到的数据集：
$$
R_{I}(h) = \dfrac{1}{I} \sum_{i=1}^{I} \mathcal{L}(h(\mathbf{x}_i), y_i)
\tag{3}
$$
此处的经验风险(3)就可以近似期望风险(1)的近似进行最小化了（当然，在实践中通常需要加上正则项）。

我们进行以下三种表示：
$$
\hat{h} = \arg \min_{h} R(h)
\tag{4}
$$

$$
h^{*} = \arg \min_{h \in \mathcal{H}} R(h)
\tag{5}
$$

$$
h_{I} = \arg \min_{h \in \mathcal{H}} R_{I}(h)
\tag{6}
$$

其中(4)表示最小化期望风险得到的理论上最优的假设$\hat{h}$，(5)表示在指定的假设空间$h \in \mathcal{H}$中最小化期望风险得到的约束最优假设$h^{*}$，(6)表示在指定的数据量为$I$的数据集上进行优化，并且在指定的假设空间$h \in \mathcal{H}$下最小化经验风险得到的最优假设$h_I$。

因为我们没办法知道$p(\mathbf{x},y)$，因此我们没办法求得$\hat{h}$，那么作为近似，$h^*$是在假定了特定假设空间时候的近似，而$h_I$是在特定的数据集和特定假设空间里面的近似。进行简单的代数变换，我们有(7):
$$
\mathbb{E}[R(h_I)-R(\hat{h})] = \mathbb{E}[R(h^*)-R(\hat{h})+R(h_I)-R(h^*)] =  \\ \mathbb{E}[R(h^*)-R(\hat{h})]+\mathbb{E}[R(h_I)-R(h^*)]
\tag{7}
$$
其中用$\mathcal{E}_{app}(\mathcal{H}) = \mathbb{E}[R(h^*)-R(\hat{h})]$， $\mathcal{E}_{est}(\mathcal{H}, I) = \mathbb{E}[R(h_I)-R(h^*)]$。$\mathcal{E}_{app}(\mathcal{H})$表征了在期望损失下，在给定的假设空间$\mathcal{H}$下的最优假设$h^*$能多接近最佳假设$\hat{h}$。而$\mathcal{E}_{est}(\mathcal{H, I})$表示了在给定假设空间$\mathcal{H}$下，对经验风险进行优化，而不是对期望风险进行优化造成的影响。不失特别的，我们用$D_{train}$表示整个训练集，有$D_{train} = \{\mathbf{X}, \mathbf{Y}\}, \mathbf{X} = \{\mathbf{x}_1,\cdots,\mathbf{x}_n\}, \mathbf{Y} = \{y_1,\cdots,y_n\}$。

我们不难发现，整个深度模型算法的效果，最后取决于假设空间$\mathcal{H}$和训练集中数据量$I$。换句话说，为了减少总损失，我们可以从以下几种角度进行考虑：

1. 数据，也就是$D_{train}$。

2. 模型，其决定了假设空间$\mathcal{H}$。

3. 算法，如何在指定的假设空间$\mathcal{H}$中去搜索最佳假设以拟合$D_{train}$。

   

通常来说，如果$D_{train}$数据量很大，那么我们就有充足的监督信息，在指定的假设空间$h \in \mathcal{H}$中，最小化$h_I$得到的$R(h_I)$就可以提供对$R(h^*)$的一个良好近似。然而，在few-shot learning (FSL)中，某些类别的样本数特别少，不足以支撑起对良好假设的一个近似。其经验风险项$R_{I}(h)$和期望风险项$R(h)$可能有着很大的距离，从而导致假设$h_I$过拟合。事实上，这个是在FSL中的核心问题，即是 经验风险最小假设$h_I$变得不再可靠。整个过程如Fig 1所示，左图有着充足的样本，因此其经验风险最小假设$h_I$和$h^*$相当接近，在$\mathcal{H}$设计合理的情况下，可以更好地近似$\hat{h}$。而右图则不同，$h_I$和$h^*$都比较远，跟别说和$\hat{h}$了。

![fsl_sufficient_samples][fsl_sufficient_samples]

<div align='center'>
    <b>
        Fig 1. 样本充足和样本缺乏，在学习过程中结果的示意图。
    </b>
</div>

为了解决在数据量缺少的情况下的不可靠的经验风险问题，也就是FSL问题，我们必须要引入先验知识，考虑到从数据，模型，算法这三个角度分别引入先验知识，现有的FSL工作可以被分为以下几种：

1. 数据。在这类型方法中，我们利用先验知识去对$D_{train}$进行数据增广(data augment)，从数据量$I$提高到$\widetilde{I}$，通常$\widetilde{I} >> I$。随后标准的机器学习算法就可以在已经增广过后的数据集上进行。因此，我们可以得到更为精确的假设$h_{\widetilde{I}}$。如Fig 2 (a)所示。
2. 模型。这类型方法通过先验知识去约束了假设空间 $\mathcal{H}$ 的复杂度，得到了各位窄小的假设空间$\widetilde{\mathcal{H}}$。如Fig 2 (b) 所示。灰色区域已经通过先验知识给排除掉了，因此模型不会考虑往这些方向进行更新，因此，往往需要更少的数据就可以达到更为可靠的经验风险假设。
3. 算法。这类型的方法考虑使用先验知识，指导如何对$\theta$进行搜索。先验知识可以通过提供一个好的参数初始化，或者指导参数的更新步，进而影响参数搜索策略。对于后者来说，其导致的搜索更新步由先验知识和经验风险最小项共同决定。

![data_model_algo][data_model_algo]

<div align='center'>
    <b>
        Fig 2. 分别从数据，模型和算法三个角度去引入先验知识。
    </b>
</div>

# Reference

[1]. Wang Y, Yao Q, Kwok J, et al. Generalizing from a few examples: A survey on few-shot learning[M]//arXiv: 1904.05046. 2019.







[fsl_sufficient_samples]: ./imgs/fsl_sufficient_samples.jpg
[data_model_algo]: ./imgs/data_model_algo.jpg



