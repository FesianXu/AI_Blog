<div align="center">
    DoReMi——一种通过代理模型估计大模型预训练最佳数据配比的方法
</div>


<div align="right">
    FesianXu 20250105 at Wechat Search Team
</div>

# 前言

LLM的预训练是决定其底座能力的至关重要的步骤，其预训练数据通常会包含有多种领域的数据，如何调整不同领域的数据配比（可以理解为采样频率）是极其重要的大模型预训练研究点。本文介绍DeepMind提出的一种基于代理模型去估计最佳数据配比的方法，希望对读者有所帮助。**如有谬误请见谅并联系指出，本文遵守[CC 4.0 BY-SA](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

- **关键字**：LLM预训练数据配比、代理模型估计数据配比、大模型预训练
- **发表会议**：NIPS 2024

$\nabla$ 联系方式：

- e-mail： FesianXu@gmail.com
- github： https://github.com/FesianXu
- 知乎：https://www.zhihu.com/people/FesianXu
- 微信公众号：机器学习杂货铺3号店

![qrcode][qrcode]



-----



为了提高LLM底座的通用能力，通常预训练数据都会包含有各种领域的数据，比如The Pile [2] 就是一个800GB大小的，涵盖了22个不同领域的常用预训练数据集，如Fig 1所示。对于LLM预训练而言（采用next token prediction的自回归方式进行预训练），不同领域数据的配比很重要，之前的工作大部分基于启发式的方法拍定配比（比如均匀采样，或者根据不同领域数据的数据量大小进行采样），由于不同领域数据的学习难度各不相同，这些启发式的配比方法不能保证LLM在预训练阶段充分利用数据。本文旨在利用一个小规模的代理模型（280M参数量），通过Group DRO (Distributionally Robust Optimization) [3-4] [^1]的方式去寻找最佳的数据配比，然后在全量参数的LLM（8B参数量）上进行预训练，通过这种方式，作者发现预训练速度能加速2.6倍，并且最终能获得6.5%的效果提升，最终整个方法被称之为 DoReMi （Domain Reweighting with Minimax Optimization），接下来让我们具体看看这个方法是怎么实施的。

![fig2_the_pile_domains][fig2_the_pile_domains]

<div align="center">
    <b>
        Fig 1. The Pile数据集中包含有22个不同领域的数据。
    </b>
</div>

首先，我们先理解一个点：LLM在通过自回归进行预训练的时候，其**最理想状态应该是能拟合不同领域数据的分布**，也就是各个领域数据的困惑度都最小化 [^2]。因此，如果我们能在“监控”预训练过程中的各个领域数据中，通过损失大小确定拟合程度**最差**的样本来自于哪个领域，然后适当提高这个领域数据的采样率，那么理论上LLM就能有更多机会去学习较难拟合的领域的数据分布，从而以更小的训练代价实现更好的拟合效果。当然，完整的LLM参数量动辄10亿（B）规模，训练一次的成本较高，我们假设参数量较小的（亿级别，如280M）LLM模型的**预训练趋势**和完整参数量（8B）的LLM模型是类似的，因此可以共享相同的数据配比，我们称此处的较小参数量模型为代理模型，通过代理模型我们能找出最佳数据配比，然后给完整参数模型进行预训练，整个过程可以参考Fig 2.示意。

由于需要判断拟合程度最差的样本，这是一个“比较”的过程，比较的一方是代理模型（proxy model），而比较的另一方则是一个参考模型（reference model） ，参考模型指的是在默认的启发式数据配比下，用和代理模型同样参数量的模型（也就是280M），预训练得到的模型。

![fig3_framework][fig3_framework]

<div align="center">
    <b>
        Fig 2. DoReMi的基本框架示意图，通过一个代理模型对预训练数据集合中不同领域数据的采样率进行重新参数化，得到最佳数据配比后去训练最终的LLM。
    </b>
</div>

至此，我们需要了解的DoReMi的基本思路已经介绍完了，我们开始深入细节。首先先进行形式化表示，假设预训练数据集中具有$k$个域的数据，表示为$D_{i}, i=1,\cdots,k$，每个域的权重记为$\alpha \in \Delta^k$，可以理解为在$k$个域中的采样概率分布，也就是$\sum_{i}^k \alpha_i = 1$。那么训练数据的分布可以表示为：
$$
P_{\alpha} = \sum_{i=1}^k \alpha_i \cdot \mathrm{unif}(D_i)
\tag{1}
$$
其中$\mathrm{unif}(D) = \dfrac{1}{|D|} \sum_{x \in D} \delta_{x}$是一个在数据集合$D$上的均匀分布，如果在$x = x^{\prime}$ 的时候 $\delta_x(x^{\prime}) = 1$，否则$\delta_x(x^{\prime}) = 0$。整个DoReMi的过程，可以描述为：

**第一步，训练一个参考模型**

首先采用初始化的数据配比$\alpha_{ref}$先训练一个280M参数量的参考模型$p_{ref}$，此处的数据配比可以根据启发式的方式选取，比如采用数据集中不同领域数据的数据量比例作为配比，见Fig 1.的baseline一列。



**第二步，通过Group DRO的方式训练一个代理模型并且得到新的域权重**

有了参考模型$p_{ref}$之后，就可以开始训练代理模型和学习新的域权重了（也即是学习得到$\bar{\alpha}$），整个过程采用DRO-LM的框架 [5] ，采用Group DRO优化器进行训练，其中的$\theta$为代理模型的参数。整个框架的公式如公式(2)所示，不难看出，这是一个`minmax`过程，其优化目标正如我们前文讨论的，优化（体现在min目标上）各域上最差拟合程度样本（体现在max目标上）的损失，我们着重讲解下这个公式。
$$
\min_{\theta} \max_{\alpha \in \Delta^k} L(\theta, \alpha) := \sum_{i=1}^{k} \alpha_i \cdot \Big [ \dfrac{1}{\sum_{x \in D_i} |x|} \sum_{x \in D_i} \ell_{\theta}(x) - \ell_{ref}(x) \Big ]
\tag{2}
$$
其中的$\ell_{\theta}(x) = - \log{p_{\theta}(x)}$和$\ell_{ref}(x) = -\log{p_{ref}(x)}$为代理模型和参考模型的负对数似然概率 [^3]，$|x|$是当前样本$x$的token数量。$\ell_{\theta}(x) - \ell_{ref}(x)$ 可视为是**超额损失**（excess loss），这个损失度量了对于代理模型而言，当前样本$x$的可优化空间。超额损失越大，说明该样本需要更多训练才能达到参考模型的效果；超额损失越小，则有两种可能，第一种可能是$\ell_{ref}$很大，这表明了这个是一个高熵的样本，其生成会趋向于均匀分布，因此难以学习，第二种可能是$\ell_{\theta}$很小，这说明对于样本$x$而言，代理模型已经学习得足够好了，属于简单样本，这两种情况下都需要适当减少权重，减少该域样本的采样。首先从这个`minmax`的内层目标（也即是`max`）看起，此时代理模型不进行参数更新，优化项是$\alpha \in \Delta^k$， 就是根据超额损失的大小去启发式更新权重，然后看到外层的`min`目标，此时优化项是代理模型参数$\theta$，也即是从内层找到了最大的超额损失后，尝试让代理模型去拟合这个超额损失。通过交替地进行最小最大优化，从而训练得到一个代理模型和新的域权重$\bar{\alpha}$。 最终的域权重，则是从每一步的训练过程中进行累计平均得到，即是$\bar{\alpha} = \dfrac{1}{T} \sum_{t=1}^{T} \alpha_t$。



这个过程，用算法描述的形式表述则是：

**输入**：各个域的数据$\mathbf{D} = \{D_1,\cdots,D_k\}$，训练步数$T$，batch size大小$b$和更新步长$\eta$ ，平滑系数$c \in [0, 1]$，本实现采用的$c=10^{-3}$。

- 初始化代理模型参数$\theta_0$
- 初始化域权重$\alpha_0 = \dfrac{1}{k} \mathbf{1} \in \mathbb{R}^{k}$

**对于$t$从1到$T$开始循环**：

1. 从 $P_u$ 中采样一个大小为 $b$ 的小批量$B = \{x_1,\cdots,x_b\}$，其中$u = \dfrac{1}{k} \mathbf{1}$ 

2. 令$|x|$为样本$x$的token长度

3. 计算每个域$i \in \{1,\cdots,k\}$的超额损失（$\ell_{*,j}(x)$是第$j$个token的损失），此处的$\max(\cdots, 0)$就是在实现公式(2)里面提到内层`max`过程，需要保证超额损失的非负性，其中的$\ell_{*,j}(x) = -\log{p_{*}(x_j | x_1,\cdots,x_{j-1})}$。
   $$
   \lambda_{t}[i] \leftarrow \dfrac{1}{|x|} \sum_{x \in B \cap D_i} \sum_{j=1}^{|x|} \max(\ell_{\theta_{t-1}, j}(x) - \ell_{ref,j}(x), 0)
   \tag{3}
   $$

4. 启发式地更新域权重（指数更新）： $\alpha^{\prime} \leftarrow \alpha_{t-1} \odot \exp(\eta \lambda_{t})$

5. 更新归一化和平滑的域权重：$\alpha_t \leftarrow (1-c) \dfrac{\alpha^{\prime}_t}{\sum_{i=1}^{k} \alpha^{\prime}_t[i]} + cu, \alpha_{t} \in \mathbb{R}^{k}$， 此处采用平滑更新的方式，是希望整个域权重更新的过程更加平滑（因为每次只见到了当前batch的数据，因此可能存在噪声），最好是在先验分布$u$的基础上进行增量更新。

6. 使用目标$L(\theta_{t-1}, \alpha_{t})$更新代理模型的参数$\theta_{t}$（可以采用Adam优化器）。

**结束循环**

**返回**：$\bar{\alpha} = \dfrac{1}{T} \sum_{t=1}^{T} \alpha_t$



**第三步，用新的权重训练一个完整的LLM**

采用第二步得到的$\bar{\alpha}$构建新的训练分布$P_{\bar{\alpha}}$，从中采样数据去预训练最终需要的完整参数量的LLM（本实验中是8B）。



**迭代式的DoReMi**

第一轮迭代的DoReMi的$\alpha_{ref}$是启发式得到的，不是最优的选择，因此整个DoReMi过程是可以迭代进行的，即是当第一轮迭代中得到了$\bar{\alpha}_1$之后，可以将第二轮迭代的$\alpha_{ref} := \bar{\alpha}_1$，然后重复整个DoReMi的过程，直到域权重收敛为止，在本文收敛的定义是$||\bar{\alpha} - \alpha_{ref}||_{\infty} \lt 10^{-3}$，此处的无穷范数即是$\bar{\alpha}$和$\alpha_{ref}$差的最大值。从作者的经验看，在GLaM数据集上只需要3轮迭代即可收敛。



---



介绍完了整个DoReMi的操作过程，我们看下实验结果。作者是在The Pile和GLaM [6] 这两个预训练数据集上进行预训练的，The Pile的情况前文介绍了，GLaM是一个具有8个域的文本数据集，由于这个数据集的域权重根据下游任务的效果而调节得到，因此GLaM的域权重可以视为是真实标签，由此检验DoReMi的域权重调整结果。从Fig 1.中可以看出，经过DoReMi后，在Pile数据集上不同领域的权重已经有了很大的变化，有些域的权重存在大幅度的下降，然而如果我们看到DoReMi (280M -> 8B) 在Pile数据集上保留验证集上所有域的困惑度，如Fig 3.所示，则会发现DoReMi在所有域上的困惑度都是明显下降。这并不是很符合直觉，因为某些域（如`Arxiv`、`PubMed central`）的权重下降了很多，意味着LLM预训练过程中采样到这些域数据的几率下降了，为什么还能得到困惑度的下降呢？

一种可能性是，正如在前文讨论的，超额损失低的那部分样本都是难样本（接近均匀分布）或者简单样本，简单样本容易学习，不需要那么多样本支持学习，而难样本则由于难度太高，即便提高了样本量也无法学习出来，因此降低采样也没带来效果损失。并且很幸运的，提高了其他中度难度样本的采样比例后，让模型泛化能力得到了进步，因此在各个域数据上的表现都没折损，都有提升。

![fig4_log_perplexity_on_pile][fig4_log_perplexity_on_pile]

<div align="center">
    <b>
        Fig 3. 对比基线，采用了DoReMi后在Pile数据集得到所有域的保留验证集上都得到了困惑度的明显下降。
    </b>
</div>

让我们看看DoReMi在下游任务上的表现，从Fig 4. (a) 中能发现，在利用The Pile预训练集的情况下，采用了DoReMi后，在所有的训练步中，性能（下游任务指标，用的是精准匹配的比例，笔者觉得可能类似ROUGE的指标）都能得到持续的提升。从Fig 4. (b) 是采用GLaM数据集作为预训练的结果，有以下结论：

- 采用多轮迭代的DoReMi（round 1 vs round 2），采用多轮迭代的效果会持续比单轮的好。
- 采用了单轮的DoReMi效果没有基线好，可能是由于GLaM数据集本身的域只有8个，DoReMi的发挥空间不如The Pile数据集。而采用了多轮DoReMi的效果能超越基线，可能说明对于域比较少的数据集，需要多轮迭代才能得到较好效果。
- 采用多轮迭代的DoReMi，其效果接近最佳权重（通过下游任务调优得到）的效果。

再看到Fig 4. (c)， 这是在GLaM数据集中多轮迭代DoReMi的权重和最佳权重的对比，可以发现采用了DoReMi后，的确权重都趋向于了最佳权重了，这也证实了DoReMi的合理性和有效性。

![fig5_downstream_perf][fig5_downstream_perf]

<div align="center">
    <b>
        Fig 4. DoReMi在下游任务中的模型性能对比
    </b>
</div>

作者同样做了消融试验，去回答DoReMi中几个关键问题：

- Q1：是否挑选最难的样本或者最简单的样本，而不是超额损失最大的样本效果会更好？

  A1：当前的挑选标准是超额损失，即是$\ell_{\theta}(x) - \ell_{ref}(x)$，如果挑选标准变成最难样本，即是$\ell_{\theta}(x)$，或者挑选最简单样本，即是$-\ell_{ref}(x)$，试验效果会如何呢？见Fig 5. 右侧所示，我们发现采用了最简单样本或者最难样本的试验，效果都不如DoReMi，并且采用了最简单样本的方案，对比基线都远远落后。这说明了，学习最简单样本明显不是一个好主意，这使得大模型的底座能力缺失严重，单纯学习最难的样本也能提升LLM的能力，但是光是关注最难的样本，而忽略了中等难度的样本，则也不是最优的方案。这个也和前面分析困惑度的试验结论遥相呼应了。

![fig6_ablation1][fig6_ablation1]

<div align="center">
    <b>
        Fig 5. 在 The Pile 数据集上训练的模型的平均下游准确度。左侧: 在 DoReMi 中将参考/代理模型的规模从 70M 增加到 280M，可以提高 8B 主模型的下游准确度，但这一趋势在 1B 代理模型中并未继续。我们假设 Group DRO 优化器在更大规模的代理模型中表现不佳。右侧: 仅针对最难或最容易的领域进行优化，而不是超额损失（结合了两者），并不能实现与 DoReMi（280M 模型）相同的平均下游准确度。
    </b>
</div>

- Q2：提高代理模型的参数尺寸，是否能获得最后效果的提升？

  A2：考虑采用不同的代理模型尺寸，如70M、150M、280M、1B参数量，而最终模型的尺寸仍然是8B，是否会观察到scaling law呢？如Fig 5. 左侧图片所示，适当提升代理模型尺寸（`70M -> 150M -> 280M`）可以提高最终模型的效果，但是当代理模型尺寸达到一定大小（如1B）后，反而出现了性能下降。因此对于代理模型而言，也并不是尺寸越大越好。

- Q3：如果代理模型达到和最终模型同样的尺寸，代理模型和最终模型的效果对比如何？

  A3：这个问题其实也很符合直觉，代理模型和最终模型采用的采样策略是不同的（损失重参数化 vs 重采样）。作者尝试将代理模型和最终模型的参数量都设置为相同（为了试验对比公平），然后对比基线、DoReMi (x -> x)和代理模型的表现，如Fig 6所示，我们发现采用了**代理模型的表现都低于最终的主模型，并且随着模型尺寸增大，性能差别则越大**。并且在1B规模的代理模型中，甚至性能还不如基线（但是其DoReMi结果还是比基线好），这意味即便代理模型没有训练得很好，在整个DoReMi体系下仍然能提升最终模型的效果。

![fig7_ablation2][fig7_ablation2]

<div align="center">
    <b>
        Fig 6. DoReMi 主模型和相同规模代理模型的困惑度，尽管 1B 代理模型的质量相对较低，但由此产生的领域权重仍然能够改善主模型。 
    </b>
</div>



# Reference

[1]. Xie, Sang Michael, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy S. Liang, Quoc V. Le, Tengyu Ma, and Adams Wei Yu. "Doremi: Optimizing data mixtures speeds up language model pretraining." *Advances in Neural Information Processing Systems* 36 (2024). **aka DoReMi**

[2]. Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gb dataset of diverse text for language modeling. arXiv, 2020.  **aka The Pile**

[3]. Arkadi Nemirovski, Anatoli Juditsky, Guanghui Lan, and Alexander Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on optimization, 19(4):1574–1609, 2009.  

[4]. Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In International Conference on Learning Representations (ICLR), 2020.  

[5]. Yonatan Oren, Shiori Sagawa, Tatsunori Hashimoto, and Percy Liang. Distributionally robust language modeling. In Empirical Methods in Natural Language Processing (EMNLP), 2019.  **aka DRO-LM**

[6]. Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, M. Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, K. Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V. Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. GLaM: Efficient scaling of language models with mixture-of-experts. arXiv, 2021.  **aka GLaM**



[^1]: Group DRO 的关键在于通过最小化最坏情况下的损失来优化领域权重，从而使得模型在所有领域上都能达到较好的性能。
[^2]:  在自然语言处理（NLP）中，困惑度（Perplexity）是评估语言模型性能的一个重要指标。它衡量模型对测试数据的预测能力，具体计算方法如下：对于一个给定的词序列 $W=(w_1,w_2,\cdots,w_T)$，困惑度 $PP(W)$ 的计算公式为：$PP(W) = P(w_1,w_2,\cdots,w_T)^{-\dfrac{1}{T}}$，其中的$P(w_1,\cdots,w_T)$为语言模型建模的联合概率分布，而$T$为序列长度。**低困惑度**: 如果语言模型能够准确预测词序列，那么它给出的联合概率会较高，从而导致困惑度较低。这意味着模型对测试数据的预测能力较强。**高困惑度**: 如果模型对词序列的预测能力较差，给出的联合概率会较低，导致困惑度较高

[^3]: 此处计算的方法是，给定一个样本$x=\{x_1, x_2,\cdots,x_N \}$，其中$N$表示序列长度，$x_i$表示token，那么$p(x_{i}|x_{i-1},x_{i-2},\cdots,x_1)$就是当前token $x_{i}$被预测正确的概率。通过求对数似然和，能得到$\log{p(x)}$，也即是对每个token的对数概率进行加和。这代表了这个序列$x$被当前语言模型采样出来的概率。



[fig2_the_pile_domains]: ./imgs/fig2_the_pile_domains.png
[fig3_framework]: ./imgs/fig3_framework.png

[fig4_log_perplexity_on_pile]: ./imgs/fig4_log_perplexity_on_pile.png
[fig5_downstream_perf]: ./imgs/fig5_downstream_perf.png
[fig6_ablation1]: ./imgs/fig6_ablation1.png

[fig7_ablation2]: ./imgs/fig7_ablation2.png
[qrcode]: ./imgs/qrcode.png

