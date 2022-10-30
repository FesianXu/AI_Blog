<div align='center'>
  在多模态模型训练时，如何合适地融合单模态损失
</div>

<div align='right'>
  FesianXu 20220420 at Baidu Search Team
</div>

# 前言

文章[1]的作者发现在多模态分类模型中，经常出现最好的单模态模型比多模态模型效果还好的情况，作者认为这是由于多模态模型的容量更大，因此更容易过拟合，并且由于不同模态的信息过拟合和泛化有着不同的节奏，如果用同一个优化策略进行优化，那么很可能得到的不是一个最佳的结果。也就是说作者认为目前的多模态融合方式还不是最合适的，因此在[1]中提出了一种基于多模态梯度混合的优化方式。本文是笔者对该文的读后感和结合业务的一些认识，**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

---

假如一个多模态分类模型由$M$个模态信息组成（如RGB，光流，音频，深度信息等等），每一个模态的输入记为$x_i$，每一个模态的特征提取网络记为$f_i = g_i(x_i)$，其中$i=1,\cdots,M$，那么对于一个后融合（Late-fusion）[2]的多模态分类模型来说，如Fig1.1(c)所示，其后融合的多模态特征由拼接（concatenate）操作构成，因此多模态特征表示为$f_{m} = f_1 \bigoplus f_2 \cdots f_M$，其中$\bigoplus$表示拼接操作。最后将会用$f_m$进行训练和分类。假设训练集为$\mathcal{T}=\{X_{1,\cdots,n}, y_{1,\cdots,n}\}$，其中$X_i$为第$i$个训练样本而$y_i$为第$i$个训练样本的标签，那么对于多模态分类而言，其损失为：
$$
\mathcal{L}_{multi} = \mathcal{L}(\mathcal{C}(f_1 \bigoplus f_2 \cdots f_M), y)
\tag{1-1}
$$

容易知道对于单模态分类而言，其损失为：
$$
\mathcal{L}_{uni} = \mathcal{L}(\mathcal{C}(f_{m}), y)
\tag{1-2}
$$
![multimodal_joint_training][multimodal_joint_training]

<div align='center'>
  <b>
    Fig 1.1 多模态联合训练，采用后融合的方式进行不同模态的信息融合。
  </b>
</div>
从理想情况看，由于多模态特征是由各个模态的特征拼接而成的，通过训练学习出合适的分类器参数$\Theta_{\mathcal{C}}^{*}$，那么多模态损失(1-1)就可以崩塌到单模态损失(1-2)，也就是说最坏情况下多模态训练得到的结果，都应该要比单模态训练的要好。然而结果并不是如此，如Fig 1.2(a)所示，以在Kinetics上的结果为例，最好的单模态结果总是要显著比多模态结果（Audio，RGB，Optical Flow三者的任意组合）要好。不仅如此，如Fig 1.2(b)所示，即便采用了一些流行的正则手段，也无法得到有效的效果提升。这不是偶然，[1]的作者认为这是由于不同模态的信息陷入过拟合的节奏是不同的，而通过相同的训练策略对多模态特征进行训练，可能对于整体而言并不能达到最优的状态。为此，对于多模态损失而言需要适当地进行加权，去适应不同模态学习的节奏，假设权系数$w_k$满足$\sum_kw_k=1$，其中的$k$是第$k$个模态，那么最终的损失为：
$$
\mathcal{L}_{blend} = \sum_{i=1}^{K+1} w_i \mathcal{L}_i
\tag{1-3}
$$
其中的$K+1$模态表示的是拼接起来后的多模态特征，也即是式子(1-1)所示的损失。关键问题有两个：

1. 这些模态均衡系数$w_i$应该怎么确定
2. 这些模态均衡系数是在线计算（动态更新）还是离线计算（静态计算后使用）

显然，均衡系数是一个超参数，单纯靠网格搜索或人工调参肯定不显示，而且无法解决关键问题2，也即是动态更新。因此作者提出了一种确定多模态均衡系数的方法。

![unimodal_vs_multimodal_acc_reg][unimodal_vs_multimodal_acc_reg]

<div align='center'>
  <b>
    Fig 1.2 (a)多模态训练得到的模型总是比最优的单模态训练模型更差；(b) 采用了一些常用的正则手段也无法获得有效的效果提升。
  </b>
</div>

首先需要定义出一个度量以衡量该模态的过拟合与泛化情况，如Fig 1.3所示，作者定义了一种综合度量模型的过拟合与泛化情况的指标，其定义为过拟合程度与泛化程度的比值的绝对值，如式子(1-4)所示。其中$\Delta O_{N,n} = O_{N+n}-O_{N}$，而$O_{N}=\mathcal{L}_{N}^{V}-\mathcal{L}_{N}^{T}$，表示为训练损失和验证损失的差值，其可被认为是**过拟合大小**，显然该值越大，过拟合程度越大。而$\Delta O_{N,n}$表示第$N$个epoch与第$N+n$个epoch之间的过拟合程度差值。那怎么表示泛化能力呢？可以通过第$N$个epoch与第$N+n$个epoch之间的验证损失$\mathcal{L}^{*}$的差值表示两个checkpoint之间的泛化能力差值。也就是说可以将式子(1-4)认为是两个epoch的checkpoint之间的过拟合程度与泛化程度比值的差分。显然我们希望OGR指标越小越好。注意此处的$\mathcal{L}^{*}$表示理想中的真实验证损失，通常会用有限的验证集损失去近似，表示为$\mathcal{L}^{V}$。后续我们都用$\mathcal{L}^{V}$代替$\mathcal{L}^{*}$。
$$
OGR = \Bigg |\dfrac{\Delta O_{N,n}}{\Delta G_{N,n}} \Bigg | = \Bigg | \dfrac{O_{N+n}-O_{N}}{\mathcal{L}^{*}_{N} - \mathcal{L}^{*}_{N+n}} \Bigg |
\tag{1-4}
$$
显然有
$$
\Delta O_{N,n} = L^{V}_{N+n}-L^{T}_{N+n}-(L^{V}_{N}+L^{T}_{N}) = \Delta L^{V} - \Delta L^{T}
\tag{1-5}
$$
然而对于欠拟合的模型来说，可能$\Delta O_{N,n}$足够小也会导致OGR指标也很小，但是这并没有意义，因为模型仍然未学习好。因此此处用无穷小量进行衡量，也即是有：
$$
\lim_{n \rightarrow 0} \Bigg |\dfrac{\Delta O_{N,n}}{\Delta G_{N,n}} \Bigg |  = \Bigg |\dfrac{\partial O_{N,n}}{\partial G_{N,n}} \Bigg | 
\tag{1-6}
$$
当然，由于此处的$n$有实际的模型含义（一个step），也就是说其实应该是$n\rightarrow 1$，也就是只有1个step的参数更新。对此我们对损失进行一阶泰勒展开有：
$$
\begin{aligned}
\mathcal{L}^{T}(\Theta+\eta \hat{g}) &\approx \mathcal{L}^{T}(\Theta)+\eta<\nabla \mathcal{L}^{T}, \hat{g}> \\
\mathcal{L}^{V}(\Theta+\eta \hat{g}) &\approx \mathcal{L}^{V}(\Theta)+\eta<\nabla \mathcal{L}^{V}, \hat{g}>
\end{aligned}
\tag{1-7}
$$
结合(1-5)和(1-7)我们有：
$$
\begin{aligned}
\partial O_{N,n} &= \eta<\nabla \mathcal{L}^{V}-\nabla\mathcal{L}^{T}, \hat{g}> \\
\partial G_{N,n} &= \eta <\nabla \mathcal{L}^{V}, \hat{g}>
\end{aligned}
\tag{1-8}
$$
因此有：
$$
OGR^2 = \Bigg ( \dfrac{<\nabla \mathcal{L}^{V}-\nabla\mathcal{L}^{T}, \hat{g}>}{<\nabla \mathcal{L}^{V}, \hat{g}>} \Bigg )^2
\tag{1-9}
$$
![overfitting_to_generalization_ratio][overfitting_to_generalization_ratio]

<div align='center'>
  <b>
    Fig 1.3 定义出OGR以描述该模态模型下的过拟合与泛化情况。
  </b>
</div>

此时我们对每个模态的梯度$\{\hat{g}_i\}_{i=1}^M$进行预估，这个预估通过各模态对应的分类器梯度反向求导得到，表示为$\{v_k\}_{1}^M$，当满足$\mathbb{E}[<\nabla \mathcal{L}^T-\nabla \mathcal{L}^{V}, v_k><\nabla \mathcal{L}^T-\nabla \mathcal{L}^{V}, v_j>] = 0$，其中$j \neq k$时，并且给定约束$\sum_k w_k=1$，我们的对$OGR^2$求最小值以求得最佳的模态均衡参数，表示为(1-10):
$$
w^{*} = \arg\min_{w} \mathbb{E} \Bigg [ \Bigg ( \dfrac{<\nabla \mathcal{L}^{T}-\nabla \mathcal{L}^{V}, \sum_k w_k v_k>}{<\nabla \mathcal{L}^V, \sum_kw_kv_k>} \Bigg )^2 \Bigg ]
\tag{1-10}
$$
原文[1]中对其进行了解析解的证明，这里就不展开了，其解析解如(1-11):
$$
w^{*}_k = \dfrac{1}{Z} \dfrac{<\nabla \mathcal{L}^V, v_k>}{\sigma^2_k}
\tag{1-11}
$$
其中$\sigma^2_k = \mathbb{E}[<\nabla \mathcal{L}^T - \nabla \mathcal{L}^V, v_k>^2]$， $Z = \sum_k \dfrac{<\nabla \mathcal{L}^V, v_k>}{2\sigma^2_k}$是标准化常数项。由此可计算出最佳的模态均衡系数，回答了我们之前提出的第一个问题。

在实践中，再强调下，正如一开始所说的，$\nabla \mathcal{L}^*$无法得到，因此通常会从训练集中划出一部分$V$作为子集去验证，得到$\nabla \mathcal{L}^V$，用此去近似$\nabla \mathcal{L}^*$。此时我们可以正式去描述Gradient-Blending（GB）算法了，我们的数据集包括训练集$T$，训练集中划出来的验证集$V$，$k$个输入模态$\{m_i\}^k_{i=1}$以及一个多模态拼接得到的特征$m_{k+1}$。对于GB算法来说，有两种形式：

1. 离线Gradient-Blending： 只计算一次模态均衡参数，并且在以后的训练中都一直固定。
1. 在线Gradient-Blending： 将会定期（比如每n个epoch-也称之为super epoch）更新，并且用新的模态均衡参数参与后续的训练。

![GB_algorithm][GB_algorithm]

<div align='center'>
  <b>
    Fig 1.4 Gradient-Blending用于模态均衡系数估计；离线与在线Gradient-Blending。
  </b>
</div>

离在线GB算法和GB估计模态均衡参数的算法见Fig 1.4，显然，由于在线G-Blend需要隔一个super epoch就更新一次均衡参数，因此计算代价更大。作者发现采用了GB估计模态均衡参数后，无论是离线还是在线的G-Blend结合了多模态分类模型训练后，效果都比单模态模型有着显著的提升，并且离线效果仅仅比在线效果差一些，而在线G-Blend的计算代价比离线高，因此后续的消融实验都是用离线G-Blend展开的。

![kinetics_acc_gb][kinetics_acc_gb]

<div align='center'>
  <b>
    Fig 1.5 采用了G-Blend之后，多模态分类效果比单模态训练有着明显提升。数据集是Kinetics。
  </b>
</div>

作者同样对比了不同epoch下在线G-Blend学习出的模态均衡参数的分布，如Fig 1.6(a)所示，可以发现其在不同epoch下其参数分布都不同，在15-20和20-25的时候甚至出现了Video部分和Audio-Video部分独占鳌头的情况，作者认为这是由于在不同训练阶段其过拟合和泛化行为特征都会改变，导致均衡参数也在一直变化，但是不管怎么样，其效果都会比不采用G-Blend的多模态分类训练更好，如Fig 1.6(b)所示。

![online_Gblend][online_Gblend]

<div align='center'>
  <b>
    Fig 1.6 (a)对在线G-Blend算法的均衡参数的探索；(b)采用了G-Blend后 VS 未采用的多模态训练效果对比。
  </b>
</div>

同样的，G-Blend不仅仅适用于Video/Audio这两个模态，还能在其他模态下生效，如Fig 1.7所示。

![diff_modality_exp][diff_modality_exp]

<div align='center'>
  <b>
    Fig 1.7 光流，音频，RGB模态的任意混合中，G-Blend都能取得较大的效果提升。
  </b>
</div>

笔者从业务的角度上看，在进行图-文/视频-文匹配的时候，经常会采用双塔多模态匹配，最后对匹配损失进行优化的实践，如[3,4,5]。这个时候由于任务只关注了多模态匹配任务，而没有考虑维持单模态内的特征空间稳定性，此时容易导致单模态内的特征空间破损。如Fig 1.8所示，其中的Fig 1.8(a)图片虽然都和猫有关，但是显然一种是真的猫，一种是猫相关的书法作品，但是这两类的文本信息可能都包含有猫，在进行多模态匹配的时候如果不考虑单模态的损失，就会导致如Fig 1.8(b)所示的单模态特征空间破损，将一些细粒度的单模态知识给『遗忘』了。

![all_cat][all_cat]

<div align='center'>
  <b>
    Fig 1.8 (a)单模态之间有着更为细粒度的知识；(b)在多模态训练中容易被『遗忘』。
  </b>
</div>




# Reference

[1]. Wang, W., Tran, D., & Feiszli, M. (2020). What makes training multi-modal classification networks hard?. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 12695-12705).

[2]. https://blog.csdn.net/LoseInVain/article/details/105545703， 《万字长文漫谈视频理解》 by FesianXu

[3]. https://fesian.blog.csdn.net/article/details/120364242， 《图文搜索系统中的多模态模型：将MoCo应用在多模态对比学习上》 by FesianXu

[4]. https://fesian.blog.csdn.net/article/details/119516894， 《CLIP-对比图文多模态预训练的读后感》 by FesianXu

[5]. https://fesian.blog.csdn.net/article/details/121699533, 《WenLan 2.0：一种不依赖Object Detection的大规模图文匹配预训练模型 & 数据+算力=大力出奇迹》 by FesianXu









[multimodal_joint_training]: ./imgs/multimodal_joint_training.png
[overfitting_to_generalization_ratio]: ./imgs/overfitting_to_generalization_ratio.png
[qrcode]: ./imgs/qrcode.jpg
[unimodal_vs_multimodal_acc_reg]: ./imgs/unimodal_vs_multimodal_acc_reg.png

[GB_algorithm]: ./imgs/GB_algorithm.png
[kinetics_acc_gb]: ./imgs/kinetics_acc_gb.png
[online_Gblend]: ./imgs/online_Gblend.png
[diff_modality_exp]: ./imgs/diff_modality_exp.png
[all_cat]: ./imgs/all_cat.png











