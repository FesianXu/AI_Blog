<div align='center'>
    Shift-GCN网络论文笔记
</div>

<div align='right'>
    FesianXu 20201105 at UESTC
</div>

# 前言

近日笔者在阅读Shift-GCN[2]的文献，Shift-GCN是在传统的GCN的基础上，用Shift卷积算子[1]取代传统卷积算子而诞生出来的，可以用更少的参数量和计算量达到更好的模型性能，笔者感觉蛮有意思的，特在此笔记。**如有谬误请联系指出，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**QQ**: 973926198

github: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)



----



# ST-GCN网络

骨骼点序列数据是一种天然的时空图结构数据，具体分析可见[5,6]，针对于这类型的数据，可以用时空图卷积进行建模，如ST-GCN[4]模型就是一个很好的代表。简单来说，ST-GCN是在空间域上采用图卷积的方式建模，时间域上用一维卷积进行建模。

骨骼点序列可以形式化表达为一个时空图$G = (V,E)$，其中有着$N$个关节点和$T$帧。骨骼点序列的输入可以表达为$\mathbf{X} \in \mathbb{R}^{N \times T \times d}$，其中$d$表示维度。为了表示人体关节点之间的连接，我们用邻接矩阵表达。按照ST-GCN原论文的策略，将人体的邻接矩阵划分为三大部分：1）离心群；2）向心群；3）根节点。具体的细节请参考论文[4]。每个部分都对应着其特定的邻接矩阵$\mathbf{A}_p, p\in\mathcal{P}$，其中$p$表示划分部分的索引。用符号$\mathbf{F} \in \mathbb{R}^{N \times C}$和$\mathbf{F}^{\prime} \in \mathbb{R}^{N \times C^{\prime}}$分别表示输入和输出的特征矩阵，其中$C$和$C^{\prime}$是输入输出的通道维度。那么，根据我们之前在GCN系列博文[7,8,9]中介绍过的，我们有最终的人体三大划分的特征融合为：
$$
\mathbf{F}^{\prime} = \sum_{p \in \mathcal{P}} \bar{\mathbf{A}}_p \mathbf{F} \mathbf{W}_p
\tag{1.1}
$$
其中$\mathcal{P} = \{根节点，离心群，向心群\}$，$\bar{\mathbf{A}}_p = \Lambda_p^{-\frac{1}{2}} \mathbf{A}_p \Lambda_p^{-\frac{1}{2}} \in \mathbb{R}^{N \times N}$是标准化后的邻接矩阵，其中$\Lambda_p^{ii} = \sum_j(\mathbf{A}_p^{ij})+\alpha$，具体这些公式的推导，见[7,8,9]。其中的$\mathbf{W}_p \in \mathbb{R}^{1 \times 1 \times C \times C^{\prime}}$是每个人体划分部分的1x1卷积核的参数，需要算法学习得出。



![stgcn][stgcn]

<div align='center'>
    <b>
        Fig 1.1 STGCN的示意图，通过不同的邻接矩阵可以指定不同的身体划分部分，通过1x1卷积可以融合通道间的信息，最后融合不同划分部分的信息就形成了新的输出向量。
    </b>
</div>



# Reference

[1]. Wu, B., Wan, A., Yue, X., Jin, P., Zhao, S., Golmant, N., … & Keutzer, K. (2018). Shift: A zero flop, zero parameter alternative to spatial convolutions. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 9127-9135).

[2]. Cheng, K., Zhang, Y., He, X., Chen, W., Cheng, J., & Lu, H. (2020). Skeleton-Based Action Recognition With Shift Graph Convolutional Network. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 183-192).

[3]. https://fesian.blog.csdn.net/article/details/109474701

[4]. Sijie Yan, Yuanjun Xiong, and Dahua Lin. Spatial temporal graph convolutional networks for skeleton-based action
recognition. In Thirty-Second AAAI Conference on Artificial
Intelligence, 2018.  

[5]. https://fesian.blog.csdn.net/article/details/105545703

[6]. https://blog.csdn.net/LoseInVain/article/details/87901764

[7]. https://blog.csdn.net/LoseInVain/article/details/88373506

[8]. https://fesian.blog.csdn.net/article/details/90171863

[9]. https://fesian.blog.csdn.net/article/details/90348807





[stgcn]: ./imgs/stgcn.png



