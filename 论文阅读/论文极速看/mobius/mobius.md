∇ 联系方式：

**e-mail:** FesianXu@gmail.com

**github:** https://github.com/FesianXu

**知乎专栏:** 计算机视觉/计算机图形理论与应用

**微信公众号：**
![qrcode][qrcode]

----

在搜索，计算广告和推荐系统中，通常有着海量的用户数据，这类型的数据各种类型混杂，比如用户点击数据，用户浏览时长，还有各种用户行为信息等，如何根据这些数据构造出合适的数据集给模型训练，是一件核心问题。百度在论文[1]中提出了一种称之为MOBIUS的负样本构建思路。通常来说，搜索广告推荐（统称为推广搜系统）都会由『召回』『排序』两大步骤组成，比如论文中提到的百度『凤巢』广告系统，其可以看成是一个『倒三角形』的漏斗形结构，由上到下分别是『召回/匹配（matching）』，『粗排，精排』，『上层排序』等。
![tri][tri]
在漏斗的顶端是在海量（亿级别）的数据中召回足够相关的项目（item），这个时候通常只考虑用『相关性（relevance）』作为标准进行匹配，相关性指的是用户和项目的相关程度，对于信息检索系统来说就是用户Query和网页Doc的相关程度，对于广告系统就是用户Query和广告Ad的相关程度。在召回足够的项目之后，再根据更多的特征进行粗排序和精排序等。最后，考虑到业务需求，比如竞价，点击模型等，需要结合这些因素和相关性进行上层排序，得到最后的展现给用户的结果。在论文中，MOBIUS以CPM（Cost Per Mile，千人展现花费）作为业务的一个上层指标进行讨论，$CPM = CTR \times Bid$，也就是说CPM是由预估点击率和竞价决定的。

论文中提到，在实际场景中，很多时候高频物体会有着CTR偏高的倾向，即便用户和该项目没有太高的相关性。这个时候就会出现这种情况，用户搜索一个Query，投放出来的广告相关性不高，但是因为该物体被很多人点击，然后导致CTR偏高。这种情况即是『低相关性，高CTR』，如下图所示
![frame][frame]
百度凤巢提出的方法是构建负样本，也就是将这种『低相关性，高CTR』的负样本在数据构建阶段就产生出来，然后通过这种数据去学习得到的模型，可以区分低相关性的同时，还有对CTR感知的能力。可以认为之前在召回阶段，模型只考虑了相关性，如式子(1)所示。
$$
\mathcal{O}_{Matching} = \max{\dfrac{1}{n} \sum_{i=1}^n \mathrm{Relevance}(query_i, item_i)}
\tag{1}
$$
而在百度凤巢提出的系统中，希望是如式子(2)所示
$$
\begin{aligned}
\mathcal{O}_{Mobius} &= \max{\sum_{i=1}^n \mathrm{CTR}(user_i, query_i, item_i) \times bid_i}  \\
& s.t. \dfrac{1}{n} \sum_{i=1}^n \mathrm{Relevance}(query_i, item_i) \geq threshold
\end{aligned}
$$

至于构建的方法也很朴素，在点击日志里，找出诸多的用户-项目的点击对，然后进行直积构建出『生成数据对』，如 $Query \otimes Item$，比如Query有`[A,B,C,D]`，而Item有`[a,b,c]`，那么生成的数据对就有$4 \times 3 = 12$个，为`<A,a>,<A,b>...<D,c>`。我们用已经训练好的相关性模型对每一个生成数据对进行相关性预测，同时设定一个阈值，将小于阈值的数据对取出，送给点击模型预测CTR，同时通过数据采样方法进行采样（此时的目的是挑选合适CTR等上层目标的样本），最后回归送到数据集中合并。整个流程下来，我们的『低相关性，高CTR（或其他上层指标）』的负样本就构建好了。构建好后，继续迭代更新点击模型的模型参数（如下图的粉色箭头所示），整个流程见下图。
![liuchen][liuchen]
总的来说，MOBIUS是百度凤巢实际使用的系统，用于负样本的构建与生成，并且希望将上层排序信号引入到相关性中。


# Reference
[1]. Fan, M., Guo, J., Zhu, S., Miao, S., Sun, M., & Li, P. (2019, July). MOBIUS: towards the next generation of query-ad matching in baidu's sponsored search. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2509-2517).



[qrcode]: ./imgs/qrcode.png

[tri]: ./imgs/tri.png
[frame]: ./imgs/frame.png
[liuchen]: ./imgs/liuchen.png

