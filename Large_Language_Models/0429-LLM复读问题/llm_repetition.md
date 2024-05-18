<div align="center">
【论文极速读】DITTO: 引入复读负样本，一种打破LLM复读问题的方法 
</div>



<div align="right">
    FesianXu 20240429 at Tencent Wechat search team
</div>

# 前言

最近工作里面遇到了LLM复读的问题，去翻了下论文，看到有一篇尝试通过引入负样本解决复读问题的工作，有所启发，在此简单介绍下，希望对大家有所帮助。**如有谬误请见谅并联系指出，本文遵守[CC 4.0 BY-SA](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

e-mail: FesianXu@gmail.com

github: https://github.com/FesianXu

github page: https://fesianxu.github.io/

知乎专栏: 计算机视觉/计算机图形理论与应用

微信公众号：机器学习杂货铺3号店



----



LLM的复读问题，一般有几种级别的复读，如下所示

- 字粒度的复读：

> User: 你喜欢北京么？
>
> AI: 北京是中国的首都，有很多名胜古迹，如长城，故宫，天坛等，我十分喜欢欢欢欢欢欢欢欢欢欢欢欢….

- 词粒度的复读：

> User: 你喜欢北京么？
>
> AI: 北京是中国的首都，有很多名胜古迹，如长城，故宫，天坛等，我十分喜欢喜欢喜欢喜欢….

- 句子粒度的复读：

> User: 你喜欢北京么？
>
> AI: 北京是中国的首都，有很多名胜古迹，如长城，故宫，天坛等，我十分热爱北京，我十分热爱北京，我十分热爱北京，…..

贪婪搜索解码（greedy search）由于其解码结果是固定的（deterministic），并且解码速度快等优点，是在实际应用中经常使用的解码方法。在清华大学的一篇论文 [1]中，介绍了一种在贪婪搜索解码的前提下对复读问题的解决方案。如Fig 1 (b)所示，在Wikitext-103 dev数据集上，作者统计了模型生成结果和人类结果在不同粒度（word、 phrase、sentence）下的连续复读占比情况。不难发现人类编写的结果的连续复读占比随着粒度的增大，会快速减少，而模型生成的结果则在句子粒度上的复读中达到了惊人的最大值（~35%）。这说明在贪婪解码中，句子粒度的复读是最常见的复读模式，因此作者对这种模式的形成原因进行了研究。

![fig_1_repetition_granularity][fig_1_repetition_granularity]

<div align="center">
    <b>
        Fig 1. 模型生成结果和人类结果在各种粒度上的连续复读占比。
    </b>
</div>

作者研究这个现象的基本方法，可以简单理解为手动重复一个句子，然后研究每个token的生成概率的变化，举个例子就是$P_{\theta}("orange" | "I\ love")$和$P_{\theta}("orange" | "I\ love\ orange,\ I\ love")$的关系，作者发现，随着手动重复的句子次数越多，其下一个句子出现复读的概率就会越大。如Fig 2 (a)所示，在该图，作者手动复读了"She is named a proxy **general** under Gaara"这一句自然句子，并观察"general"一词在不同复读次数情况下的概率，即是$p(x_t | x_{<t})$，同时观察当前最大概率的令牌（token），即是$\max p(\cdot|x_{<t})$。从图中不难看出，在初始阶段（即是复读次数为0），$p("general"|x_{<t})$约为0.05左右，而最大概率的token则是"for"，但是随着复读开始，$p("general"|x_{<t})$得到了极为迅速的增长，并且很快$\arg \max p(\cdot|x_{<t}) = “general”$，即是最大概率词崩塌到了$x_t$。这说明，一旦句子复读模式开始出现，哪怕只有一次，整个模型就会出现自增强（self-reinforcement）现象，模型将会倾向于输出之前出现过的句子，导致模型结果崩塌到复读结果，如Fig 1 (a)所示。我们不妨举个具体例子，排比手法是LLM经常使用的罗列观点、事实的手法，如下所示

> - **用户：**和马有关的成语有哪些？
>
> - **模型：**和马有关的成语非常丰富，以下是一些常见的例子：
>   马到成功：形容事情顺利，刚开始就取得成功。
>   指鹿为马：指着鹿，说是马。比喻故意颠倒黑白，混淆是非。
>   马到功成：形容事情顺利，刚开始就取得成功。

此处的第1个成语出现了“马到”，而第3个成语也出现了“马到”，此时如果模型没有训练好，由我们之前得出的自增强结论，将会很容易出现复读现象。这种类型的case在一些排比手法的问答中经常出现，或者是输出有些其他特定的模式的问答中也可能出现。

作者在论文中，同时还对非自然句子进行了观察，如Fig 2 (b)所示，“fría backed rounds Manganiello Benzedrine Magruder Crego Stansel Zemin compressus  ”这个句子是随机从词表中挑选而来的，其$p(x_t|x_{<t})$的基本规律和Fig 2 (a)类似，但是其最大概率的词不会崩塌到$x_{t}$，同时$p(x_t|x_{<t})$随着复读次数增加，最多到0.10左右，对比图(a)的0.90左右有着巨大差别，这一点的差别可能来自于初始化阶段，$p(x_t)$大小的区别。

![fig_2_normal_random_sentence][fig_2_normal_random_sentence]

<div align="center">
    <b>
        Fig 2. 在正常语句和随机语句情况下的，下一个token为复读token的概率随着复读次数的变化。
    </b>
</div>

当然，以上都是基于采样的一句话进行观察得到的结论，为了让结论更具有普遍性，作者在公开数据集上进行了观察，采用的数据有以下三种：

1. $D_{random}$: 从词表中随机采样得到的句子集合，数量为1000个句子。
2. $D_{wiki}$: 从Wikitext-103 dev数据集中随机采样得到的1000个句子。
3. $D_{book}$： 从BookCorpus数据集中随机采样得到的1000个句子。

为了更好的定量分析模型在复读下的情况，作者定义了三种不同的指标，首先为了方便理解，需要约定一些符号。假定给定一个数据集$D$，将其复读$N$次构建一个序列$\mathbf{x}=(\mathbf{s}^{0},\mathbf{s}^{1},\cdots,\mathbf{s}^{N})$，其中$\mathbf{s}^{n} = (x_{n,1},\cdots,x_{n,L_s})$，$x_{n,l}$是第$n$次复读的第$l$个令牌，$L_s$表示句子$s$的长度。不难理解，$x_{n,l}$在句子中上文即可表示为$\mathbf{x}_{n, <l} = (x_{n,l},\cdots,x_{n,l-1})$。 将这个序列$\mathbf{x}$输入模型$P_{\theta}$，可以得到$P_{\theta}(x_{n,l}|\mathbf{x}_{<n, l})$，其中$\mathbf{x}_{<n,l} = (\mathbf{x}^0, \cdots, \mathbf{s}^{n-1}, \mathbf{x}_{n, <l})$。

1. 平均令牌概率（Average Token Probability）： 如公式(1)所示，用以表征在$n$次复读情况下生成的句子，其句子的所有令牌的令牌概率和，TP越高表示复读产生该句子的概率越高。
$$
TP(\mathbf{s}^{n})= \dfrac{1}{L_s} \sum_{l=1}^{L_s} P_{\theta}(x_{n,l}|\mathbf{x}_{<n, l})
\tag{1}
$$


2. 令牌概率增加率（Rate of Increased Token Probability）: 用于表示在复读n次的情况下，$\mathbf{s}^n$中有多少令牌的概率比其在初始的$\mathbf{s}^{0}$时候高。IP越高，表示当前复读的现况给后续带来出现复读的概率越大。
   $$
   IP(\mathbf{s}^{n}) = \dfrac{1}{L_s} \sum_{l=1}^{L_s} \mathbb{I}(P_{\theta}(x_{n,l}|\mathbf{x}_{<n,l}) > P_{\theta}(x_{0, l} | \mathbf{x}_{<0, l}))
   \tag{2}
   $$
   
3. 胜出率（Winner Rate）：如果$P_{\theta}(x_{n,l}|\mathbf{x}_{<n,l}) > P_{\theta}(x_{0, l} | \mathbf{x}_{<0, l})$并且$x_{n,l} = \arg \max P(\cdot|x_{<n, l})$，那么称$x_{n,l}$为胜出者。可以定义出胜出率如公式(3)，一个更高的胜出率意味着在基于最大概率的解码中（如greedy search），更容易出现复读。
   $$
   WR(\mathbf{s}^n) = \dfrac{1}{L_s} \sum_{l=1}^{L_s} \mathbb{I}(x_{n,l}\ is\ a\ winner)
   \tag{3}
   $$

![fig_3_metric_cal][fig_3_metric_cal]

<div align="center">
    <b>
        Fig 3. 通过复读一个句子100次构造样本，统计模型的TP、IP和WR指标变化趋势。
    </b>
</div>
如图Fig 3.所示，这三张图就是在以上三个数据集上统计TP、IP和WR指标的变化，从中我们能得到几个结论：

1. 为什么会出现句子粒度的复读？ 从Fig 3. (b)可知，$IP_{1}$已经大于0.9了，这意味着即便只出现一次句子复读，那么复读的概率将会在大部分情况中增加。举个例子就是， $P_{\theta}("orange" | "I\ love\ orange,\ I\ love") > P_{\theta}("orange" | "I\ love\ ")$，**模型倾向于从历史前文中找到“捷径”**，导致其复读前文出现过的句子。
2. 为什么模型一旦开始复读，就难以自愈了呢？从Fig 3.中可知，TP、IP和WR都是随着复读次数的增加而单调递增，这意味着**句子粒度的复读具有自增强效应**，一旦出现复读现象的苗头，就只会越演越烈，难以回头。
3. 什么类型的句子容易出现复读呢？作者对比了$D_{random}, D_{wiki},D_{book}$三类型数据，发现有着更高$TP_{0}$的后两者的自增强效应更加明显，这似乎说明了有着较强语义的自然句子更容易出现复读。

从结论来看，模型生成结果有从历史前文中找“捷径”的倾向，由于采用了贪婪搜索解码，导致模型生成结果容易出现复读，而复读具有自增强效应，因此一旦落入复读“陷阱”就难以脱身。既然知道了出现复读的原因，那么怎么去解决呢？


![fig_4_ditto_loss][fig_4_ditto_loss]

<div align="center">
    <b>
        Fig 4. DITTO损失函数约束了复读n-1次和复读n次下的令牌生成概率。
    </b>
</div>
作者提出了一种称呼为DITTO（Pseudo repetition penalization，伪复读惩罚）的方法，其主要思想是手动构建具有复读模式的样本（负样本），混合正常样本（正样本）给模型进行学习，辅以惩罚损失指导模型避免出现复读。 惩罚损失函数如公式(4)所示，首先需要构建伪复读样本，可以考虑从训练集中抽取部分锚点样本，然后对锚点样本复读N次，产生如$\mathbf{x} = (\mathbf{s}^0,\cdots,\mathbf{s}^N)$的样本，直到达到模型的输入截断长度为止。这个损失函数其实也很容易理解，其本质就是约束第n次复读和第n-1次复读中的令牌生成概率，打破其自增强趋势，如Fig 4所示，$P_{\theta}(x_{n,l}|\mathbf{x}_{<n,l})$和$P_{\theta}(x_{n-1,l}|\mathbf{x}_{<n-1,l})$分别表示在第n次和第n-1复读中的同一个令牌$x_{l}$的生成概率，约束让两者接近，可以打破复读的自增强趋势。其中得到$P^{*}_{\theta}$表示该处无梯度的反向传播。
$$
\mathcal{L}_{DITTO}^{n,l}(P_{\theta}(x_{n,l}|\mathbf{x}_{<n,l})) = -\log(1-|P_{\theta}(x_{n,l}|\mathbf{x}_{<n,l})-\lambda \cdot P_{\theta}^{*}(x_{n-1,l}|\mathbf{x}_{<n-1,l})|)
\tag{4}
$$
作者在基准测试集中对比了若干方法，如Fig 5. (a)所示，其中困惑度和准确率是DITTO的效果最好，看到复读率指标`repetition-4`和`repetition-sen`，由于基准测试集有人工标注因此可以统计人工的复读程度，和这个值接近可以认为效果越好。 我们看到虽然DITTO复读指标上的效果不如UL-token+seq，但是后者牺牲了部分的通用性能。DITTO是以MLE为基线试验的，可以看出确实比起MLE有着很大的提升，因此可以确定DITTO策略的有效性。Fig 5. (b)则是在非贪婪搜索解码中的效果对比，可以发现是DITTO效果最好，但是其实提升就没有第一个试验来得大了，这是因为解码过程采用了采样策略后，复读问题本就会得到极大缓解。


![fig_5_exp_result][fig_5_exp_result]

<div align="center">
    <b>
        Fig 5. 实验对比结果。
    </b>
</div>

我们再看到采用了DITTO后，TP、IP、WR等指标的变化，如Fig 6所示，我们能发现红线确实符合预期的接近0，这说明引入惩罚之后，复读问题的自增强效应得到了打压。

![fig_6_ditto_metric][fig_6_ditto_metric]

<div align="center">
    <b>
        Fig 6. 采用了DITTO后，TP、IP和WR确实符合预期，说明自增强效应得到了打压。
    </b>
</div>






# Reference

[1]. Xu, Jin, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, and Jian Li. "Learning to break the loop: Analyzing and mitigating repetitions for neural text generation." *Advances in Neural Information Processing Systems* 35 (2022): 3082-3095.



[fig_1_repetition_granularity]: ./imgs/fig_1_repetition_granularity.jpg
[fig_2_normal_random_sentence]: ./imgs/fig_2_normal_random_sentence.jpg
[fig_3_metric_cal]: ./imgs/fig_3_metric_cal.jpg

[fig_4_ditto_loss]: ./imgs/fig_4_ditto_loss.jpg
[fig_5_exp_result]: ./imgs/fig_5_exp_result.jpg
[fig_6_ditto_metric]: ./imgs/fig_6_ditto_metric.jpg





