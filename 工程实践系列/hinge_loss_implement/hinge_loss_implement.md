<div align="center">
    hinge loss的一种实现方法
</div>

<div align="right">
    FesianXu 20220820 at Baidu Search Team
</div>
hinge loss是一种常用损失[1]，常用于度量学习和表征学习。对于一个模型$\hat{y}=f(x), \hat{y} \in [0,1]$，如果给定了样本$x$的标签$y$ （假设标签是0/1标签，分别表示负样本和正样本），那么可以有两种选择进行模型的表征学习。第一是`pointwise`形式的监督学习，通过交叉熵损失进行模型训练，也即是如式子(1-1)所示。
$$
\begin{aligned}
\mathcal{L}_{ce} &= -\sum_{i=1}^{C} y_i \log(s(\hat{y}_i)) \\
s(\hat{y}) &=  \dfrac{\exp(\hat{y}_i)}{\sum_{j=1}^{C} \exp(\hat{y}_j)}
\end{aligned}
\tag{1-1}
$$
其中的$s(\cdot)$是softmax函数。第二种方式是将样本之间组成如$<s^{+}, s^{-}>$的pair，通过hinge loss进行pair的偏序关系学习，其hinge loss可以描述为式子(1-2)：
$$
\mathcal{L}_{hinge} = \max(0, s^{-} - s^{+} + m)
\tag{1-2}
$$
其中的$s^{-}$和$s^{+}$分别表示负样本和正样本的打分，而$m$这是正样本与负样本之间打分的最小间隔。如Fig 1.所示，我们发现$s^{+}-s^{-}_{1} < m$，而$s^{+}-s^{-}_{2} > m$，从式子(1-2)中可以发现，只有$s_{1}^{-}$会产生loss，而$s^{-}_{2}$则不会产生loss，这一点能防止模型过拟合一些简单的负样本，而尽量去学习难负例。

![hinge_loss][hinge_loss]

<div align="center">
    <b>
        Fig 1. hinge loss的图示。
    </b>
</div>

从实现的角度出发，我们通常可以采用下面的方式实现，我们简单介绍下其实现逻辑。

```python
import torch 
import torch.nn.functional as F

margin = 0.3
for data in dataloader():
    inputs, labels = data
    score_orig = model(inputs) # score_orig shape (N, 1)
    N = score_orig.shape[0]
    score_1 = score_orig.expand(1, N) # score_1 shape (N, N)
    score_2 = torch.transpose(score_1, 1, 0) 

    label_1 = label.expand(1, N) # label_1 shape (N, N)
    label_2 = label_1.transpose(label_1, 1, 0)
	label_diff = F.relu(label_1 - label_2)
    score_diff = F.relu(score_2 - score_1 + margin)
    hinge_loss = score_diff * label_diff
    ...
```

为了实现充分利用一个batch内的样本，我们希望对batch内的所有样本都进行组pair，也就是说当batch size为$N$的时候，将会产出$N^2-N$个pair（样本自身不产生pair），为了实现这个目的，就需要代码中`expand`和`transpose`这两个操作，如Fig 2.所示，通过这两个操作产出的`score_1`和`score_2`之差就是batch内所有样本之间的打分差，也就可以认为是batch内两两均组了pair。

![score_diff][score_diff]

<div align="center">
    <b>
        Fig 2. 对score的处理流程图
    </b>
</div>

与此相似的，如Fig 3.所示，我们也对label进行类似的处理，但是考虑到偏序已经预测对了的pair不需要产生loss，而只有偏序错误的pair需要产出loss，因此是`label_1-label_2`产出`label_diff`。通过`F.relu()`我们替代`max()`的操作，将不产出loss的pair进行屏蔽，将`score_diff`和`label_diff`相乘就产出了hinge loss。

![label_diff][label_diff]

<div align="center">
    <b>
        Fig 3. 对label处理的流程图。
    </b>
</div>

即便我们的label不是0/1标签，而是分档标签，比如相关性中的0/1/2/3四个分档，只要具有高档位大于低档位的这种物理含义（而不是分类标签），同样也可以采用相同的方法进行组pair，不过此时`label_1-label_2`产出的`label_diff`中会出现大于1的item，可视为是对某组pair的loss加权，此时需要进行标准化，代码将会改成如下:

```python
import torch 
import torch.nn.functional as F

margin = 0.3
epsilon = 1e-6
for data in dataloader():
    inputs, labels = data
    score_orig = model(inputs) # score_orig shape (N, 1)
    N = score_orig.shape[0]
    score_1 = score_orig.expand(1, N) # score_1 shape (N, N)
    score_2 = torch.transpose(score_1, 1, 0) 

    label_1 = label.expand(1, N) # label_1 shape (N, N)
    label_2 = label_1.transpose(label_1, 1, 0)
	label_diff = F.relu(label_1 - label_2)
    score_diff = F.relu(score_2 - score_1 + margin)
    hinge_loss = torch.sum(score_diff * label_diff) / (torch.sum(label_diff) + epsilon) # 标准化处理，加上epsilon防止溢出
    ...
```







# Reference

[1]. https://blog.csdn.net/LoseInVain/article/details/103995962, 《一文理解Ranking Loss/Contrastive Loss/Margin Loss/Triplet Loss/Hinge Loss》



[hinge_loss]: ./imgs/hinge_loss.png
[score_diff]: ./imgs/score_diff.png
[label_diff]: ./imgs/label_diff.png



