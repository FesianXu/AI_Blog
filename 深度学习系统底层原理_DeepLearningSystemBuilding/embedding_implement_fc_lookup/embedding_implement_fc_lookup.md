<div align='center'>
    【Debug危机系列】Embedding层的千层套路
</div>

<div align='right'>
    FesianXu 20220916 at Baidu Search Team
</div>

# 前言

这次的debug案例来自于朋友的一个问题，Embedding层的前向和反向速度是否会随着token的增多而增加呢？本文对这个问题进行讨论。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢** 。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**： 机器学习杂货铺3号店



----



前几天土豆收到朋友的一个问题，问题内容如下图所示。这个问题理解起来不难，对于一个Embedding层来说，token的数量会影响前向和反向的速度吗？我们接下来看看土豆的分析和一些试验。

![question][question]

这个问题从直观上看，Embedding层的前向和反向过程是不会收到token数量的影响的，除非token实在太多导致内存占用太大，不断地出现缺页异常导致换页，从而影响访存速度。问题中有1000w个token，按照维度768，float32类型计算，也就30多G内存，对于服务器而言不算太多。而Embedding层我们都知道，可以通过两种方式实现，如Fig 1. 所示，通常来说我们可以考虑采用对`Lookup Table`查表的方式，将ID对应的某一行取出就得到了该ID的Embedding。还可以将这个ID转化为one-hot编码向量，矩阵乘以Embedding参数矩阵后，也可以得到该ID的Embedding。

对于查表的方式得到的Embedding，由于整个过程只需要对ID对应的某一行进行检索，因此计算复杂度是$\mathcal{O}(1)$，理论上说不会受到token数量增加带来的影响。对于查表而言，反向过程类似于前向过程，同样计算复杂度是$\mathcal{O}(1)$。后者由于通过矩阵乘法实现，one-hot向量中极为稀疏，仅有一个值为1，其他都为0，在计算前向和反向过程的时候同样会对这些无效的0进行计算，因此计算复杂度会随着token的增加而增加，复杂度为$\mathcal{O}(n)$，其中的$n$为token数量。理论上如此，在我们的工程实践中，真的如此吗？从朋友的问题上看有两种可能：

1. 他采用FC层进行one-hot向量矩阵乘法的方式实现Embedding，但是在这种情况下，前向过程和反向过程应该会同步增加耗时，不会存在“反向过程比前向过程两倍还多”的情况。
2. 他采用查表的方式实现Embedding，但是由于某种未知的框架机制，导致了题目中的情况，即计算耗时随着token数量增加而增加，并且反向传播耗时明显比前向传播耗时长。

为了验证这两种假设，我们得进行试验，让我们开始撸代码跑实验吧~

![Embedding_inplement][Embedding_inplement]

<div align='center'>
    <b>
        Fig 1. 采用lookup table查表的方式实现Embedding层 以及 通过one-hot编码向量矩阵相乘得到方式实现Embedding层。
    </b>
</div>

首先，我们采用FC层进行Embedding的耗时试验，代码如Append Code A. 所示，从实验中我们发现，随着token数量n的逐步增加（100 -> 5000），其总耗时time（前向+反向）呈现线性上涨（红色曲线），而前向（fwd_time）和反向（bwd_time）也呈现线性上涨，但是其反向时间/前向时间（bwd_time/fwd_time）的比例基本维持在1，因此并不会出现朋友问题中的那种情况，可以初步排除是采用FC层进行Embedding提取的可能性。

![exp_a_fc_emb][exp_a_fc_emb]

<div align='center'>
    <b>
        Fig 2. 采用FC层进行Embedding的耗时试验。
    </b>
</div>

那么可以初步判断朋友是采用查表的方式实现的，我们用Appendix Code B.的代码进行验证。我们可以发现，总耗时同样随着token数量增加而线性上涨，但是前向时间却保持恒定（~0.2s），而反向耗时则随着token数量增加而线性上涨，反向耗时/前向耗时同样呈现线性上涨，这个现象满足朋友的描述。可以断定朋友是采用了类似于Appendix Code B.的代码进行模型训练的。

![exp_b_lookup_emb][exp_b_lookup_emb]

<div align='center'>
    <b>
        Fig 3. 采用查表的方式实现的Embedding耗时试验。
    </b>
</div>


这个和土豆之前的想法有部分矛盾，首先其前向过程的确是计算复杂度为$\mathcal{O}(1)$的，这个也被刚才的试验验证了。但是为何其反向复杂度是会随着token数量增加而增加的呢，计算复杂度看来是$\mathcal{O}(n)$，倒像是通过FC层进行反向传播的样子。我们通过以下代码，打印出通过查表方式得到的Embedding层参数的梯度，进行观察，我们发现虽然梯度只在第1,3,5,7行为非0，但是其他行虽然为0.0同样会作为一个有效的梯度，参与反向梯度传播（即便此时梯度值为0，参与了梯度反向传播也不会影响到对应ID的Embedding参数更新）。此时的反向传播过程，其实和FC层进行Embedding提取的反向传播过程是一致的，会随着token数量的增加而增加反向传播的计算复杂度$\mathcal{O}(n)$，这就解释了朋友观察到的现象。

![grad][grad]

<div align='center'>
    <b>
        Fig 4. 通过查表方式得到的Embedding的参数梯度。
    </b>
</div>

怎么解决呢？我们看到pytorch的`nn.Embedding`层中有个叫`sparse`的参数，这个参数如果指定为真，则表示梯度对于权重矩阵而言，以稀疏矩阵的方式进行，此时梯度是稀疏的，将只考虑有效的ID对应行的权重矩阵的梯度更新，此时那些为0.0的梯度就不会再被参与反向传播计算了，从而将计算量维持在$\mathcal{O}(1)$。

![sparse_grad][sparse_grad]

<div align='center'>
    <b>
        Fig 5. 在nn.Embedding层中指定sparse为真，将采用稀疏梯度进行Embedding参数的更新。
    </b>
</div>

让我们用Appendix Code C.的代码进行试验，我们发现此时无论是前向耗时，还是反向耗时都是$\mathcal{O}(1)$级别的了，此时符合我们对于Embedding层的预期。

![exp_c_sparse_grad][exp_c_sparse_grad]

<div align='center'>
    <b>
        Fig 6. 采用了稀疏梯度之后，其总耗时，前向耗时和反向耗时都是常数级别的。
    </b>
</div>






# Appendix 



## Code A. 采用FC层进行Embedding的耗时试验

```python
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
time_list = []
bw_time_list = []
fwd_time_list = []
ratio_list = []

for n in (10000,20000,30000,40000,50000,60000,70000,80000,90000,100000):
    n = n // 100
    emb = torch.rand((n, 256))
    emb = Variable(emb, requires_grad=True)
    inputs = torch.randint(0, n, (10000,))
    label = torch.rand((10000, 256))
    inputs_onehot = Variable(F.one_hot(inputs, num_classes=n).float(), requires_grad=False)

    begin = time.time()
    bw_time = 0
    fwd_time = 0
    for i in range(100):
        begin_fwd = time.time()
        pred = torch.matmul(inputs_onehot, emb)
        end_fwd = time.time()
        fwd_time += end_fwd - begin_fwd
        
        loss = (pred-label).mean()
        begin_bw = time.time()
        loss.backward()
        end_bw = time.time()
        bw_time += end_bw - begin_bw
    end = time.time()
    print("n={}, time={}, bw_time={:.4f}, fwd_time={:.4f}, bwd_time/fwd_time={:.4f}".format(
        n, end-begin, bw_time, fwd_time, bw_time/fwd_time
    ))
    time_list.append(end-begin)
    bw_time_list.append(bw_time)
    fwd_time_list.append(fwd_time)
    ratio_list.append(bw_time/fwd_time)
plt.plot(time_list, color='r',label='Total Time')
plt.plot(bw_time_list, color='b',label='Backward time')
plt.plot(fwd_time_list, color='g',label='Forward time')
plt.legend()
```

## Code B. 采用查表的方式进行Embedding的耗时试验

```python
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
time_list = []
bw_time_list = []
fwd_time_list = []
ratio_list = []

for n in (10000,20000,30000,40000,50000,60000,70000,80000,90000,100000):
    emb = nn.Embedding(n, 256)
    inputs = torch.randint(0, n, (10000,))
    label = torch.rand((10000, 256))
    begin = time.time()
    bw_time = 0
    fwd_time = 0
    for i in range(100):
        begin_fwd = time.time()
        pred = emb(inputs)
        end_fwd = time.time()
        fwd_time += end_fwd - begin_fwd
        
        loss = (pred-label).mean()
        begin_bw = time.time()
        loss.backward()
        end_bw = time.time()
        bw_time += end_bw - begin_bw
    end = time.time()
    print("n={}, time={}, bw_time={}, fwd_time={}, bwd_time/fwd_time={}".format(
        n, end-begin, bw_time, fwd_time, bw_time/fwd_time
    ))
    time_list.append(end-begin)
    bw_time_list.append(bw_time)
    fwd_time_list.append(fwd_time)
    ratio_list.append(bw_time/fwd_time)
plt.plot(time_list, color='r',label='Total Time')
plt.plot(bw_time_list, color='b',label='Backward time')
plt.plot(fwd_time_list, color='g',label='Forward time')
plt.legend()
```

## Code C. 采用稀疏梯度之后的耗时试验

```python
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
time_list = []
bw_time_list = []
fwd_time_list = []
ratio_list = []

for n in (10000,20000,30000,40000,50000,60000,70000,80000,90000,100000):
    emb = nn.Embedding(n, 256, sparse=True)
    inputs = torch.randint(0, n, (10000,))
    label = torch.rand((10000, 256))
    begin = time.time()
    bw_time = 0
    fwd_time = 0
    for i in range(100):
        begin_fwd = time.time()
        pred = emb(inputs)
        end_fwd = time.time()
        fwd_time += end_fwd - begin_fwd
        
        loss = (pred-label).mean()
        begin_bw = time.time()
        loss.backward()
        end_bw = time.time()
        bw_time += end_bw - begin_bw
    end = time.time()
    print("n={}, time={}, bw_time={:.4f}, fwd_time={:.4f}, bwd_time/fwd_time={:.4f}".format(
        n, end-begin, bw_time, fwd_time, bw_time/fwd_time
    ))
    time_list.append(end-begin)
    bw_time_list.append(bw_time)
    fwd_time_list.append(fwd_time)
    ratio_list.append(bw_time/fwd_time)
plt.plot(time_list, color='r',label='Total Time')
plt.plot(bw_time_list, color='b',label='Backward time')
plt.plot(fwd_time_list, color='g',label='Forward time')
plt.legend()
```







[question]: ./imgs/question.png
[Embedding_inplement]: ./imgs/Embedding_inplement.png
[exp_a_fc_emb]: ./imgs/exp_a_fc_emb.png
[exp_b_lookup_emb]: ./imgs/exp_b_lookup_emb.png
[grad]: ./imgs/grad.png
[sparse_grad]: ./imgs/sparse_grad.png
[exp_c_sparse_grad]: ./imgs/exp_c_sparse_grad.png









