<div align='center'>
    正确使用pytorch，以确保实验的可复现性
</div>

<div align='right'>
    FesianXu 20201214 at UESTC
</div>

# 前言

本文翻译自官方文档[1]，主要阐述了如何正确地使用`pytorch`，以确保实验的可复现性(reproducibility)。**如有谬误请联系指出，本文遵守[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**QQ**: 973926198

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]



----

通常来说，不同`pytorch`发行版，不同硬件平台乃至不同的版本`commits`之间都有着很多细节上的区别，并不能完全保证实验结果的可复现性。并且，在不同硬件构架上进行实验，比如CPU和GPU之间的实验结果，也有可能不能完全复现，即便采用了固定的随机种子。

然而，作为实验者，我们还是可以采用一些手段去减少实验之中的不确定性，以提高在不同特定平台，设备和发行版之间的可复现性。首先，我们可以控制随机性，去控制参数初始化，数据随机加载过程中的随机性，这些随机性通常会导致多次实验之间的差别。其次，我们可以通过配置`pytorch`去避免在一些操作中使用一些非确定性（nondeterministic）的算法，因此可以确保多次调用同一个函数，在给定了同一个输入时，都可以返回相同的结果。

**注意**： 确定性操作/算法通常要比非确定性的操作要慢一些，因此模型的单次运行速度可能会受到损失。然而，采用确定性行为的算子，可以节省在调试，开发过程中的时间。



# 控制随机性

## pytorch随机数生成

你可以使用`torch.manual_seed()`去设置，固定所有设备上的随机数种子，如code 1所示，则可以有效地控制随机数生成，使得每次重复实验中的随机数总是一致的。

```python
import torch
torch.manual_seed(0)
```

<div align='center'>
    <b>
        code 1. 固定随机数种子。
    </b>
</div>

## 其他库中的随机数生成器

如果使用中的任何库依赖于`numpy`，那么通过设置全局的numpy随机数种子也可以达到固定随机数的目的。

```python
import numpy as np
np.random.seed(0)
```

# CUDA卷积基准化

The cuDNN library, used by CUDA convolution operations, can be a source of nondeterminism across multiple executions of an application. When a cuDNN convolution is called with a new set of size parameters, an optional feature can run multiple convolution algorithms, benchmarking them to find the fastest one. Then, the fastest algorithm will be used consistently during the rest of the process for the corresponding set of size parameters. Due to benchmarking noise and different hardware, the benchmark may select different algorithms on subsequent runs, even on the same machine.

Disabling the benchmarking feature with `torch.backends.cudnn.benchmark = False` causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.

However, if you do not need reproducibility across multiple executions of your application, then performance might improve if the benchmarking feature is enabled with `torch.backends.cudnn.benchmark = True`.

Note that this setting is different from the `torch.backends.cudnn.deterministic` setting discussed below.

CUDA卷积库调用了cuDNN库，有可能在多次重复的操作执行中，成为非确定性操作的源头。当一个cuDNN卷积操作，在以一些新的尺寸参数被调用时，系统会尝试去运行多种卷积算法，以便于找到一种最快的，然后依此为基准。然后这个最快的卷积算法将会在剩下的流程中一直被调用（当然，你得保证对应的参数尺寸大小不能改变）。因为不同硬件之间的架构差异，甚至存在基准噪声，这个基准可能会在你多次运行时选取不同的算法作为最快算法，甚至在同一个机器上也可能发生这种事情。

可以通过`torch.backends.cudnn.benchmark = False`去避免这个基准的特性，这会使得cuDNN确定性地去选择一个算法，可能会造成一些性能上的损失，但是至少能保证每次的运行结果一致性。





# 避免非确定性操作









# Reference

[1]. https://pytorch.org/docs/stable/notes/randomness.html#reproducibility





[qrcode]: ./imgs/qrcode.jpg





