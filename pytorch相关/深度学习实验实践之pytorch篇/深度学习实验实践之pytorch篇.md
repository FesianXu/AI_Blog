<div align=center>
<font size="6"><b>
《深度学习实践》 之 基于pytorch的实验模型搭建模式
</b></font> 
</div>


[TOC]

****

# 前言

**我们在进行深度学习实践过程中，通常是会采用成熟的深度学习框架，如目前最火的PyTorch[2]和TensorFlow[1]。然而，不同的框架有着不同的特性，有着不同的模型搭建模式，本文基于PyTorch框架，简单介绍下笔者在深度学习实践过程中的一些小经验，望能抛砖引玉，与各位共同探讨更为有效的深度学习模型搭建模式。**
**如有谬误，请联系指正。转载请注明出处。**
*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`

****

PS: 在深度学习实践中，特别是模型的原型搭建的过程中，因为通常要求“短平快”，也就是在短时间内能最快验证模型的效果，因此通常是会选择python作为编程语言进行搭建模型的，而在实际生产和模型部署中，才会选择C++等更为运行高效的语音进行重写。因此，本文将会是基于python作为基础进行的。

# 深度学习实践常用工具介绍

在深度学习实践中，基于python，有很多常用的工具可以大大加快你的模型搭建效率，这里介绍一二，以供参考。

1. Anaconda[3]，是一个集成了诸多科学计算工具包的一个大集合，分为python2和python3两个版本，里面的conda[4]类似于pip，是一个python的包管理工具，可以很方便的进行虚拟开发环境的管理，克隆，移植等。如果安装了anaconda，就不需要安装python了。
2. 利用conda的install功能，我们可以很容易地安装我们所需要的几乎所有包，这里我们需要pytorch，其为主要的深度学习开发框架，其次，我们需要matplotlib，其为可视化绘图的有效工具，最后，强烈建议安装jupyter notebook，其为一个交互式开发环境，对于模型的开发起到了加速的作用，更关键的是，其支持作为远程服务端，方便你在笔记本上调试服务器上的程序。

因此，以下的内容将会假设各位已经安装了pytorch，anaconda和pytorch，因为笔者是在远端服务器上进行模型的训练和测试的，而在本地笔记本上进行代码编写，因此假设各位有着自己的代码编辑软件，笔者是pycharm和vim皆可，其中pycharm的远程解释器和同步功能可能更适合刚入门的同学。

# 基于pytorch的模型和实验模式

有些同学可能会疑惑为什么需要在意模型搭建模式呢？直接构思出模型的结构，直接撸代码，然后训练不就得了吗？的确，这样能搭建出一个模型，运气好的话也许还能正常的跑起来。但是我们知道，在深度学习中最不可避免的就是调参，因此在计算力足够的情况下，通常会开很多个相同的程序进行训练，以得到最好的超参数[5]，这个过程中，如果按照刚才那种直接改模型代码的方式去进行，很容易陷入紊乱，因为要跑的超参数太多，你一直改动，就很可能忘了一开始的超参数是什么了，这样子很容易导致实验记录错误。**因此，我们对深度模型的搭建和实验这两个过程进行解构，分为模型搭建，实验和超参数挑选这两大点。**这两个点，所用的工具和思想不同。我们下面分别介绍。

## 模型搭建

我们留意到，当我们确定了一个模型的结构的时候，如果这个模型确定能work，但是不确定是否是最优模型结构，需要经常地去调试超参数，这个时候我们模型的基本框架其实是没有太大的变化的，我们只需要这里改下参数，或者那里改下参数就行了，也就是说我们的模型代码是相对固定的。我们可以说，这个时候，我们需要**结构化**我们的模型代码，这个对于TensorFlow来说，其实我们在[6]已经有所介绍了。

在笔者的实践过程中，对于这种类型的代码，为了更好的结构化，我选择用pycharm的远程解释器和同步功能，或者直接用vim对远程模型代码进行编写，对于pytorch框架来说，大体来说无非是需要继承`DataLoader`类和`nn.Module`类，使得程序可以异步加载自己的数据集和运行自己的模型，对于继承`DataLoader`类，这个较为简单，各位不妨去其他博客学习，我们主要介绍的是如何**结构化**继承`nn.Module`类。

pytorch有个很方便的特点就是，只要你继承了`nn.Module`类，那么就可以把这个子类看成是个模型了，可以由别的模型调用了，这里简单说不好理解，我们用代码示意下，注意这个代码不是完全的。

```python
import torch
import torch.nn as nn
import second_model.model as xxnet
# 这里second_model 可以是你自己定义的其他模型，这里可以把它看成子模型使用

class MyModel(nn.Module):
    def __init__(self,
                 alpha,
                 beta):
        super().__init__()
        # 注意，子模型的定义，如conv1 conv2一般都在__init__()方法中完成，这样才能保证你的参数是只分配了一次内存的。而且，只有在__init__()的子模型，在使用model.cuda()的时候才能继承，一次性进入显存。
        self.alpha = alpha
        self.beta = beta
        self._is_first = True
        
        self.conv1 = nn.Conv2d(...)
        self.conv2 = bb.Conv2d(...)
        
    
    def forward(self, inputvs, labels):  # 我们需要重写forward方法，这里定义了模型的操作实体
        # 注意，如果你的模型需要在这里定义子模型，如fc，那么确保只初始化一次，不然会导致OOM问题。
        if self._is_first:
            self.fc = nn.Linear(...)
            self._is_first = False
        # 然后这里对操作实体进行定义
        conv = self.conv1(inputvs)
        conv = self.conv2(conv)
        return conv
        
```





****

# Reference

[1]. [TensorFlow](https://www.tensorflow.org/)

[2]. [PyTorch](https://pytorch.org/)

[3]. [Anaconda](https://www.anaconda.com/download/)

[4]. [conda](https://conda.io/docs/)

[5]. [训练集，测试集，检验集的区别与交叉检验](https://blog.csdn.net/LoseInVain/article/details/78108955)

[6]. [tensorflow编程实践：结构化你的模型](https://blog.csdn.net/LoseInVain/article/details/82085185)