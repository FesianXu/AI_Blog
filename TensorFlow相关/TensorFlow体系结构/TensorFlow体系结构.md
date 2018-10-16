<h1 align = "center">TensorFlow的体系结构</h1>

## 前言
** 本文翻译自官方的体系结构介绍，有利于理解TensorFlow系统的整体框架结构，有利于自行后续阅读源码，因此翻译为中文，以飨国人，原文出自[TensorFlow Architecture](https://www.tensorflow.org/extend/architecture) **

**如有谬误，请联系指正。转载请注明出处。**

*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`

*******************************************************

我们设计的TensorFlow（下面简称为TF）可应用于大规模分布式训练和推理，同时，他也足够灵活，可以支持开发新的机器学习算法和系统层面上的优化过程。
这份文档描述了这个系统的整体框架，正是得益于这个框架，TF才能同时兼具有灵活性和多尺度性（指的是可以在大规模数据下分布式训练）。读这份文档之前，假设读者已经有了对TF的基本了解，知道怎么用TF的编程模式，如计算图编程，操作（operation）和对话（session）等，如果你不了解也没关系，读[这份文档](https://www.tensorflow.org/guide/low_level_intro)了解些入门知识吧，同时，了解一些[分布式TF](https://www.tensorflow.org/deploy/distributed)同样是大有裨益的。
这份文档是为了那些想要扩展TF的功能的软件工作者，或者是想要优化TF的硬件性能，加速大规模数据下的训练和机器学习算法实现的硬件工程师，亦或是任何一个想要探索TF的底层的人士准备的。在读到这篇文章的最后，你应该能够理解TF的基本结构，足够帮助你阅读并且修改TF的核心代码。

## 概览
TF的运行库是一个跨平台的库，下图表示了他的通用结构。一个基于C的API底层分隔开了不同的用户级的代码，这些代码可以是来自于核心运行库中的多种语言的，如常用的python，c++等，包括近期的tensorflow.js。
![layers][layers]
这个文档将会关注下列几个方面：

1. **用户端(client)**
 * 	用于在图(graph)中定义数据流(dataflow)中的计算过程
 * 	用对话(session)初始化图的执行过程(execution)

2. **分布式主机(distributed master)**
 * 从图中进行剪枝出一些特定的子图，作为`Session.run()`中的参数
 * 将子图划分到多个地方，用于在不同的进程或者设备中运行
 * 分布这些图的片段到工作者服务(Worker services)中
 * 通过工作者服务初始化这些图的片段执行过程

3. **工作者服务(Worker services)**
 * 利用核心实现(kernel implementations)将图中的操作调度到可用的硬件设备上(CPU,GPU,TPU,etc)
 * 从另一个工作者服务中接受操作结果或者将结果发送到另一个工作进程服务中。

4. **核心实现(kernel implementations)**
 * 为单独的图上的操作进行执行计算。


![figure2][figure2]

如上图，这张图展示了上述部件之间的交互关系。其中，`/job:worker/task:0`和`/job:ps/task:0`都是在工作者服务下的任务，字母中的`ps`表示了`parameter server`，既是参数服务器，他是一个负责储存和更新模型参数的任务。当其他的任务在最优化模型的参数的时候，他们将会发送更新参数的信号给这些参数们。这里在任务之间的分号并不是必须的，但是他在分布式训练中相当的普遍。
注意到这里的分布式主机和工作者服务只会出现在分布式的TF中。在单进程版本的TF中，包含了一个特殊的对话(Session)实现，他可以完成任何分布式主机可以完成的东西，但是只会和本地进程的设备进行通信。

****

## 客户机（Client）
作为用户的我们，我们一般情况下只会关心我们的模型的图该怎么构建，这个时候我们用TF的Client进行计算图的搭建。TF不仅支持直接用分立的操作符进行组装构建模型，而且可以利用一些方便的库，如Estimation APIs等组装出一个神经网络和其他更加高阶的抽象的网络。是的，TF由C写成，但是C过于复杂，你一般不会想在这个语言中构造复杂的模型并且调试，这将会是个痛苦的过程，因此TF封装出了多种面向客户的客户机语言，用于简单明了地构建计算图。TF支持多种用户机语言，优先推荐的是python和c$++$，前者适合制作模型的原型后者适合部署模型。随着功能逐步确立下来，我们将他们封装成了C$++$，这样用户就可以从所有的客户机语言中得到最好的实现。尽管大部分的训练库都还只有python的接口，但是C$++$确实是支持更为高效的实现的。
client负责创建一个对话，而对话可以通过`tf.GraphDef`协议，将计算图的定义发送到分布式主机中。当client需要计算(evaluate)一个节点的值的时候，这个计算过程将会触发一个对分布式主机的调用，用于初始化这个计算过程。
正如下图所示，这个clien构建了一个应用了权值w和特征向量x的计算图，并且加上了一个偏置项b，并且将其保存在一个变量(variable)s中。
![figure3][figure3]

**相关代码参考**：
* [tf.Session](https://www.tensorflow.org/api_docs/python/tf/Session)

## 分布式主机(Distributed master)
分布式主机可以：
* 从整个计算图中进行剪枝，得到可以计算由client请求的节点计算用的子图。（翻译的比较乱，其实意思就是可以将整个图给剪枝分配给不同从机进行分布式工作。）
* 切分计算图以得到计算图的片段给每一个设备，同时缓存这些片段以至于可以在接下来的步骤中一直重利用。

因为主机可以一次性观察到整个计算图，因此它将采用标准的优化措施，如公共子表达式消去(common subexpression elimination)和常数折叠(constant folding)。
![figure4][figure4]

下图展示了一个可能存在的一个任务切分，作为一个例子。这个分布式主机聚合了模型的参数，以便于可以在参数服务器中同时安置他们。
![figure5][figure5]

当计算图的边缘被切分的时候，这个分布式主机将会插入和接受到一些在分布式任务中用于传递信息的节点。
![figure6][figure6]

然后分布式主机搬运这些图的片段到分布式任务中。
![figure7][figure7]

**相关代码参考**：
* [MasterService API definition](https://www.github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/protobuf/master_service.proto)
* [Master interface](https://www.github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/distributed_runtime/master_interface.h)

****
## 工作者服务(Worker Service)
每个任务的工作者服务指的是：
* 处理来在于主机的请求。
* 对构成本地子图的操作，调度内核的执行指令，并且作为任务之间的直接沟通的中间件。

我们为了以小代价来运行大规模的计算图，优化了工作者服务。我们的当前实现可以在一秒钟内执行成千上万的子图，这个使得在训练阶段可以对大规模的相同操作进行快速的处理。工作者服务分派内核给当前的本地设备，并且尽可能的并行运行内核，比如在多核CPU和GPU的情况下就可以这样。
我们针对每一对源和目的设备类型，详细说明了发送(Send)和接受(Recv)操作：
* 在本地CPU和GPU设备间进行迁移，可以采用`cudaMemcpyAsync()` API以覆盖计算和数据迁移。
* 直接在两个本地GPU之间进行对等的DMA数据迁移，可以避免通过host CPU的高昂的复制代价（经过了中转必然导致一些代价。）

对于任务之间的迁移，TF采用了不同的协议，包括：
* 基于TCP协议的gRPC
* 基于Converged Ethernet的RDMA

在多核CPU的通信中，我们同样有对英伟达的NCCL库的初步支持，见[tf.contrib.nccl](https://www.tensorflow.org/api_docs/python/tf/contrib/nccl)
![figure8][figure8]

**参考代码**：
* [WorkerService API definition](https://www.github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/protobuf/worker_service.proto)
* [Worker interface](https://www.github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/distributed_runtime/worker_interface.h)
* [Remote rendezvous (for Send and Recv implementations)](https://www.github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h)

****

## 核心实现（Kernel Implementations）
运行环境中包括了操作200个标准的操作，包括了数学运算，矩阵操作，控制流和状态管理操作等。每一个操作都有核心实现，而且这些实现都为不同的设备而专门优化过。许多这些操作核心是通过使用`Eigen::Tensor`实现的，这个可以实现利用C$++$模版用来生成高效并行代码以为多核CPU和GPU使用。然而，我们一般直接使用一些库，像是cuDNN，它具有一个更为高效的核心实现。我们同样实现了[Quantization](https://www.tensorflow.org/performance/quantization)，这个东西可以加速在一些移动设备上的推理速度，并且使用[gemmlowp](https://github.com/google/gemmlowp)，一个低精度的矩阵运算库可以加速量化计算过程。

如果在通过把一个计算表示为一系列已有的操作的组合存在困难或者低效率（也就是通过现有的客户机接口，比如python难以实现或者难以高效实现一些操作），那么用户可以通过利用C++来编码一个更为高效的实现，然后将其登记到内核中去。例如，我们推荐对于一些性能要求苛刻的操作，如ReLU和Sigmoid激活函数和他们相应的梯度等，将其进行登记。[XLA Compiler](https://www.tensorflow.org/performance/xla/index)有过一个实验性质的自动内核融合的实现。

**参考代码**：
* [OpKernel interface](https://www.github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/framework/op_kernel.h)


****

## 写到最后
这篇文章实在不好翻译，很多地方翻译的不好，有需要的同学请自行翻看原文，有一些地方其实没有太理解的，翻译也没有翻译好，需要以后进一步理解。


[layers]: ./imgs/layers.png
[figure2]: ./imgs/fig2.png
[figure3]: ./imgs/fig3.png
[figure4]: ./imgs/figure4.png
[figure5]: ./imgs/figure5.png
[figure6]: ./imgs/figure6.png
[figure7]: ./imgs/figure7.png
[figure8]: ./imgs/figure8.png
