<div align='center'>
    [darknet源码系列] darknet源码中的常见数据结构
</div>

<div align='right'>
    FesianXu 20201117 at UESTC
</div>

# 前言

最近笔者在好奇如何从最底层开始搭建一个深度学习系统，之前都是采用现成的成熟深度学习框架，比如`PyTorch`，`TensorFlow`等进行模型的搭建，对底层原理了解不是特别深刻。因此笔者最近在阅读darknet的源码，希望能从中学习到一些底层的知识，本文主要是对darknet中常见的数据结构进行记录和分析。**如有谬误请联系指出，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**QQ**: 973926198

github: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

----

# DarkNet

`darknet` [1]是一个纯由C语言编码而成的轻量级深度学习框架，在最小运行状态下（即是不使用GPU，不使用多线程和opencv）时，可以实现无外部库依赖实现深度学习建模，训练，测试等基础功能。如果从`caffe`,`pytorch`,`tensorflow`等大型深度学习库开始去研究底层，因为代码结构复杂，而且依赖项太多，使得入门者望而却步，由于`darknet`可以在无依赖的情况下运行，因此是研究深度学习框架底层代码的一个很好的资源。

`darknet`中定义了很多必要的数据结构去表示网络结构，网络层结构等，这些数据结构为网络配置解析提供了方便，是阅读源码过程中必不可少的，本文主要分析`darknet`中主要的数据结构。

# 数据结构

## 为了解析方便定义的数据结构 

类似于在`caffe`中利用`prototxt`文件去表示一个网络的结构，每一层的超参数以及整个网络的超参数（例如学习率，优化器参数等），在`darknet`中采用了`cfg`文件去配置网络的这一切参数，可以视为是简单版本的`prototxt`文件，以下以`resnet18.cfg`为例子，截取了其中的头尾部分的`cfg`片段：

```shell
[net]
# Training
# batch=128
# subdivisions=1

# Testing
batch=1
subdivisions=1

height=256
width=256
channels=3
min_crop=128
max_crop=448

burn_in=1000
learning_rate=0.1
policy=poly
power=4
max_batches=800000
momentum=0.9
decay=0.0005

angle=7
hue=.1
saturation=.75
exposure=.75
aspect=.75

[convolutional]
batch_normalize=1
filters=64
size=7
stride=2
pad=1
activation=leaky

[maxpool]
size=2
stride=2

# Residual Block
[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=linear

[shortcut]
activation=leaky
from=-3

[convolutional]
filters=1000
size=1
stride=1
pad=1
activation=linear

[softmax]
groups=1
```
<div align='center'>
    <b>
        code 1. resnet18.cfg的节选片段示例。
    </b>
</div>

其中每一个`[xxx]`代表一个**片区(section)**，通常表示的是一个 **层（layer）**。其中的第一个片区`[net]`或者`[network]`是表示该网络的基本参数，比如图片长度，高度通道数，`batch_size`，学习率，优化器参数等。每个配置文件必须以`[net]`起始。

为了解析该文本配置文件的方便，我们需要用一种数据结构将整个网络配置串联起来，最理想的莫过于是 **链表(linked list)** 了，其中链表中的每个元素都是一个片区，该链表的数据结构定义如下，为了更好地组织数据，定义的是双向链表，因此每个节点`node`都有前项`prev`和后继`next`节点指针，当然还有内容负载指针 `void* val`。

```c
typedef struct node{
    void *val; // 节点内容负载
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front; // 双向链表头结点
    node *back;  // 双向链表尾节点
} list;
```
<div align='center'>
    <b>
        code 2. 关于双向链表和节点的定义。
    </b>
</div>

因此可视化出来Fig 1所示，其中的负载既可以是`section`，也可以是其他元素，比如`key-val`（键值）对，此处暂时以`section`为例子。

![double_linked_list][double_linked_list]

<div align='center'>
    <b>
        Fig 1. 以内容负载为section作为例子的双向链表，其中头尾元素需要特别用指针标记出。
    </b>
</div>

其中每个`section`都是神经网络的一个层（除了第一个`section`，第一个是特殊的`[net]`参数组），`section`的数据结构定义如code 3所示，其中的`char* type`表示该层的名字，比如`[convolutional]`，`[avgpool]`等；其中的`list* options`表示的是该层中的每一个参数的 **键值对（key-val pair, kvp）**,每一个键值对都是该层的一个具体参数，比如`[convolutional]`的`stride=1`就表示其步进长度，我们需要用一个键值对数据结构进行表示，如code 4所示。

```c
typedef struct{
    char *type;
    list *options;
}section;
```
<div align='center'>
    <b>
        code 3. 片段section数据结构的定义。
    </b>
</div>

```c
typedef struct{
    char *key; // 索引键
    char *val; // 对应值
    int used;  // 是否被使用
} kvp;
```
<div align='center'>
    <b>
        code 4. 键值对kvp数据结构的定义。
    </b>
</div>
如Fig 2所示，当考虑到每个`section`内的参数kvp之后，我们有新的"链表套链表"的结构，采用该结构就足以表达整个网络的层次与参数了。

![section_linked_list][section_linked_list]

<div align='center'>
    <b>
        Fig 2. 当把section内部的kvp考虑了之后，结合Fig 1形成了最终的网络结构化表达形式。
    </b>
</div>



## 网络结构的数据结构

之前谈到的数据结构是为了解析`cfg`文件方便而定义的，考虑到网络的计算（包括前向和后向计算），参数保存等，我们还需要一些其他数据结构，这些数据结构可以基于解析得到的双向网络配置链表，初始化整个神经网络的参数和结构。这些数据结构中，最重要的莫过于`layer`和`network`。

`layer`的数据结构定义如code 5所示，我们发现这个定义中有着非常多的元素（其实我还省略掉了一些和GPU计算有关的元素，因为在本文中假设只使用CPU进行计算，这样又便于我们分析整体代码的结构。），其中元素那么多的原因在于，作者采用的纯C语言的编写方式，没办法直接采用C++的面向对象的思想设计代码，这意味着不能通过继承的方式，设置一个`layer`父类，然后不同的`convolution_layer`,`rnn_layer`,`crnn_layer`,`cost_layer`,`connected_layer`等子类继承这个共有的父类，因为不同子类（也就是不同层）的参数类别千差万别，因此可以做到比较好的隔离。然而，C语言是面向过程编程的，因此作者设计的`layer`类就必须包括该框架中需要的所有层的所有参数类别，这使得该数据结构异常的臃肿，而且难以扩展，定制其他层。不过暂且不讨论这些缺点，我们先看看该层中有哪些参数需要注意的吧。

```c
struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;
    
    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu; 

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
	
    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;
    tree *softmax_tree;
    size_t workspace_size;
};
```

<div align='center'>
    <b>
        code 5. layer的定义，其中为了简便，省略掉了和GPU计算有关的元素。
    </b>
</div>

第一个元素`LAYER_TYPE`是一个枚举类型，用于指定该层的类型；第二个元素`ACTIVATION`也是一个枚举类型，指定激活函数类型；第三个元素`COST_TYPE`同样是枚举类型，指定损失函数类型。5-7行的三个函数指针（指向函数的指针）比较特别，








```c
typedef struct network{
    int n;
    int batch;
    size_t *seen; // how many pictures have been processed already
    int *t; // ? 
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;
} network;

```








# Reference

[1]. https://pjreddie.com/darknet/

[2]. 





[double_linked_list]: ./imgs/double_linked_list.png

[section_linked_list]: ./imgs/section_linked_list.png





