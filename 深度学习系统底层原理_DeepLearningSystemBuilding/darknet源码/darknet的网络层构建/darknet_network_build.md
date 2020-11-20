<div align='center'>
    [darknet源码系列-2] darknet源码中的cfg解析
</div>



<div align='right'>
    FesianXu 20201118 at UESTC
</div>

# 前言

笔者在[1]一文中简单介绍了在`darknet`中常见的数据结构，本文继续上文的节奏，介绍如何从`cfg`文本文件中解析出整个网络的结构与参数。**如有谬误请联系指出，本文遵守[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**QQ**: 973926198

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

-----

**注意：阅读本文之前，建议阅读[1]，以便对`darknet`的数据结构定义有所了解。** 为了简便，本文暂时不考虑GPU下的运行，只考虑CPU运行的情况。

# 初探

最主要的网络结构和参数解析函数在`/src/parser.c`里，该函数名为`network *parse_network_cfg`[2]，此函数完成了`cfg`文件的解析，并且通过解析得到的网络结构与参数初始化`network`结构体，以便于后续的网络计算。我们从该函数开始进行剖析，部分代码见coda 1.1，传入参数很简单，就是`cfg`文件的名字`char* filename`，而返回的就是解析并初始化后的`network*`。为了让读者回顾`darknet`的基本数据结构，我们展示Fig 1.1，该链表承载了作为解析过程中的主要数据负载作用。具体的函数分析见code 1.1的注释。

![section_linked_list][section_linked_list]

<div align='center'>
    <b>
        Fig 1.1 该双向链表承载了网络配置解析过程中的主要内容负载作用，此时注意到无论是front还是back，其终端都是NULL，这点需要去看函数list_insert[3]。
    </b>
</div>



```c
network *parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename); // 解析cfg配置文件，返回配置链表，如Fig 1.1所示
    node *n = sections->front; // 获取sections的首节点，一般是[network]或者[net]类型的section
    if(!n) error("Config file has no sections"); // 判断是否存在section
    network *net = make_network(sections->size - 1); //构建network,其中的sections->size是包括了[net]之后的长度，因此要减去1，一共有sections->size-1层。
    net->gpu_index = gpu_index;
    size_params params; // 该数据结构承担了初始化网络结构，参数的重任，该结构体的定义如code 1.2所示。

    section *s = (section *)n->val;
    // 取得第一个section节点的指针，因为第一个section一般是一些全局的配置，具体见[1]。
    list *options = s->options; // 第一个section内容负载中的双向链表
    // 第一个section类型必须是[net]或者[network]
    if(!is_network(s)) error("First section must be [net] or [network]");
    // 如果没啥问题，就开始解析网络的全局配置，也即是[net]中的内容，将解析出的内容赋值到net数据结构中
    parse_net_options(options, net); 

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;
    // 这一段都是在对对应的值进行赋值，没太多可说的

    size_t workspace_size = 0;
    // workspace_size 指明了所有层（layer）中最大的内存需求，从而提前开辟出整块内存以便后续计算，同一时间内，GPU或者CPU只有一个层在计算，因此只需要满足最大内存需求即可。
    n = n->next; // 我们已经把[net]结构体解析完啦，现在我们考虑之后的层，一般就是实际的神经网络层了。
    int count = 0;
    free_section(s); 
    fprintf(stderr, "layer     filters    size              input                output\n");
    while(n){  // 当 NULL == n 的时候退出解析
        params.index = count; // 每一层的计数器，从0开始
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val; // 每一层的实际负载，也即是section本体
        options = s->options; // 实际section的双向链表，其中的元素
        layer l = {0}; // 解析出的section将初始化到layer中，此处先定义layer。
        LAYER_TYPE lt = string_to_layer_type(s->type); // 将字符串格式的层名，比如[convolutional]等解析为枚举类型的数据
        
        // 针对不同类型的层lt，有着不同的解析函数，格式为 parse_xxxx(options, params)，我们后续以卷积层为例子，其他层也是类似的。
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params);
        }else if(lt == DECONVOLUTIONAL){
            l = parse_deconvolutional(options, params);
        }
        .... 
        // 为了简便，这里省略了很多相似的parse解析函数，分别解析不同的层
        else if(lt == DROPOUT){
            l = parse_dropout(options, params);
            l.output = net->layers[count-1].output;
            l.delta = net->layers[count-1].delta;
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        
        l.clip = net->clip;
        l.truth = option_find_int_quiet(options, "truth", 0);
        ....
        // 这里省略了类似的参数解析过程
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        // 针对每个特定层，还可以指定其特定的学习率learning_rate，是否停止梯度stopbackward,平滑smooth,保存参数与否dontsave，加载参数与否dontload,等细节参数，这些参数将会在该层覆盖全局[net]的设置。这里的代码在解析这些参数。
        
        option_unused(options); // 将没有使用到的参数打印（因为有可能cfg文件写错了，某些层不需要某些参数，但是又多写了，这里需要提示出来）
        net->layers[count] = l; // 将解析得到的layer赋值到network中。
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        // 每个特定层，需要的内存空间是不一样的，这个在定义某个层的时候是需要预先定义（计算）出来的，通过这个代码纪录整个网络需要的最大内存空间。
        free_section(s); // 释放解析临时内存
        n = n->next; // 走，哥们我们去下一个section
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    
    free_list(sections);
    layer out = get_network_output_layer(net); // 取得输出层的句柄
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    // 网络的输出就是输出层的最终输出
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    // 给网络输入分配空间,大小取决于每个样本的维度net->inputs和批次大小net->batch
    net->truth = calloc(net->truths*net->batch, sizeof(float));
    // ground truths 的内存分配
    if(workspace_size){
        //printf("%ld\n", workspace_size);
    	net->workspace = calloc(1, workspace_size);
    }
    // 正如之前说的，在任意时刻 GPU或者CPU中只有一个层在运行，因此只需要预先分配所有层的最大内存需求即可了。
    return net;
}
```

<div align='center'>
    <b>
        code 1.1 parse_network_cfg的函数定义，注意到为了显示简便，省略了许多重复类似的语句，读者可以查阅源代码或者根据上下文得知含义。
    </b>
</div>

整个`parse_network_cfg`的全景就如code 1.1所示，其中涉及到了一个临时的中间数据结构`size_params`，我们之前没有谈到，该结构定义如code 1.2所示。我们如果认真分析整个过程，发现其实这个函数分为两大阶段：解析，网络初始化。因此后续的章节也按照这两个部分分别剖析。

```c
typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;
```

<div align='center'>
    <b>
        code 1.2 该数据结构是临时数据结构，用以储存解析-定义网络过程中的数据。
    </b>
</div>

# cfg解析

与`cfg`解析有关的函数有很多，主要的是

1. `list* read_cfg(char* filename)`：用于解析主要的section列表等，见Fig 1.1。
2. `void parse_net_options(list *options, network *net)`：该函数用于解析`[net]`中的每个键值对。
3. `layer parse_xxxxx(list *options, size_params params)`：该类型函数用于解析每个特定的神经网络层的参数，比如`convolutional_layer parse_convolutional(list *options, size_params params)`，本文会以这个函数作为例子进行剖析。

## read_cfg

`read_cfg`输入`cfg`文件名，输出网络配置解析列表，如Fig 1.1。`read_cfg`的具体注释见code 2.1。

```c
list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r"); // 读文件
    if(file == 0) file_error(filename); // 判空
    char *line; // 每一行的指针，记得用完要释放内存
    int nu = 0; // 行数计数器
    list *options = make_list(); // 新建一个双向链表，这个链表是主链，用于储存section，该结构定义见[1]
    section *current = 0; // 每一个section的指针
    while((line=fgetl(file)) != 0){ // 读取每一行的数据，如果为0表示读完了
        ++ nu; // 行计数器加一
        strip(line); // 去除头尾的空格
        switch(line[0]){ 
         	// 如果每一行的第一个字符是'['，那么确定是标志了新的section的开始。
            case '[':
                // 因此需要对新的section进行内存分配
                current = malloc(sizeof(section));
                list_insert(options, current); // 将新的section插入options链表
                current->options = make_list(); // 新建链表，该链表是用于储存键值对的。
                current->type = line; // [xxxx] 表示了该层的类型，将其存入type，比如[convolutional]
                break;
            case '\0': // 忽略新行
            case '#': // 忽略注释
            case ';':
                // 这些无关的标志位，可以开始释放line内存了
                free(line);
                break;
            default:
                // 如果都不是，那么意味着是开始section的真实内容负载了，开始解析，并且将解析出的键值对放置到section中。
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file); // 关闭文件
    return options;
}
```

<div align='center'>
    <b>
        code 2.1 read_cfg函数注释，其是解析cfg文件的主要函数。
    </b>
</div>

该函数中有一个最为关键的调用函数，就是`int read_option(char *s, list *options)`，该函数用于解析每一个section内的键值对（注意到此时仍然只是字符串），具体定义见code 2.2的详细注释。通过`read_cfg()`函数，我们将配置解析到了链表中，之后就可以关闭`cfg`文件，直接读取链表进行网络初始化即可，这样不仅提高了效率，而且减少了因为读取网络过程中，意外修改配置文件导致错误出现的可能性。

```c
int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0; // 键值对中的数值
    for(i = 0; i < len; ++i){ 
        if(s[i] == '='){ // 以'='作为截断的标志，中间不能出现空格
            s[i] = '\0'; // 截断key-val，将'='替换成截断位，也即是'\0'。
            val = s+i+1; // 将val指向值的地址位置，因为已经做过了截断，因此需要+1
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s; // 显然之前的一段是键值，因为加了'\0'作为截断，因此现在key和val都是提取出来了
    option_insert(options, key, val); // 添加到section链表中,见code 2.3
    return 1;
}
```

<div align='center'>
    <b>
        code 2.2 read_option用以解析键值对，并且将其添加到指定的section链表中。
    </b>
</div>

```c
void option_insert(list *l, char *key, char *val)
{
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}
```

<div align='center'>
    <b>
        code 2.3 option_insert将键值对添加到section链表中。
    </b>
</div>



## parse_net_options

这个函数负责解析`[net]`部分的参数，这类型的参数和一般的section不同，其是全局的网络配置，因此独立成了一个函数。其没有太难理解的东西，基本上就是调用一系列字符串解析函数，这系列的函数会去读取之前解析到的参数链表，将其中section中的键值对解析出特定的数据类型（比如`int`,`float`），因此对应有很多类似的函数，比如：

1.  `int option_find_int(list *l, char *key, int def)`
2. `float option_find_float(list *l, char *key, float def)`
3. `char *option_find_str(list *l, char *key, char *def)`

```c
void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->notruth = option_find_int_quiet(options, "notruth",0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, "random", 0);

    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .0000001);
    }
    ... 
    // 省略了类似参数的解析过程
}
```

<div align='center'>
    <b>
        code 2.4 parse_net_options将`[net]`中的全局参数进行解析，并且放置到`net`数据结构中。
    </b>
</div>

## parse_xxx

该类型的函数用于解析某个特定的层，比如`convolutional`卷积层，`deconvolutional`转置卷积层，`activation`激活层等等，其中的`xxx`表明了层的种类，如果想要定制某个新的层，需要进行类似的注册（也即是需要书写自己的`parse_new_layer()`函数等）。本文以`parse_convolutional()`作为例子进行讲解。从code 2.5中可以发现，其中最主要的函数是`make_convolutional_layer()`，用以通过解析得到的卷积层参数去构造卷积层。

```c
convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    int groups = option_find_int_quiet(options, "groups", 1);
    // 解析和卷积层相关的参数，比如滤波器通道数，大小，步进，填充等
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic"); 
    ACTIVATION activation = get_activation(activation_s); // 得到指定的激活层，枚举类型

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c; // 数据集图片的基本参数，包括长宽，通道数
    batch = params.batch; // 批次大小
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam); // 构造卷积层
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}
```

<div align='center'>
    <b>
        code 2.5 parse_convolutional的函数注解，其中最主要的函数是make_convolutional_layer，用以通过解析得到的卷积层参数去构造卷积层。
    </b>
</div>

每个特定的神经网络层都有其特定的`make_xxx_layer()`函数，用以将解析得到的参数构造出特定的层。这个内容我们留到下一篇博文进行剖析，至此我们已经完全将`cfg`文件进行了解析。

# 该系列其他文章

1. [[darknet源码系列-1] darknet源码中的常见数据结构](https://fesian.blog.csdn.net/article/details/109779812)

# Reference

[1]. [[darknet源码系列-1] darknet源码中的常见数据结构](https://fesian.blog.csdn.net/article/details/109779812)

[2]. https://github.com/pjreddie/darknet/blob/4a03d405982aa1e1e911eac42b0ffce29cc8c8ef/src/parser.c#L742

[3]. https://github.com/pjreddie/darknet/blob/4a03d405982aa1e1e911eac42b0ffce29cc8c8ef/src/list.c#L40



[qrcode]: ./imgs/qrcode.jpg

[section_linked_list]: ./imgs/section_linked_list.png



