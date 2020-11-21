<div align='center'>
    [卷积算子加速] im2col优化
</div>

<div align='right'>
    FesianXu 20201121 at UESTC
</div>

# 前言

在深度学习模型中，卷积是非常重要的工具，然而卷积的计算复杂度很高，因此需要对此进行特定的优化，`im2col`与`winograd` [5]，`fourier` [4]是非常常见的优化方法，本文介绍基于`im2col`的优化方法。**如有谬误请联系指出，本文遵守[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$  联系方式：

**e-mail**: FesianXu@gmail.com

**QQ**: 973926198

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

----



# 耗时的卷积算子

卷积操作如Fig 1.1所示，通常涉及到了非常多的浮点运算，示例代码如code 1.1所示。

![conv2d][conv2d]

<div align='center'>
    <b>
        Fig 1.1 卷积操作的示意图。
    </b>
</div>



```c
for (int batch = 0; batch < B; ++batch) { // batch 批次
    for (int ox = 0; ox < Xout; ++ox) {   // 输出height大小
        for (int oy = 0; oy < Yout; ++oy) { // 输出width大小
            for (int oc = 0; oc < N; ++oc) { // 输出通道大小
                for (int kx = 0; kx < K; ++kx) { // kernel的height大小
                    for (int ky = 0; ky < K; ++ky) { // kernel的width大小
                        for (int ic = 0; ic < M; ++ic) { // 输入通道大小
                            const int iy = oy * SY + ky * DY - PY;
                            const int ix = ox * SX + kx * DX - PX;
                            if (0 <= ix && ix < X && 0 <= iy && iy <= Y) {
                                output[n][oy][ox][oc] += input[n][iy][ix][ic] * filter[oc][ky][kx][ic];
                            }
                        }
                    }
                }
            }
        }
    }
}
```

<div align='center'>
    <b>
        code 1.1 通常的直接卷积计算可以视为是7层嵌套循环的计算。
    </b>
</div>

code 1.1其中的`SY`和`SX`是width和height方向的`stride`，`DY`和`DX`是width和height方向的`dilate`大小（一般不设置dilate的话都为1）, `PY`和`PX`是width和height方向的`padding`大小。 显然，这个朴素的直接计算过程存在很多可以优化的地方，比如进行向量化，然而，直接卷积涉及到了很多超参数，比如卷积核大小，步进大小等，单一的优化方式不能对所有的超参数都适用，因此`cuDNN` [6] , `HexagonNN`，`MACE` [7] 等神经网络库中，对特定尺寸的卷积核（通常是最为常用的）进行了优化，比如$1 \times 5$，$5 \times 1$，$3 \times 3$，步进为2的卷积等等，如果涉及到其他更为普遍的卷积核，就只能采用原始的未优化的默认实现了。显然，这不是一种通用的做法。

后续提出过很多尝试优化直接卷积的方案，基于傅立叶变换的算法是其中一种[4]，称之为**快速卷积算法 (Fast Convolution Algorithm)**，其原理就是通过傅立叶变换将卷积计算转换成乘法计算，从而减少了大量的运算量。但是不幸的是，该算法提速受限于特定的卷积参数（大尺寸的卷积核，单位步进和dilation，足够大的输入尺寸和输入输出通道数等），对于比较小规模的计算就力不从心了，因此也是一种非通用的做法。

比较普遍的通用直接卷积优化方案是通过`im2col`和`GEMM`实现。`GEMM`全称`General Matrix Multiplication`通用矩阵乘法，是`BLAS`(Basic Linear Algebra subroutine，基础线性代数库)的一部分，其通过很多方式（比如矩阵分区块，多线程，向量化等等）实现了优化，具体我们以后的博文再讨论。总而言之，只要我们将卷积操作以某种方式转换成矩阵相乘的方式，就能从现成的`GEMM`中获得极大的提速裨益。后文我们谈谈如何通过`im2col`的方式将卷积操作转换成矩阵乘法。



# im2col

正如我们刚才code 1.1的代码所示，我们使用了一大堆嵌套的循环去实现卷积，这对于学习算法而言是很好的，因为这足够直接。但是实际中，计算速度并不够快。`im2col`（或者`im2row`，类似因此不独立讨论）将高阶张量的卷积转换成矩阵乘法。我们不妨先进行一个观察：卷积核与输入图片/特征图的某个局部区域（patch）之间进行点乘，并且通过滑动窗口的采样方式，去更新局部区域的信息。如果我们在内存中，把所有可能的局部区域拼成一个矩阵会怎么样呢？然后我们就可以通过矩阵乘法去进行卷积运算了，结合`GEMM`，可以提供200x以上的加速（取决于特定硬件）。这个就是`im2col`的基本思路，如Fig 2.1所示。

![im2col_operation][im2col_operation]

<div align='center'>
    <b>
        Fig 2.1 im2col将图片数据按照滑动块（patch）进行展开。
    </b>
</div>

举例而言，假设输入是$227 \times 227 \times 3$的张量，卷积核尺寸是$11 \times 11 \times 3$，`stride = 4`，`padding = 0`。那么我们先将每个卷积核展开成向量，维度为$K^2C = 11 \times 11 \times 3 = 363$，假设输出特征图通道为$D$，那么用`im2col`展开卷积核后的矩阵$\mathbf{A}$的尺寸为$D \times 363$。计算有多少个滑动窗口区块，我们有$((227-11)/4)+1=55$，我们一共有55个区块（在长宽方向上各有55个，整个图片上就是有$55^2=3025$个），那么`im2col`之后的输入特征图矩阵$\mathbf{B}$的尺寸为$363 \times 3025$，最终得出的结果就是矩阵乘法$\mathbf{A} \cdot \mathbf{B}$，得出的输出矩阵$\mathbf{C}$的尺寸为$D \times 3025$，通过逆运算`col2im`将$\mathbf{C}$塑性为$55 \times 55 \times D$即完成了最终的卷积输出结果。整个过程如Fig 2.2所示。假设$\mathbf{\tilde{A}}$为卷积核，$\mathbf{\tilde{B}}$为输入特征图，$\mathbf{C}$为卷积输出结果，那么有：
$$
\begin{aligned}
\mathbf{C} = \mathrm{col2im}(\mathrm{im2col}(\mathbf{\tilde{A}}) \cdot \mathrm{im2col}(\mathbf{\tilde{B}}))
\end{aligned}
\tag{2.1}
$$
![Convolution_With_Im2col][Convolution_With_Im2col]

<div align='center'>
    <b>
        Fig 2.2 im2col将卷积核和输入特征图进行展开后，用矩阵乘法取代了卷积操作。
    </b>
</div>

在`caffe` [1]和`darknet` [2]中都提供了相应的实现，我们主要观察下`darknet`的实现，如code 2.1所示，主要的是#19行，我们发现其实是通过一系列计算（通过超参数计算区块的地址位置），对输入张量进行访存地址的重定位。

```c
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}
```

<div align='center'>
    <b>
        code 2.1 darknet中的im2col操作代码。
    </b>
</div>





# Reference

[1]. https://github.com/BVLC/caffe/blob/master/src/caffe/layers/im2col_layer.cpp

[2]. https://github.com/pjreddie/darknet/blob/master/src/im2col.c

[3]. Dukhan M. The Indirect Convolution Algorithm[J]. arXiv preprint arXiv:1907.02129, 2019.

[4]. Vasilache N, Johnson J, Mathieu M, et al. Fast convolutional nets with fbfft: A GPU performance evaluation[J]. arXiv preprint arXiv:1412.7580, 2014.

[5]. Andrew Lavin and Scott Gray. Fast algorithms for convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4013–4021, 2016.  

[6]. Sharan Chetlur, Cliff Woolley, Philippe Vandermersch, Jonathan Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer. cudnn: Efficient primitives for deep learning. arXiv preprint arXiv:1410.0759, 2014.  

[7]. Xiaomi. MACE. https://github.com/XiaoMi/mace. [Online; accessed 8-April-2019].  







[qrcode]: ./imgs/qrcode.jpg
[conv2d]: ./imgs/conv2d.jpg
[im2col_operation]: ./imgs/im2col_operation.png
[Convolution_With_Im2col]: ./imgs/Convolution_With_Im2col.png

