<div align='center'>
    双目三维重建——层次化重建思考
</div>


<div align='right'>
    FesianXu 2020.7.22 at ANT FINANCIAL intern
</div>

# 前言

本文是笔者阅读[1]第10章内容的笔记，本文从宏观的角度阐述了双目三维重建的若干种层次化的方法，包括投影重建，仿射重建和相似性重建到最后的欧几里德重建等。本文作为介绍性质的文章，只提供了这些方法的思路，并没有太多的细节，细节将会由之后的博文继续展开。如有谬误，请联系作者指出，转载请注明出处。

$\nabla$ 联系方式：
**e-mail**:      [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**:             973926198
github:       https://github.com/FesianXu

----

**注**: 在阅读本文之前，强烈推荐读者先阅读[4]和[5]， 以了解[几何变换的层次——投影变换，仿射变换，度量变换和欧几里德变换的具体区别]和[conic圆锥线和quadric二次曲锥面的定义和应用]，在本文中将会基于这些前置知识进行讨论。同时作为成像的基础知识，相机内外参数的知识[6]也是必须了解的。

# 双目三维重建简介

作为三维重建，我们希望能够得到被重建物体的结构（也就是三维世界中的点的位置信息）。在双目三维重建中，正如其“双目”所言，我们假设有两个摄像机对某个物体进行观测，得到了多视角对同一个物体描述的图片。通常，在三维重建任务中，我们通过一系列的前置算法，可以得到这些多视角图片之间的对应点关系，如Fig 1所示，点$\mathbf{p}_l$和点$\mathbf{p}_r$都是三维世界客体点$\mathbf{P}$的投影，也就是说它们是一对 **对应点**（correspondence），通常我们用$\mathbf{x}_i \leftrightarrow \mathbf{x}_i^{\prime}$表示一对对应点。笔者之前的博文[2,3]中曾经根据对极线约束和图像矫正对对应点的影响进行过介绍，读者有兴趣可自行查阅。

![stereovision][stereovision]

<div align='center'>
    <b>
        Fig 1. 双目多视角图片之间的对应点，其都是对三维世界中客体点P的投影。
    </b>
</div>

通常来说，在三维重建任务中，我们假设对应点对应的三维点$\mathbf{P}$的位置是未知的，需要我们求出，并且我们也不知道相机的方向，位置和内参数校准矩阵等（也就是外参数和内参数都未知）。整个重建任务就是需要去计算相机矩阵$\mathrm{P}$和$\mathrm{P}^{\prime}$，使得对于三维点$\mathbf{X}_i$有：
$$
\mathbf{x}_i = \mathrm{P} \mathbf{X}_i,   \mathbf{x}_i^{\prime} = \mathrm{P}^{\prime} \mathbf{X}_i \\ \forall i
\tag{1}
$$
其中的$i$表示的是对应点的编号。当给定的对应点太少时，这个任务是显然不能完成的，但是如果给定了足够多的对应点，那么我们就有足够的约束去唯一确定出一个基础矩阵（Fundamental Matrix）[2] 出来，如式子(2)所示（当然，求解这个基础矩阵也并没有那么简单，这个并不在本文中进行阐述）。此时，整个三维重建将会存在一种称之为 **投影歧义** （projective ambiguity）的现象，我们将会在下文进行介绍。作为无校准的相机来说，在不引入任何先验知识进行约束的情况下，这是双目三维重建能得到的最好结果。在添加了其他对场景的先验知识后（比如平行线约束，垂直线约束，摄像机内参数相同假设等），投影歧义可以被减少到仿射歧义甚至是相似歧义的程度。
$$
\mathbf{x}^{\mathrm{T}} \mathcal{F} \mathbf{x}^{\prime} = 0
\tag{2}
$$
总的来说，重建过程的三部曲可以分为：

1. 根据对应点计算得到基础矩阵。
2. 根据基础矩阵计算得到相机矩阵$\mathrm{P},\mathrm{P}^{\prime}$。
3. 对于每对对应点$\mathbf{x}_i \leftrightarrow \mathbf{x}_i^{\prime}$来说，通过三角测量法，计算其在空间上的三维点坐标位置。

注意到本文对这些重建方法的介绍只是一个概念性的方法，读者需要明晰的是，不要尝试单纯地实现本文介绍的方法去实现重建，对于真实场景的图片而言，重建过程存在着各种噪声（比如对应点对可能不准确，存在噪声等，需要鲁棒估计），这些具体的方法我们将会在之后的博文介绍。

这里需要单独拎出来提一下的是 **三角测量法**（Triangulation），如Fig 2所示，在计算得到了相机矩阵$\mathrm{P},\mathrm{P}^{\prime}$之后，我们知道$\mathbf{x},\mathbf{x}^{\prime}$满足对极约束$\mathbf{x}^{\mathrm{T}} \mathcal{F} \mathbf{x}^{\prime} = 0$，换句话说，我们知道$\mathbf{x}$在对极线$\mathcal{F}\mathbf{x}^{\prime}$上，反过来这意味着从图像点$\mathbf{x},\mathbf{x}^{\prime}$反向投影得到的射线共面，因此它们的反向射线将会交于一点$\mathbf{X}$。通过三角测量的方法，我们可以测量除了基线上的任意一个三维空间点，原因在于基线上的点的反向射线是共线的，因此不能唯一确定其相交点。

![triangulation][triangulation]

<div align='center'>
    <b>
        Fig 2. 通过三角测量法去确定三维空间中的点。
    </b>
</div>


# 重建歧义性

>  **重建过程存在或多或少的歧义性，为了解决歧义性，我们需要引入额外的信息。**

单纯地从对应点去进行场景，物体的三维重建必然是有一定的歧义性的，只是说引入了对场景一定的先验知识后，这种歧义性会得到缓解。举个例子，光从成对的对应点对（甚至可能是多个视角的点对），都不能计算出场景的绝对位置（指的是地球上的经纬度）和绝对朝向，这点很容易理解，例如Fig 3所示，即便是给定了相机内参数等，也不可能决定b和c这两个走廊的具体的东西走向，还是南北走向，亦或是其经纬度，这些涉及到地理位置的绝对信息无法光从相机重建得到。

![corridor][corridor]

<div align='center'>
    <b>
        Fig 3. 不引入其他任何知识，只从相机得到的照片，无法判断场景的绝对地理信息。
    </b>
</div>

一般来说，基于相机的重建，我们称它最好的情况下，对于世界坐标系来说，都只能是欧几里德变换（包括这旋转和偏移）。当然，如果我们的相机没有标定，也就是内参数未知，就Fig 3的走廊为例子，我们无法确定走廊的宽度和长度，它可能是3米，也可能只是一个玩具走廊，只有30厘米，这些都有可能。在不引入对场景尺度的任何先验而且相机没有标定的情况下，我们称基于相片的场景重建只能最好到相似性变换（也就是是存在旋转，偏移和尺度缩放）。

如果用数学形式去解释这个现象，用$\mathbf{X}_i$表示一系列场景中的三维点，$\mathrm{P},\mathrm{P}^{\prime}$表示一对相机，其将三维点投影到$\mathbf{x}_i, \mathbf{x}_i^{\prime}$。假设我们有相似性变换$\mathrm{H}_S$
$$
\mathrm{H}_S = 
\left[\begin{matrix}
\mathrm{R} & \mathbf{t} \\
\mathbf{0}^{\mathrm{T}} & \lambda
\end{matrix}\right]
\tag{3}
$$
其中的$\mathrm{R}$是旋转矩阵，$\mathbf{t}$是偏移，$\lambda^{-1}$是尺度放缩。假设我们对三维点进行相似性变换，那么我们用$\mathrm{H}_S\mathbf{X}_i$取代$\mathbf{X}_i$，并且用$\mathrm{P}\mathrm{H}_S^{-1}$和$\mathrm{P}^{\prime}\mathrm{H}_S^{-1}$取代原来的相机参数$\mathrm{P},\mathrm{P}^{\prime}$。我们发现，因为有$\mathrm{P}\mathbf{X}_i = (\mathrm{P}\mathrm{H}_S^{-1})(\mathrm{H}_S \mathbf{X}_i)$，因此在图像上的投影点位置是不会改变的。 **通常来说，在不引入其他先验的情况下，我们会发现，在重建过程中，算法只会保证图像上的投影点的位置是投影正确的，没法保证三维空间的其他信息了。**



## 相似歧义性

更进一步，我们对相机参数进行分解，有$\mathrm{P} = \mathrm{K}[\mathrm{R}_P | \mathbf{t}_P]$，那么经过相似性变换之后，有：
$$
\mathrm{P}\mathrm{H}_S^{-1} = \mathrm{K} [\mathrm{R}_P\mathrm{R}^{-1} | \mathbf{t}^{\prime}]
\tag{4}
$$
一般情况下，我们不是很关心偏移$\mathbf{t}^{\prime}$。我们会发现，相似性变换并不会改变其相机内参数，$\mathrm{K}$是不会改变的，也就是说，即便是对于校准后的相机，最好的重建结果也会存在相似性歧义，我们称之为 **相似性重建** (Similarity reconstruction, metric reconstruction)。 如Fig 4的图a所示，这就是相似性歧义的示意图，我们无法确定场景的绝对大小。

![proj_ambiguity][proj_ambiguity]

<div align='center'>
    <b>
        Fig 4. 重建的相似性歧义性和投影歧义性
    </b>
</div>
如果两个相机的内参数中的焦距都是已知的，那么这个重建最好能到 **相似性重建** (similarity reconstruction) 的程度，其重建过程引入的是 **相似歧义性** (similarity ambiguity)。也就是说，重建出来的场景只会在尺度大小上和真实场景的有差别。

## 投影歧义性

如果我们对内参数一无所知，也不知道相机之间的相对位置关系，那么整个场景的重建将会陷入 **投影歧义性** (projective ambiguity)，如Fig 4的图b所示。同样我们可以假设一个不可逆的矩阵 $\mathrm{H}_P \in \mathbb{R}^{4 \times 4}$ 作为投影矩阵，用我们在之前介绍的方法，我们会发现将投影矩阵同时作用在三维点和相机上时，不影响图像上的投影点位置。因此实际的重建三维点将会是投影歧义的。这个称之为 **投影重建** (Projective reconstruction)，投影重建和相似重建的不同之处在于，相似重建因为相机内参数已经知道，因此相机焦点位置是确定的，而投影重建因为没有校准机参数，因此焦点位置可能会变化，Fig 4示意图就明确了这一点。

## 仿射歧义性

如果两个相机只是存在位置偏移上的变化，而内参数完全相同（可以视为是同一个相机的在不同位置拍摄的相片），那么重建过程是最好能达到 **仿射重建** （affine reconstruction）的程度，相对应的，这个重建过程会引入 **仿射歧义性** （affine ambiguity）。举个例子，假如我们知道不同相机之间的焦距都是一样的（焦距是内参数的一部分），因此整个场景可能存在旋转，偏移和尺度放缩或者切变（Shear）[7]。但是不会存在如同Fig 4所示的投影歧义的那么严重的歧义性。

## 度量重建和欧几里德重建

一般来说，我们同样可以用 **度量重建**  (metric reconstruction) 去表述相似性重建，因为相似性重建过程中的某些量 比如 线与线的角度，线段比例等度量在重建场景和真实场景中应该是一致的。另外，当我们说到 **欧几里德重建** (Euclidean reconstruction) 的时候，一般我们也是把它当做度量重建或者是相似性重建的别称，因为没有其他额外的知识，比如场景的世界坐标上的朝向，景深甚至是世界坐标经纬度等，我们是无法真正地实现欧几里德重建的，而这些额外知识已经脱离了基于相机的重建了。



# 理想点，无限远平面和IAC

我们曾经在博文[4,7]中提到过理想点（Ideal point），无限远平面（The plane at infinity）和IAC （Image of Absolute Conic）[8]。这些几何元素是用于描述投影空间中的一些性质，包括变换前后的不变性。这里进行知识回顾并且进一步的介绍。

简单来说，在投影变换过程中，平行线的平行性质将得不到保留，因此可能存在 **透视** (perspective) 现象，具体体现出来如同Fig 5所示。这里的平行线交汇的无限远处的消失点我们称之为 **理想点**， 由所有理想点所组成的平面称之为 **无限远平面** ，我们用符号$\Pi_{\infty}$表示。

![parallel_road][parallel_road]

<div align='center'>
    <b>
        Fig 5. 本来应该是平行的马路，在相机成像的时候，则变成了“不平行”，汇聚于无限远处的消失点(vanishing point)。
    </b>
</div>

在谈到**绝对圆锥曲线** (Absolute Conic, AC) 和 **绝对圆锥曲线的投影** (Image of Absolute Conic, IAC) 之前，我们可以举个日常生活中的例子，我们都知道月亮离我们很远，可以视为是在无限远处的一个圆形（圆锥曲线的一种特例），想象你开车行驶在一条笔直的道路上，右侧空中高悬着圆月，你会发现不管你怎么疾驰，月亮仿佛都跟随着你，而且位置不变，大小也不变。 

这个浪漫的“月亮跟着我”的例子，正是欧几里德变换对于处在无限远处平面上的绝对圆锥线的影响。 **绝对圆锥曲线AC** 是指的处在无限远平面上的圆锥曲线，而 **绝对圆锥曲线的投影IAC** 指的是绝对圆锥曲线在成像平面上的投影，如Fig 6所示，通常我们用$\Omega$表示IAC，用$\omega$表示AC，其中$C$表示的是焦点。当距离足够远，可以视为无限远时，我们直观上会发现，只要成像平面是欧几里德变换的，那么AC将不会改变，原因很简单，因为任何的旋转，平移对于无限远而言，都太过渺小，因此可以视为没有任何欧式变换能够影响这些性质。因此，我们知道 **欧几里德变换不影响IAC的形状大小** 。

![IAC_AC][IAC_AC]

<div align='center'>
    <b>
        Fig 6. 在无限远平面上的AC和在成像平面上的IAC。
    </b>
</div>

我们为什么要在这里讨论这些概念呢？原因在于，不管是几何变换也好，三维重建也好，都会涉及到“变”与“不变”的量，当需要对某些变化的量约束到不变的量时，我们需要添加条件，比如固定住无限远平面的位置，固定AC的形状大小等，从而可以减少歧义性，实现更精确的三维重建。

# 层次化三维重建

> 通过在投影重建的基础上，添加一系列的信息，我们可以分别得到场景的仿射重建，相似性重建。

我们考虑双目三维重建的层次化过程。首先作为基础的，假设我们的相机是没有校准的，我们需要基于对应的两张图像之间的对应点，求得场景的投影重建后，再添加若干信息，可以分别得到场景的仿射重建，相似性重建。

对于未校准的相机而言，假设给定了两张图片之间的对应点$\mathbf{x}_i \leftrightarrow \mathbf{x}_i^{\prime}$，通过式子(2)我们可以计算出基础矩阵$\mathcal{F}$。通过之前的分析，我们知道存在投影矩阵 $\mathrm{H}_P$ 可以使得场景重建存在投影歧义性，如Fig 5所示。

需要注意的是，我们这里提到的对应点对不能在两台相机的焦点连线上（也就是基线），这个我们之前也提到过。

![proj_rec][proj_rec]

<div align='center'>
    <b>
        Fig 5. 投影重建带来的投影歧义性，对于单次重建来说，投影重建的每个可能的场景都是投影一致性的。
    </b>
</div>


## 仿射重建

从投影重建到仿射重建，回忆下我们在[4]中曾经讨论的：

> 仿射变换不会影响无限远平面的位置

也就是说，为了消除投影歧义性，我们需要添加约束 **固定** 无限远平面的位置。用数学形式化地表达我们整个过程，假设我们现在已经有了一个对场景的投影重建结果，包括一个三元组$(\mathrm{P}, \mathrm{P}^{\prime}, \{\mathbf{X}_i\})$，其中$\mathrm{P}, \mathrm{P}^{\prime}$是相机矩阵，$\mathbf{X}_i$为场景坐标点集。进一步假设我们确定平面$\pi$作为真正的无限远平面，那么这个平面将会用一个齐次坐标下的向量表示，有$\pi \in \mathbb{R}^4$。我们需要把这个平面挪到$(0,0,0,1)^{\mathrm{T}}$去，因此需要找到一个投影矩阵，将$\pi$映射到$(0,0,0,1)^{\mathrm{T}}$，也就是有：$\mathrm{H}^{-1}\pi = (0,0,0,1)^{\mathrm{T}}$。有：
$$
\mathrm{H} = 
\left[
\begin{matrix}
\mathrm{I} | \mathbf{0} \\
\pi^{\mathrm{T}}
\end{matrix}
\right]
\tag{5}
$$

此时的$\mathrm{H}$可以作用在所有的三维重建后的点集和两个相机矩阵上，注意到，公式(5)将会在$\pi^{\mathrm{T}} = \mathbf{0}$的情况下失效。 通过求得这个投影矩阵，我们得到了 **仿射重建** 。 

然而，正如我们说的，除非我们添加一些额外的信息，否则无限远平面 $\pi$ 是不能确定下来的，我们接下来给出几个例子，说明什么类型的信息是足够确定这个平面的。

### 偏移运动

**偏移运动** (Translational motion) 指的是我们已知拍摄出来的两张照片来自于同一个相机，只不过是这两张照片来自于不同的视角下拍摄的，而且这个视角变化只是由于相机矩阵的$\mathbf{t}^{\mathrm{T}}$ 也即是偏移造成的。简单来说，就是如Fig 6 所示，黄色平面就是只是存在平移的相机，而灰色平面则同时存在平移和旋转。

![rectification_coplanar][rectification_coplanar]

<div align='center'>
    <b>
        Fig 6. 绿色物体为成像客体，黄色平面是只存在偏移变化的相机，而灰色平面则存在着偏移和旋转。
    </b>
</div>

回忆下我们刚才提到的“月亮跟着我”的例子，对于无限远处的物体来说，只是存在平移变化，是不会影响该物体的位置的（因为平移的距离对比他们之间的距离来说，实在是微不足道）。因此，相机的平移不会影响两张照片中的处在无限远处平面的点的位置，让我们用$\mathrm{X}$表示这个处在无限远处平面的点。如Fig 7所示，两幅图像的对极点位置在图像中的相机坐标系是一致的，就是因为相机的平移大小对于走廊的深度来说是微不足道的，因此可以视为是无限远平面，因此对极点不变。我们可以知道，投影形成这些不变点的三维空间点，是处在无限远处的，通过匹配点和图片的像素位置（通过两个条件，既是匹配点又是图片上不变的点去筛选），我们可以寻找得到三组以上的这种无限远处的点，并且通过最小二乘法或者解析法求出这个无限远处平面$\pi$。

![corridor][corridor]

<div align='center'>
    <b>
        Fig 7. 纯平移运动不会改变远处（可以视为无限远处）的对极点位置。
    </b>
</div>

虽然这样原理上是可行的，但是实际上计算过程中存在较大的数值问题，事实上我们这样计算出的基础矩阵是一个反对称(skew-symmetric)矩阵，这意味着我们还需要对基础矩阵进行约束。 事实上，在实际中最常用的约束还是下面谈到的平行线约束。



### 平行线约束

我们知道仿射变换不改变平行线的平行性，而投影变换则可能会改变，那么根据这个知识，我们可以从场景中寻找三组以上的本应该在实际三维空间中平行，却因为投影的透视现象在图像中相交的平行线，将它们的交点视为是无限远处的点，因此可以确定出无限远处平面$\pi$。

![parallel_lines][parallel_lines]

<div align='center'>
    <b>
        Fig 8. 寻找场景中三组本应该在实际三维空间中平行，却在图像中因为存在透视而相交的平行线，它们的交点作为无限远处的点，可以确定出无限远处平面。
    </b>
</div>

这个过程听起来挺理想，然而因为存在噪声，多组不同的平行线在图片中不一定会交于一个点，这个时候我们需要鲁棒估计进行数值问题上的求解，我们之后再讨论。

### 线段比例

我们知道仿射变换是不会改变变换前后线段之间的比例，见[4]中的具体描述。这为我们计算无限远处的消失点(vanishing point)又提供了一种思路：我们可以通过引入真实三维世界中某个直线上的线段比例长度去确定消失点的位置，如Fig 9和Fig 10所示。具体的计算过程我们之后的博文进行介绍。

![equal_length][equal_length]

<div align='center'>
    <b>
        Fig 9. 根据实际世界中的线段比例去计算消失点位置。
    </b>
</div>

![affine_rectification][affine_rectification]

<div align='center'>
    <b>
        Fig 10. 通过引入线段比例的先验知识，从而将投影歧义性消除。
    </b>
</div>



### 无限单应性矩阵

一旦无限远处平面被确定下来，我们就确定了仿射重建，随后我们就有一个被称之为“无限单应性矩阵”（The infinite homography）的特殊矩阵。这个矩阵负责把两个相机的图像中的无限远处的消失点进行映射，是一个2D的单应性矩阵。假设相机$\mathrm{P}$对应的拍摄的图片的消失点$\mathbf{x}$对应在无限远处平面的客体点是$\mathbf{X}$，然后假设该客体点在另一个相机$\mathrm{P}^{\prime}$对应拍摄的图片的投影为$\mathbf{x}^{\prime}$。那么这个无限单应性矩阵存在有以下性质：
$$
\mathbf{x}^{\prime} = \mathrm{H}_{\infty} \mathbf{x}
\tag{6}
$$
假设我们知道两个相机矩阵
$$
\begin{align}
\mathrm{P} &= [\mathrm{M} | \mathbf{m}] \\
\mathrm{P}^{\prime} &= [\mathrm{M}^{\prime} | \mathbf{m}^{\prime}] \\
\end{align}
\tag{7}
$$
他们是符合仿射重建的相机矩阵，那么我们有无限单应性矩阵$\mathrm{H}_{\infty} = \mathrm{M}^{\prime} \mathrm{M}^{-1}$。这个并不难证明得到，留个读者自证。结合其(6)，我们有：
$$
\mathbf{x}^{\prime} = \mathrm{M}^{\prime} \mathrm{M}^{-1} \mathbf{x}
\tag{8}
$$
也即是说，通过寻找两个图片的对应的消失点对，我们可以计算得到无限单应性矩阵$\mathrm{H}_{\infty}$。我们接下来可以对相机矩阵中的某一个进行标准化，因此(7)变化为：
$$
\begin{align}
\mathrm{P} &= [\mathrm{I} | \mathbf{0}] \\
\mathrm{P}^{\prime} &= [\mathrm{M}^{\prime} | \mathbf{e}^{\prime}] \\
\end{align}
\tag{8}
$$
此时$\mathrm{H}_{\infty} = \mathrm{H}^{\prime}$，也就是说，通过计算得到无限单应性矩阵，我们可以恢复从仿射重建的相机矩阵。



### 其中一个相机是仿射相机

假设我们确定两个相机之中的其中一个是仿射相机[10]，当然，仿射相机只是对投影相机的一种近似，其近似的基本假设就是被拍摄物体的表面纹理深度对于拍摄的距离来说可以忽略不计，也就是[11]中所说的弱深度纹理，low-relief。我们知道仿射相机进行的是仿射变换，因此不会移动无限远处平面的位置，而且我们知道仿射相机的主平面(principle plane)就是无限远处平面，并且它就可以用相机矩阵的第三行向量表示。那么假设最简单的情况，我们把这个仿射相机的相机矩阵$\mathrm{P}$标准化为$\mathrm{P} = [\mathrm{I} | \mathbf{0}]$，第三行为$(0,0,1,0)^{\mathrm{T}}$，因此要把这个无限远处平面固定到$(0,0,0,1)^{\mathrm{T}}$，只需要：

1. 同时简单地交换两个相机矩阵的最后两列；
2. 同时交换每个三维客体点$\mathbf{X}_i$的最后两个坐标即可。



## 相似性重建/度量重建









# Reference

[1]. Hartley R, Zisserman A. Multiple view geometry in computer vision[M]. Cambridge university press, 2003. Chapter 10

[2]. https://blog.csdn.net/LoseInVain/article/details/102665911

[3]. https://blog.csdn.net/LoseInVain/article/details/102775734

[4]. https://blog.csdn.net/LoseInVain/article/details/104533575

[5]. https://blog.csdn.net/LoseInVain/article/details/104515839

[6]. https://blog.csdn.net/LoseInVain/article/details/102632940

[7]. https://blog.csdn.net/LoseInVain/article/details/102756630

[8]. http://www.cs.unc.edu/~marc/tutorial/node87.html

[9]. https://blog.csdn.net/richardzjut/article/details/10473051

[10]. https://blog.csdn.net/LoseInVain/article/details/102883243

[11]. https://blog.csdn.net/LoseInVain/article/details/102739778

[12]. 



[stereovision]: ./imgs/stereovision.jpg
[triangulation]: ./imgs/triangulation.jpg

[corridor]: ./imgs/corridor.jpg
[proj_ambiguity]: ./imgs/proj_ambiguity.jpg

[proj_rec]: ./imgs/proj_rec.jpg

[parallel_road]: ./imgs/parallel_road.jpg
[moon_follow_me]: ./imgs/moon_follow_me.jpg
[IAC_AC]: ./imgs/IAC_AC.png

[rectification_coplanar]: ./imgs/rectification_coplanar.jpg
[parallel_lines]: ./imgs/parallel_lines.jpg

[affine_rectification]: ./imgs/affine_rectification.jpg
[equal_length]: ./imgs/equal_length.jpg









