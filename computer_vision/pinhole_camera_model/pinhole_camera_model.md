<div align='center'>
    相机的针孔模型及其内参数，外参数的理解
</div>

<div align='right'>
    2019.10.18 FesianXu
</div>

[TOC]



----



# 前言

在相机校准中，我们经常会提到**内参数**，**外参数**，这些参数决定了一个相机的成像的效果，是后续一系列计算机视觉问题的基础中的基础，然而因为较为底层的原因，现在却比较少人关心它，笔者最近在学习底层的计算机视觉理论，感觉有所裨益，希望能在此进行笔记，作为备忘，如果能对读者有所帮助，则是更好不过了。**如有谬误，请联系指正。转载请注明出处。**

 $\nabla$ 联系方式：
**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**: 973926198
github: https://github.com/FesianXu 

# 相机的针孔模型



为了简单地解释一个相机为什么能够成像，我们通常会引入相机的针孔模型(pinhole model)。如Fig 1.1所示，在针孔模型中，相机呈现的都是倒像，这点其实很好理解，因为光线都是直线传播的，因此实体(entity)在相机中的像必然是倒过来的。这里，为了让光只能通过一束（因为只有一束才能确保实体到像的一对一关系，然而实际中不可能做到理想的情况。），我们通常假设这个针孔是无限小的，然而因为无限小的针孔不能透光，为了使得成像有着充足的光线，针孔又必须足够的大，这俩要求显然是个矛盾，因此一般我们需要在针孔处安置透镜，而透镜的引入，包括透镜的厚度，透光度等等不理想的因素，使得成像分析变得复杂起来，但是我们这里还是按照针孔模型的结构去理解，以简化分析。（透镜这里的作用是为了更好的聚集光线。）

![image_pin][image_pin]

<div align='center'>
    <b>
    Fig 1.1 相机的成像。
    </b>
</div>

我们需要知道的是，理想的相机模型是不需要透镜的，因为没有透镜的引入，因此成像没有因透镜产生的几何变形和模糊。在这个模型中，我们其实是在描述从**实体的3D坐标到成像平面的2D坐标之间的映射关系**。如Fig 1.2所示，现实中的实体点$X$坐标为$(x,y,z)$，其光线通过焦点$C$聚集在成像平面上，但是这个像是倒像，不方便分析，为了方便，我们通常假设和倒像的成像平面对称的一端也有个成像平面，这个平面成像是正面的，其特性和真实的成像平面一模一样，除了呈现的是正像之外，因此我们正式地将其称为**成像平面(image plane)**。其真实实体的映射点坐标为$x = (u,v)$。

![imageplane][imageplane]

<div align='center'>
    <b>
    Fig 1.2 相机的针孔模型。
    </b>
</div>

这里，为了方便接下来的讨论，我们将定义和解释以下术语：

1. 焦点(camera center, optic center)： 所有光线都会聚集的点，比如Fig 1.2中的点C。
2. 成像平面(image plane)：相机的CCD平面，图像在这个平面上形成，注意后续讨论的image plane一般会是指的呈现正像的那个平面。
3. 光轴(principal axis)：经过焦点，并且与成像平面垂直的线。
4. 光轴面(principal plane)： 包含着焦点，并且和成像平面平行的面。
5. 焦距(focal length)： 通常表示为$f$，指的是焦点到成像平面的距离。
6. 帧(frame): 这里提到的帧和我们通常视频处理里面的帧不太一样，这里提到的帧指的是一种度量，用于衡量一个特定的坐标系系统。
7. 世界坐标系(world frame, world coordinate system)：一个固定的坐标系，用于表示现实实体的坐标（比如点线面等等）。
8. 相机坐标系(camera frame, camera coordinate system)：将相机的焦点作为其原点，光轴作为其Z轴的坐标系。
9. 外参数(extrinsic parameters): 外参数描述了如何将实体的3D点（以世界坐标系描述）映射到以相机坐标系描述的3D点上，显然，这个是坐标系的平移和旋转过程。
10. 内参数(intrinsic parameters)：内参数描述了如何将已经是用相机坐标系描述的3D点投射到成像平面上。
11. 视网膜平面（image, retina plane）：图像在这个平面上成像，注意到，图像平面用相机坐标系度量，其单位是mm，毫米，属于物理单位。
12. 图像帧(image frame)：这个帧和我们通常理解的帧一致，其用像素(pixel)去描述图像平面，而不是mm了，属于逻辑单位。（比如一个像素对应多少mm的距离是不同的。）
13. 光心(principal point)： 指的是光轴和成像平面的交点。



这里我们给出一个图取参数上面谈到的一些概念，注意到的是其中的virtual image plane其实是本文中谈到的成像平面。[1]

![camera_calibration_focal_point][camera_calibration_focal_point]

<div align='center'>
    <b>
    Fig 1.3 相机针孔成像过程及其术语解析。
    </b>
</div>



# 坐标系的改变

为了将一个在世界坐标系中表示的点，以相机坐标系的形式进行表达，我们需要进行坐标系的平移和旋转变化。比如Fig 2.1所示，我们需要通过平移和旋转将$(X_C, Y_W, Z_W)$转换到$(X_C,Y_C,Z_C)$，容易知道，在不同坐标系中，对于同一个实体点$P$来说，其表达形式都不同。我们接下来考虑怎么进行这个坐标系转换。

![TandR][TandR]

<div align='center'>
    <b>
    Fig 2.1 世界坐标系 到 相机坐标系的转换过程。
    </b>
</div>

通常来说，这个过程可以简单表示为，平移向量和旋转矩阵的操作，如：
$$
\hat{\mathbf{X}}_C = \mathbf{R}(\mathbf{X}_W-C) 
\tag{2.1}
$$
其中，$\mathbf{X}_W = (X_W,Y_W,Z_W)$是世界坐标系坐标，$\hat{\mathbf{X}}_C = (X_C, Y_C, Z_C)$是相机坐标系坐标，$\mathbf{R} \in \mathbb{R}^{4 \times 4}$是旋转矩阵（注意这里是齐次坐标系的表达方法），$C = (X_0, Y_0, Z_0)$是用世界坐标系描述的焦点。

我们考虑到在中心投影中，如Fig 2.2中，我们根据相似三角形的规律有，其中以相机坐标系描述的点$\hat{\mathbf{X}}_C$投影到成像平面上有$\mathbf{X}_C = (x_c, y_c)^{\mathrm{T}}$
$$
\begin{aligned}
x_c &= \frac{f X_c}{Z_c} \\
y_c &= \frac{f Y_c}{Y_c}
\end{aligned}
\tag{2.2}
$$
![central_proj][central_proj]

<div align='center'>
    <b>
    Fig 2.2 中心投影，符合相似三角形的比例关系。
    </b>
</div>

用矩阵形式表达就是:
$$
\mathbf{x}_C =  
\left[
 \begin{matrix}
   f & 0 & 0 \\
   0 & f & 0 \\
   0 & 0 & 1 
  \end{matrix} 
\right]
\hat{\mathbf{X}}_{C}
\tag{2.3}
$$
可知此时有: $\mathbf{x}_c = (f X_C, f Y_C, Z_C)^{\mathrm{T}}$，其是用齐次坐标系表达的，等价于非齐次形式的$\mathbf{x}_c = (fX_C/Z_c, f Y_C/Z_c)^{\mathrm{T}}$。

考虑到公式(2.1)和(2.3)，我们能够把一个3D点映射成2D点：
$$
\begin{aligned}
\mathbf{x}_C &=
\left[
 \begin{matrix}
   f & 0 & 0 \\
   0 & f & 0 \\
   0 & 0 & 1 
  \end{matrix} 
\right]
\hat{\mathbf{X}}_C = 
\left[
 \begin{matrix}
   f & 0 & 0 \\
   0 & f & 0 \\
   0 & 0 & 1 
  \end{matrix} 
\right]
\mathbf{R} [\mathbf{I} | -\mathbf{C}]
\left(
\begin{matrix}
{\mathbf{X}}_W \\
1
\end{matrix}
\right) \\
&= 
\left[
 \begin{matrix}
   f & 0 & 0 \\
   0 & f & 0 \\
   0 & 0 & 1 
  \end{matrix} 
\right]
\mathbf{R} [\mathbf{I} | -\mathbf{C}] \hat{\mathbf{X}}_W
\end{aligned}
\tag{2.4}
$$
其中$\hat{\mathbf{X}}_W$是${\mathbf{X}}_W$的齐次表达。

这里的$\mathbf{R} [\mathbf{I} | -\mathbf{C}]$称之为外参数(extrinsic parameters)，这些参数描述了如何将世界坐标系的实体3D点转换到以相机坐标系描述的3D点。

那么总结来说，其实对于坐标系的平移和旋转，我们可以用下面的几副图来表示：

![p1][p1]

<div align='center'>
    首先，我们有两个不同的坐标系，左边的世界坐标系(X,Y,Z)和右边的相机坐标系(u,v,w)
</div>

![p2][p2]

<div align='center'>
    然后，我们通过将两者的原点O和C以平移的方式挪到一起，我们通过平移矩阵T去实现。
</div>

![p3][p3]

![p4][p4]

<div align='center'>
    最后，利用旋转矩阵，将其进行坐标轴的旋转和对齐即可。
</div>



-----

# 考虑更多因素

注意到通过上面的讨论，我们转换得到的$\mathbf{x}_c$的单位仍然是物理单位mm，如果我们需要用像素去度量（实际上也是用像素度量的），我们仍需要进行其他处理。（内参数的协助） $\mathbf{x}_c$在这里是以光心作为其原点的，而传统的表示中，我们一般以左上角的作为原点进行描述。因为一些制造工艺上的不精确性，我们的成像传感器CCD通常不是完美的矩形网格，可能会有变形。比如偏斜(skewness)用于描述CCD单元的变形程度，见Fig 2.3。

![skewness][skewness]

<div align='center'>
    <b>
    Fig 2.3 CCD单元的偏斜。
    </b>
</div>

那么经过矫正，其正确的坐标应该是:
$$
\begin{aligned}
x &= x^{\prime}-y^{\prime}\cot(\theta) \\
y &= \dfrac{y^{\prime}}{\sin(\theta)}
\end{aligned}
\tag{2.5}
$$
考虑到CCD的偏斜，和物理单位到像素单位的转变，我们有以下公式：
$$
\begin{aligned}
\mathbf{x} &= 
\left[
\begin{matrix}
m_x & 0 & x_0 \\
0 & m_y & y_0 \\
0 & 0 & 1
\end{matrix}
\right]
\left[
\begin{matrix}
1 & -\cot(\theta) & 0 \\
0 & \dfrac{1}{\sin(\theta)} & 0 \\
0 & 0 & 1
\end{matrix}
\right] \mathbf{x}_C \\
&= 
\left[
\begin{matrix}
m_x & 0 & x_0 \\
0 & m_y & y_0 \\
0 & 0 & 1
\end{matrix}
\right]
\left[
\begin{matrix}
1 & -\cot(\theta) & 0 \\
0 & \dfrac{1}{\sin(\theta)} & 0 \\
0 & 0 & 1
\end{matrix}
\right]
\left[
\begin{matrix}
f & 0 & 0 \\
0 & f & 0 \\
0 & 0 & 1
\end{matrix}
\right]
\mathbf{R}[\mathbf{I}|-\mathbf{C}] \hat{\mathbf{X}}_W \\
&= 
\left[
\begin{matrix}
m_x f & -m_x f \cot(\theta) & x_0 \\
0 & \dfrac{m_y f}{\sin(\theta)} & y_0 \\
0 & 0 & 1
\end{matrix}
\right] 
\mathbf{R}[\mathbf{I}|-\mathbf{C}] \hat{\mathbf{X}}_W \\
&= 
\left[
\begin{matrix}
\alpha_x & s & x_0 \\
0 & \alpha_y & y_0 \\
0 & 0 & 1
\end{matrix}
\right] 
\mathbf{R}[\mathbf{I}|-\mathbf{C}] \hat{\mathbf{X}}_W \\
&= 
\mathbf{K} \mathbf{R}[\mathbf{I}|-\mathbf{C}] \hat{\mathbf{X}}_W \\
&= 
\mathbf{P} \hat{\mathbf{X}}_W \\
\end{aligned}
\tag{2.6}
$$

在这个公式(2.6)中，我们发现有很多陌生的符号，其中我们将：
$$
\left[
\begin{matrix}
m_x & 0 & x_0 \\
0 & m_y & y_0 \\
0 & 0 & 1
\end{matrix}
\right] 和
\left[
\begin{matrix}
1 & -\cot(\theta) & 0 \\
0 & \dfrac{1}{\sin(\theta)} & 0 \\
0 & 0 & 1
\end{matrix}
\right] 和
\left[
\begin{matrix}
f & 0 & 0 \\
0 & f & 0 \\
0 & 0 & 1
\end{matrix}
\right]
$$
中的参数称之为内参数(intrinsic parameters)，我们这里讨论下这些参数：

1. $m_x$和$m_y$是在x轴和y轴（指的是有偏斜过后的），每个单位长度的像素数量。通过这俩参数可以将物理单位mm转换为像素。
2. $f$ 是相机的焦距。
3. $x_0$和$y_0$是在偏斜的图像帧中的光心（以像素为单位）。
4. $s$是偏斜系数(skewness factor)，当像素是矩形的时候其为0。
5. $\theta$是两个图像SSD平面边缘之间的偏斜角度，见Fig 2.3。

这三个内参数矩阵可以合为一个矩阵$\mathbf{K}$，通过这个矩阵，我们可以将用相机坐标系表示的3D点映射到成像平面上，从而得到我们目标需要的2D点。



-----

# 总结

在这篇博文中，我们讨论了相机的针孔模型，其中涉及到了相机的内参数和外参数等，我们将会在以后的文章中发现，这些参数对于相机的呈像是很重要的，因此需要去通过相机标定（camera calibration）去计算这些参数。

----

# Reference

[1].   https://jp.mathworks.com/help/vision/ug/camera-calibration.html 

[2].  Forsyth D , JeanPonce, 福赛斯, et al. Computer vision : a modern approach[M]. 电子工业出版社, 2012. 

[3]. 电子科技大学自动化学院 杨路 老师 计算机视觉课程课件。





[camera_calibration_focal_point]: ./imgs/camera_calibration_focal_point.png
[image_pin]: ./imgs/image_pin.jpg
[imageplane]: ./imgs/imageplane.png
[TandR]: ./imgs/TandR.jpg
[central_proj]: ./imgs/central_proj.jpg

[skewness]: ./imgs/skewness.jpg
[p1]: ./imgs/p1.jpg

[p2]: ./imgs/p2.jpg
[p3]: ./imgs/p3.jpg
[p4]: ./imgs/p4.jpg