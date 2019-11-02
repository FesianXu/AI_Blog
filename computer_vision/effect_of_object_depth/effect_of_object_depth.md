<div align='center'>
   	讨论物体的表面深度对相机成像的影响
</div>

<div align='right'>
    2019.11.02 FesianXu
</div>

# 前言

对于不同的物体来说，其表面纹理，或者凸出凹陷各有不同，这些对于相机成像而言都会造成影响，笔者在这篇博文中尝试对此进行讨论。**如有谬误，请联系指正。转载请注明出处。**

 $\nabla$  联系方式：
**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**: 973926198
github: https://github.com/FesianXu 



----



显然，我们常见的物体都并不是一个简单的平面，如Fig 1所示，其表面深度是各有不同的，联想到我们以前在[1]谈到的相机的针孔模型，和在[2]中谈到的投影相关的内容，我们发现，对于某个物体的投影而言，其投影坐标满足公式(1)
$$
\begin{aligned}
x^{\prime} &= \dfrac{xf}{z} \\
y^{\prime} &= \dfrac{yf}{z} \\
z^{\prime} &= f
\end{aligned}
\tag{1}
$$
![persgeometry][persgeometry]

我们发现，因为投影是从3D到2D的转换，因此显然在2D图片中，所有的深度都变成了焦距$z^{\prime} = f$，而像素点的2D坐标$(x^{\prime}, y^{\prime})$则是3D点深度$z$和本身3D坐标$(x,y,z)$的函数。因此，如果物体本身的深度不一致，比如如Fig 1中的，某些点离相机比较近，某些点离相机比较远，那么，其就不在满足线性变换的性质了，因为分母$z$都一直在变化。

![geometry][geometry]

<div align='center'>
    <b>
        Fig 1. 不同形状的几何体其表面凸出凹陷各种各样。
    </b>
</div>

再如Fig 2所示，因为物体本身的深度不一致引起的非线性扭曲，我们称之为投影缩放(foreshortening)，**注意到，我们这里谈到的非线性，指的是物体在2D平面上的投影的长度，和真实的长度不呈线性比例，也就是说投影长度“不可信”了，不能真实地表示实际物体**。

如图Fig 3所示，我们容易推想出，如果在不同深度下，即使物体本身的长度（红线长度）可能差别巨大，但是因为存在投影缩放，使得在平面上显示出来的投影大小相似，也就是有着较大的非线性了。

这里的物体本身的深度是物体的属性，和相机的位置无关，有些文献将其称之为`belief`[3].



![scale_foreshorten][scale_foreshorten]

<div align='center'>
    <b>
        Fig 2. 因为物体本身的深度引起的非线性，称之为投影缩放(foreshortening)。
    </b>
</div>



![diff_depth][diff_depth]

<div align='center'>
    <b>
        Fig 3. 不同的深度导致其严重的非线性。
    </b>
</div>



为了以后的分析方便，我们可以假设相机到物体的距离远远大于物体本身的深度(10倍以上)，也就是[3]所说的`low-belief`的情况，在这种情况下，投影缩放造成的非线性可以省略，就有了所谓的弱透视投影[2]。



-----

# Reference

[1].  https://blog.csdn.net/LoseInVain/article/details/102632940 

[2].  https://blog.csdn.net/LoseInVain/article/details/102698703 

[3].  https://blog.csdn.net/LoseInVain/article/details/102739778 







[geometry]: ./imgs/geometry.jpg
[persgeometry]: ./imgs/persgeometry.png
[scale_foreshorten]: ./imgs/scale_foreshorten.png
[diff_depth]: ./imgs/diff_depth.png