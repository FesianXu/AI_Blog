<div align='center'>
    人体动作捕捉与SMPL模型 (mocap and SMPL model)
</div>
<div align='right'>
    FesianXu 2020.7.5 at Alibaba internship
</div>





# 前言

笔者最近在做和motion capture动作捕捉相关的项目，学习了一些关于人体3D mesh模型的知识，其中以SMPL模型最为常见，笔者特在此进行笔记，希望对大家有帮助，如有谬误，请在评论区或者联系笔者指出，转载请注明出处，谢谢。

$\nabla$ 联系方式：
**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**: 973926198
github: https://github.com/FesianXu

-----



# 人体动作捕捉与人体3D mesh模型

人体动作捕捉（motion capture，以下简称mocap），我们在这个任务里面的目标是通过传感器（可以是RGB摄像头，深度摄像头或者光学标记，3D扫描仪）对人体的一段时间的某个动作进行捕捉，从而可以实现三维的人物建模。 注意到这里的“动作”一词有时候也可以用“姿态（pose）”一词描述， 在具体的表现形式上可以有以下若干种形式（也就是我们应该如何去表示某个动作的方法）：

1.  通过人体关节点表示，如Fig 1的第二张图所示。
2. 通过人体铰链结构表示，如Fig 1的第三张图所示。 （其中1和2我们都称之为人体姿态估计问题，其问题的关键在于对人体关节点的位置的预测，这里的位置可以是图片上的2D像素位置，也可能是3D空间位置）
3. 通过人体3D mesh表示，但是这个mesh并不包括人体的细节，比如表情，手势，脚踝的转动等，如Fig 1第四张图所示。
4. 通过人体细节3D mesh表示，这个mesh包含着人体的脸部表情，手势和脚踝转动等细节，如Fig 1第五张图所示。 （3和4的方法考虑了人体的形态特征，比如胖瘦，高矮等，因此表征能力更加丰富。）

![exp_hand_face][exp_hand_face]

<div align='center'>
    <b>
        Fig 1. 描述人体动作/姿态的若干种方法，原图出自[1]，其中的第四张图是本文需要介绍的SMPL模型，第五张图是在SMPL模型上扩展得到更多人体细节的SMPL-X模型。
    </b>
</div>
现有的人体姿态估计（human pose estimation）和mocap关系密切，现有很多关于人体姿态估计的工作已经可以在较为复杂的多人环境里面对2D 人体关节点进行准确估计了，如[2,3,4]等。但是为了能够利用捕捉到的关节点对人物动作3D建模，我们光利用2D人体关节点是不足够的，因为2D关节点到3D空间点的映射是具有歧义性的（ambiguous），因此对于同样一个2D关节点，在空间上就有可能有多种映射的可能性，如Fig 2所示，除非用多视角的图像去消除这种歧义性。
![2d3d_ambiguity][]

![2d3d_ambiguity_2][2d3d_ambiguity_2]

![2d3d_am][2d3d_am]

<div align='center'>
    <b>
        Fig 2 如果不对3D模型进行约束，那么单纯的单视角图像将会存在2D到3D投影的歧义性，如最后一张图的(a)是原始的2D节点，其到3D的投影有非常多的可能性。这里的歧义性可以由一定数量的多视角图像消除，或者通过对人体姿态的先验进行降低，原图出自[5]和[6]。
    </b>
</div>

然而，如果想要只是用单视角的图片去进行人体动作，我们就必须引入其他对人体3D姿态的先验知识，可以考虑引入的常见的先验知识有以下几种：

1. 可以对人体关节旋转的极限进行建模[6]，如果2D到3D投影过程中，我们能排除掉某些显然人体做不到的姿态（可以用关节与关节的角度极限表示），那么我们就去除了一定的歧义性。因此[6]作者收集了很多瑜伽表演者的人体极限姿态的角度数据集。

   ![joints_limits][joints_limits]

2. 可以收集大规模正常人体活动的3D 扫描数据，如AMASS[7]，是一个集成了15个人体真实的3D扫描数据集，并且用SMPL人体模型进行标准参数化的大型数据库，通过它，我们可以隐式地学习到人体的正常姿态，很大程度上减少歧义性，具体见HMR模型[8]，VIBE模型[8]。

   ![amass][amass]

   

现在流行的，而且效果不错的方法都是第二种方法，也就是通过建立大规模的真实人体3D数据集，这个数据集需要进行标准的参数化成数字人体模型，比如SMPL模型。然后通过对抗学习进行人体姿态正则的引入。比如HMR模型[5]，其示意图见Fig 3。

![hmr][hmr]

<div align='center'>
    <b>
        Fig 3 在HMR模型中，作者通过在大规模的真实人体mesh数据上进行对抗学习，从而引入了人体姿态正则，原图出自[5]。
    </b>
</div>

不管是为了对人体3D模型进行标准化的参数化，从而建模成3D数字人体模型，还是为了对人体3D模型进行渲染，亦或是为了引入人体姿态先验知识，我们都需要想办法设计一个可以数字化表示人体的方法，SMPL模型就是其中一种最为常用的。接下来我们简要介绍下SMPL模型。



# SMPL模型

SMPL模型在[9]提出，其全称是**Skinned Multi-person Linear Model**，其意思很简单，Skinned表示这个模型不仅仅是骨架点了，其是有蒙皮的，其蒙皮通过3D mesh表示，3D mesh如Fig 4所示，指的是在立体空间里面用三个点表示一个面，可以视为是对真实几何的采样。具体描述见[10]。那么Multi-person表示的是这个模型是可以表示不同的人的，是通用的。Linear就很容易理解了，其表示人体的不同姿态或者不同升高，胖瘦（我们都称之为形状shape）是一个线性的过程，是可以控制和解释的（线性系统是可以解释和控制的）。那么我们继续探索SMPL模型是怎么定义的。

![3d_mesh][3d_mesh]

<div align='center'>
    <b>
        Fig 4 不同解析率的兔子模型的3D mesh。
    </b>
</div>









[exp_hand_face]: ./imgs/exp_hand_face.jpg

[2d3d_ambiguity]: ./imgs/2d3d_ambiguity.jpg
[2d3d_ambiguity_2]: ./imgs/2d3d_ambiguity_2.jpg
[2d3d_am]: ./imgs/2d3d_am.jpg

[joints_limits]: ./imgs/joints_limits.jpg
[amass]: ./imgs/amass.jpg
[hmr]: ./imgs/hmr.jpg
[3d_mesh]: ./imgs/3d_mesh.png







# Reference

[1]. Pavlakos G, Choutas V, Ghorbani N, et al. Expressive body capture: 3d hands, face, and body from a single image[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 10975-10985.

[2]. Cao Z , Hidalgo G , Simon T , et al. OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018.

[3]. Fang H S, Xie S, Tai Y W, et al. Rmpe: Regional multi-person pose estimation[C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 2334-2343.

[4]. Chen Y, Wang Z, Peng Y, et al. Cascaded pyramid network for multi-person pose estimation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7103-7112.

[5]. Kanazawa A, Black M J, Jacobs D W, et al. End-to-end recovery of human shape and pose[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 7122-7131.

[6]. Akhter I, Black M J. Pose-conditioned joint angle limits for 3D human pose reconstruction[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 1446-1455.

[7]. Mahmood N, Ghorbani N, Troje N F, et al. AMASS: Archive of motion capture as surface shapes[C]//Proceedings of the IEEE International Conference on Computer Vision. 2019: 5442-5451.

[8]. Kocabas M, Athanasiou N, Black M J. VIBE: Video inference for human body pose and shape estimation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 5253-5263.

[9]. Loper M, Mahmood N, Romero J, et al. SMPL: A skinned multi-person linear model[J]. ACM transactions on graphics (TOG), 2015, 34(6): 1-16.

[10]. https://whatis.techtarget.com/definition/3D-mesh

[11]. 



