<div align=center>
<font size="6"><b>《SVM笔记系列之二》SVM的对偶问题</b></font> 
</div>

# 前言
**支持向量机的对偶问题比原问题容易解决，在符合KKT条件的情况下，其对偶问题和原问题的解相同，这里我们结合李航博士的《统计学习方法》一书和林轩田老师的《机器学习技法》中的内容，介绍下SVM的对偶问题。**
**如有谬误，请联系指正。转载请注明出处。**
*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`
**有关代码开源**: [click][click]

*****

# SVM的原问题的无约束表示
　　我们在上一篇博文《SVM笔记系列1，SVM起源与目的》中，谈到了SVM的原问题，这里摘抄如下：
$$
\min_{W,b} \frac{1}{2}||W||^2
$$
$$
s.t. 1-y_i(W^Tx_i+b) \leq 0, \ i=1,\cdots,N
$$
这是一个有约束的最优化问题，我们利用广义拉格朗日乘子法(我们将在接下来的文章再继续讨论这个)，将其转换为无约束的形式：
$$
L(W,b,\alpha) = \frac{1}{2}||W||^2 + \alpha_i\sum_{i=1}^N(1-y_i(W^Tx_i+b)), \ \alpha_i \geq 0
$$
变形为：
$$
L(W,b,\alpha) = \frac{1}{2}||W||^2 + \sum_{i=1}^N {\alpha_i}-\sum_{i=1}^N{\alpha_iy_i(W^Tx_i+b)} , \ \alpha_i \geq 0
$$
这里我们假设原问题为$\theta_P(x)$，我们将会得到原问题的无约束表述为：
$$
\theta_P(x) = \min_{W,b} \max_{\alpha} L(W, b, \alpha)=\min_{W,b} \max_{\alpha} \frac{1}{2}||W||^2 + \sum_{i=1}^N {\alpha_i}-\sum_{i=1}^N{\alpha_iy_i(W^Tx_i+b)},, \ \alpha_i \geq 0
$$
这里我觉得有必要解释下为什么上式可以表征SVM的原问题的约束形式。
假设我们有一个样本点$x_i$是不满足原问题的约束条件$1-y_i(W^Tx_i+b) \leq 0$的，也就是说$1-y_i(W^Tx_i+b) \gt 0$，那么在$\max_{\alpha}$这个环节就会使得$\alpha_i \rightarrow +\infty$从而使得$L(W,b,\alpha) \rightarrow +\infty$。如果$x_i$是满足约束条件的，那么为了求得最大值，因为$1-y_i(W^Tx_i+b) \leq 0$而且$\alpha_i \geq 0$，所以就会使得$\alpha_i = 0$。由此我们得知：
$$
L(W,b,\alpha) = \begin{cases}  
\frac{1}{2}||W||^2 & 1-y_i(W^Tx_i+b) \gt 0 满足约束条件\\
+\infty & 1-y_i(W^Tx_i+b) \leq 0 不满足约束条件
\end{cases}
$$
因此在满足约束的情况下，
$$
\theta_P(x)=\min_{W,b} \frac{1}{2}||W||^2
$$
不满足约束条件的样本点则因为无法对正无穷求最小值而自然抛弃。


****
# SVM的对偶问题
　　从上面的讨论中，我们得知了SVM的原问题的无约束表达形式为：
$$
\theta_P(x) = \min_{W,b} \max_{\alpha} L(W, b, \alpha)=\min_{W,b} \max_{\alpha} \frac{1}{2}||W||^2 + \sum_{i=1}^N {\alpha_i}-\sum_{i=1}^N{\alpha_iy_i(W^Tx_i+b)}
$$
设SVM的对偶问题为$\theta_D(x)$，可知道其为：
$$
\theta_D(x) = \max_{\alpha} \min_{W,b} L(W,b,\alpha)=\max_{\alpha} \min_{W,b} \frac{1}{2}||W||^2 + \sum_{i=1}^N {\alpha_i}-\sum_{i=1}^N{\alpha_iy_i(W^Tx_i+b)}
$$
求解$\min_{W,b} L(W,b,\alpha)$，因为$L(W,b,\alpha)$是凸函数，我们对采用求梯度的方法求解其最小值：
$$
\frac{\partial{L}}{\partial{W}}=W-\sum_{i=1}^N\alpha_iy_ix_i=0, i=1,\cdots,N
$$
$$
\frac{\partial{L}}{\partial{b}}=\sum_{i=1}^N\alpha_iy_i=0,i=1,\cdots,N
$$
得出：
$$
W=\sum_{i=1}^N\alpha_iy_ix_i,　\sum_{i=1}^N\alpha_iy_i=0,　\alpha_i \geq0,i=1,\cdots,N
$$
将其代入$\theta_D(x)$，注意到$\sum_{i=1}^N\alpha_iy_i=0$,得：
$$
\theta_D(x) = \max_{\alpha}
\frac{1}{2} \sum_{i=1}^N \alpha_iy_ix_i \sum_{j=1}^N a_jy_jx_j+\sum_{i=1}^N\alpha_i
-\sum_{i=1}^N\alpha_iy_i(\sum_{j=1}^N \alpha_jy_jx_j \cdot x_i+b)= \max_{\alpha}
-\frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_jy_iy_j(x_i \cdot x_j)+ \sum_{i=1}^N\alpha_i
$$
整理为:
$$
\theta_D(x) = \max_{\alpha}
-\frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_jy_iy_j(x_i \cdot x_j)+ \sum_{i=1}^N\alpha_i
$$
$$
s.t. \ \sum_{i=1}^N\alpha_iy_i=0
$$
$$
\alpha_i \geq0,i=1,\cdots,N
$$
等价为求最小问题:
$$
\theta_D(x) = \min_{\alpha}
\frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_jy_iy_j(x_i \cdot x_j)- \sum_{i=1}^N\alpha_i
$$
$$
s.t. \ \sum_{i=1}^N\alpha_iy_i=0
$$
$$
\alpha_i \geq0,i=1,\cdots,N
$$

根据Karush–Kuhn–Tucker(KKT)条件（我们以后单独介绍KKT条件）,我们有：
$$
\nabla_WL(W^*,b^*,\alpha^*)=W^*-\sum_{i=1}^N\alpha_i^*y_ix_i=0 \Longrightarrow W^* = \sum_{i=1}^N\alpha_i^*y_ix_i
$$
$$
\nabla_bL(W^*,b^*,\alpha^*) = 
-\sum_{i=1}^N \alpha^*_i y_i=0
$$
$$
\alpha^*_i(1-y_i(W^*x_i+b^*))=0
$$
$$
1-y_i(W^*x_i+b^*) \leq0
$$
$$
\alpha^*_i \geq0
$$
所以得知:
$$
W^* = \sum_{i=1}^N\alpha_i^*y_ix_i
$$
并且其中至少有一个$\alpha_j^* \gt 0$，对此$j$有，$y_j(W^*x_j+b^*)-1=0$
代入刚才的$W^*$，我们有
$$
b^*=y_j-\sum_{i=1}^N\alpha^*_iy_i(x_i \cdot x_j)
$$
所以决策超平面为：
$$
\sum_{i=1}^N \alpha^*_iy_i(x_i \cdot x)+b^*=0
$$
分类超平面为：
$$
\theta(x)=sign(\sum_{i=1}^N \alpha^*_iy_i(x_i \cdot x)+b^*)
$$
其中$\alpha^*_i=0$的是普通向量，而$\alpha^*_i >0$的是支持向量，因为当$\alpha^*_i >0$时，我们有$1-y_i(W^*x_i+b)=0$。



[click]: https://github.com/FesianXu/AI_Blog/tree/master/SVM%E7%9B%B8%E5%85%B3

