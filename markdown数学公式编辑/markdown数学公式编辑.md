<div align=center>
<font size="6"><b>markdown数学公式编辑
</b></font> 
</div>

# 目录
1. markdown公式编辑基础知识
2. 希腊字母
3. 常见操作大全
4. 数学符号大全
5. 其他

*********************************************************************************

# markdown公式编辑基础知识
在CSDN和大多数markdown编辑器中，用`$`做为行内公式标志，`$$`作为行间公式标志，如：
示例：
`$a^2+b^2$`
效果：
$a^2+b^2$
示例：
`$$a^2+b^2$$`
效果：
$$a^2+b^2$$

在行间公式末加上`\tag{}`可以做公式标记，如：
示例：
`$$ a^2+b^2 \tag{1.1}$$`
效果：
$$ a^2+b^2 \tag{1.1}$$



*********************************************************************************

# 希腊字母
编号 | 大写代码 | 大写渲染结果 | 小写代码 | 小写渲染结果
:-: | :-: | :-: | :-: | :-:
1  | NA |  A | \alpha | $\alpha$
2  | NA | B | \beta | $\beta$
3  | \Gamma | $\Gamma $ | \gamma | $\gamma$
4  | \Delta | $\Delta$ | \delta | $\delta$
5  | NA | E | \epsilon | $\epsilon$
6  | NA | Z | \zeta | $\zeta$ 
7  | NA | H | \eta | $\eta$
8  | \Theta | $\Theta$ | \theta | $\theta$
9  | NA | I | \iota | $\iota$
10 | NA | K | \kappa | $\kappa$
11 | \Lambda | $\Lambda$ | \lambda | $\lambda$
12 | NA | M | \mu | $\mu$
13 | NA | N | \nu | $\nu$
14 | \Xi | $\Xi$ | \xi | $\xi$
15 | NA | O | NA | o
16 | \Pi | $\Pi$ | \pi | $\pi$
17 | NA | P | \rho | $\rho$
18 | \Sigma | $\Sigma$ | \sigma | $\sigma$
19 | NA | T | \tau | $\tau$
20 | NA | Y | \upsilon | $\upsilon$
21 | \Phi | $\Phi$ | \phi | $\phi$
22 | NA | X | \chi | $\chi$
23 | \Psi | $\Psi$ | \psi | $\psi$
24 | \Omega | $\Omega$ | \omega | $\omega$


*********************************************************************************
# 常见操作大全
-----------------------------------------
## 上下标
**用^表示上标，_表示下标，如果上（下）标内容多于一个字符就需要使用{}隔离。**
示例：`$\sum_{i=1}^N$`
效果: $\sum_{i=1}^N$
如果是字符前面有上下标，则需要使用`\sideset`语法：
示例： `$\sideset{^1_2}{^3_4} \Omega$`
效果：$\sideset{^1_2}{^3_4} \Omega$

-----------------------------------------
## 括号和分隔符
**( )和[ ]就是自身了，由于{ }是Tex的元字符，所以表示它自身时需要转义。**
示例：`$f(x,y) = x^2 + y^2, x \in [0,100]$`
效果：$f(x,y) = x^2 + y^2, x \in [0,100]$

有时候括号需要大号的，普通括号不好看，此时需要使用`\left`和`\right`加大括号的大小。
示例：`$(\frac{x}{y})^8$`，`$\left(\frac{x}{y}\right)^8$`

效果：$(\frac{x}{y})^8$,  $\left(\frac{x}{y}\right)^8$
`\left`和`\right`必须成对出现，对于不显示的一边可以使用 . 代替。

示例：`$\left.\frac{{\rm d}u}{{\rm d}x} \right| _{x=0}$`
效果：$\left.\frac{{\rm d}u}{{\rm d}x} \right| _{x=0}$

-----------------------------------------
## 分数
使用\frac{分子}{分母}格式，或者 分子\over 分母。
示例：`$\frac{1}{2x+1}$` 或者 `$1\over{2x+1}$`
效果：$\frac{1}{2x+1}$, $1\over{2x+1}$

-----------------------------------------
## 开方
示例：`$\sqrt[9]{3}$` 和 `$\sqrt{3}$`
效果：$\sqrt[9]{3}$, $\sqrt{3}$

-----------------------------------------
## 省略号
有两种省略号，`\ldots` 表示语文本底线对其的省略号，`\cdots`表示与文本中线对其的省略号。
示例：`$f(x_1, x_2, \ldots, x_n)=x_1^2 + x_2^2+ \cdots + x_n^2$`
效果：$f(x_1, x_2, \ldots, x_n)=x_1^2 + x_2^2+ \cdots + x_n^2$

上下文省略号`\vdots`
示例：
`$$ x_1,x_2,\cdots,x_n \\ \vdots \\ x_1,x_2,\cdots,x_n $$`
效果：
$$ x_1,x_2,\cdots,x_n \\ \vdots \\ x_1,x_2,\cdots,x_n $$

-----------------------------------------
## 矢量
示例：`$\vec{a} \cdot \vec{b}=0$`
效果: $\vec{a} \cdot \vec{b}=0$

-----------------------------------------
## 积分
示例：`$\int_0^1x^2{\rm d}x $`
效果： $\int_0^1x^2{\rm d}x $

-----------------------------------------
## 极限
示例：`$$\lim_{n\rightarrow+\infty}\frac{1}{n(n+1)}$$`
效果： 
$$\lim_{n\rightarrow+\infty}\frac{1}{n(n+1)}$$
这个效果用行间显示比较美观。

-----------------------------------------
## 累加、累乘
示例：`$\sum_1^n\frac{1}{x^2}$`， `$\prod_{i=0}^n\frac{1}{x^2}$`
示例：`$$\sum_1^n\frac{1}{x^2}$$`， `$$\prod_{i=0}^n\frac{1}{x^2}$$`
效果：$\sum_1^n\frac{1}{x^2}$， $\prod_{i=0}^n\frac{1}{x^2}$
效果: 
$$\sum_1^n\frac{1}{x^2}$$
$$\prod_{i=0}^n\frac{1}{x^2}$$

*********************************************************************************
# 数学符号大全

## 常见运算符

编号 | Tex代码 | 渲染效果 | 备注
:-: | :-: | :-:|:-:
1  | \pm | $\pm$ 
2  | \times | $\times$
3  | \div | $\div$
4  | \mid | $\mid$
5  | \cdot | $\cdot$
6  | \circ | $\circ$
7  | \ast  | $\ast$
8  | \bigodot | $\bigodot$
9  | \bigotimes  | $\bigotimes$
10 | \bigoplus | $\bigoplus$
11 | \leq | $\leq$
12 | \geq | $\geq$
13 | \neq | $\neq$
14 | \approx | $\approx$
15 | \equiv | $\equiv$
16 | \sum | $\sum$
17 | \prod | $\prod$
18 | \coprod | $\coprod$

## 集合运算符

编号 | Tex代码 | 渲染效果 | 备注
:-: | :-: | :-:|:-:
1  | \emptyset | $\emptyset$
2  | \in | $\in$
3  | \notin | $\notin$
4  | \subset | $\subset$
5  | \supset | $\supset$
6  | \subseteq | $\subseteq$
7  | \supseteq | $\supseteq$
8  | \bigcap | $\bigcap$
9  | \bigcup | $\bigcup$
10 | \bigvee | $\bigvee$
11 | \bigwedge | $\bigwedge$
12 | \biguplus | $\biguplus$
13 | \bigsqcup | $\bigsqcup$
14 | \varnothing | $\varnothing$

## 对数运算符

编号 | Tex代码 | 渲染效果
:-: | :-: | :-:|
1  | \log | $\log$
2  | \lg | $\lg$
3  | \ln | $\ln$

## 三角运算符
编号 | Tex代码 | 渲染效果 | 备注
:-: | :-: | :-:|:-:
1  | \bot | $\bot$
2  | \angle | $\angle$
3  | 45^\circ | $45^\circ$
4  | \sin | $\sin$
5  | \cos | $\cos$
6  | \tan | $\tan$
7  | \cot | $\cot$
8  | \sec | $\sec$
9  | \csc | $\csc$


## 微积分运算符
编号 | Tex代码 | 渲染效果 | 备注
:-: | :-: | :-:|:-:
1  | \prime | $\prime$ | 求导符号
2  | \int | $\int$ |
3  | \iint | $\iint$ |
4  | \iiint | $\iiint$ |
5  | \iiiint | $\iiiint$ |
6  | \oint | $\oint$
7  | \lim | $\lim$
8  | \infty | $\infty$
9  | \nabla | $\nabla$
10 | \partial | $\partial$ | 偏微分符号

## 逻辑运算符
编号 | Tex代码 | 渲染效果 | 备注
:-: | :-: | :-:|:-:
1  | \because | $\because$
2  | \therefore | $\therefore$
3  | \forall | $\forall$
4  | \exists | $\exists$

## 戴帽符号
编号 | Tex代码 | 渲染效果 | 备注
:-: | :-: | :-:|:-:
1  | \hat{y} | $\hat{y}$
2  | \check{y} | $\check{y}$
3  | \breve{y} | $\breve{y}$

## 连线符号
编号 | Tex代码 | 渲染效果 | 备注
:-: | :-: | :-:|:-:
1  | \overline{a+b+c+d}  | $\overline{a+b+c+d} $
2  | \underline{a+b+c+d} | $\underline{a+b+c+d} $
3  | \overbrace{a+\underbrace{b+c}_{1.0}+d}^{2.0} | $\overbrace{a+\underbrace{b+c}_{1.0}+d}^{2.0}$


## 箭头符号
编号 | Tex代码 | 渲染效果 | 备注
:-: | :-: | :-:|:-:
1  | \uparrow | $\uparrow $
2  | \downarrow | $\downarrow$
3  | \Uparrow | $\Uparrow$
4  | \Downarrow | $\Downarrow$
5  | \rightarrow | $\rightarrow$
6  | \leftarrow | $\leftarrow$
7  | \Rightarrow | $\Rightarrow$
8  | \Leftarrow | $\Leftarrow$
9  | \longrightarrow | $\longrightarrow$
10 | \longleftarrow | $\longleftarrow$
11 | \Longrightarrow | $\Longrightarrow$
12 | \Longleftarrow | $\Longleftarrow$

*********************************************************************************
# 其他
编号 | Tex代码 | 渲染效果 | 备注
:-: | :-: | :-:|:-:
1  | \sim | $\sim$  | 符合分布符号
2  | \aleph | $\aleph$ | 阿列夫符号
3  | \Im | $\Im$ | 虚数集合
4  | \Re | $\Re$ | 实数集合
5  | \simeq | $\simeq$ |
6  | \cong | $\cong$
7  | \prec | $\prec$
8  | \lhd | $\lhd$




# 使用指定字体
`\rm text`使用罗马体书写text，效果如：$\rm text$
`\cal text`使用花体书写text，效果如：$\cal text$
其他的字体还有：

编号 | Tex代码 | 备注
:-: |:-:|:-:
1  | \rm | 罗马体
2  | \bf | 黑体
3  | \sl | 倾斜体
4  | \mit | 数学斜体
5  | \sc　| 小体大写字母
6  | \it | 意大利体
7  | \cal | 花体
8  | \sf | 等线体
9  | \tt | 打字机字体


# 引用
1. [《[CSDN_Markdown]使用LaTeX基本数学公式》][ref_2]
2. [《MathJax basic tutorial and quick reference》][ref_1]


[ref_1]: https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference
[ref_2]: http://blog.csdn.net/bendanban/article/details/44196101
