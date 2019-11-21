<div align='center'>
   C语言中的内存布局(memory layout)
</div>

<div align='right'>
    2019.11.20 FesianXu
</div>



# 前言

最近看了关于内存布局的文章[1]，感觉讲的很好，结合他这里的原文，这里做大部分的翻译和理解注释等，希望对各位有所帮助。

 $\nabla$联系方式：
**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**: 973926198
github: https://github.com/FesianXu 



-----

# 内存布局

根据经典的计算机冯洛伊曼模型，内存储存着计算过程中的代码和数据等。一般来说，内存是称之为DRAM，其数据是掉电易失的，我们为了简化编程过程，通常会把内存空间当作是连续的一大块，也就是说如果给每个内存小块进行编址的话，可以从0直接编码到最大的内存空间上限，我们通常把这个一大块连续的内存空间称之为**虚拟内存空间**，为什么称之为“虚拟”呢？那是因为物理硬件上的内存上不一定是连续的，其通过了一系列的映射才把可能是非连续的物理内存空间映射成了连续的虚拟内存空间，不过这个已经不在我们这篇文章的讨论范畴了，我们这里知道我们编程中，我们的变量，代码其实都是储存在这一大块的连续的虚拟内存空间就够了。

当然，这么一大块内存空间为了能够被更好地管理，我们通常要对内存进行布局，也就是划分功能块，我们称之为 **内存布局（memory layout）** 我们这里以c语言为例。通常我们的划分是连续的，如Fig 1所示，通常我们把连续的虚拟内存空间，从低地址位到高地址位，划分为五大段(segment):

1. 文本段(test segment)
2. 初始化后的数据段(initialized data segment)
3. 未初始化的数据段(uninitialized data segment)
4. 栈(stack)
5. 堆(heap)

我们接下来分别介绍。

![memoryLayoutC][memoryLayoutC]

<div align='center'>
    <b>
        Fig 1. 内存布局的示意图。
    </b>
</div>

## 文本段

文本段又被称之为代码段，其中包含着程序代码的可被执行的指令（CPU中的译码器将解释这些指令，从而实现数值计算或逻辑计算等）。我们发现文本段是从最低地址位开始分配的，那是因为，如果放在堆栈的后面，如果堆栈溢出(overflow)了，那么文本段就可能会被覆盖掉，造成不可预料的程序错误，因此为了避免这个问题，我们把文本段放在了最低位。

通常来说，代码段中的代码是可以被共享的（感觉有点像动态链接的意思，多个程序动态链接同一个库的程序，而不尝试去进行集成在一起，因为集成在一起将会造成多个同样指令的多个副本，造成浪费），因此，对于同一个模块（同一个库），我们只需要在文本段保留一个副本就够了。文本段通常是只读的，从而避免程序在意外情况下改变了其中的指令。（如果真的造成了溢出，真的可能会不可预料地改变文本段的指令，这个通常是很危险的，会导致这个系统的崩溃）



## 初始化后的数据段

初始化后的数据段(initialized data segment)，通常简称为数据段(data segment)。数据段中储存的是程序中的全局变量或者是静态变量，而这些变量是被程序员初始化过了的。注意到，数据段的数据并不意味着只是只读的，其中的变量可能在程序运行中被改变。数据段又可以被划分为初始化过了的只读区(initialized read-only area)和初始化过了的读写区(initialized read-write area)，这个由程序中的关键字进行修饰。举例而言：

```c
char s[] = "hello world";
```

如果这个语句在函数之外，定义了一个全局的字符数组，其储存在了数据段的初始化过了的读写区。如果像是：

```c
char *string = "hello world";
```

那么，这个字符实体`"hello world"`将会被储存在初始化过了的只读区，但是其指针`&string`本身储存在了读写区。



## 未初始化的数据段

未初始化的数据段(Uninitialized data segment)，也被称之为`BSS`段，其名字以一个古老的汇编操作符命名，其代表了“以符号为始的块（Block Started by Symbol）”。在程序执行之前，在这个段的数据都会内核初始化成0。

未被初始化的这些数据从初始化过的数据段（也即是Initialized data segment）的结尾处开始，其中包含着所有的全局变量和静态变量，注意到这些变量未曾在代码中进行任何的显式的初始化。例如：

```c
static int i; // 未经过初始化的静态变量，将会储存在BSS中
int j; // 定义的全局变量j，其未经过初始化，也是会储存在BSS中
```



## 栈区

栈区（stack）用于储存自动变量，其里面是在函数每次被调用的时候，都会被保存的一些信息。每次当函数被调用的时候，一些信息，例如

1. 应该在何处返回的地址
2. 调用者的环境信息，比如一些寄存器信息等

将会被储存在栈区中（保留现场信息）。这个被调用的函数则会在栈区中申请分配内存给函数里面定义的自动变量和临时变量以供使用。这个就是为什么在C语言中迭代函数可以工作的原因了，每次迭代函数都调用了其自身的时候，其会使用一个新的栈区内存，因此不同栈区内存之间的内容不会相互干扰，即便他们从源代码上看起来的确是同一个函数，但是他们的实际内存上的内容却得到了隔离。

栈区（stack）一般是在堆区（heap）的邻边，并且栈区其数据地址的增长方式和堆区是相反的，也就是说堆区的数据按照初始化的顺序，可能是从**低地址位到高地址位**分配的， 而栈区的数据可能按照 **从高地址位到低地址位**的方向分配，这种策略减少了数据溢出造成的危害。当堆区的指针和栈区指针相碰时，我们容易知道，已经没有空余的内存可以分配了。（在现代大规模的地址空间和虚拟内存技术的帮助下，栈区和堆区可能被安置在任何地方，但是他们一般还是从相反的方向进行分配）

栈区包含着程序栈（program stack），其是一个LIFO(Last In First Out)的结构，一般会被安置在内存的高地址位。在标准的x86结构计算机上，它朝着地址0（也就是地址起始点）方向增长；然而在其他的一些结构的计算机中，它朝着反方向增长。一个“栈区指针”寄存器将会一直跟踪着栈区的头部(top of the stack)，在每次数据压入栈区的时候，它将会自动地调整。为了一个函数而压入栈区的一系列值，我们称之为**栈帧**(stack frame)，一个栈帧至少要包括了返回地址，不然将会无法返回被调用函数，导致出错。



## 堆区

堆区（heap）是用于分配动态内存的段。我们用代码`malloc(), realloc(), new`等分配的内存都储存在堆区。堆区在BSS段的结尾处开始，并且其朝着高地址位的方向增长。正如我刚才所说的，堆区通过`malloc(),realloc(),free`等进行管理着内存的分配和释放，其可能会使用`brk`或者`sbrk`系统调用进行调整其大小（注意到`brk/sbrk`的使用和一个最小堆区并不足以满足`malloc/realloc/free`这些命令功能的完整要求，其也许还需要通过`mmap`内存映射去潜在地预定一些非连续的虚拟内存区域到进程的虚拟内存空间中）。堆区是被进程中的所有共享库和动态加载模组所共享的，比如动态链接库(`.dll, .so`)等。



# 例子

现在有c语言代码如：

```CQL
// file name memory-layout.c
#include <stdio.h> 
int main(void) 
{ 
    return 0; 
} 
```

我们可以通过指令`size`对其使用的各部分的内存进行报告，如下所示：

```shell
[narendra@CentOS]$ gcc memory-layout.c -o memory-layout
[narendra@CentOS]$ size memory-layout
text       data        bss        dec        hex    filename
960        248          8       1216        4c0    memory-layout
```

我们在原来代码的基础上添加一个全局变量，其未曾被初始化：

```c
#include <stdio.h> 
  
int global; /* Uninitialized variable stored in bss*/
  
int main(void) 
{ 
    return 0; 
} 
```

同样地我们观察其内存报告：

```shell
[narendra@CentOS]$ gcc memory-layout.c -o memory-layout
[narendra@CentOS]$ size memory-layout
text       data        bss        dec        hex    filename
 960        248         12       1220        4c4    memory-layout
```

我们发现`BSS`区增大了4个字节，那个正是新定义的全局变量的大小。

我们再添加一个未曾初始化的静态变量试试看：

```c
#include <stdio.h> 
  
int global; /* Uninitialized variable stored in bss*/
  
int main(void) 
{ 
    static int i; /* Uninitialized static variable stored in bss */
    return 0; 
} 
```

同样观察报告，发现`BSS`区增大到了16.

```shell
[narendra@CentOS]$ gcc memory-layout.c -o memory-layout
[narendra@CentOS]$ size memory-layout
text       data        bss        dec        hex    filename
 960        248         16       1224        4c8    memory-layout
```

如果对这个静态变量进行初始化，那么其多出来的内存将会在数据段中，而不是在`BSS`段中：

```c
#include <stdio.h> 
  
int global; /* Uninitialized variable stored in bss*/
  
int main(void) 
{ 
    static int i = 100; /* Initialized static variable stored in DS*/
    return 0; 
} 
```

```shell
[narendra@CentOS]$ gcc memory-layout.c -o memory-layout
[narendra@CentOS]$ size memory-layout
text       data        bss        dec        hex    filename
960         252         12       1224        4c8    memory-layout
```





# Reference

[1]. https://www.geeksforgeeks.org/memory-layout-of-c-program/

[2]. https://www.geeksforgeeks.org/common-memory-pointer-related-bug-in-c-programs/

[3]. https://www.tutorialspoint.com/compiler_design/index.htm





[memoryLayoutC]: ./imgs/memoryLayoutC.jpg



