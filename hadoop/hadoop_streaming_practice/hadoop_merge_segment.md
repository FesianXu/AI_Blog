<div align='center'>
    【Hadoop Streaming实践系列】 大规模字段提取的实践
</div>

<div align='right'>
    FesianXu 20220829 at Baidu Search Team
</div>

# 前言

本文介绍如何利用Hadoop Streaming任务进行数以十亿计的大规模字段提取。**如有谬误请联系指出，本文遵守[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**github**: https://[github](https://so.csdn.net/so/search?q=github&spm=1001.2101.3001.7020).com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode](https://img-blog.csdnimg.cn/61deeb71d05b4a81afba5883c34bfc7e.jpg#pic_center)

----

假设现在有两个数据库A和B，在A里面存储了几十亿的url，而在B中储存了数以百亿计的url对应的正排信息，这些正排信息通过json格式进行组织。如下所示

```
# each row in A database
https://www.bilibili.com/video/BV1Dg41167zC
https://www.bilibili.com/video/BV1PU4y1678L
...
https://www.bilibili.com/video/BV1jV4y147jb
```

```
# each row in B database
{"loc":"https://www.bilibili.com/video/BV1Dg41167zC","loc_info":{"data_size":100,"timestamp":20200810}}
{"loc":"https://www.bilibili.com/video/BV1PU4y1678L","loc_info":{"data_size":120,"timestamp":20200512}}
...
{"loc":"https://www.bilibili.com/video/BV1jV4y147jb","loc_info":{"data_size":123,"timestamp":20210302}}
```

当然，数据库B里面的数据显然不是按照和数据库A一样的顺序排列的，这里只是为了举例方便而让他们具有一样的序而已。显然不可能对于每个来自于数据库A的url都在B里面遍历一遍以查找对应的正排信息，这样的计算复杂度将会是无法接受的$\mathcal{O}(NM)$，其中$N$和$M$分别是数据库A和B的数据量大小。那么如何通过MapReduce解决这个问题呢？我们知道在Map阶段，会对输入数据进行划分到不同节点中，由海量的节点进行相同的操作，等所有mapper任务结束后，会对输出进行汇聚，并且排序（sort），然后进行划分后在分配给每个reducer进行归并处理。我们发现，可以在mapper阶段，对数据库B的所有数据进行解析，并且把目标正排字段提取出来，同时对于数据库A的url，我们想办法让它在输出的时候总是和数据库B解析得到的正排字段“挨着”，这样我们的计算复杂度就缩减到了$\mathcal{O}(N+M)$。对于A的每条输出，我们可以为其加上一个标识符，表示其来自于数据库A，并且为了让其能一直排在B解析结果的前面，可以考虑将这个标识符设定为`AAAAA`(在sort阶段是按照字符的字典序进行排序的)，因此A的mapper输出结果如（url和标识符之间通过`\t`隔开）：

```
# each row in A database
https://www.bilibili.com/video/BV1Dg41167zC AAAAA
https://www.bilibili.com/video/BV1PU4y1678L AAAAA
...
https://www.bilibili.com/video/BV1jV4y147jb AAAAA
```

同时解析出来的B数据和其正排信息，如下所示：

```
https://www.bilibili.com/video/BV1Dg41167zC loc_info:100
https://www.bilibili.com/video/BV1PU4y1678L loc_info:120
...
https://www.bilibili.com/video/BV1jV4y147jb loc_info:123
```

那么将所有mapper的输出进行合并，并且sort后，其输出如：

```
https://www.bilibili.com/video/BV1Dg41167zC AAAAA
https://www.bilibili.com/video/BV1Dg41167zC loc_info:100
https://www.bilibili.com/video/BV1PU4y1678L AAAAA
https://www.bilibili.com/video/BV1PU4y1678L loc_info:120
...
https://www.bilibili.com/video/BV1jV4y147jb AAAAA
https://www.bilibili.com/video/BV1jV4y147jb loc_info:123
```

可以看到，如果B里有A对应url的正排信息的话，来自于A的数据总是在B的前面，通过这个线索即可完成url的正排字段提取。mapper和reducer的示意代码如下：

```python
# mapper code, the input of mapper could come from both A and B
import sys 
import json 

for line in sys.stdin:
    line = line.strip()
    line_seg = line.split("\t")
    if len(line_seg) == 1 and line[:4] == "http":
        # if come from A
        print("{}\t{}".format(
        	line_seg[0], "AAAAA"
        ))
    else:
        # if come from B
        try:
            datapack = json.loads(line) 
        except:
            continue
        
        try:
            target_seg = extract_seg(datapack)
            loc = extract_loc(datapack)
            print("{}\t{}".format(
            	loc, target_seg
            ))
        except:
            continue
```



```python
# reducer code, the input come from the sorted result of A+B output
import sys 

old_key = ""
content = ""
is_from_A = False

def output(key, content, is_from_A):
    if key == "" or content == "" or (not is_from_A):
        # key and content must be valid and is_from_A must be True
        return 
    print("{}\t{}".format(
    	key, content
    ))
# 初始化
for line in sys.stdin:
    line = line.strip()
    line_seg = line.split("\t")
    key = line_seg[0]
    if key != old_key:
        output(old_key, content, is_from_A)
        old_key = key
        content = ""
        is_from_A = False
    
    if line_seg[1] == "AAAAA":
        is_from_A = True
    else:
        content = line_seg[1]
    
    # 循环末尾
        
output(old_key, content, is_from_A) # 后处理
```

这里需要简单对reducer的代码进行解释，我们从上面的分析已经可以知道reducer的输入，来自于对A和B的解析结果，并且进行排序后的输出。因此reducer的目的很显然，就是判断当前url是否来自于A，同时其后继行来自于B并且url与A一致，那么就将其正排信息输出即可。同时，由于标识符为“AAAAA”，对于同一个url数据而言，来自于A的数据总是排序在前，来自于B的数据（由于是字典序更小的`loc_info`开头）则总是接续其后。因此，我们的reducer有两件事情需要做的，第一判断`old_key`是否和当前的`key`相同，如果是，那么就认为A的url在B中匹配到了。其次还得考虑当前数据是否来自于A，当然这个判断简单，只要通过标识符判断即可。最后别忘了对最后一个结果进行输出即可，我们的输出`output()`是在切换新的key的时候进行输出的，因此最后一行输出在循环中是不会输出的，需要在循环外进行输出。给一个运行例子，考虑我们的模拟输入（其中的AAAAA为标识符，content为正排内容）：

```
url0 AAAAA
url1 content
url2 AAAAA
url2 content
url3 content
url4 AAAAA
url4 content
```

那么我们的reducer运行结果和关键变量的结果如下表所示（运行到循环末尾处作为断点）

| 循环数 | old_key | key    | content | is_from_A | 当前是否输出output                    |
| ------ | ------- | ------ | ------- | --------- | ------------------------------------- |
| 初始化 | ""      | ""     | ""      | False     | N                                     |
| 1      | "url0"  | "url0" | ""      | True      | N                                     |
| 2      | "url1"  | "url1" | content | False     | N                                     |
| 3      | "url2"  | "url2" | ""      | True      | N                                     |
| 4      | "url2"  | "url2" | content | True      | N                                     |
| 5      | “url3”  | “url3” | ""      | False     | Y （根据上一次结果，也即是4进行输出） |
| 6      | “url4”  | “url4” | ""      | True      | N                                     |
| 7      | “url4”  | “url4” | content | True      | N                                     |
| 后处理 | “url4”  | “url4” | content | True      | Y（根据上一次结果，也即是7进行输出）  |

通过这种处理方法，能够将匹配不上的结果跳过，而将能匹配上的结果进行输出。