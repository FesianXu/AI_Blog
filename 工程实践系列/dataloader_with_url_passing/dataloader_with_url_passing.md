<div align='center'>
  【工程实践系列】在paddle数据加载器中返回URL或者其他文本信息
</div>



<div align='right'>
  FesianXu 20220521 at Baidu Search Team
</div>

# 前言

最近笔者在进行全量数据的特征提取，其中需要产出的数据如`<url, score>`这种二元组，但是paddle的Dataloader并不支持直接返回字符串，只支持float,int,bool,uint等数值类型的tensor返回，而需要的url是一串字符串，因此需要进行一些特殊处理才能返回需要的产出，本文记录笔者的解决思路。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

----

最近笔者工作中遇到一个需求，需要计算数以百亿行计的url的模型打分，最后产出每个url的模型打分，格式如二元组`<url, score>`，这个过程笔者用`paddle`，自己实现了一套简单的推理框架，可以实现QPS=64000的计算速度。但是还有个小问题，`paddle`的Dataloader并不支持对Dataset返回的字符串进行组装，否则会报错如下：
```
ValueError: (InvalidArgument) Input object type error or incompatible array data type. tensor.set() supports array with bool, float16, float32, float64, int8, int16, int32, int64, uint8 or uint16, please check your input or input array data type. (at /paddle/paddle/fluid/pybind/tensor_py.h:355)
```

可以发现paddle的Dataloader仅对`bool,float,int,uint`等数值类型支持组装，如果需要实现对字符串的组装，那么首先需要想办法将字符串转成数值类型。这个时候可以简单用`ord()`将字符串转换成对应的十进制整数数值，然后进行组装。然而当字符串中有可能出现一些特殊字符符号，比如控制字符等的时候，如果需要保留这些特殊字符，可以考虑首先先将字符串转化为base64编码的结果，例如以下的url：

```
https://www.baidu.com/  -> aHR0cHMlM0EvL3d3dy5iYWlkdS5jb20v
```

由于base64编码的数值范围是有限的，如Fig 1所示，因此能用一个词表把所有base64字符与数值索引进行对应，如`char2index_map`所示，此时我们就成功把字符串转换成了数值，而Dataloader支持对数值的组装。

![base64_encoding][base64_encoding]

<div align='center'>
  <b>
    Fig 1. base64编码的编码字符范围。
  </b>
</div>

```python
char2index_map = {
            'a': 0, 'H': 1, 'R': 2, '0': 3, 'c': 4, 'D': 5, 'o': 6, 'v': 7, 'L': 8, '2': 9, 'h': 10, 'b': 11, 't': 12, 'i': 13, '5': 14, 'Y': 15, 'W': 16, 'l': 17, 'k': 18, 'd': 19, 'S': 20, 'j': 21, '9': 22, 'w': 23, 'Z': 24, '1': 25, '3': 26, 'X': 27, 'N': 28, 'm': 29, 'F': 30, 'J': 31, 'C': 32, 'Q': 33, 'O': 34, 'T': 35, 'I': 36, 'M': 37, '4': 38, 'z': 39, 'A': 40, 'n': 41, 'u': 42, 'K': 43, 'p': 44, '7': 45, '/': 46, 'f': 47, 'P': 48, 'r': 49, 's': 50, 'g': 51, 'x': 52, 'e': 53, '+': 54, '6': 55, 'y': 56, 'q': 57, 'E': 58, '8': 59, 'V': 60, 'U': 61, 'G': 62, 'B': 63, '=': 64
        }
index2char_map = dict([(v, k) for k,v in char2index_map.items()])
look_up_char = lambda idx: self.token2char_map[idx]
look_up_token = lambda idx: self.char2token_map[idx]
```

注意到由于Dataloader对于不同batch之间的数据组装，要求组装完后的tensor维度是一致的，比如`[N, seq]`，对于这个场景而言就是字符长度需要一致，因此通常需要在`collect_fn`里面进行填充操作（假设多余部分全部填充为-1），使得Dataloader返回的tensor形状如`[N, seq]`。在解码恢复成字符串的时候，按照以下代码进行即可

```python
def id2char(loc_ids):
	return list(map(look_up_char, loc_ids))

for pack in loader:
	locs, ... = pack
	# loc是dataloader
  for ind, loc in enumerate(locs):
    valid_loc_id = np.where(loc == -1)[0][0]
    loc = np.squeeze(loc[:valid_loc_id]).tolist()
    locchar = id2char(loc)
    locstr = ''.join(locchar)
    locstr = bytes(base64.b64decode(locstr))
    locstr = locstr.decode(encoding="utf-8")
```

返回的`locstr`即是原先的字符串了。

当然，也还有一个小方法，就是直接对字符串求md5值，由于md5值无法复原为原先的字符串，因此跑完打分后还需要提个Hadoop任务进行md5值和原先字符串的匹配，不过这个过程速度也很快，偷懒的时候可以采用这个方法。



[qrcode]: ./imgs/qrcode.jpg

[base64_encoding]: ./imgs/base64_encoding.png



