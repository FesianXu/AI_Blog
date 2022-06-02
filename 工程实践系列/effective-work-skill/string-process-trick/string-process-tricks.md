工作中遇到的数据很多都是按照字段进行组织的，分隔符通常是制表符`\t`，如下所示：
```shell
seg_A	seg_B	seg_C
...
...
...
```
这种格式可以组织各种数据，比如日志文件，训练/测试数据，用户行为数据等，这些数据都是字符串的形式，并且按照每行一个样本，每列一个字段的形式进行储存。linux中有很多工具和命令处理此类数据非常方便，这里简单笔记下。

# 打印某一列
采用`awk`可以做到。
```shell
cat your_file | awk -F "\t" '{print $1}' # 打印第一列
cat your_file | awk -F "\t" '{print $NF}' # 打印最后一列
```

# 查找字符串
采用`sed`, `grep` ，大杀器`awk`当然也可，不过没必要。
```shell
cat your_file | sed -n '/string_pattern/p'
cat your_file | grep "your_string"
cat your_file | grep -E "string_pattern"
```

# 新增一列
采用`awk`， 假如原本的文件如：
```shell
# file: your_file
1	2	3
4	5	6
7	8	9
```
那么在最后一列添加上字符`A`，可以用以下脚本
```shell
cat your_file | awk -F "\t" '
BEGIN{string="A"}
{
	for(i = 0; i < NF; i++)
	{
		printf($i);
		printf("\t");
	}
	printf("%s\n", string);
}'
```
其输出结果是:
```shell
1	2	3	A
4	5	6	A
7	8	9	A
```
我们发现`awk`的可执行代码非常类似于C语言，并且awk是对每一行为单位进行处理的，这意味着以上的代码会对文件中的每一行进行相同的操作。

# 替换字符串
替代字符串这个操作可以由非常多的工具进行，比如`sed`, `tr`, 万能的`awk`等。个人喜欢用`sed`。
```shell
cat your_file | sed -n 's/old_string/new_string/p'
```

# 大小写字母转换
`tr`命令适合用于字符串的转换，压缩和删除。
```shell
echo "AbC" | tr -t [A-Z] [a-z]
```
输出为`abc`

# 数据筛选
有时候需要对每一列的某些数值指标（离散的或者连续的）进行筛选，可以采用`awk`轻松搞定。原数据如：
```shell
# file: your_file
data_a	data_b	1.0
data_a	data_b	0.5
data_a	data_b	0.8
data_a	data_b	0.3
data_a	data_b	None
```
那么挑选所有最后一列大于0.5的行，可以
```shell
cat your_file | awk -F "\t" '$3>0.5'
```
输出为:
```shell
data_a	data_b	1.0
data_a	data_b	0.8
data_a	data_b	None
```
为了筛出掉缺省值`None`，也可以选择同时筛选多个条件，通过与（&&）或（||）非（！）连接起来，如下面的第一条
```shell
cat your_file | awk -F "\t" '$3>0.5 && $3!="None"'
cat your_file | awk -F "\t" '$3>0.5 && $3<0.8'
cat your_file | awk -F "\t" '$3<0.5 || $3>0.8'
cat your_file | awk -F "\t" '$3!=0'
```

# 求每一行数字的加和并且求平均值
```shell
cat your_file | awk -F "\t" '
BEGIN{sumv=0;line=0}
{
	sumv+=$2;
	line++;
}
END{print sumv/line}
'
```

# awk引用shell环境变量
有时候需要awk访问shell环境变量，此时不能简单用`${VAR}`进行访问，可以考虑以下两种方式：
```shell
ENVVAR="..." 
cat file | awk -v var="${ENVVAR}" '{print var}' 
```
或者通过访问内置的`ENVIRON`变量实现
```shell
export ENVVAR="..."
cat file | awk '{print ENVIRON["ENVVAR"]}'
unset ENVVAR # 为了后续的数据引用安全，unset掉
```

# awk文件间查重
有时候需要对两个文件之间某个字段的重复部分进行筛选，然后对提取出重复字段的整行部分。比如我们有以下数据：
**file_1.data**
```shell
http://haokan.baidu.com/v?pd=wisenatural&vid=7217158148519997092 A
https://zhidao.baidu.com/question/2054086297537323627/answer/3367935303.html B
http://m.bilibili.com/video/BV1Zw411d7m6 C
https://haokan.baidu.com/v?pd=wisenatural&vid=6619006578881460543 D
http://haokan.baidu.com/v?pd=wisenatural&vid=5651466905742673647 E
```
**file_2.data**
```shell
https://haokan.baidu.com/v?pd=wisenatural&vid=7217158148519997092 A
http://zhidao.baidu.com/question/2054086297537323627/answer/3367935303.html B
http://v.qq.com/boke/page/t/0/w/t01451r5euw.html C
http://3g.163.com/v/video/V5KLT7ESE.html D
http://haokan.baidu.com/v?pd=wisenatural&vid=5651466905742673647 E
```

那么可以通过以下awk脚本进行：
```shell
awk -F " " 'FNR==NR{a[$1];next} $1 in a {print $1}' file_1.data  file_2.data
```
那么会输出:
```shell
http://haokan.baidu.com/v?pd=wisenatural&vid=5651466905742673647 E
```
其中的`NR` (Number of Records)表示程序开始累计读取的记录数，而`FNR` （File Number of Records）表示当前文件读取的记录数，当单文件执行时，两者相同；当存在多文件时，`FNR`会在读取新文件的时候重新置位为0，而`NR`会一直累计，因此可以用`FNR==NR`来判断是否在读取第一份文件。
我们会发现协议头`https://`和`http://`即便不同，其url主体还是一致的，这种情况下需要进行`split`去除协议头后进行对比，脚本如下：
```shell
awk -F " " -v seg="://" '
FNR==NR{split($1, b, seg);a[b[2]];next}  {
split($1, c, seg); if (c[2] in a) print $0
}' file_1.data  file_2.data
```
将会输出：
```shell
https://haokan.baidu.com/v?pd=wisenatural&vid=7217158148519997092 A
http://zhidao.baidu.com/question/2054086297537323627/answer/3367935303.html B
http://haokan.baidu.com/v?pd=wisenatural&vid=5651466905742673647 E
```

# awk文件去重
awk也可以用于当前文件中，对于某个字段进行去重处理，加入目前输入文件如：
**file.data**
```shell
https://haokan.baidu.com/v?pd=wisenatural&vid=7217158148519997092 A
http://v.qq.com/boke/page/t/0/w/t01451r5euw.html C
http://zhidao.baidu.com/question/2054086297537323627/answer/3367935303.html B
http://v.qq.com/boke/page/t/0/w/t01451r5euw.html C
http://3g.163.com/v/video/V5KLT7ESE.html D
http://haokan.baidu.com/v?pd=wisenatural&vid=7217158148519997092 A
https://zhidao.baidu.com/question/2054086297537323627/answer/3367935303.html B
http://m.bilibili.com/video/BV1Zw411d7m6 C
http://haokan.baidu.com/v?pd=wisenatural&vid=7217158148519997092 A
https://haokan.baidu.com/v?pd=wisenatural&vid=6619006578881460543 D
http://haokan.baidu.com/v?pd=wisenatural&vid=5651466905742673647 E
http://haokan.baidu.com/v?pd=wisenatural&vid=7217158148519997092 A
http://haokan.baidu.com/v?pd=wisenatural&vid=5651466905742673647 E
```
那么在不考虑协议头的差别的情况下（也就是只看url主体），去重脚本如下：
```shell
awk -F " " -v seg="://" '{
split($1,a,seg); if (a[2] in b) {next} else {b[a[2]]};print $0
}' file.data
```
输出为：
```shell
https://haokan.baidu.com/v?pd=wisenatural&vid=7217158148519997092 A
http://v.qq.com/boke/page/t/0/w/t01451r5euw.html C
http://zhidao.baidu.com/question/2054086297537323627/answer/3367935303.html B
http://3g.163.com/v/video/V5KLT7ESE.html D
http://m.bilibili.com/video/BV1Zw411d7m6 C
https://haokan.baidu.com/v?pd=wisenatural&vid=6619006578881460543 D
http://haokan.baidu.com/v?pd=wisenatural&vid=5651466905742673647 E
```

# 单文件查重
有时候需要进行单文件查重，比如对某个字段进行查重，如下所示
**file_1.data**
```
part-00001.attempt_000.gz
part-00001.attempt_001.gz
part-00002.attempt_000.gz
part-00003.attempt_000.gz
part-00004.attempt_000.gz
part-00004.attempt_001.gz
part-00004.attempt_002.gz
```
其中的`attempt_xxx`是失败重试的次数，那么对其进行查重可以用以下脚本:
```shell
cat file_1.data | awk '{
	split($0, tmp, ".");
	part_name = tmp[1];
	if (part_name in part_set) {
		print part_name, $0;
	} else {
		part_set[part_name];
	}
}'
```
由此可以将重复的part打印出来，当然由此也可以选择未重复的part。

# 两文件查重
有时候需要简单统计两个文件之间的某个字段重复程度，比如统计两个文件重复的url数量，那么可以用`grep`实现，通过`-x`指定完全字符串匹配，`-F`将匹配模式指定为固定字符串的列表，用`-f`指定规则文件，其内容含有一个或多个规则样式，让grep查找符合规则条件的文件内容，格式为每行一个规则样式。通过`awk`首先将字段进行转储，如：
```shell
cat url_1.data | awk -F "\t" '{print $1}' > tmp_url_1.data
cat url_2.data | awk -F "\t" '{print $1}' > tmp_url_2.data
grep -xFf tmp_url_1.data tmp_url_2.data
```
这样可以统计`tmp_url_2.data`中的url有多少是在`tmp_url_1.data`出现过的。


# 更换分割字段的分隔符
有时候需要更改文件的分隔符，比如从`"\t"`转成`" "`，那么可以用如下脚本：
```shell
# file.data: your_file
1	2	3
4	5	6
7	8	9
```

```shell
cat file.data | awk -F "\t" -v OFS=" " '{$1=$1; print $0}'
```
这里有个值得注意的就是： `$0`是awk中对于输入record的记录，不会由于设置了`OFS`输出分隔符（Output Field Seperator ）而变化，因此需要通过`$1=$1`进行`$0`值的重建。

# 提取括号内的值
有时候遇到的数据如下所示：
```shell
# file.data: your_file
date(20220114)time(0419pm)
date(20220114)time(0839pm)
...
```
需求是提取括号内的内容，那么可以用以下命令：
```shell
cat file.data | awk -F"[()]" '{print $2,$4}'
```
也还有很多命令可以实现这类型的需求，笔者以后继续整理下

# 打印特定行的字符串
用`awk`可以解决，但是最快的还是采用`sed`进行：
```shell
sed -n '2,$p' data.file  # 第二行到最后一行的所有数据
sed -n '100p' data.file # 第100行数据
sed -n '4,6p' data.file # 第4到第5行数据
```

# 字符串替换（正则模板）
通过正则表达式可以实现更为灵活的字符串查找和匹配，以`sed`为例子，假如当前文档如：
```shell
# file.data
\mathcal{J} = \sum_{i=0]^{N-1} \mathcal{L}_i 
\tag{1-1}
```
有时候需要把所有`\tag{}`的字符串都去除，最好的方法就是采用正则表达式：
```shell
cat file.data | sed -e 's/\(.*\)\\tag{.*}\(.*\)/\1\2/p'
```
其中`-e`表示扩展正则表达式，`s/reg_pattern/replace_str/p`表示用`replace_str`去替换符合`reg_pattern`的字符串，其中`\(\)`是对括号的转义，而`()`是表示一组字符串（在后续会用`\1 \2`进行指定），那么除去正则表达式，这个正则表达式的意思是`(.*)\tag{.*}(.*)` 也就是查找符合该模式的字符串。在`replace_str`域，`\1 \2`代表符合正则表达式的字符串组，那么其实`\1` = "\mathcal{J} = \sum_{i=0]^{N-1} \mathcal{L}_i ", `\2` = ""。[1]

# 字段挑选
如以下输入，不同字段用空格隔开，但是由于某些原因，可能并不仅仅是一个空格，其中可能有若干个空格隔开了不同字段，可以考虑结合`cut`和`tr`进行字段挑选
```shell
NO Name SubjectID Mark 备注

1  longshuai 001  56 不及格
2  gaoxiaofang  001 60 及格
3  zhangsan 001 50 不及格
4  lisi    001   80 及格
5  wangwu   001   90 及格
```
```shell
cat abc.sh | tr -s " " | cut -d " " -f2,4
```
以上脚本对第2和4列字段进行打印，其中的`tr -s`将对重复的空格进行压缩，输出结果如：
```
Name Mark
longshuai 56
gaoxiaofang 60
zhangsan 50
lisi 80
wangwu 90
```

`cut`命令的参数有：
```shell
-b：按字节筛选；
-n：与"-b"选项连用，表示禁止将字节分割开来操作；
-c：按字符筛选；
-f：按字段筛选；
-d：指定字段分隔符，不写-d时的默认字段分隔符为"TAB"；因此只能和"-f"选项一起使用。
-s：避免打印不包含分隔符的行；
--complement：补足被选择的字节、字符或字段（反向选择的意思或者说是补集）；
--output-delimiter：指定输出分割符；默认为输入分隔符。
```

当然，这个字段挑选的功能也可以由大杀器`awk`完成，但是有时候用`cut`会更精炼一些。

# 删除重复字符
有些场景中，可能会出现重复字符，这些字符可能是用户的不规范输入，或者其他各种原因产生的，比如最常见的是重复空格，或者重复的制表符等等，可以采用`tr -s `命令进行重复字符的去除，如:
```shell
echo "sssssss" | tr -s "s" 
# 输出为 s
```

---


# Reference
[1]. https://unix.stackexchange.com/questions/78625/using-sed-to-find-and-replace-complex-string-preferrably-with-regex
[2]. https://article.itxueyuan.com/m9bPp