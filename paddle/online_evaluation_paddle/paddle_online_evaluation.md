<div align='center'>
Paddle静态图训练时在线验证
</div>
<div align='right'>
  FesianXu 20220312 at Baidu Search Team
</div>

# 前言

在使用paddle静态图进行模型训练的时候，可以同时进行在线模型验证，实现自动化的最优checkpoint挑选。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：
**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

----



在训练模型的时候，通常需要在验证集上进行最佳的checkpoint挑选。有些同学会考虑将隔若干步的checkpoint都dump到本地，然后离线对每个checkpoint进行验证，从中挑出最佳的checkpoint。这个做法有点麻烦，其实可以考虑在训练时，隔若干步进行训练时的在线验证。基本的伪代码框架如下（基于paddle）：

```python
import paddle.fluid as fluid
import paddle
from optim import optimization
from model import ErnieModel
from reader import ErnieDataReader

def create_model(pyreader_name, is_test, config):
  if config.is_sync_reader: # 同步数据读取器
    src_ids = fluid.layers.data(name='src_ids',
                shape=[-1, config.max_seq_len, 1], dtype='int64')
   	...
    ## 此处定义同步数据接口，类似于tensorflow里面的placeholder
    pyreader = fluid.io.DataLoader.from_generator(
                feed_list=[src_ids, pos_ids, sent_ids, input_mask, labels],
                capacity=70, 
                iterable=False)
    ## 此处定义同步数据loader
  else: # 异步数据读取器
    input_shapes = [
      [-1, max_seq_len, 1], # src_ids
      ...
    ] ## 定义数据输入shape
    input_dtypes = ["int64", ...] ## 定义数据类型
    lod_levels = [0] * len(dtypes)
    pyreader = fluid.layers.py_reader(
            capacity=30,
            shapes=input_shapes,
            dtypes=input_dtypes,
            lod_levels=lod_levels,
            name=pyreader_name,
            use_double_buffer=True)
    inputs = fluid.layers.read_file(pyreader)
    src_ids, ... = inputs
  
  model = ErnieModel(inputs=[src_ids,...], config, )
  logit = fluid.layers.fc(input=text_ernie.pool_feat(), 
                          size=2)
  loss = fluid.layers.softmax_with_cross_entropy(
            logits=logit,
            label=labels
        )
  return loss


def train(config):
  paddle.enable_static()
  train_program = fluid.Program()
  startup_prog = fluid.Program()
  
  with fluid.program_guard(train_program, startup_prog):
    with fluid.unique_name.guard():
      train_pyreader, loss = create_model(..., is_test=False)
      optimization.optimization(loss, ...)
  exe = fluid.Executor(place)
  exe.run(startup_prog)
  train_exe = exe
  eval_exe = exe
  
  if config.use_online_eval:
    test_program = fluid.Program()
    test_startup_prog = fluid.Program()
    with fluid.unique_name.guard():
      with fluid.program_guard(test_program, test_startup_prog):
        create_model(..., is_test=True)
    test_program = test_program.clone(for_test=True)
    
 
  train_data_reader = ErnieDataReader(..., is_test=False)
  if config.is_sync_reader:
    	train_pyreader.set_batch_generator(train_data_reader.data_generator())
  else:
      train_pyreader.decorate_tensor_provider(train_data_reader.data_generator())
  
  train_pyreader.start()
  
  while step < TOTAL_STEP:
    # for train
    fetch_list = [
      ...
    ]
    ret = train_exe.run(fetch_list=fetch_list, program=train_program)
    
    
    if config.use_online_eval and step % config.step_online_eval and step != 0:
      eval_data_reader = ErnieDataReader(..., is_test=True)
      if config.is_sync_reader:
          eval_pyreader.set_batch_generator(eval_data_reader.data_generator())
      else:
          eval_pyreader.decorate_tensor_provider(eval_data_reader.data_generator())

      eval_pyreader.start()
      while True:
        # for eval
        try:
          fetch_list = [
            ...
          ]
          ret = eval_exe.run(fetch_list=fetch_list, program=eval_program)
        except fluid.core.EOFException:
          test_pyreader.reset()
   
   except fluid.core.EOFException:
    train_pyreader.reset()
```





[qrcode]: ./imgs/qrcode.jpg







