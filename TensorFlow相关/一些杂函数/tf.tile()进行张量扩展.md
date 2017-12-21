# tf.tile()进行张量扩展

在一些应用场景中，比如有一个这样的输入向量[`batch_size`, `video_size`, `feature_size`]，现在要对每一个batch（**batch-wise**）进行矩阵乘法，乘数形状如[`feature_size`, `new_size`]，最后需要得到一个









