# Pitch/Duration 修改器

这是一个音高/时长修改器。该修改器针对帧级别的音频（而不是想 sox or ffmpeg那样只针对整条音频）

该修改器基于 praat 的实现，在上层进行了 python 封装，以方便调用。

算法原理是基于 psola 进行波形的重建，可以很方便的用于音色转换/声码器训练等需要数据增强的场景。

更多原理和帮助文档如下:

- https://parselmouth.readthedocs.io/en/latest/examples/pitch_manipulation.html#
- https://www.fon.hum.uva.nl/praat/manual/Types_of_objects.html

一些更多的介绍和 demo 图示，也可以参考我的博客 [Pitch Tuner](https://liu-feng-deeplearning.github.io/2021/06/30/PitchTuner/)
 
### 使用方法

- 可以参考  tuner.py 文件中的 "test" 函数


 