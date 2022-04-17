# torch.backends.cudnn.benchmark

> 参考链接：https://zhuanlan.zhihu.com/p/73711222

pytorch默认会使用cuDNN加速。但是，在使用cuDNN的时候，`torch.backends.cudnn.benchmark`为`False`，意味着我们可以继续提速

## 背景知识

实现卷积操作的方式多样，每种都有其特定优势。有的算法在卷积核大的情况下，速度很快，有的算法在某些情况下，内存使用比较小。在给定一个神经网络的前提下，可以预先进行一些简单的优化测试，为每个卷积层选择最适合（最快）的卷积算法，决定好每层最快的算法之后，再运行整个网络，效率就会提升不少。

## cudnn.benchmark=True的作用

将`cudnn.benchmark`设为`true`，可以在pytorch中对模型里的卷积层进行预先的优化，可以在每一个卷积层中测试cuDNN提供的所有卷积实现算法，然后选择最快的那个，这样在模型启动的时候，就可以较大幅度地减少训练时间

## 哪些因素会影响到卷积层的运行时间：

1、卷积层本身的参数，常见的包括卷积核大小、stride、dilation、padding、输出通道的个数等

2、输入的相关参数，包括输入的宽和高、输入通道的个数等

3、一些其他的因素，比如硬件平台、输入输出精度、布局等

## 设置`cudnn.benchmark=True`需要满足的条件

1、网络模型不变

2、输入大小不变

需要满足以上条件，否则每次都需要重新选择合适算法，花费时间，降低效率

## 代码加到哪里？

一般加在开头

比如在使用gpu的同时，后面补一句

```python
if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

......
......
```

## 源码理解

1、`torch.backends.cudnn.deterministic`

这个flag置为`True`的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置Torch的随机种子为固定值的话，应该可以保证每次运行网络的时候相同的输入的输出是固定的

2、`cuDnnGetConvolutionForwardAlgorithm_v7`和`cudnnFindConvolutionForwardAlgorithmEx`

pytorch默认调用的是前者，设置`benchmark=True`会调用后者

`Get`函数使用一些人为设置的启发式的方法（heuristic）去选择程序所认为的最合适的算法

`Find`函数是穷尽式的（exhaustive search），即会遍历所有可选的卷积进行比较

pytorch默认也是会对每层的卷积算法进行预先选择，速度比较快，但是选择出来的结果不是那么好