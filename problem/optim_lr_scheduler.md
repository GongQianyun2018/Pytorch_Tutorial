# torch.optim.lr_scheduler

> https://blog.csdn.net/qyhaill/article/details/103043637

## lr_scheduler综述

`torch.optim.lr_scheduler`提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。一般情况下我们会设置随着epoch增大而逐渐减小学习率从而达到更好的训练效果。而`torch.optim.lr_scheduler.ReduceLROnPlateau`则提供了基于训练中某些测量值使学习率动态下降的方法。

```python
>>> scheduler = ...
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()
```

注意：`lr_scheduler`更新optimizer的lr，是更新的`optimizer.param_groups[n]['lr']`而不是`optimizer.defaults['lr']`

## lr_scheduler调整策略：根据训练次数

### torch.optim.lr_scheduler.LambdaLR

语法：

```python
class torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
```

更新策略：
$$
new\_lr=\lambda \times initial\_lr
$$
参数：

1、optimizer(Optimizer)：要更改学习率的优化器

2、lr_lambda (function or list)：根据epoch计算$\lambda$的函数；或者是一个`list`这样的function，分别计算各个parameter groups的学习率更新用到的$\lambda$

3、last_epoch (int)：最后一个epoch的index，如果是训练了epoch后中断了，继续训练，这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始

### torch.optim.lr_scheduler.StepLR

```python
class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

更新策略：

每过`step_size`个epoch，做一次更新：
$$
new\_lr=initial\_lr \times \gamma ^{epoch // step\_size}
$$
参数：

1、optimizer (Optimizer)：要更改学习率的优化器

2、step_size (int)：每训练step_size个epoch，更新一次参数

3、gamma (float)：更新lr的乘法因子

4、last_epoch (int)：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的模型epoch。默认为-1表示从头开始训练，即从epoch=1开始

### torch.optim.lr_scheduler.MultiStepLR

```python
class torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

更新策略：

每次遇到`milestones`中的epoch，做一次更新：
$$
new\_lr=initial\_lr \times \gamma^{bisect\_right(milestones, epoch)}
$$
参数：

1、optimizer (Optimizer)：要更改学习率的优化器

2、milestones (int)：递增的list，存放要更新lr的epoch

3、gamma (float)：更新lr的乘法因子

4、last_epoch (int)：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的模型epoch。默认为-1表示从头开始训练，即从epoch=1开始

### torch.optim.lr_scheduler.ExponentialLR

```python
class torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
```

更新策略：

每个epoch都做一次更新：
$$
new\_lr=initial\_lr \times \gamma^{epoch}
$$
参数：

1、optimizer (Optimizer)：要更改学习率的优化器

2、gamma (float)：更新lr的乘法因子

3、last_epoch (int)：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的模型epoch。默认为-1表示从头开始训练，即从epoch=1开始

### torch.optim.lr_scheduler.CosineAnnealingLR

```python
class torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
```

更新策略：

让lr随着epoch的变化图类似于cos：
$$
new\_lr=eta\_min+(initial\_lr-eta\_min) \times (1+cos(\frac{epoch}{T\_max}\pi))
$$
其中，`eta_min`表示最小学习率，`T_max`表示cost周期的1/4。

参数：

1、optimizer (Optimizer)：要更改学习率的优化器

2、T_max (int)：lr的变化是周期性的，T_max是周期的1/4

3、eta_min (float)：lr的最小值，默认为0

4、last_epoch (int)：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的模型epoch。默认为-1表示从头开始训练，即从epoch=1开始

## lr_scheduler调整策略：根据训练中某些测量值

### torch.optim.lr_scheduler.ReduceOnPlateau

```python
class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
```

更新策略：

给定一个metric，当metric停止优化时减小学习率
$$
new\_lr=\lambda \times old\_lr
$$
$\lambda$是参数`factor`

参数：

1、optimizer (Optimizer)：要更改学习率的优化器

2、mode(str)：只能是`'min'`或者`'max'`，默认`'min'`

`'min'`：当metrics不再下降时减小lr

`'max'`：当metrics不再增长时减小lr

3、factor (float)：lr减小的乘法因子，默认为0.1

4、在metric停止优化`patience`个epoch后减小lr，例如，如果`patience=2`，那metric不再优化的前两个epoch不做任何事，第三个epoch后metric仍然没有优化，那么更新lr，默认为10

5、verbose (bool)：如果为True，在更新lr后print一个更新信息，默认为False