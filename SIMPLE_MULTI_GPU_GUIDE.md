# 使用指南

## 🎯 多GPU支持

扩散模型项目支持单机多GPU训练，使用PyTorch的DataParallel自动并行化。

## 🚀 使用方法

### 手动配置
如果您需要自定义配置，可以修改Main.py中的设置：

```python
modelConfig = {
    "state": "train", # 运行状态："train" 训练模式 或 "eval" 评估模式
    # ... 其他配置
    "device": "cuda",  # 自动使用所有GPU
    "batch_size": 80,  # 总批次大小（会在多个GPU间自动分配）
    # ... 其他配置
}
```

### 自动多GPU训练
现在无需任何额外配置！当您运行训练时：

```python
# 默认配置会自动使用所有可用GPU
python Main.py
```

## ⚡采样
### 配置

可以修改Main.py中的设置：

```python
modelConfig = {
    "state": "eval", # 运行状态："train" 训练模式 或 "eval" 评估模式
    # ... 其他配置
    "device": "cuda",  # 自动使用所有GPU
    "batch_size": 16,  # 总批次大小（会在多个GPU间自动分配）
    # ... 其他配置
}
```
采样标签为随机生成，参考Train.py(line 205)

### 运行

```python
# 默认配置会自动使用所有可用GPU
python Main.py
```
### 结果存储
采样结果会保存在 `SampledImgs/` 目录下
[labels.txt](test\SampledImgs\labels.txt) 为随机生成的标签
[NoisyNoGuidenceImgs.png](test\SampledImgs\NoisyNoGuidenceImgs.png) 为无指导的噪声图像
[SampledNoGuidenceImgs.png](test\SampledImgs\SampledNoGuidenceImgs.png) 为生成的采样图像
