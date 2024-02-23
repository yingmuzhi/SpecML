
Spectrum Machine Learning

---

## x.1 工作日志

---

20240223

1. 文献调研


2. 环境搭建

python 3.11 ，pip内容见`env/requirements.txt`，conda环境见`env/env_py311.yaml`.

3. 代码书写

`dataset.py`: 选取蛋白质主链数据；从FTIR上扫描128次循环，测量1000cm-1到4000cm-1的数据；先做主链区域的测试；区域包含amide I, II, III；`dataset.py`文件主要重载了Pytorch的`Dataset`类，并使用`Dataloader`类进行测试；

提取 1300cm^-1 到 2060cm^-1 的数据，有的如 G-actin 有约 1066 个数据点，最终选取前1000个数据传入网络模型；作为一个小demmo，使用数据写死的方法读取数据，读取G-actin(未聚集蛋白)，F-actin(聚集蛋白)，D2O(未聚集蛋白)；

`model.py`: 使用了一个非常简单的线性神经网络TinyNet 

网络模型如下：

```shell
TinyNet(
  (prior): Sequential(
    (0): Linear(in_features=1000, out_features=784, bias=True)
  )
  (encoder): Sequential(
    (0): Linear(in_features=784, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=3, bias=True)
  )
  (decoder): Sequential(
    (0): Linear(in_features=3, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=784, bias=True)
  )
  (rear): Sequential(
    (0): Linear(in_features=784, out_features=196, bias=True)
    (1): ReLU()
    (2): Linear(in_features=196, out_features=49, bias=True)
    (3): ReLU()
    (4): Linear(in_features=49, out_features=1, bias=True)
  )
)
```

`train.py`: 使用MSE损失函数；使用SGD优化函数；增加读取测试集，每一次都会验证测试集的损失程度；