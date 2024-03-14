
Spectrum Machine Learning

---

## x.1 工作日志

---

20240314

1. 文献调研

2. 环境搭建

3. 代码书写

- 将数据更改后，发现有时候读取的信号长度到不了[1, 1000]，将缺少的地方用0进行填充。
- 成功预测不同温度下的蛋白结构。

4. 展望

- 现在loss很大，期望增加预处理阶段，只提取需要的波数的数据，将数据堆叠成.npy数据格式；再根据npy数据格式求出一些统计值如均值，方差，对整个数据集归一化；
- 希望用多线程处理数据预处理；
- 期望用GPU进行网络训练；
- 期望增加一些评价指标；
- 期望使用.yml修改参数；
- 期望将代码重构为包含Trainer的格式；
- 期望代码能够使用importlib自动导入网络模型；


---

20240227

1. 文献调研

寻找光谱，红外，机器学习文章，参考南开大学`https://doi.org/10.1016/j.trac.2024.117612`

随机种子设置为3407，参考`https://arxiv.org/abs/2109.08203v2`

2. 环境搭建
t
3. 代码书写

`train.py`: 增加存取和读取模型参数的部分，save_pth(); 增加设置随机种子部分; 

删除`def validata_data():`方法，使用`dataset_utils.py`的方法读取数据；

`dataset_utils.py / read_one_signal`: 增加获取数据，取了1300\~2060cm-1（150\~850）数据；取了2780\~3050cm-1（50\~350）数据；使用magic number获取数据；

4. 展望

预处理，增加数据归一化；



---

20240226

1. 文献调研

想寻找一维信号和CAM，做一维红外的可解释性，参考`https://doi.org/10.1016/j.measurement.2021.110242`，并不适用。

调研生物，红外，机器学习结合的文章，很少，有用于预测蛋白质二级结构的文章，参考2020JACS`Cite This: J. Am. Chem. Soc. 2020, 142, 19071−19077`; 2024 JACS`DOI:10.1021/jacs.3c12258`。

想找一些生物信息：蛋白(BSA)，变量因素：温度，方法：machine learning之间结合的文章。

2. 环境搭建

3. 代码书写

`dataset.py`: 改头换面，使得数据能够直接读取FTIR出来的原始数据`.dat`文件。数据组织格式如下：

```
Data
|----FTIR
      |---- 1_data_mapping.csv
      |---- 1
            |---- G-actin_1.dat
            |---- G-actin_2.dat
            |---- ...
      |---- 2
            |---- F-actin_1.dat
            |---- F-actin_2.dat
            |---- ...
      |---- ...  
```

原始数据存放在FTIR下面，具有多个类别，文件名1, 2, ...就是类别标签target；该类别下具有多种文件，如G-actin_1.dat就是signal；signal和target是一一对应的关系；

增加了一些为dataset服务的utils脚本；



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