README
======

DTS402TC Group Coursework – Support Vector Machine (SVM) Model


1. Overview
-----------
This file accompanies the Support Vector Machine (SVM) implementation for the
DTS402TC Group Coursework. The notebook demonstrates a classical machine learning
approach to food image classification using handcrafted visual features.


2. File Included
----------------
SVM.ipynb

This Jupyter Notebook contains:
- Data loading and preprocessing
- Handcrafted feature extraction for image classification
- Support Vector Machine (SVM) model implementation from scratch
- Model training and evaluation
- Visualization and reporting of classification results


3. Environment Requirements
---------------------------
Python Version:
- Python 3.6 or above

Required Libraries:
- numpy
- pandas
- matplotlib

Note:
High-level machine learning libraries such as scikit-learn and deep learning
frameworks are not used, in accordance with the coursework coding policy.


4. How to Run
-------------
1. Open `SVM.ipynb` using Jupyter Notebook or JupyterLab.
2. Run all cells sequentially from top to bottom.
3. All experimental results and evaluation outputs are generated within the
   notebook and preserved in the saved output.


5. Implementation Notes
-----------------------
- The SVM classifier is implemented using basic Python and NumPy operations.
- Feature extraction is based on handcrafted image descriptors rather than
  end-to-end deep learning models.
- The notebook is self-contained and does not require additional configuration
  files.
- Results reported in the notebook are consistent with those presented in the
  final PDF report.


6. Academic Integrity Statement
-------------------------------
This notebook is an original implementation created solely for the DTS402TC
coursework. The work complies with the academic integrity guidelines and coding
restrictions specified in the module documentation.


DTS402TC Group Coursework – Random Forest Model


1. Overview
-----------
This file accompanies the Random Forest implementation for the DTS402TC Group Coursework.
The notebook provides a complete pipeline for food image classification, including data
processing, model construction, training, and evaluation.


2. File Included
----------------
random+forest.ipynb

This Jupyter Notebook contains:
- Data loading and preprocessing
- Feature handling for image-based classification
- Random Forest model implementation from scratch
- Model training and evaluation
- Visualization of results, including:
  * Confusion matrix
  * ROC curves
  * Feature importance analysis
  * Performance metrics


3. Environment Requirements
---------------------------
Python Version:
- Python 3.6 or above

Required Libraries:
- numpy
- pandas
- matplotlib

Note:
High-level machine learning and deep learning libraries such as scikit-learn,
TensorFlow, and PyTorch are not used, in compliance with the coursework coding policy.


4. How to Run
-------------
1. Open `random+forest.ipynb` using Jupyter Notebook or JupyterLab.
2. Run all cells sequentially from top to bottom.
3. All results, figures, and evaluation outputs are generated within the notebook
   and are preserved in the saved output.


5. Implementation Notes
-----------------------
- The Random Forest model is implemented using basic Python and NumPy operations.
- The notebook is self-contained and does not require additional configuration files.
- Experimental results shown in the notebook correspond to those reported in the
  final PDF report.


6. Academic Integrity Statement
-------------------------------
This notebook is an original implementation created solely for the DTS402TC coursework.
The work complies with the academic integrity guidelines and coding restrictions specified
in the module documentation.

DTS402TC Group Coursework – MLP Model
# Food Image Classification – Hand-crafted Features + NumPy MLP  
**(个人模型：NumPy 实现的多层感知机 MLP)**

> 重要说明：如果在助教/老师的环境里运行本 notebook 出现 `ImportError: numpy.core.multiarray failed to import` 或 “A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 …” 这类错误，请先看下面的【环境与兼容性说明】。这是环境中 `numpy` 与 Anaconda 里安装的 `matplotlib`（以及其他 C 扩展）版本不兼容导致的，不是本 notebook 代码的问题。

---

## 0. 项目概述（个人部分）

本 notebook 实现了一个**基于手工特征（HOG + 颜色直方图 + LBP）+ 自实现多层感知机（MLP）的食物图像分类模型**，用于区分三类食物：`bibimbap`, `chicken-curry`, `hamburger`。

本 README 对应的个人模型和代码文件为：

- 个人模型：**NumPy 实现的多层感知机（MLPClassifier）**
- 个人代码文件：`Food Classification.ipynb`
- 个人主要贡献标注方式：
  - 在 notebook 中第 **2 节 “Implement MLP”** 及之后所有以 `mlp = MLPClassifier(...)` 为核心的训练、评估和可视化代码，均为本人的个人实现部分。
  - MLP 整个前向传播、反向传播、动量优化、学习率衰减和早停（early stopping）逻辑均为 NumPy 自己实现，未使用现成深度学习框架（如 PyTorch / TensorFlow）。

---

## 1. 环境与兼容性说明（务必阅读）

本 notebook 中大量使用了 `numpy` 和 `matplotlib`。在某些 Anaconda / Jupyter 环境下，可能会出现如下错误：

> A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 …  
> `ImportError: numpy.core.multiarray failed to import`

这类错误并不是 notebook 代码本身的问题，而是**环境中的 numpy 版本与已安装的 matplotlib（以及其它 C 扩展）不兼容**导致的：

- 当前环境的 `numpy` 版本为 **2.2.6**
- 但环境中的 `matplotlib` 二进制扩展是用 **numpy 1.x** 编译的
- 结果是在 `numpy`=2.x 下导入这些扩展时会失败，引发 `ImportError: numpy.core.multiarray failed to import`

### 推荐解决方案（选一条即可）

**方案一（推荐）：在当前 Jupyter 环境中将 numpy 降级到 < 2.0**

在对应的环境（例如使用 `conda`）中执行：

```bash
# 示例：将 numpy 降级到 1.26.x（与大部分现有 matplotlib 兼容）
conda install numpy<2.0
# 或者使用 pip（根据实际环境选择）
pip install "numpy<2.0"
```

降级完成后重新启动 Jupyter 内核，再运行 notebook 即可。

**方案二：升级/重装 matplotlib（以及依赖）以适配 numpy 2.x**

如果必须保留 `numpy 2.2.6`，则需要：

```bash
conda install -f matplotlib
# 或
pip install --force-reinstall --no-binary :all: matplotlib
```

确保新安装的 matplotlib 是针对当前 numpy 版本重新编译的。

---

## 2. 数据与特征

本 notebook 假设项目目录结构如下（相对 `MLP.ipynb` 所在位置）：

```text
Food Classification/
├── Code/
│   └── notebooks/
│       └── MLP.ipynb   # 本人 notebook
└── Data/
    └── features/
        ├── train_features.npy
        ├── train_labels.npy
        ├── val_features.npy
        ├── val_labels.npy
        ├── test_features.npy
        └── test_labels.npy
        └── class_mapping.npy
```

特征已在前置步骤中由小组统一提取（非本 notebook 的内容），包括：

- HOG（Histogram of Oriented Gradients）
- 颜色直方图
- LBP（Local Binary Pattern）

这些特征被拼接成一个高维向量，保存在 `*.npy` 文件中。本 notebook 的输入维度为：

- `input_dim = 6190`
- 类别数为 `num_classes = 3`，对应 `['bibimbap', 'chicken-curry', 'hamburger']`

---

## 3. How to Run / 运行说明

1. **Activate environment / 激活环境**

   ```bash
   conda activate foodml
   ```

2. **Open Jupyter / 打开 Jupyter**

   ```bash
   jupyter notebook
   ```

   在浏览器中选择 `Python (foodml)` Kernel，并打开 `MLP.ipynb`。  
   In the browser, choose `Python (foodml)` kernel and open `MLP.ipynb`.

3. **Run all cells in order / 按顺序运行所有代码单元**

   主要步骤如下：  
   Main steps by sections:

   - **Section 1** – Data Loading & Standardization / 数据加载与标准化  
     - 从 `Data/features` 目录加载 train/val/test 特征与标签  
     - 按训练集统计量对特征进行标准化（减均值除方差）

   - **Section 2–3** – MLP Components & Classifier / MLP 组件与分类器  
     - 实现 `Dense`、`ReLU`、`Dropout`、`SoftmaxWithLoss`  
     - 实现 `MLPClassifier`（前向传播、反向传播、Momentum SGD、早停）

   - **Section 4** – Evaluation Tools / 评估工具  
     - 纯 NumPy 实现：混淆矩阵、Accuracy、Macro Precision/Recall/F1  
     - ROC 曲线 & AUC，Precision-Recall 曲线 & AP（多类别 OvR）

   - **Section 4 (search)** – Hyperparameter Search / 超参数搜索  
     - 在若干配置上训练并记录 train/val 精度

   - **Section 5** – Final Training & Learning Curves / 最终训练与学习曲线  
     - 使用选定超参数重新训练最终 MLP，并绘制 loss/accuracy 曲线

   - **Section 6** – Test Evaluation & Confusion Matrix / 测试集评估与混淆矩阵  
     - 在 train/val/test 上评估精度与 Macro-F1  
     - 打印每类 ROC-AUC 与 AP  
     - 打印测试集混淆矩阵

如无路径或环境问题，Notebook 可以一键从头运行到尾，不需要 sklearn。  
If paths and environment are set correctly, the notebook can be run top-to-bottom without sklearn.

---

## 4. Model Implementation Details / 模型实现细节

### 4.1 MLP Architecture / MLP 结构

- 输入维度：`6190`（预训练 CNN 特征）  
- 输出类别数：`3`（例如：`bibimbap`, `chicken-curry`, `hamburger`）  
- 隐藏层：可配置，例如最终模型使用 `[64]` 隐藏单元  
- 激活函数：`ReLU`  
- 正则化：`Dropout`（最终模型使用 `p_drop = 0.7`）  
- 损失函数：`Softmax + Cross-Entropy`（多分类）  
- 优化器：SGD + Momentum（纯 NumPy 实现），带学习率衰减 `lr_decay`

### 4.2 Training / 训练设置

- Epoch 数：最多 60（含早停机制）  
- Batch Size：32  
- Early Stopping：根据验证集 loss，`patience = 5`  
- 学习率初始值：例如最终模型为 `5e-4`，并按 `lr_decay = 0.995` 衰减  
- 训练中记录：  
  - `train_loss`, `val_loss`  
  - `train_acc`, `val_acc`  
  - 每个 epoch 的学习率 `lr`

---

## 5. Evaluation / 评估指标

所有评估均采用 **纯 NumPy** 实现，不调用 sklearn。  
All evaluation metrics are implemented in pure NumPy (no sklearn).

### 5.1 Metrics / 指标

- Accuracy（准确率）  
- Macro Precision / Recall / F1  
- Confusion Matrix（混淆矩阵）  
- ROC 曲线 & AUC（多类别 One-vs-Rest）  
- Precision-Recall 曲线 & AP（多类别 One-vs-Rest）

通过函数 `evaluate_classifier_np` 对任意分类器进行统一评估：  
Using `evaluate_classifier_np`, any classifier can be evaluated uniformly:

```python
results = evaluate_classifier_np(
    name="MLP (test)",
    clf_predict=mlp.predict,
    clf_proba=mlp.predict_proba,
    X=X_test_std,
    y=y_test,
    num_classes=num_classes,
)
```

函数会输出：  
The function prints:

- overall accuracy & macro-F1  
- per-class ROC-AUC & AP  
- confusion matrix（混淆矩阵）

并返回包含详细曲线数据的字典（便于后续画图或分析）。  
And returns a dict with detailed curve data for further plotting / analysis.

---

## 6. Reproducibility Notes / 复现与注意事项

1. **Environment name / 环境名称**

   - 请务必在说明文档或评分环境中使用 `foodml` 环境，以保证依赖版本与此代码一致。  
   - Please use the `foodml` environment when grading / running, to match dependency versions.

2. **No sklearn / 不使用 sklearn**

   - 当前版本的代码 **完全不依赖** `sklearn`，因此即使老师环境中缺少 sklearn，也不会影响运行。  
   - All critical components (MLP, metrics) are implemented from scratch with NumPy.

3. **File paths / 文件路径**

   - 若更改了项目结构，需同步修改 Notebook 中的数据路径部分。  
   - If you change folder layout, adjust paths in the notebook accordingly (`DATA_DIR`, `FEATURE_DIR`).

4. **Randomness / 随机性**

   - 训练过程包含随机初始化与 batch 打乱，若需要完全可复现，可在 Notebook 顶部显式设置 NumPy 随机种子：  
   - Training includes random init and shuffling; to make it fully reproducible, you can set a NumPy seed at the top:

   ```python
   import numpy as np
   np.random.seed(42)
   ```

---

## 7. Summary / 总结

本项目实现了一个 **基于预训练 CNN 特征的多层感知机（MLP）图像分类器**，完整覆盖了：  
This project implements a **Multi-Layer Perceptron (MLP) classifier on top of pre-trained CNN features**, including:

- 纯 NumPy 实现的前向 / 反向传播、Dropout、SoftmaxWithLoss  
- Momentum SGD + 学习率衰减 + 早停  
- 纯 NumPy 实现的多分类 ROC/AUC 与 PR/AP 评估工具  
- 在 train / val / test 上的性能评估与混淆矩阵展示  

环境建议使用 `foodml`，只需安装 `numpy`, `matplotlib`, `pandas` 等基础包，即可完整运行 `MLP.ipynb`。  
With the `foodml` environment and only basic packages (`numpy`, `matplotlib`, `pandas`), the entire notebook `MLP.ipynb` can be executed end-to-end without sklearn.