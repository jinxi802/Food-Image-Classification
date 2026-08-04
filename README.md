# DTS402TC_ML_GroupCW_Food_Classification

## Project Overview
This group machine learning coursework implements a three-class food image classification task (bibimbap / chicken-curry / hamburger). Strict coding constraints: **No scikit-learn / PyTorch / TensorFlow**, all models & evaluation metrics coded purely with NumPy, Pandas, Matplotlib from scratch.
Two data sources are used: public Kaggle Food-101 dataset (≥2000 samples) + self-collected real-world food images (1020 samples). The group builds three entirely different independent ML models for fair comparison, completes unified preprocessing, feature engineering, cross-dataset testing, and comprehensive quantitative analysis via Accuracy, Macro Precision/Recall/F1, ROC-AUC, confusion matrix, training curve visualization.

## Clear Team Division of Labor (3 Members, Individual Independent Implementation)
### Member 1: Hongrui Ji — Regularized Linear SVM Model (Individual 30% Score Part)
1. Design & implement L2 regularized multi-class SVM (One-vs-Rest) entirely with NumPy mini-batch gradient descent, hinge loss.
2. Handcrafted HOG feature extraction, image augmentation pipeline (flip, brightness adjust, CLAHE contrast).
3. SVM hyperparameter tuning (regularization λ, learning rate), early stopping strategy.
4. Train & validate SVM on train/val/test splits; generate accuracy/loss curves, confusion matrix, multi-class ROC-AUC plots.
5. Analyze SVM generalization gap, domain shift performance on open-source vs self-collected food data.

### Member 2: Fei Ren — Random Forest Ensemble Model (Individual 30% Score Part)
1. Build decision tree & Random Forest from scratch based on Gini impurity, bootstrap sampling, random feature subset split.
2. Implement ensemble majority voting prediction logic for multi-class food classification.
3. Extract concatenated handcrafted visual features (gradient + color + texture descriptors).
4. Evaluate RF performance; output feature importance ranking, confusion matrix, ROC curves.
5. Measure training/inference time & memory consumption; analyze ensemble robustness against domain variation.

### Member 3: Tianqi Gao — NumPy MLP Neural Network Model (Individual 30% Score Part)
1. Full scratch MLP implementation: forward propagation, backpropagation, SGD with momentum, learning rate decay, Dropout regularization.
2. Use pre-trained ResNet-50 as fixed feature extractor to generate deep feature vectors as MLP input.
3. Design anti-overfitting pipeline: hidden layer size tuning, high dropout rate, validation-based early stopping.
4. Compute Precision-Recall curves, AP metrics, normalized confusion matrix for test set evaluation.
5. Compare neural network nonlinear fitting capacity vs traditional linear/ensemble models.

### Shared Group Collaborative Work (70% Group Mark Part, All Three Members Jointly Complete)
1. Problem definition: Justify food classification practical value & intra-class visual similarity challenges.
2. Dataset collection & processing:
   - Download Kaggle Food-101 open dataset (2100 food images, 3 categories).
   - Self-shoot & label 1020 real food images with complex lighting/background.
   - Unified image resizing, normalization, data augmentation, missing value cleaning, domain alignment.
3. Cross-dataset statistical comparison: brightness distribution, class balance, feature variance bias analysis.
4. Unified evaluation framework: Shared NumPy functions for Accuracy, Macro P/R/F1, ROC-AUC calculation.
5. Cross-model comparative analysis: Summarize strengths, weaknesses, speed, memory, generalization of SVM / RF / MLP.
6. Final group report writing (≤2000 words), integrate all three models’ charts, metrics & discussion.
7. Organize full submission package, standardize README & reproduction guide.

## Core Task Breakdown (Assignment Marking Structure)
1. Problem & Dataset (20% Group Work)
   - Task1: Real-world food classification problem justification
   - Task2: Dual dataset collection (open-source + self-collected, meet sample quantity requirement)
   - Task3: Statistical & visual comparison between two datasets
   - Task4: Unified cross-dataset data preprocessing pipeline

2. Independent Model Implementation (30% Individual Marks, Separately Graded Per Person)
   Each student completes a distinct model category, fully independent code without sharing implementation logic:
   - Linear Margin Model: SVM (Hongrui Ji)
   - Tree Ensemble Model: Random Forest (Fei Ren)
   - Neural Network Model: MLP (Tianqi Gao)

3. Result Evaluation & Comparative Analysis (30% Group Work)
   - Unified multi-class evaluation metrics (Accuracy, Macro F1, ROC-AUC, confusion matrix)
   - Separate test on open-source data, self-collected data, mixed combined dataset
   - Trade-off analysis: prediction accuracy, training speed, memory usage, interpretability, cross-domain robustness
   - Failure case discussion & improvement proposals

4. Formal Group Report (20% Group Work, ≤2000 Words)
   Standard academic layout with all experimental figures, metric tables, code snippets and critical discussion.

## Repository File Structure
```text
Food-Image-Classification/
├── README.md # Project introduction, team division & running guide
├── requirements.txt # Dependencies list
├── train_labels.csv # Training set label file
├── val_labels.csv # Validation set label file
├── test_labels.csv # Test set label file
├── data/ # Dataset folder (open-source + self-collected food images)
│
├── code/ # All source code & notebooks
│ ├── SVM.ipynb # SVM model implementation
│ ├── random_forest.ipynb # Random Forest implementation
│ └──  MLP.ipynb # Numpy-based MLP implementation
│
├── json/ # Model outputs, metrics & optimized result json files
│ ├── optimized_results.json
│ ├── optimized_results_with_evaluation.json
│ └──  random_forest_results.json
│
└── result/ # All visualized experimental charts & output images
├── confusion_matrix_test.png
├── feature_importance.png
├── loss_accuracy_tradeoff.png
├── metrics_comparison.png
├── performance_metrics.png
├── roc_curves_test.png
├── trade_off_analysis.png
└── svm_regularized_results/
│ ├── advanced_analysis.png
│ ├── final_analysis.json
│ └── final_performance.png
```

## Tech Stack & Strict Coding Restrictions
Permitted libraries only: `numpy`, `pandas`, `matplotlib`, OpenCV (image IO/resize only)
Forbidden high-level libraries: scikit-learn, PyTorch, TensorFlow, XGBoost, pre-built ML APIs
All core model training, loss calculation, evaluation metrics are manually implemented with pure NumPy as coursework required.
Python ≥3.6, all deliverables are self-contained Jupyter Notebooks with all outputs retained (no re-run needed for marking).


## Dataset Information
The dataset consists of two parts:
1. Public dataset: **Kaggle Food-101**
2. Supplementary data: Manually collected online food images
All images are unified preprocessed and split into train / validation / test sets.

### Environment Installation
```bash
conda create -n foodml python=3.9
conda activate foodml
pip install -r requirements.txt
```

### Full Reproduction Steps
1. Place open-source & self-collected food images under data/ folder
2. Run shared preprocessing scripts to extract unified feature vectors
3. Execute each member’s independent Jupyter Notebook separately:
   - Run code/SVM.ipynb for SVM training & visualization
   - Run code/random+forest.ipynb for Random Forest experiments
   - Run code/Food Classification.ipynb for NumPy MLP training
4. All loss curves, confusion matrices, ROC charts auto-saved to exp_output/
5. Cross-model comparison table generated for group report analysis

### Group Contributors & Student IDs
- Selena Ji — SVM Model
- Estelle Ren — Random Forest Model
- Kiki Gao — NumPy MLP Model
