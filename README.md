# PJM电力负荷预测系统

## 项目简介

本项目实现了一个基于机器学习的电力负荷预测系统，使用XGBoost和CatBoost等算法对PJM电力市场负荷数据进行预测。系统提供API服务和可视化功能。

## 功能特点

- 支持XGBoost和CatBoost两种预测模型
- 提供REST API接口进行预测
- 支持数据可视化
- 包含完整的模型训练、评估和调优流程
- 支持五种不同特征集的实验对比

## 安装指南

### 依赖安装

```bash
pip install -r requirements.txt
```

### 主要依赖

- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- xgboost>=1.5.0
- scikit-learn>=0.24.0
- fastapi>=0.68.0
- uvicorn>=0.15.0
- catboost

## 使用方法

### 启动API服务

```bash
python src/api.py
```

### 使用Jupyter Notebook进行实验

1. 启动Jupyter Notebook
2. 打开notebooks/目录下的相应文件：
   - 01_数据探索.ipynb - 数据分析和可视化
   - 02_XGBoost预测.ipynb - XGBoost模型实验
   - 03_CatBoost预测.ipynb - CatBoost模型实验

## 项目结构

```
PJM_Power_Forecast/
├── analysis_results/        # 分析结果图表
├── data/                    # 数据文件
├── hyperparameter_tuning/   # 超参数调优结果
├── models/                  # 训练好的模型文件
├── notebooks/               # Jupyter Notebook实验文件
│   ├── 01_数据探索.ipynb
│   ├── 02_XGBoost预测.ipynb
│   └── 03_CatBoost预测.ipynb
├── src/                     # 源代码
│   ├── api.py               # FastAPI服务
│   ├── data_loader.py       # 数据加载和预处理
│   ├── lstm_model.py        # LSTM模型实现
│   └── xgb_model.py         # XGBoost模型实现
└── visualizations/          # 可视化结果
```

## 模型说明

### XGBoost模型

- 实现文件: src/xgb_model.py
- 默认参数:
  - n_estimators=1000
  - max_depth=6
  - learning_rate=0.1
- 支持特征重要性分析

### CatBoost模型

- 实现文件: notebooks/03_CatBoost预测.ipynb
- 支持五种特征集实验:
  1. 基础实验
  2. 全部特征
  3. 消融实验
  4. 增强实验
  5. 综合实验
- 包含超参数调优功能

## 结果展示

系统生成的预测结果和可视化图表保存在以下目录:
- analysis_results/ - 数据分析图表
- visualizations/ - 预测结果可视化
- hyperparameter_tuning/ - 超参数调优结果
