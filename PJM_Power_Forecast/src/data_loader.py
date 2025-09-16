"""
数据加载与预处理模块
对应第2章 数据分析与预处理
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """加载PJM负荷数据"""
    df = pd.read_csv(file_path, parse_dates=['Datetime'])
    df = df.set_index('Datetime').sort_index()
    # 统一列名为PJM_Load
    if 'PJM_Load_MW' in df.columns:
        df = df.rename(columns={'PJM_Load_MW': 'PJM_Load'})
    return df

def detect_anomalies(df, window=24, n_sigma=3):
    """异常值检测（滑动窗口+3σ原则）"""
    rolling = df.rolling(window=window)
    mean = rolling.mean()
    std = rolling.std()
    anomalies = (df - mean).abs() > n_sigma * std
    return anomalies

def preprocess_data(df):
    """数据预处理流程"""
    # 异常值处理
    anomalies = detect_anomalies(df)
    df[anomalies] = np.nan
    df = df.fillna(method='ffill')
    
    # 标准化
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), 
                           index=df.index, 
                           columns=df.columns)
    
    # 添加时间特征
    df_scaled['hour'] = df_scaled.index.hour
    df_scaled['day_of_week'] = df_scaled.index.dayofweek
    df_scaled['month'] = df_scaled.index.month
    
    return df_scaled, scaler

def create_sequences(data, window_size=24, horizon=24):
    """创建时序样本"""
    X, y = [], []
    for i in range(len(data)-window_size-horizon):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+horizon])
    return np.array(X), np.array(y)
