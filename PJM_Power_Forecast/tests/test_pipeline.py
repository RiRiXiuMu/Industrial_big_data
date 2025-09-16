"""
端到端测试脚本
验证从数据加载到预测的完整流程
"""
import sys
import os
import numpy as np
import pandas as pd

# 添加项目根目录到PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data, preprocess_data, create_sequences
from src.xgb_model import XGBForecaster
from src.lstm_model import LSTMWrapper

def test_data_loading():
    """测试数据加载和预处理(使用实际数据)"""
    print("Testing data loading with real data...")
    try:
        # 测试加载实际数据
        loaded = load_data("data/raw/PJM_Load_hourly.csv")
        assert len(loaded) > 0, "数据文件不能为空"
        assert 'PJM_Load' in loaded.columns or 'PJM_Load_MW' in loaded.columns, "缺少负荷数据列"
        assert not loaded.isnull().values.any(), "数据包含空值"
        
        # 测试预处理
        processed, _ = preprocess_data(loaded)
        assert 'hour' in processed.columns
        assert 'day_of_week' in processed.columns
        
        # 测试序列创建
        X, y = create_sequences(processed.values)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 24  # 窗口大小
        assert y.shape[1] == 24  # 预测步长
        
        print("✅ Data loading tests passed")
        return True
    except Exception as e:
        print(f"❌ Data loading tests failed: {str(e)}")
        return False

def test_xgb_model():
    """测试XGBoost模型"""
    print("Testing XGBoost model...")
    try:
        # 创建测试数据
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_test = np.random.rand(10, 10)
        y_test = np.random.rand(10)
        
        # 测试训练和预测
        model = XGBForecaster(n_estimators=10)
        model.train(X_train, y_train, X_test, y_test)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)
        
        # 测试评估
        metrics = model.evaluate(X_test, y_test)
        assert 'mae' in metrics
        assert 'rmse' in metrics
        
        print("✅ XGBoost tests passed")
        return True
    except Exception as e:
        print(f"❌ XGBoost tests failed: {str(e)}")
        return False

def test_lstm_model():
    """测试LSTM模型"""
    print("Testing LSTM model...")
    try:
        # 创建测试数据 (batch, seq_len, features)
        X_train = np.random.rand(10, 24, 1)
        y_train = np.random.rand(10, 24)
        X_test = np.random.rand(2, 24, 1)
        y_test = np.random.rand(2, 24)
        
        # 测试训练和预测
        model = LSTMWrapper(input_size=1, hidden_size=16)
        model.train(X_train, y_train, X_test, y_test, epochs=2)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        
        # 测试评估
        metrics = model.evaluate(X_test, y_test)
        assert 'mae' in metrics
        assert 'rmse' in metrics
        
        print("✅ LSTM tests passed")
        return True
    except Exception as e:
        print(f"❌ LSTM tests failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("\nRunning end-to-end tests...")
    data_ok = test_data_loading()
    xgb_ok = test_xgb_model()
    lstm_ok = test_lstm_model()
    
    if all([data_ok, xgb_ok, lstm_ok]):
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n⚠️ Some tests failed, please check the output above")
        sys.exit(1)
