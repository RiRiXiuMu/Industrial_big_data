"""
XGBoost预测模型实现
对应第3章 基于XGBoost预测设计与实现
"""
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

class XGBForecaster:
    def __init__(self, n_estimators=1000, max_depth=6, learning_rate=0.1):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            early_stopping_rounds=50
        )
        self.feature_importances_ = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """模型训练"""
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=10
        )
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X):
        """预测未来24小时负荷"""
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        """模型评估"""
        y_pred = self.predict(X)
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': self.model.score(X, y_true)
        }

    def plot_feature_importance(self, feature_names=None):
        """可视化特征重要性"""
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(len(self.feature_importances_))]
            
        sorted_idx = np.argsort(self.feature_importances_)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_idx)), 
                self.feature_importances_[sorted_idx],
                align='center')
        plt.xticks(range(len(sorted_idx)), 
                  np.array(feature_names)[sorted_idx], 
                  rotation=90)
        plt.title('Feature Importance')
        plt.tight_layout()
        return plt
