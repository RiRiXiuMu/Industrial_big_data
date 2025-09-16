"""
基于PyTorch的LSTM模型实现
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=24):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)  # out shape: (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步
        return out

class LSTMWrapper:
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        # 转换数据为PyTorch张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            
            # 打印训练进度
            if (epoch+1) % 10 == 0:
                train_loss = loss.item()
                val_loss = self.evaluate(X_val, y_val)['mse'] if X_val is not None else None
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}" + 
                      (f", Val Loss: {val_loss:.4f}" if val_loss else ""))

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            preds = self.model(X).cpu().numpy()
        return preds
    
    def evaluate(self, X, y):
        if X is None or y is None:
            return {}
            
        y = torch.FloatTensor(y).cpu().numpy()
        preds = self.predict(X)
        
        return {
            'mae': mean_absolute_error(y, preds),
            'rmse': np.sqrt(mean_squared_error(y, preds)),
            'mse': mean_squared_error(y, preds)
        }
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
