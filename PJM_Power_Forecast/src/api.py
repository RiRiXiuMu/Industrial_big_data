"""
FastAPI后端服务实现
对应第5章 系统实现
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from typing import Optional
from datetime import datetime

# 导入自定义模块
from data_loader import load_data, preprocess_data
from xgb_model import XGBForecaster
from lstm_model import LSTMWrapper

app = FastAPI(title="PJM电力负荷预测系统")

# 加载预训练模型
MODELS = {
    "xgb": XGBForecaster(),
    "lstm": LSTMWrapper()
}

class PredictionRequest(BaseModel):
    model_type: str = "xgb"
    start_date: str
    end_date: str

@app.on_event("startup")
async def load_models():
    """启动时加载模型"""
    # 这里应该替换为实际的模型加载逻辑
    print("Loading pre-trained models...")

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = "xgb"):
    """预测端点"""
    try:
        # 1. 加载和预处理数据
        df = load_data(file.file)
        df_processed, scaler = preprocess_data(df)
        
        # 2. 调用模型预测
        model = MODELS[model.lower()]
        # 确保使用处理后的数据列
        predictions = model.predict(df_processed[['PJM_Load', 'hour', 'day_of_week', 'month']].values)
        
        # 3. 反标准化结果
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # 4. 准备响应数据
        timestamps = pd.date_range(
            start=df.index[-1] + pd.Timedelta(hours=1),
            periods=24,
            freq='H'
        )
        
        return JSONResponse({
            "timestamps": [str(ts) for ts in timestamps],
            "predictions": predictions.tolist(),
            "model": model
        })
    
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=400
        )

@app.get("/plot")
async def plot_predictions(request: PredictionRequest):
    """可视化预测结果"""
    try:
        # 1. 获取数据
        df = load_data("data/raw/PJM_Load_hourly.csv")
        df = df.loc[request.start_date:request.end_date]
        # 确保使用正确的列名
        if 'PJM_Load' not in df.columns:
            df = df.rename(columns={'PJM_Load_MW': 'PJM_Load'})
        
        # 2. 预测
        predictions = MODELS[request.model_type].predict(df.values)
        
        # 3. 创建可视化
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df.values, label="Actual")
        plt.plot(pd.date_range(
            start=df.index[-1] + pd.Timedelta(hours=1),
            periods=24,
            freq='H'
        ), predictions, label="Predicted")
        plt.legend()
        
        # 4. 保存并返回图片
        img_path = "static/prediction_plot.png"
        plt.savefig(img_path)
        plt.close()
        
        return FileResponse(img_path)
        
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=400
        )

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
