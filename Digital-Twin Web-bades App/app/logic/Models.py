from fastapi import HTTPException
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import RobustScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR =os.path.dirname(BASE_DIR)
# Load model components once
model_dir = os.path.join(BASE_DIR, "ml_models")
imputer = joblib.load(os.path.join(model_dir, "imputer.pkl"))
scaler = RobustScaler()
encoder = joblib.load(os.path.join(model_dir, "encoder.pkl"))
selector = joblib.load(os.path.join(model_dir, "selector.pkl"))


class ImprovedHFNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)


# Load model weights
model_path = os.path.join(model_dir, "model.pth")
model = ImprovedHFNet(43, 3) 

