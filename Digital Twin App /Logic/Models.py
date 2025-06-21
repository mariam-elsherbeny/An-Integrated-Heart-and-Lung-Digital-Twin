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

#####################################################################
def classify_scenario(patient_df: pd.DataFrame) -> str:
    """
    Given a single‐row DataFrame with the renamed columns, return one of:
      • "Normal Heart & Normal Lung"
      • "Normal Heart & Pneumonia Lung"
      • "CAD Heart & Normal Lung"
      • "CAD Heart & Pneumonia Lung"
      • "Heart Failure Heart & Normal Lung"
      • "Heart Failure Heart & Pneumonia Lung"
    """

    row = patient_df.iloc[0]

    # 1) Pull out the key flags
    HF = bool(row["HEART FAILURE"])
    CAD = bool(row["CAD"])
    CI = str(row["CHEST INFECTION"]).lower() == "yes"
    PE = bool(row["PULMONARY EMBOLISM"])

    # 2) Define “normal” ranges (tweak to your real thresholds)
    normal_ranges = {
        "BNP":       (0,   100),
        "EF":        (50,  100),
        "CREATININE":(0.6, 1.2),
        "UREA":      (7,   20),
        "Function Class": (1, 2),
        "Edema":     (0,   0),   # 0 means no edema
        "RAISED CARDIAC ENZYMES": (0, 0)
    }

    def in_normal(field):
        lo, hi = normal_ranges[field]
        return lo <= row[field] <= hi

    # 3) Encode each scenario in order
    # Scenario 1
    if (not HF and not CAD
        and in_normal("BNP") and in_normal("EF") and in_normal("RAISED CARDIAC ENZYMES")
        and not CI and not PE
        and in_normal("Function Class") and in_normal("Edema")
        and in_normal("CREATININE") and in_normal("UREA")
    ):
        return "Normal Heart & Normal Lung"

    # Scenario 2
    if (not HF and not CAD
        and (CI or PE
             or not in_normal("Function Class")
             or not in_normal("Edema")
             or not in_normal("CREATININE")
             or not in_normal("UREA")
        )
    ):
        return "Normal Heart & Pneumonia Lung"

    # Scenario 3
    if (CAD and not HF
        and in_normal("BNP") and in_normal("EF") and in_normal("RAISED CARDIAC ENZYMES")
        and not CI and not PE
    ):
        return "CAD Heart & Normal Lung"

    # Scenario 4
    if (CAD and not HF
        and (CI or PE
             or not in_normal("Edema")
             or not in_normal("Function Class")
             or not in_normal("CREATININE")
             or not in_normal("UREA")
        )
    ):
        return "CAD Heart & Pneumonia Lung"

    # Scenario 5
    if (HF
        and not CI and not PE
        and not in_normal("Function Class")  # you might interpret dyspnea/Fx-class > normal as HF sign
        and row["RAISED CARDIAC ENZYMES"]  # True if >0
        and not CAD
    ):
        return "Heart Failure Heart & Normal Lung"

    # Scenario 6
    if (HF
        and (CI or PE
             or not in_normal("Function Class")
             or not in_normal("Edema")
             or not in_normal("CREATININE")
             or not in_normal("UREA")
        )
    ):
        return "Heart Failure Heart & Pneumonia Lung"

    return "Unknown Scenario"
