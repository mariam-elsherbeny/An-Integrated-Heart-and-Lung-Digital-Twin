from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import FastAPI, File, Form, HTTPException, Body
from logic.Models import imputer, scaler, encoder, selector, model, model_path
import os
import sqlite3
import random
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# Create the FastAPI app
app = FastAPI()

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define templates early before use!
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Mount static and model directories
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/models", StaticFiles(directory=os.path.join(BASE_DIR, "models")), name="models")

# CORS middleware (allow all origins for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to database
DB_PATH = os.path.join(BASE_DIR, "database.db")


# Database connection
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Request models
class DoctorLogin(BaseModel):
    username: str
    password: str

class PatientData(BaseModel):
    heart_rate: float
    blood_pressure: float
    cholesterol: float

# Condition mapping
condition_to_model = {
    0: ("Normal", "/models/NormalMotion.glb"),
    1: ("Abnormal", "/models/Abnormal.glb"),
    2: ("Heart Failure", "/models/HeartFailure.glb"),
}

# Routes

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/index")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login")
def login(request: Request):
    return templates.TemplateResponse("login_page.html", {"request": request})

@app.get("/patients")
async def patients_page(request: Request):
    return templates.TemplateResponse("patients.html", {"request": request})

@app.get("/dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/simulation")
async def simulation(request: Request):
    return templates.TemplateResponse("simulation.html", {"request": request})

@app.get("/medical-images")
async def medical_images(request: Request):
    return templates.TemplateResponse("medical_images.html", {"request": request})

@app.get("/patient-info")
async def patient_info(request: Request):
    return templates.TemplateResponse("patient_info.html", {"request": request})

@app.get("/blood-tests")
async def blood_tests(request: Request):
    return templates.TemplateResponse("blood_tests.html", {"request": request})

@app.get("/vitals")
async def vitals(request: Request):
    return templates.TemplateResponse("vitals.html", {"request": request})

@app.get("/contactus")
async def contactus(request: Request):
    return templates.TemplateResponse("contactus.html", {"request": request})


@app.get("/api/patients")
async def get_patients(doctor_id: str):
    conn = get_db_connection()
    patients = conn.execute(
        """
        SELECT p.* 
        FROM patients p
        WHERE p.doctor_id = ?
        """, 
        (doctor_id,)
    ).fetchall()
    conn.close()

    if patients:
        return {
            "patients": [
                {
                    "patient_id": row["patient_id"],
                    "name": row["Name"],
                    "age": row["Age"],
                    "sex": row["Sex"],
                } for row in patients
            ]
        }
    return JSONResponse(
        content={"error": "Doctor not found or no patients."}, status_code=404
    )

@app.get("/api/patient/{patient_id}")
async def get_patient_details(patient_id: int):
    conn = get_db_connection()
    patient = conn.execute(
        "SELECT * FROM Patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()
    conn.close()

    if patient:
        return {
            "patient_id": patient["patient_id"],
            "name": patient["Name"],
            "EF": patient["EF"],
            "edema": patient["edema"],
            "dyspnea": patient["dyspnea"],
            "hemoglobin": patient["hemoglobin"],
            "Na": patient["Na"],
            "K": patient["K"],
            "age": patient["Age"],
            "sex": patient["Sex"],
            "weight": patient["Weight"],
            "blood_type": patient["Blood_Type"],
            "contact": patient["Contact"],
            "family_history": patient["Family_History"],
            "white_cells": patient["White_Cells"],
            "cholesterol": patient["Cholesterol"],
            "glucose": patient["Glucose"],
            "urea": patient["UREA"],
            "creatinine": patient["CREATININE"],
            "airway_resistance": patient["Airway_Resistance"],
            "mri": patient["MRI"],
            "smoking": patient["SMOKING"],
            "alcohol": patient["ALCOHOL"],
            "prior_cmp": patient["PRIOR_CMP"],
            "ckd": patient["CKD"],
            "bnp": patient["BNP"],
            "raised_cardiac_enzymes": patient["RAISED_CARDIAC_ENZYMES"],
            "acs": patient["ACS"],
            "stemi": patient["STEMI"],
            "heart_failure": patient["HEART_FAILURE"],
            "hfref": patient["HFREF"],
            "hfnef": patient["HFNEF"],
            "valvular": patient["VALVULAR"],
            "chb": patient["CHB"],
            "sss": patient["SSS"],
            "aki": patient["AKI"],
            "af": patient["AF"],
            "vt": patient["VT"],
            "cardiogenic_shock": patient["CARDIOGENIC_SHOCK"],
            "pulmonary_embolism": patient["PULMONARY_EMBOLISM"],
            "chest_infection": patient["CHEST_INFECTION"],
            "bmi": patient["BMI"],
            "dm_y": patient["DM_Y"],
            "htn_y": patient["HTN_Y"],
            "obesity": patient["Obesity"],
            "dlp": patient["DLP"],
            "function_class": patient["Function_class"],
            "fbs": patient["FBS"],
            "cr": patient["CR"],
            "tg": patient["TG"],
            "ldl": patient["LDL"],
            "hdl": patient["HDL"],
            "bun": patient["BUN"],
            "hb_y": patient["HB_Y"],
            "vhd": patient["VHD"],
        }
    return JSONResponse(
        content={"error": "Patient not found."}, status_code=404
    )

@app.post("/api/predict")
def predict(data: dict = Body(...)):
    try:
        new_patient = data
        new_patient = pd.DataFrame([new_patient])
        types = {
            "AGE": "int64",
            "SMOKING ": "int64",
            "ALCOHOL": "int64",
            "PRIOR CMP": "int64",
            "CKD": "int64",
            "GLUCOSE": "int64",
            "UREA": "int64",
            "CREATININE": "int64",
            "BNP": "int64",
            "RAISED CARDIAC ENZYMES": "int64",
            "ACS": "int64",
            "STEMI": "int64",
            "HEART FAILURE": "int64",
            "HFREF": "int64",
            "HFNEF": "int64",
            "VALVULAR": "int64",
            "CHB": "int64",
            "SSS": "int64",
            "AKI": "int64",
            "AF": "int64",
            "VT": "int64",
            "CARDIOGENIC SHOCK": "int64",
            "PULMONARY EMBOLISM": "int64",
            "CHEST INFECTION": "object",
            "Weight": "int64",
            "Sex": "object",
            "BMI": "int64",
            "DM_y": "int64",
            "HTN_y": "int64",
            "Obesity": "object",
            "DLP": "object",
            "Function Class": "int64",
            "FBS": "int64",
            "CR": "float64",
            "TG": "int64",
            "LDL": "int64",
            "HDL": "int64",
            "BUN": "int64",
            "HB_y": "float64",
            "VHD": "object",
            "EF": "float64",
            "PR": "int64",
            "BP": "int64",
            "Dyspnea": "object",
            "Edema": "int64",
            "Hemoglobin": "float64",
            "Na": "int64",
            "K": "float64",
        }
        rename_dict = {
            'AGE': 'AGE',
            'SMOKING': 'SMOKING ',
            'ALCOHOL': 'ALCOHOL',
            'PRIOR_CMP': 'PRIOR CMP',
            'CKD': 'CKD',
            'GLUCOSE': 'GLUCOSE',
            'UREA': 'UREA',
            'CREATININE': 'CREATININE',
            'BNP': 'BNP',
            'RAISED_CARDIAC_ENZYMES': 'RAISED CARDIAC ENZYMES',
            'ACS': 'ACS',
            'STEMI': 'STEMI',
            'HEART_FAILURE': 'HEART FAILURE',
            'HFREF': 'HFREF',
            'HFNEF': 'HFNEF',
            'VALVULAR': 'VALVULAR',
            'CHB': 'CHB',
            'SSS': 'SSS',
            'AKI': 'AKI',
            'AF': 'AF',
            'VT': 'VT',
            'CARDIOGENIC_SHOCK': 'CARDIOGENIC SHOCK',
            'PULMONARY_EMBOLISM': 'PULMONARY EMBOLISM',
            'CHEST_INFECTION' : 'CHEST INFECTION',
            'Weight': 'Weight',
            'Sex': 'Sex',
            'BMI': 'BMI',
            'DM_y': 'DM_y',
            'HTN_y': 'HTN_y',
            'Obesity': 'Obesity',
            'DLP': 'DLP',
            'functionClass': 'Function Class',
            'FBS': 'FBS',
            'CR': 'CR',
            'TG': 'TG',
            'LDL': 'LDL',
            'HDL': 'HDL',
            'BUN': 'BUN',
            'HB_y': 'HB_y',
            'VHD': 'VHD',
            'EF': 'EF',
            'PR': 'PR',
            'BP': 'BP',
            'dyspnea': 'Dyspnea',
            'edema': 'Edema',
            'Hemoglobin': 'Hemoglobin',
            'Na': 'Na',
            'K': 'K'
        }
        new_patient.columns = new_patient.columns.str.strip()
        new_patient = new_patient.rename(columns=rename_dict)
        new_patient = new_patient.astype(types)
        new_patient = new_patient[
            [
                "EF",
                "AGE",
                "SMOKING ",
                "ALCOHOL",
                "PRIOR CMP",
                "CKD",
                "GLUCOSE",
                "UREA",
                "CREATININE",
                "BNP",
                "RAISED CARDIAC ENZYMES",
                "ACS",
                "STEMI",
                "HEART FAILURE",
                "HFREF",
                "HFNEF",
                "VALVULAR",
                "CHB",
                "SSS",
                "AKI",
                "AF",
                "VT",
                "CARDIOGENIC SHOCK",
                "PULMONARY EMBOLISM",
                "CHEST INFECTION",
                "Weight",
                "Sex",
                "BMI",
                "DM_y",
                "HTN_y",
                "Obesity",
                "DLP",
                "BP",
                "PR",
                "Edema",
                "Dyspnea",
                "Function Class",
                "FBS",
                "CR",
                "TG",
                "LDL",
                "HDL",
                "BUN",
                "HB_y",
                "K",
                "Na",
                "VHD",
                "Hemoglobin",
            ]
        ]
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        new_imputed = imputer.transform(new_patient.select_dtypes(include='number'))
        new_scaled = scaler.fit_transform(new_imputed)
        new_cat = encoder.transform(new_patient.select_dtypes(include='object')).toarray()
        new_selected = selector.transform(new_scaled)
        new_processed = np.hstack([new_selected, new_cat])
        new_tensor = torch.tensor(new_processed, dtype=torch.float32)
        with torch.no_grad():
            output = model(new_tensor)
            predicted = torch.argmax(output, dim=1).item()
        return {"predicted": predicted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/doctor-login")
async def doctor_login(data: DoctorLogin):
    conn = get_db_connection()
    doctor = conn.execute(
        "SELECT doctor_id FROM doctors WHERE Username = ? AND Password = ?",
        (data.username, data.password),
    ).fetchone()

    if not doctor:
        conn.close()
        return JSONResponse(content={"error": "Invalid credentials"}, status_code=401)

    patients = conn.execute(
        "SELECT patient_id FROM patients WHERE doctor_id = ?", (doctor["doctor_id"],)
    ).fetchall()
    conn.close()

    patient_list = [row["patient_id"] for row in patients]
    return {"message": "Login successful", "patients": patient_list, "doctor_id": doctor["doctor_id"]}
