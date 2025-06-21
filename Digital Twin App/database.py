import os
import sqlite3

# Path to database

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "database.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop existing tables to start fresh
    cursor.execute("DROP TABLE IF EXISTS patients")
    cursor.execute("DROP TABLE IF EXISTS doctors")

    # Create doctors table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS doctors (
            doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            Username TEXT UNIQUE,
            Password TEXT
        )
        """
    )

    # Create patients table with new doctor_id column
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER,
            Name TEXT,
            Age INTEGER,
            Sex TEXT,
            Weight REAL,
            Blood_Type TEXT,
            Contact TEXT,
            Family_History TEXT,
            White_Cells INTEGER,
            Cholesterol REAL,
            Airway_Resistance REAL,
            MRI TEXT,
            SMOKING INTEGER,
            ALCOHOL INTEGER,
            PRIOR_CMP INTEGER,
            CKD INTEGER,
            Glucose REAL,
            UREA REAL,
            CREATININE REAL,
            BNP INTEGER,
            RAISED_CARDIAC_ENZYMES INTEGER,
            ACS INTEGER,
            STEMI INTEGER,
            HEART_FAILURE INTEGER,
            HFREF INTEGER,
            HFNEF INTEGER,
            VALVULAR INTEGER,
            CHB INTEGER,
            SSS INTEGER,
            AKI INTEGER,
            AF INTEGER,
            VT INTEGER,
            CARDIOGENIC_SHOCK INTEGER,
            PULMONARY_EMBOLISM INTEGER,
            CHEST_INFECTION TEXT,
            BMI REAL,
            DM_Y INTEGER,
            HTN_Y INTEGER,
            Obesity TEXT,
            DLP TEXT,
            Function_class INTEGER,
            FBS INTEGER,
            CR REAL,
            TG INTEGER,
            LDL INTEGER,
            HDL INTEGER,
            BUN INTEGER,
            HB_Y INTEGER,
            VHD TEXT,
            FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id)
        )
        """
    )

    # Insert default doctors
    cursor.execute(
        """
        INSERT INTO doctors (Username, Password)
        VALUES (?, ?)
        """,
        ("drsmith", "securePass123"),
    )

    cursor.execute(
        """
        INSERT INTO doctors (Username, Password)
        VALUES (?, ?)
        """,
        ("drjones", "anotherPass456"),
    )

    # Get doctor_id for drsmith
    cursor.execute("SELECT doctor_id FROM doctors WHERE Username = ?", ("drsmith",))
    doctor1_id = cursor.fetchone()[0]

    # Insert patient under doctor 1
    cursor.execute(
        """
        INSERT INTO patients (
        doctor_id, Name, Age, Sex, Weight, Blood_Type, Contact, Family_History, White_Cells, Cholesterol, Airway_Resistance, MRI,
        SMOKING, ALCOHOL, PRIOR_CMP, CKD, Glucose, UREA, CREATININE, BNP, RAISED_CARDIAC_ENZYMES,
        ACS, STEMI, HEART_FAILURE, HFREF, HFNEF, VALVULAR, CHB, SSS, AKI, AF, VT,
        CARDIOGENIC_SHOCK, PULMONARY_EMBOLISM, CHEST_INFECTION ,BMI, DM_Y, HTN_Y, Obesity, DLP, Function_class,
        FBS, CR, TG, LDL, HDL, BUN, HB_Y, VHD
    )
    VALUES (?,?,?,?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
        doctor1_id,             # doctor_id
        "Alice Smith",          # Name
        50,                     # Age
        "Female",               # Sex
        59,                     # Weight
        "A",                    # Blood_Type
        "123-123-1234",         # Contact
        "Hypertension (Father)",# Family_History
        6500,                   # White_Cells
        180,                    # Cholesterol
        0.9,                    # Airway_Resistance
        "processed_7845.png",   # MRI
        0,                      # SMOKING
        0,                      # ALCOHOL
        0,                      # PRIOR_CMP
        0,                      # CKD
        102,                    # Glucose
        40,                     # UREA
        0.87,                   # CREATININE
        330,                    # BNP
        0,                      # RAISED_CARDIAC_ENZYMES
        0,                      # ACS
        0,                      # STEMI
        0,                      # HEART_FAILURE
        0,                      # HFREF
        0,                      # HFNEF
        0,                      # VALVULAR
        0,                      # CHB
        0,                      # SSS
        0,                      # AKI
        0,                      # AF
        0,                      # VT
        0,                      # CARDIOGENIC_SHOCK
        0,                      # PULMONARY_EMBOLISM
        "0" ,                      # CHEST_INFECTION
        22.761,                 # BMI
        0,                      # DM_Y
        0,                      # HTN_Y
        "N",                    # Obesity
        "Y",                    # DLP
        0,                    # Function_class 
        84,                     # FBS
        0.9,                    # CR
        114,                    # TG
        91,                     # LDL
        31,                     # HDL
        23,                     # BUN
        13,                     # HB_Y
        "N"                     # VHD
    ),
    )

    cursor.execute(
        """
    INSERT INTO patients (
        doctor_id, Name, Age, Sex, Weight, Blood_Type, Contact, Family_History, White_Cells, Cholesterol, Airway_Resistance, MRI,
        SMOKING, ALCOHOL, PRIOR_CMP, CKD, Glucose, UREA, CREATININE, BNP, RAISED_CARDIAC_ENZYMES,
        ACS, STEMI, HEART_FAILURE, HFREF, HFNEF, VALVULAR, CHB, SSS, AKI, AF, VT,
        CARDIOGENIC_SHOCK, PULMONARY_EMBOLISM, CHEST_INFECTION ,BMI, DM_Y, HTN_Y, Obesity, DLP, Function_class,
        FBS, CR, TG, LDL, HDL, BUN, HB_Y, VHD
    )
    VALUES (?,?,?,?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doctor1_id,
            "John Doe",
            80,
            "Male",
            74,
            "B",
            "555-555-5555",
            "Heart Disease (Brother)",
            8500,
            220,
            2.5,
            "processed_0.png",
            0,                      # SMOKING
            0,                      # ALCOHOL
            0,                      # PRIOR_CMP
            0,                      # CKD
            239,                    # Glucose
            26,                     # UREA
            0.7,                    # CREATININE
            367,                    # BNP
            1,                      # RAISED_CARDIAC_ENZYMES
            1,                      # ACS
            0,                      # STEMI
            1,                      # HEART_FAILURE
            0,                      # HFREF (conflicts with EF=36%)
            1,                      # HFNEF (conflicts with EF=36%)
            0,                      # VALVULAR
            0,                      # CHB
            0,                      # SSS
            0,                      # AKI
            0,                      # AF
            0,                      # VT
            0,                      # CARDIOGENIC_SHOCK
            0,                      # PULMONARY_EMBOLISM
            "0",                    # CHEST_INFECTION
            24.44,                  # BMI
            0,                      # DM_Y
            1,                      # HTN_Y
            "N",                    # Obesity
            "Y",                    # DLP
            0,                    # Function_class
            79,                     # FBS
            1.4,                    # CR
            152,                    # TG
            79,                     # LDL
            37,                     # HDL
            17,                     # BUN
            12,                     # HB_Y (or Hemoglobin)
            "N",                     # VHD
                ),
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
