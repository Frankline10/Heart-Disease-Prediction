# Importing Libraries

import streamlit as st
import numpy as np
import pickle


# Loading Files

model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

with open("models/columns.pkl", "rb") as f:
    model_columns = pickle.load(f)


# App Title

st.title("Heart Disease Prediction App")
st.write("Enter patient details below to predict heart disease.")


# User Inputs

age = st.number_input("Age", min_value=1, max_value=120, step=1)

sex = st.selectbox("Sex", ["Male", "Female"])

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]
)

resting_bp = st.number_input("Resting Blood Pressure", step=1)

cholesterol = st.number_input("Serum Cholesterol (mg/dl)", step=1)

fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

rest_ecg = st.selectbox(
    "Resting ECG",
    ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"]
)

max_hr = st.number_input("Max Heart Rate Achieved", step=1)

exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

oldpeak = st.number_input("Oldpeak (ST depression)", step=0.1)

slope = st.selectbox(
    "Slope of Peak Exercise",
    ["upsloping", "flat", "downsloping"]
)

ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3)

thal = st.selectbox(
    "Thal",
    ["normal", "fixed_defect", "reversible_defect"]
)


# Converting inputs using manual mapping

# Binary mappings
sex = 1 if sex == "Male" else 0
exercise_angina = 1 if exercise_angina == "Yes" else 0

# Example mappings (UPDATE based on your extracted mappings)

chest_pain_map = {
    "typical angina": 0,
    "atypical angina": 1,
    "non-anginal pain": 2,
    "asymptomatic": 3
}

rest_ecg_map = {
    "normal": 0,
    "ST-T wave abnormality": 1,
    "left ventricular hypertrophy": 2
}

slope_map = {
    "upsloping": 0,
    "flat": 1,
    "downsloping": 2
}

thal_map = {
    "fixed_defect": 0,
    "normal": 1,
    "reversible_defect": 2
}

chest_pain_type = chest_pain_map[chest_pain_type]
rest_ecg = rest_ecg_map[rest_ecg]
slope = slope_map[slope]
thal = thal_map[thal]


# Prediction

if st.button("Predict"):

    import pandas as pd

    input_dict = {
            'slope_of_peak_exercise_st_segment': slope,
            'thal': thal,
            'resting_blood_pressure': resting_bp,
            'chest_pain_type': chest_pain_type,
            'num_major_vessels': ca,
            'fasting_blood_sugar_gt_120_mg_per_dl': fasting_bs,
            'resting_ekg_results': rest_ecg,
            'serum_cholesterol_mg_per_dl': cholesterol,
            'oldpeak_eq_st_depression': oldpeak,
            'sex': sex,
            'age': age,
            'max_heart_rate_achieved': max_hr,
            'exercise_induced_angina': exercise_angina
        }

    input_df = pd.DataFrame([input_dict])

    input_df = input_df[model_columns]

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)

    # Output
    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")