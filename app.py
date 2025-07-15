import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎯 Income Classification App")
st.markdown("Predict whether an individual's income is **>50K** or **<=50K** based on your inputs.")

# === Mappings ===
workclass_map = {
    "Private": 0, "Self_emp_not_inc": 1, "Self_emp_inc": 2,
    "Federal_gov": 3, "Local_gov": 4, "State_gov": 5, "Without_pay": 6
}
education_map = {
    "Bachelors": 0, "HS_grad": 1, "Some_college": 2, "Masters": 3,
    "Assoc_acdm": 4, "Assoc_voc": 5, "Doctorate": 6, "School": 7
}
marital_map = {
    "Married": 0, "Not_Married": 1, "Previously_Married": 2
}
occupation_map = {
    "Tech_support": 0, "Craft_repair": 1, "Other_service": 2, "Sales": 3,
    "Exec_managerial": 4, "Prof_specialty": 5, "Handlers_cleaners": 6,
    "Machine_op_inspct": 7, "Adm_clerical": 8, "Farming_fishing": 9,
    "Transport_moving": 10, "Priv_house_serv": 11, "Protective_serv": 12,
    "Armed_Forces": 13
}
relationship_map = {
    "Husband": 0, "Not_in_family": 1, "Own_child": 2,
    "Unmarried": 3, "Wife": 4, "Other_relative": 5
}
race_map = {
    "White": 0, "Other": 1
}
sex_map = {"Male": 0, "Female": 1}
native_country_map = {
    "United_States": 0, "Other": 1
}

# === User Inputs ===
age = st.slider("Age", 18, 90, 30)
workclass = st.selectbox("Workclass", list(workclass_map.keys()))
education = st.selectbox("Education", list(education_map.keys()))
marital_status = st.selectbox("Marital Status", list(marital_map.keys()))
occupation = st.selectbox("Occupation", list(occupation_map.keys()))
relationship = st.selectbox("Relationship", list(relationship_map.keys()))
race = st.selectbox("Race", list(race_map.keys()))
sex = st.selectbox("Sex", list(sex_map.keys()))
native_country = st.selectbox("Native Country", list(native_country_map.keys()))
hours_per_week = st.slider("Hours per Week", 1, 100, 40)

# === Fixed values ===
fnlwgt = 189664
capital_gain = 0
capital_loss = 0

# === Assemble Input ===
input_data = np.array([[
    workclass_map[workclass],
    education_map[education],
    marital_map[marital_status],
    occupation_map[occupation],
    relationship_map[relationship],
    race_map[race],
    sex_map[sex],
    native_country_map[native_country],
    age,
    fnlwgt,
    capital_gain,
    capital_loss,
    hours_per_week
]])

# === Scale and Predict ===
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    label = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: **{label}**")
