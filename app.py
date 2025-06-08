import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")
st.set_page_config(page_title="Loan Eligibility App", layout="centered")
st.title("🏦 Loan Eligibility Prediction")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Amount Term (months)", [12, 36, 60, 120, 180, 240, 300, 360])
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Encode inputs
def encode_input():
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    dependents_val = 3 if dependents == "3+" else int(dependents)
    education_val = 0 if education == "Graduate" else 1
    self_employed_val = 1 if self_employed == "Yes" else 0
    property_area_val = {"Urban": 2, "Rural": 0, "Semiurban": 1}[property_area]

    return [gender_val, married_val, dependents_val, education_val, self_employed_val,
            applicant_income, coapplicant_income, loan_amount, loan_term,
            credit_history, property_area_val]

input_data = np.array(encode_input()).reshape(1, -1)

# Predict
if st.button("Check Eligibility"):
    prediction = model.predict(input_data)
    st.subheader("✅ Loan Approved" if prediction[0] == 1 else "❌ Loan Not Approved")
