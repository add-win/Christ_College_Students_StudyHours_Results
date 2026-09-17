import streamlit as st
import joblib
import pandas as pd

model = joblib.load("log_reg_hours_model.pkl")

st.title("Student Pass / Fail Prediction")

hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

attendance = st.number_input(
    "Enter Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "StudyHours": [hours],
        "Attendance": [attendance]
    })

    prediction = model.predict(input_data)

    probabilities = model.predict_proba(input_data)

    pass_probability = probabilities[0][1]

    if prediction[0] == 1:
        st.success("🎉 Student is likely to PASS!")
    else:
        st.error("❌ Student is likely to FAIL!")

    st.write(f"✅ Pass Probability: {pass_probability * 100:.2f}%")
