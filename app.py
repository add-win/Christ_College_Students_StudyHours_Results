import streamlit as st
import joblib
import pandas as pd

model = joblib.load("log_reg_model.pkl")

st.title("Student Pass / Fail Based on Study Hours")

hours = st.number_input(
    "Enter the Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.5
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "StudyHours": [hours]
    })

    prediction = model.predict(input_data)

    probabilities = model.predict_proba(input_data)

    fail_probability = probabilities[0][0]
    pass_probability = probabilities[0][1]

    if prediction[0] == 1:
        st.success("🎉 Student is likely to PASS!")
        st.info(f"Pass Probability: {pass_probability * 100:.2f}%")
    else:
        st.error("❌ Student is likely to FAIL!")
        st.info(f"Fail Probability: {fail_probability * 100:.2f}%")
