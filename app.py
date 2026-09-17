import Streamlit as st
import joblib

model = joblib.load("log_reg_model.pkl")

st.title("Student Pass / Fail Based on Study Hours")
hours = st.number_input("Enter the Study Hours", min_value = 0.0, max_value = 15.0, value = 5.0)

if st.button("Predict"):
    
    prediction = model.predict([[hours]])

    if prediction[0] == 1:
        st.success("🎉 Student is likely to PASS!")
    else:
        st.error("❌ Student is likely to FAIL!")
