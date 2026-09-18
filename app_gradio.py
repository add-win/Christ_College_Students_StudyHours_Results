import gradio as gr
import joblib
import pandas as pd
import os

# Load model
model = joblib.load("log_reg_model.pkl")


def predict_result(study_hours):

    input_data = pd.DataFrame({
        "StudyHours": [study_hours]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        result = "PASS"
        confidence = probability[1] * 100
    else:
        result = "FAIL"
        confidence = probability[0] * 100

    return f"Student Result: {result}\nProbability: {confidence:.2f}%"


demo = gr.Interface(
    fn=predict_result,
    inputs=gr.Number(
        label="Enter Study Hours",
        minimum=0,
        maximum=24,
        value=5
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="Student Result Prediction",
    description="Predict Pass or Fail based on Study Hours."
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
