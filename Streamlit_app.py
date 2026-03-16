import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("student_model.pkl", "rb"))

st.title("Student Score Prediction System")

study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100)

if st.button("Predict Score"):

    df = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["study_hours", "attendance"]
    )

    prediction = model.predict(df)

    st.success(f"Predicted Score: {prediction[0]:.2f}")