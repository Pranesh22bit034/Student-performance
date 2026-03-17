import streamlit as st
import pandas as pd
import pickle
import os

# 1. Setup absolute paths to avoid FileNotFoundError in the cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "encoders.pkl")

# 2. Load model and encoders using the dynamic paths
@st.cache_resource
def load_model():
    model = pickle.load(open(MODEL_PATH, "rb"))
    encoders = pickle.load(open(ENCODER_PATH, "rb"))
    return model, encoders

model, encoders = load_model()

# 3. App Header
st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict overall score.")

# 4. UI Layout - Demographics & Background
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 14, 19, 16)
    gender = st.selectbox("Gender", ["male", "female", "other"])
    school_type = st.selectbox("School Type", ["public", "private"])
    parent_education = st.selectbox(
        "Parent Education",
        ["no formal", "high school", "diploma", "graduate", "post graduate", "phd"]
    )
    internet_access = st.selectbox("Internet Access", ["yes", "no"])

with col2:
    study_hours = st.slider("Study Hours", 0.5, 8.0, 3.0)
    attendance_percentage = st.slider("Attendance %", 50.0, 100.0, 85.0)
    travel_time = st.selectbox(
        "Travel Time",
        ["<15 min", "15-30 min", "30-60 min", ">60 min"]
    )
    extra_activities = st.selectbox("Extracurricular", ["yes", "no"])
    study_method = st.selectbox(
        "Study Method",
        ["notes", "textbook", "group study", "coaching", "mixed", "online videos"]
    )

# 5. UI Layout - Subject Scores
st.subheader("Subject Scores")
c1, c2, c3 = st.columns(3)

with c1:
    math_score = st.number_input("Math Score", 0.0, 100.0, 65.0)

with c2:
    science_score = st.number_input("Science Score", 0.0, 100.0, 65.0)

with c3:
    english_score = st.number_input("English Score", 0.0, 100.0, 65.0)

# 6. Prediction Logic
if st.button("Predict Score"):
    
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "school_type": [school_type],
        "parent_education": [parent_education],
        "study_hours": [study_hours],
        "attendance_percentage": [attendance_percentage],
        "internet_access": [internet_access],
        "travel_time": [travel_time],
        "extra_activities": [extra_activities],
        "study_method": [study_method],
        "math_score": [math_score],
        "science_score": [science_score],
        "english_score": [english_score]
    })

    # Encode categorical columns using the loaded encoders dictionary
    for col, encoder in encoders.items():
        if col in input_data.columns:
            input_data[col] = encoder.transform(input_data[col])

    # Generate prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(f"Predicted Overall Score: {prediction[0]:.2f}")
