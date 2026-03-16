import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load Model, Encoders, Features
# -----------------------------
@st.cache_resource
def load_assets():
    model = pickle.load(open("model/student_model.pkl", "rb"))
    encoders = pickle.load(open("model/encoders.pkl", "rb"))
    features = pickle.load(open("model/features.pkl", "rb"))
    return model, encoders, features

model, encoders, features = load_assets()

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Performance Predictor")

st.write("Enter student details to predict overall score.")

# -----------------------------
# Input UI
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 14, 19, 16)

    gender = st.selectbox(
        "Gender",
        ["male", "female", "other"]
    )

    school_type = st.selectbox(
        "School Type",
        ["public", "private"]
    )

    parent_education = st.selectbox(
        "Parent Education",
        ["no formal", "high school", "diploma",
         "graduate", "post graduate", "phd"]
    )

    internet_access = st.selectbox(
        "Internet Access",
        ["yes", "no"]
    )

with col2:
    study_hours = st.slider(
        "Study Hours",
        0.5, 8.0, 3.0
    )

    attendance_percentage = st.slider(
        "Attendance %",
        50.0, 100.0, 85.0
    )

    travel_time = st.selectbox(
        "Travel Time",
        ["<15 min", "15-30 min", "30-60 min", ">60 min"]
    )

    extra_activities = st.selectbox(
        "Extracurricular Activities",
        ["yes", "no"]
    )

    study_method = st.selectbox(
        "Study Method",
        ["notes", "textbook", "group study",
         "coaching", "mixed", "online videos"]
    )

# -----------------------------
# Subject Scores
# -----------------------------
st.subheader("Subject Scores")

c1, c2, c3 = st.columns(3)

with c1:
    math_score = st.number_input(
        "Math Score",
        0.0, 100.0, 65.0
    )

with c2:
    science_score = st.number_input(
        "Science Score",
        0.0, 100.0, 65.0
    )

with c3:
    english_score = st.number_input(
        "English Score",
        0.0, 100.0, 65.0
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Overall Score"):

    input_data = pd.DataFrame({
        "age":[age],
        "gender":[gender],
        "school_type":[school_type],
        "parent_education":[parent_education],
        "study_hours":[study_hours],
        "attendance_percentage":[attendance_percentage],
        "internet_access":[internet_access],
        "travel_time":[travel_time],
        "extra_activities":[extra_activities],
        "study_method":[study_method],
        "math_score":[math_score],
        "science_score":[science_score],
        "english_score":[english_score]
    })

    # Encode categorical columns
    for col, encoder in encoders.items():
        if col in input_data.columns:
            input_data[col] = encoder.transform(input_data[col])

    # BONUS LINE (Fix feature order)
    input_data = input_data[features]

    # Predict
    prediction = model.predict(input_data)

    st.success(f"Predicted Overall Score: {prediction[0]:.2f}")
