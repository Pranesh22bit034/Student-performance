import streamlit as st
import pandas as pd
import pickle

# Load Model
@st.cache_resource
def load_assets():
    model = pickle.load(open("student_model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    return model, encoders

model, label_encoders = load_assets()

st.title("Student Performance Predictor 🎓")

st.write("Enter student details to predict overall score.")

# UI
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 14, 19, 16)
    gender = st.selectbox("Gender", ["male","female","other"])
    school_type = st.selectbox("School Type", ["public","private"])
    parent_education = st.selectbox(
        "Parent Education",
        ["no formal","high school","diploma","graduate","post graduate","phd"]
    )
    internet_access = st.selectbox("Internet Access", ["yes","no"])

with col2:
    study_hours = st.slider("Study Hours", 0.5, 8.0, 3.0)
    attendance_percentage = st.slider("Attendance %", 50.0, 100.0, 85.0)
    travel_time = st.selectbox("Travel Time", ["<15 min","15-30 min","30-60 min",">60 min"])
    extra_activities = st.selectbox("Extracurricular", ["yes","no"])
    study_method = st.selectbox(
        "Study Method",
        ["notes","textbook","group study","coaching","mixed","online videos"]
    )

st.subheader("Subject Scores")

col3, col4, col5 = st.columns(3)

with col3:
    math_score = st.number_input("Math",0.0,100.0,65.0)

with col4:
    science_score = st.number_input("Science",0.0,100.0,65.0)

with col5:
    english_score = st.number_input("English",0.0,100.0,65.0)


# Prediction
if st.button("Predict Overall Score"):

    input_df = pd.DataFrame({
        "age":[age],
        "sex":[gender],
        "school_type":[school_type],
        "parent_education":[parent_education],
        "study_time":[study_hours],
        "attendance_percent":[attendance_percentage],
        "internet_access":[internet_access],
        "travel_time":[travel_time],
        "extracurricular":[extra_activities],
        "study_method":[study_method],
        "math_score":[math_score],
        "science_score":[science_score],
        "english":[english_score]
    })

    # Apply label encoding
    for column, encoder in label_encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    prediction = model.predict(input_df)

    st.success(f"Predicted Overall Score: {prediction[0]:.2f}")
