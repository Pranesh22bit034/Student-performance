import streamlit as st
import pandas as pd
import pickle

# --- 1. Load the Model and Encoders ---
@st.cache_resource
def load_assets():
    model = pickle.load(open("student_model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    return model, encoders

try:
    model, label_encoders = load_assets()
except FileNotFoundError:
    st.error("Model or encoders not found! Please run your training script first.")
    st.stop()

st.title("Student Performance Predictor 🎓")
st.write("Enter the student's details below to predict their overall score.")

# --- 2. Build the User Interface (All 13 Features) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Background")
    age = st.number_input("Age", min_value=14, max_value=19, value=16)
    gender = st.selectbox("Gender", ["male", "female", "other"])
    school_type = st.selectbox("School Type", ["public", "private"])
    parent_education = st.selectbox(
        "Parent Education", 
        ["no formal", "high school", "diploma", "graduate", "post graduate", "phd"]
    )
    internet_access = st.radio("Internet Access at Home", ["yes", "no"])

with col2:
    st.subheader("Habits & Extracurriculars")
    study_hours = st.slider("Study Hours (Daily)", min_value=0.5, max_value=8.0, value=3.0, step=0.1)
    attendance_percentage = st.slider("Attendance (%)", min_value=50.0, max_value=100.0, value=85.0)
    travel_time = st.selectbox("Travel Time to School", ["<15 min", "15-30 min", "30-60 min", ">60 min"])
    extra_activities = st.radio("Extracurricular Activities", ["yes", "no"])
    study_method = st.selectbox(
        "Primary Study Method", 
        ["notes", "textbook", "group study", "coaching", "mixed", "online videos"]
    )

st.subheader("Current Academic Scores")
col3, col4, col5 = st.columns(3)
with col3:
    math_score = st.number_input("Math Score", min_value=0.0, max_value=100.0, value=65.0)
with col4:
    science_score = st.number_input("Science Score", min_value=0.0, max_value=100.0, value=65.0)
with col5:
    english_score = st.number_input("English Score", min_value=0.0, max_value=100.0, value=65.0)

# --- 3. Prediction Logic ---
if st.button("Predict Overall Score", type="primary"):
    
    # EXACT match to the 13 columns used during training
    # EXACT match to the features the model was trained on
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [gender],                               # Changed from gender
        "school_type": [school_type],
        "parent_education": [parent_education],
        "study_time": [study_hours],                   # Changed from study_hours
        "attendance_percent": [attendance_percentage], # Changed from attendance_percentage
        "internet_access": [internet_access],
        "travel_time": [travel_time],
        "extracurricular": [extra_activities],         # Changed from extra_activities
        "study_method": [study_method],
        "math_score": [math_score],
        "science_score": [science_score],
        "english": [english_score]                     # Changed from english_score
    })
    
    try:
        # Apply the exact Label Encoders saved during training
        for column, le in label_encoders.items():
            if column in input_data.columns:
                input_data[column] = le.transform(input_data[column])
                
        # Make the Prediction
        prediction = model.predict(input_data)
        
        # Display the Result
        st.success(f"### Predicted Overall Score: {prediction[0]:.2f}")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
