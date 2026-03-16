import streamlit as st
import pandas as pd
import pickle

# --- 1. Load the Model and Encoders ---
@st.cache_resource
def load_assets():
    model = pickle.load(open("student_model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    return model, encoders

model, label_encoders = load_assets()

st.title("Student Performance Predictor 🎓")
st.write("Enter the student's details below to predict their overall score.")

# --- 2. Build the User Interface ---
# Organizing inputs into columns for a cleaner UI layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & School")
    age = st.number_input("Age", min_value=14, max_value=19, value=16)
    gender = st.selectbox("Gender", ["male", "female", "other"])
    school_type = st.selectbox("School Type", ["public", "private"])
    parent_education = st.selectbox(
        "Parent Education", 
        ["no formal", "high school", "diploma", "graduate", "post graduate", "phd"]
    )
    internet_access = st.radio("Internet Access at Home", ["yes", "no"])

with col2:
    st.subheader("Study Habits & Activities")
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
    
    # 3.1 Create a DataFrame ensuring exact column order matching X_train
    # Inside your app.py, under the 'if st.button("Predict"):' section:
    
    # EXACT match to the 13 columns from X_train
    input_data = pd.DataFrame({
        "age": [age],                                   # Was missing
        "gender": [gender],                             # Was missing
        "school_type": [school_type],
        "parent_education": [parent_education],
        "study_hours": [study_hours],
        "attendance_percentage": [attendance_percentage], # Fixed from "attendance"
        "internet_access": [internet_access],
        "travel_time": [travel_time],
        "extra_activities": [extra_activities],         # Was missing
        "study_method": [study_method],
        "math_score": [math_score],
        "science_score": [science_score],
        "english_score": [english_score]                # Was missing
    })
    
    # 3.2 Apply the Label Encoders
    try:
        for column, le in label_encoders.items():
            if column in input_data.columns:
                # Use transform to map inputs based on how it was trained
                input_data[column] = le.transform(input_data[column])
                
        # 3.3 Make the Prediction
        prediction = model.predict(input_data)
        
        # Display the result
        st.success(f"### Predicted Overall Score: {prediction[0]:.2f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")