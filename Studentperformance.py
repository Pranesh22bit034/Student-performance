import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data = pd.read_csv("Student_Performance.csv")

data = data.drop(["student_id", "final_grade"], axis=1)

y = data["overall_score"]


X = data.drop("overall_score", axis=1)


label_encoders = {}

for column in X.columns:
    if X[column].dtype == "object":
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

score = r2_score(y_test, predictions)
print("Model Performance (R2 Score):", score)

pickle.dump(model, open("student_model.pkl", "wb"))
pickle.dump(label_encoders, open("encoders.pkl", "wb"))

print("Model saved successfully")


from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("student_model.pkl", "rb"))

# Load encoders
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.route("/")
def home():
    return "Student Score Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    df = pd.DataFrame([data])

    # Encode categorical columns
    for column in df.columns:
        if column in encoders:
            df[column] = encoders[column].transform(df[column])

    prediction = model.predict(df)

    return jsonify({
        "Predicted Overall Score": float(prediction[0])
    })

if __name__ == "__main__":

    import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("student_model.pkl","rb"))

st.title("Student Score Prediction")

hours = st.number_input("Study Hours")
attendance = st.number_input("Attendance")

if st.button("Predict"):
    prediction = model.predict([[hours,attendance]])
    st.success(prediction)