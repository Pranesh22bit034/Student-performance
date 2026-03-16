from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("student_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.route("/")
def home():
    return "Student Score Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    # Encode categorical variables
    for column in df.columns:
        if column in encoders:
            df[column] = encoders[column].transform(df[column])

    prediction = model.predict(df)

    return jsonify({
        "Predicted Overall Score": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
