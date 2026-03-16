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