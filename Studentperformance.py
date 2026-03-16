import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# 1. Load the dataset
data = pd.read_csv("Student_Performance.csv")

# 2. Drop irrelevant columns
# 'student_id' is just an identifier, and 'final_grade' is a letter grade version of our target
data = data.drop(["student_id", "final_grade"], axis=1)

# 3. Separate features (X) and target (y)
y = data["overall_score"]
X = data.drop("overall_score", axis=1)
pickle.dump(X.columns.tolist(), open("model/features.pkl","wb"))

print("Features being trained on:", X.columns.tolist())

# 4. Encode Categorical Variables
label_encoders = {}

for column in X.columns:
    # Check if the column contains text/categorical data
    if X[column].dtype == "object":
        le = LabelEncoder()
        # Fit the encoder to the data and transform the column to numbers
        X[column] = le.fit_transform(X[column])
        # Save the encoder for this specific column so we can use it in Streamlit
        label_encoders[column] = le

# 5. Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Initialize and Train the Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

print("Training the model...")
model.fit(X_train, y_train)

# 7. Evaluate the Model
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(f"Model Performance (R2 Score): {score:.4f}")

# 8. Save the Model and Encoders for Streamlit
pickle.dump(model, open("student_model.pkl", "wb"))
pickle.dump(label_encoders, open("encoders.pkl", "wb"))

