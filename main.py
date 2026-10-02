import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load dataset
data = pd.read_csv("data/Telco-Customer-Churn.csv")


# Clean data
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

data = data.dropna()


# Separate features and target
X = data.drop(["customerID", "Churn"], axis=1)
y = data["Churn"]


# Encode categorical features
X = pd.get_dummies(X, drop_first=True)


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Scale features for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train_scaled, y_train)

logistic_predictions = logistic_model.predict(X_test_scaled)


# Logistic Regression evaluation
logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print("\nLogistic Regression Results")
print("---------------------------")
print(f"Accuracy: {logistic_accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, logistic_predictions))

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        logistic_predictions,
        labels=["No", "Yes"]
    )
)


# Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

random_forest_predictions = random_forest_model.predict(X_test)


# Random Forest evaluation
random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

print("\nRandom Forest Results")
print("---------------------")
print(f"Accuracy: {random_forest_accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, random_forest_predictions))


# Example new customer
new_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 477.50
}


# Prepare new customer data
new_customer_df = pd.DataFrame([new_customer])

new_customer_encoded = pd.get_dummies(new_customer_df)

new_customer_encoded = new_customer_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

new_customer_scaled = scaler.transform(new_customer_encoded)


# Predict churn for new customer
prediction = logistic_model.predict(new_customer_scaled)

probability = logistic_model.predict_proba(
    new_customer_scaled
)

churn_probability = probability[0][1] * 100


# Display prediction
print("\nNew Customer Prediction")
print("-----------------------")
print("Prediction:", prediction[0])
print(f"Churn Probability: {churn_probability:.2f}%")