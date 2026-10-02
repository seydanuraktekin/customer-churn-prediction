# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to leave the company based on customer information and service usage.

## Project Overview

The goal of this project is to analyze telecom customer data and build a classification model for customer churn prediction.

The project includes data cleaning, categorical feature encoding, feature scaling, model training, model comparison, and prediction for a new customer.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information such as:

- Tenure
- Contract type
- Internet service
- Tech support
- Payment method
- Monthly charges
- Total charges
- Churn status

After data cleaning, 7,032 customer records are used in the project.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Machine Learning Models

Two classification algorithms are trained and evaluated:

### Logistic Regression

Accuracy: **80.38%**

### Random Forest

Accuracy: **78.68%**

In the current experiment, Logistic Regression also achieved better precision, recall, and F1-score for the churn class.

## Project Workflow

1. Load the customer dataset
2. Clean missing and invalid values
3. Separate features and target variable
4. Encode categorical variables
5. Split the dataset into training and test sets
6. Scale features for Logistic Regression
7. Train Logistic Regression and Random Forest models
8. Evaluate the models
9. Predict churn for a new customer

## Example Prediction

The project can also make predictions for a new customer.

Example output:

```text
New Customer Prediction
-----------------------
Prediction: Yes
Churn Probability: 72.88%