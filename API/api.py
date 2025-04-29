from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import uuid

# Load trained model and scaler
model = joblib.load("../notebooks&models/best_rf.joblib")
scaler = joblib.load("../notebooks&models/scaler.joblib")

# Define CSV file to store loan data
DATA_FILE = "C:/Users/SAMSUNG/Music/Goldsman_Sachs_Projects/P3.1_Credit_Risk/data/predicted.csv"

# Define categorical mappings
education_map = {"Bachelor's": 0, "High School": 1, "Master's": 2}
employment_map = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
marital_map = {"Divorced": 0, "Married": 1, "Single": 2}
purpose_map = {"Auto": 0, "Business": 1, "Education": 2, "Home": 3, "Other": 4}

def preprocess_data(df):
    """Preprocess input data for consistency with the training pipeline."""
    
    # Convert categorical variables using mappings
    df["Education"] = df["Education"].map(education_map)
    df["EmploymentType"] = df["EmploymentType"].map(employment_map)
    df["MaritalStatus"] = df["MaritalStatus"].map(marital_map)
    df["LoanPurpose"] = df["LoanPurpose"].map(purpose_map)

    # Convert binary categories to 0/1
    df["HasMortgage"] = df["HasMortgage"].astype(int)
    df["HasDependents"] = df["HasDependents"].astype(int)
    df["HasCoSigner"] = df["HasCoSigner"].astype(int)

    # Feature Engineering: Loan-to-Income Ratio (LTI)
    df["LTI"] = df["LoanAmount"] / df["Income"]

    # Feature Engineering: Risk Score (Custom metric)
    df["RiskScore"] = (df["CreditScore"] / df["InterestRate"]) * df["DTIRatio"]

    # Define the feature set used during training
    feature_columns = [
        "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines",
        "InterestRate", "LoanTerm", "DTIRatio", "RiskScore", "LTI",
        "Education", "EmploymentType", "MaritalStatus", "LoanPurpose",
        "HasMortgage", "HasDependents", "HasCoSigner"
    ]

    # Separate numerical features for scaling
    numeric_cols = [
        "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines",
        "InterestRate", "LoanTerm", "DTIRatio", "RiskScore", "LTI"
    ]

    # Apply Scaling (normalize numerical values)
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Ensure the correct feature order
    df = df[feature_columns]

    return df

app = Flask(__name__)

### API Endpoint for Prediction

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to predict loan default risk."""
    try:
        # Get JSON data from request
        data = request.get_json()

        # Generate a unique LoanID
        loan_id = str(uuid.uuid4())[:8]  # Short 8-character unique ID
        data["LoanID"] = loan_id  # Assign LoanID before saving or returning

        # Convert data to DataFrame
        df_input = pd.DataFrame([data])

        # Preprocess user input
        df_processed = preprocess_data(df_input)

        # Make prediction
        prediction = model.predict(df_processed)[0]
        prediction_text = "Default" if prediction == 1 else "No Default"

        # Save user input with prediction
        data["Default"] = prediction_text
        save_prediction(data)

        # Return prediction result
        return jsonify({"prediction": prediction_text, "loan_id": loan_id})

    except Exception as e:
        return jsonify({"error": str(e)})

def save_prediction(data):
    """Save predictions to a CSV file."""
    df_new = pd.DataFrame([data])

    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new

    df_updated.to_csv(DATA_FILE, index=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
