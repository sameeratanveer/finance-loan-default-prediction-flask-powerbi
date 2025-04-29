from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import uuid

# Load trained model and scaler
model = joblib.load("../notebooks&models/best_rf.joblib")
scaler = joblib.load("../notebooks&models/scaler.joblib")

# Define CSV file to store loan data
DATA_FILE = "../data/predicted.csv"

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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Get user input from the form
            user_data = {
                "Age": int(request.form["age"]),
                "Income": float(request.form["income"]),
                "LoanAmount": float(request.form["loan_amount"]),
                "CreditScore": int(request.form["credit_score"]),
                "MonthsEmployed": int(request.form["months_employed"]),
                "NumCreditLines": int(request.form["num_credit_lines"]),
                "InterestRate": float(request.form["interest_rate"]),
                "LoanTerm": int(request.form["loan_term"]),
                "DTIRatio": float(request.form["dti_ratio"]),
                "Education": request.form["education"],
                "EmploymentType": request.form["employment_type"],
                "MaritalStatus": request.form["marital_status"],
                "HasMortgage": 1 if "has_mortgage" in request.form else 0,
                "HasDependents": 1 if "has_dependents" in request.form else 0,
                "LoanPurpose": request.form["loan_purpose"],
                "HasCoSigner": 1 if "has_cosigner" in request.form else 0
            }

            # Convert to DataFrame
            df_input = pd.DataFrame([user_data])

            # Preprocess user input
            df_processed = preprocess_data(df_input)

            # Make prediction
            prediction = model.predict(df_processed)[0]
            prediction = "Default" if prediction == 1 else "No Default"

            # Save user input with prediction
            user_data["Default"] = prediction
            save_prediction(user_data)

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)

def save_prediction(data):
    """Save predictions to a CSV file."""
    # Generate a unique LoanID
    data["LoanID"] = str(uuid.uuid4())[:8]  # Short 8-character unique ID

    df_new = pd.DataFrame([data])

    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new

    df_updated.to_csv(DATA_FILE, index=False)

if __name__ == "__main__":
    app.run(debug=True)
