import os
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
uploaded_data = None

# Train Fraud Detection Model
def train_model():
    """Trains fraud detection model and returns accuracy."""
    df = pd.read_csv('cleaned_bank_transactions_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['IsFraud']), df['IsFraud'], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
    joblib.dump(model, 'fraud_model.pkl')
    
    accuracy = model.score(X_test, y_test) * 100
    return model, accuracy

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles dataset upload, detects fraud, and explains fraud reasons."""
    global uploaded_data
    uploaded_data = pd.read_csv(request.files['csvFile'])

    model, accuracy = train_model() if not os.path.exists('fraud_model.pkl') else (joblib.load('fraud_model.pkl'), None)
    
    # Add fraud reasons
    uploaded_data['FraudReason'] = uploaded_data.apply(
        lambda row: "High amount" if row['TransactionAmount'] > uploaded_data['TransactionAmount'].quantile(0.9)
        else "Frequent login attempts" if row['LoginAttempts'] > 2
        else "Time gap in transactions", axis=1)

    preds, probs = model.predict(uploaded_data.drop(columns=['TransactionID'])), model.predict_proba(uploaded_data.drop(columns=['TransactionID']))[:, 1]
    uploaded_data['IsFraud'], uploaded_data['FraudProbability'] = preds, [f"{p * 100:.2f}%" for p in probs]

    return jsonify({
        "fraud_cases": uploaded_data[uploaded_data['IsFraud'] == 1].to_dict('records'),
        "accuracy": accuracy
    })

@app.route('/remove_fraud', methods=['POST'])
def remove_fraud():
    """Removes fraud cases and saves a cleaned dataset."""
    global uploaded_data
    if uploaded_data is None:
        return jsonify({"error": "No uploaded dataset found"}), 400

    uploaded_data = uploaded_data[uploaded_data['IsFraud'] == 0]

    # Save cleaned dataset with versioning
    version = len([f for f in os.listdir() if f.startswith("cleaned_no_fraud")]) + 1
    cleaned_filename = f'cleaned_no_fraud_v{version}.csv'
    uploaded_data.to_csv(cleaned_filename, index=False)

    return jsonify({"message": f"Fraud cases removed and saved as {cleaned_filename}"}), 200

@app.route('/check_fraud', methods=['GET'])
def check_fraud():
    """Checks if fraud cases are still in the dataset."""
    fraud_count = len(uploaded_data[uploaded_data['IsFraud'] == 1]) if uploaded_data is not None else 0
    return jsonify({"fraud_cases": fraud_count}), 200

def run_fraud_explan():
    """Starts the Flask fraud detection system."""
    if not os.path.exists('fraud_model.pkl'):
        train_model()
    app.run(debug=True, host='127.0.0.1', port=5000)

if __name__ == "__main__":
    run_fraud_explan()
