from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import mysql.connector  # Use MySQL instead of SQLite
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO


app = Flask(__name__)
uploaded_data = None

# ---------- Utility Functions ----------
def clean_data(df):
    if 'TimeSinceLastTxn' not in df.columns and {'TransactionDate', 'PreviousTransactionDate'}.issubset(df.columns):
        try:
            df['TimeSinceLastTxn'] = (
                pd.to_datetime(df['TransactionDate']) - pd.to_datetime(df['PreviousTransactionDate'])
            ).dt.total_seconds() / 3600
            df.drop(['TransactionDate', 'PreviousTransactionDate'], axis=1, inplace=True)
        except Exception as e:
            print(f"Date error: {e}")
    return df

def create_sample_data():
    np.random.seed(42)
    now = datetime.now()
    df = pd.DataFrame({
        'TransactionID': [f'TXN{i:05d}' for i in range(1000)],
        'TransactionAmount': np.random.exponential(100, 1000),
        'CustomerOccupation': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Student'], 1000),
        'LoginAttempts': np.random.randint(1, 5, 1000),
        'TransactionDate': [now.strftime('%Y-%m-%d')] * 1000,
        'PreviousTransactionDate': [(now - timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%d')] * 1000
    })
    df['IsFraud'] = 0
    df.loc[df['TransactionAmount'] > df['TransactionAmount'].quantile(0.90), 'IsFraud'] = 1
    df.loc[df['LoginAttempts'] > 2, 'IsFraud'] = 1
    df['TimeSinceLastTxn'] = (
        pd.to_datetime(df['TransactionDate']) - pd.to_datetime(df['PreviousTransactionDate'])
    ).dt.total_seconds() / 3600
    df.loc[df['TimeSinceLastTxn'] > 48, 'IsFraud'] = 1
    if df['IsFraud'].mean() < 0.15:
        extra = int(0.2 * len(df) - sum(df['IsFraud']))
        if extra > 0:
            df.loc[df[df['IsFraud'] == 0].sample(extra, random_state=42).index, 'IsFraud'] = 1
    for col in df.select_dtypes(include='object').columns:
        if col != 'TransactionID':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df.drop(['TransactionDate', 'PreviousTransactionDate'], axis=1, inplace=True)
    df.to_csv('cleaned_bank_transactions_data.csv', index=False)
    return df

def train_model():
    df = pd.read_csv('cleaned_bank_transactions_data.csv') if os.path.exists('cleaned_bank_transactions_data.csv') else create_sample_data()
    X = df.drop('IsFraud', axis=1)
    y = df['IsFraud']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=3,
        class_weight='balanced', random_state=42
    ).fit(X_train, y_train)
    joblib.dump(model, 'fraud_model.pkl')
    return model

# ---------- Routes ----------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('Aboutus.html')

@app.route('/results')
def results():
    return render_template('result.html')

@app.route('/FAQs')
def FAQs():
    return render_template('FAQs.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_data
    try:
        file = request.files.get('csvFile')
        if not file or not file.filename:
            return render_template('result.html', error="No file uploaded")

        uploaded_data = pd.read_csv(file)
        required = ['TransactionID', 'TransactionAmount', 'CustomerOccupation', 'LoginAttempts']

        missing = [c for c in required if c not in uploaded_data.columns]
        if missing:
            return render_template('result.html', error=f"Missing columns: {', '.join(missing)}")

        model = joblib.load('fraud_model.pkl') if os.path.exists('fraud_model.pkl') else train_model()

        processed = clean_data(uploaded_data.copy())
        for c in processed.select_dtypes(include='object').columns:
            if c != 'TransactionID':
                processed[c] = LabelEncoder().fit_transform(processed[c].astype(str))
        X = processed.drop('TransactionID', axis=1, errors='ignore')

        for f in model.feature_names_in_:
            if f not in X.columns:
                X[f] = 0

        preds = model.predict(X[model.feature_names_in_])
        probs = model.predict_proba(X[model.feature_names_in_])[:, 1]

        display = uploaded_data[['TransactionID', 'TransactionAmount', 'CustomerOccupation', 'LoginAttempts']].copy()
        display['IsFraud'] = preds
        display['FraudProbability'] = [f"{p * 100:.2f}%" for p in probs]
        fraud, total = sum(preds), len(preds)

        return render_template('result.html',
                               fraud_count=fraud, total_count=total,
                               fraud_percentage=f"{(fraud / total) * 100:.2f}" if total else "0.00",
                               fraud_cases=display[display['IsFraud'] == 1].to_dict('records'),
                               display_data=display.to_dict('records'))
    except Exception as e:
        return render_template('result.html', error=f"Error: {str(e)}")




@app.route('/remove_fraud/<transaction_id>', methods=['POST'])
def remove_fraud(transaction_id):
    global uploaded_data
    try:
        if uploaded_data is None:
            return render_template('result.html', error="No data uploaded.")
        
        if transaction_id in uploaded_data['TransactionID'].values:
            uploaded_data.loc[uploaded_data['TransactionID'] == transaction_id, 'IsFraud'] = 0
            uploaded_data.to_csv('updated_bank_transactions_data.csv', index=False)
            return redirect(url_for('results'))

        return render_template('result.html', error="Transaction not found.")
    except Exception as e:
        return render_template('result.html', error=f"Error: {str(e)}")
def remove_fraud_from_data(data):
    """Remove all transactions suspected of fraud based on multiple behavioural patterns."""
    if data is None:
        raise ValueError("No data uploaded.")

    # Verify required fraud-related columns
    required_columns = ['TransactionAmount', 'LoginAttempts', 'TransactionFrequency', 'CustomerLocation']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Define fraud thresholds
    HIGH_TRANSACTION_AMOUNT = 10000  # Flag transactions above £10,000
    HIGH_LOGIN_ATTEMPTS = 5  # Flag users with more than 5 failed login attempts
    HIGH_TRANSACTION_FREQUENCY = 10  # Flag accounts making more than 10 transactions in an hour
    HIGH_RISK_LOCATIONS = ['Russia', 'North Korea', 'Unknown']  # Example locations flagged for risk

    # Identify fraudulent transactions
    fraud_cases = (
        (data['TransactionAmount'] >= HIGH_TRANSACTION_AMOUNT) |
        (data['LoginAttempts'] >= HIGH_LOGIN_ATTEMPTS) |
        (data['TransactionFrequency'] >= HIGH_TRANSACTION_FREQUENCY) |
        (data['CustomerLocation'].isin(HIGH_RISK_LOCATIONS))
    )

    # Remove fraudulent transactions
    data = data[~fraud_cases]

    # Save updated dataset
    data.to_csv('updated_bank_transactions_data.csv', index=False)
    return data

@app.route('/remove_all_fraud', methods=['POST'])
def remove_all_fraud():
    global uploaded_data
    try:
        uploaded_data = remove_fraud_from_data(uploaded_data)
        return redirect(url_for('results'))  # Reflect changes on results page

    except Exception as e:
        print("Error encountered:", e)  # Debugging output
        return render_template('result.html', error=f"Error: {str(e)}")



@app.route('/run_again', methods=['POST'])
def run_again():
    global uploaded_data
    try:
        if uploaded_data is None:
            return render_template('result.html', error="No data uploaded.")
        
        # Save the updated dataset
        uploaded_data.to_csv('updated_bank_transactions_data.csv', index=False)
        
        # Reload the model to reprocess the updated dataset
        model = joblib.load('fraud_model.pkl') if os.path.exists('fraud_model.pkl') else train_model()

        processed = clean_data(uploaded_data.copy())
        for c in processed.select_dtypes(include='object').columns:
            if c != 'TransactionID':
                processed[c] = LabelEncoder().fit_transform(processed[c].astype(str))
        X = processed.drop('TransactionID', axis=1, errors='ignore')

        for f in model.feature_names_in_:
            if f not in X.columns:
                X[f] = 0

        preds = model.predict(X[model.feature_names_in_])
        probs = model.predict_proba(X[model.feature_names_in_])[:, 1]

        display = uploaded_data[['TransactionID', 'TransactionAmount', 'CustomerOccupation', 'LoginAttempts']].copy()
        display['IsFraud'] = preds
        display['FraudProbability'] = [f"{p * 100:.2f}%" for p in probs]
        fraud, total = sum(preds), len(preds)

        return render_template('result.html',
                               fraud_count=fraud, total_count=total,
                               fraud_percentage=f"{(fraud / total) * 100:.2f}" if total else "0.00",
                               fraud_cases=display[display['IsFraud'] == 1].to_dict('records'),
                               display_data=display.to_dict('records'))
    except Exception as e:
        return render_template('result.html', error=f"Error: {str(e)}")

@app.route('/explain/<transaction_id>')
def explain_transaction(transaction_id):
    global uploaded_data
    if uploaded_data is None:
        return render_template('result.html', error="No data uploaded.")

    # Retrieve transaction details
    transaction = uploaded_data.loc[uploaded_data['TransactionID'] == transaction_id].to_dict(orient='records')
    if not transaction:
        return render_template('result.html', error="Transaction not found.")
    
    transaction = transaction[0]  # Convert list to dictionary

    # Generate SHAP explanation (assuming a trained model)
    model = joblib.load('fraud_model.pkl') if os.path.exists('fraud_model.pkl') else train_model()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(uploaded_data.drop(columns=['TransactionID', 'IsFraud']))  # Exclude non-numeric features
    
    # Create SHAP plot for the specific transaction
    transaction_idx = uploaded_data.index[uploaded_data['TransactionID'] == transaction_id][0]
    shap.summary_plot(shap_values[transaction_idx], uploaded_data.drop(columns=['TransactionID', 'IsFraud']))

    # Convert plot to Base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    shap_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return render_template('explain.html', transaction=transaction, shap_plot=shap_plot)

# ---------- Entry Point ----------
if __name__ == '__main__':
    if not os.path.exists('fraud_model.pkl'):
        train_model()
    app.run(debug=True, host='127.0.0.1', port=5000)
