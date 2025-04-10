import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def list_fraud_transactions():
    """Returns all fraud transaction IDs."""
    try:
        if not os.path.exists('cleaned_bank_transactions_data.csv'):
            return "Dataset not found."

        df = pd.read_csv('cleaned_bank_transactions_data.csv')
        fraud_transactions = df[df['IsFraud'] == 1]['TransactionID'].tolist()

        return fraud_transactions if fraud_transactions else "No fraud transactions found."
    
    except Exception as e:
        return f"Error: {str(e)}"

def explain_and_remove_transaction(transaction_id):
    """Explains fraud reason, removes transaction, and retrains the model."""
    try:
        if not os.path.exists('cleaned_bank_transactions_data.csv'):
            return "Dataset not found."

        df = pd.read_csv('cleaned_bank_transactions_data.csv')

        transaction = df[df['TransactionID'] == transaction_id]
        if transaction.empty:
            return f"Transaction {transaction_id} not found."

        fraud_reason = []
        if transaction['TransactionAmount'].iloc[0] > df['TransactionAmount'].quantile(0.90):
            fraud_reason.append("High transaction amount")
        if transaction['LoginAttempts'].iloc[0] > 2:
            fraud_reason.append("Multiple login attempts")
        if transaction['TimeSinceLastTxn'].iloc[0] > 48:
            fraud_reason.append("Long gap since last transaction")

        fraud_reason_text = ", ".join(fraud_reason) if fraud_reason else "No specific reason found."

        df = df[df['TransactionID'] != transaction_id]
        df.to_csv('cleaned_bank_transactions_data.csv', index=False)

        train_model()

        return f"Transaction {transaction_id} removed. Reason: {fraud_reason_text}"
    
    except Exception as e:
        return f"Error: {str(e)}"

def train_model():
    """Retrains the fraud detection model."""
    df = pd.read_csv('cleaned_bank_transactions_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(df.drop('IsFraud', axis=1), df['IsFraud'], test_size=0.2, stratify=df['IsFraud'])

    model = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'fraud_model.pkl')
