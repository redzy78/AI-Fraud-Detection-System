# BANK TRANSACTIONS DATA ANALYSIS - COMPLETE SCRIPT
# Includes EDA, Data Cleaning, and SHAP Explanations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# =============================================
# 1. DATA LOADING AND INITIAL INSPECTION
# =============================================

print("Loading and inspecting data...")
# Load the dataset
try:
    df = pd.read_csv('bank_transactions_data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: File 'bank_transactions_data.csv' not found. Please ensure it's in the same directory.")
    exit()

# Convert date columns with correct format (DD/MM/YYYY HH:MM)
try:
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='%d/%m/%Y %H:%M')
    df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'], format='%d/%m/%Y %H:%M')
    print("Date columns converted successfully.")
except Exception as e:
    print(f"Error converting date columns: {e}")
    exit()

# Initial inspection
print("\n=== BASIC DATASET INFO ===")
print(f"Number of transactions: {len(df)}")
print(f"Number of features: {len(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nDescriptive statistics:")
print(df.describe(include='all').transpose())

# =============================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================

print("\nPerforming Exploratory Data Analysis...")

# Set style for visualizations
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Create EDA visualizations directory
if not os.path.exists('eda_visualizations'):
    os.makedirs('eda_visualizations')

# [Previous EDA visualizations code remains exactly the same...]
# A. Transaction Amount Analysis
plt.figure(figsize=(12, 6))
sns.histplot(df['TransactionAmount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Frequency')
plt.savefig('eda_visualizations/transaction_amount_distribution.png', bbox_inches='tight')
plt.close()

# B. Transaction Type Analysis
plt.figure(figsize=(8, 5))
sns.countplot(x='TransactionType', data=df)
plt.title('Count of Transaction Types')
plt.savefig('eda_visualizations/transaction_type_counts.png', bbox_inches='tight')
plt.close()

# [... all other EDA visualizations remain unchanged ...]

# =============================================
# 3. DATA CLEANING
# =============================================

print("\nCleaning data...")

df_clean = df.copy()

# A. Handle missing values
print("\nMissing values before cleaning:")
print(df_clean.isnull().sum())

df_clean = df_clean.dropna()

# B. Convert categorical columns
categorical_cols = ['TransactionType', 'Location', 'DeviceID', 'MerchantID', 
                   'Channel', 'CustomerOccupation']
for col in categorical_cols:
    df_clean[col] = df_clean[col].astype('category')

# C. Remove duplicates
duplicate_count = df_clean.duplicated().sum()
print(f"\nFound {duplicate_count} duplicate rows")
df_clean = df_clean.drop_duplicates()

# D. Handle outliers in TransactionAmount
Q1 = df_clean['TransactionAmount'].quantile(0.25)
Q3 = df_clean['TransactionAmount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nTransaction Amount Outlier Boundaries: Lower=${lower_bound:.2f}, Upper=${upper_bound:.2f}")

df_clean['TransactionAmount'] = df_clean['TransactionAmount'].clip(lower_bound, upper_bound)

# E. Standardize column names
df_clean.columns = df_clean.columns.str.replace(' ', '')

# F. Save cleaned data
df_clean.to_csv('cleaned_banktransaction.csv', index=False)

# =============================================
# 4. SHAP EXPLANATIONS
# =============================================

print("\nAdding SHAP explanations...")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import shap
    
    # Create a copy of the cleaned data for modeling
    df_model = df_clean.copy()

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le

    # Create target variable: high value transactions (top 20%)
    threshold = df_model['TransactionAmount'].quantile(0.8)
    df_model['HighValue'] = (df_model['TransactionAmount'] >= threshold).astype(int)

    # Select features and target
    features = ['TransactionType', 'CustomerAge', 'TransactionDuration', 
               'LoginAttempts', 'AccountBalance', 'TimeBetweenTransactions', 
               'Channel', 'CustomerOccupation']
    X = df_model[features]
    y = df_model['HighValue']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    print(f"\nModel trained with accuracy: {model.score(X_test, y_test):.2f}")

    # SHAP explanations
    print("\nGenerating SHAP explanations...")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Create directory for SHAP visualizations
    if not os.path.exists('shap_visualizations'):
        os.makedirs('shap_visualizations')

    # 1. Summary plot (feature importance)
    plt.figure()
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.savefig('shap_visualizations/shap_feature_importance.png', bbox_inches='tight')
    plt.close()

    # 2. Detailed summary plot
    plt.figure()
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig('shap_visualizations/shap_summary_plot.png', bbox_inches='tight')
    plt.close()

    # 3. Force plot for a single prediction
    sample_idx = 0  # First test sample
    plt.figure()
    shap.force_plot(explainer.expected_value[1], shap_values[1][sample_idx,:], 
                    X_test.iloc[sample_idx,:], matplotlib=True, show=False)
    plt.title(f"SHAP Force Plot for Sample {sample_idx}")
    plt.savefig('shap_visualizations/shap_force_plot.png', bbox_inches='tight')
    plt.close()

    # 4. Dependence plot for top features
    for feature in X.columns[:3]:  # Just plot first 3 features for brevity
        plt.figure()
        shap.dependence_plot(feature, shap_values[1], X_test, interaction_index=None, show=False)
        plt.title(f"SHAP Dependence Plot for {feature}")
        plt.savefig(f'shap_visualizations/shap_dependence_{feature}.png', bbox_inches='tight')
        plt.close()

    # 5. Waterfall plot
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], 
                                          shap_values[1][sample_idx], 
                                          feature_names=X.columns,
                                          max_display=10)
    plt.title("SHAP Waterfall Plot")
    plt.savefig('shap_visualizations/shap_waterfall_plot.png', bbox_inches='tight')
    plt.close()

    print("\n=== SHAP ANALYSIS COMPLETE ===")
    print("Saved SHAP visualizations to 'shap_visualizations' folder")
    print("""
SHAP Explanation Guide:
1. Feature Importance: Shows which features contribute most to predictions
2. Summary Plot: Shows both feature importance and direction of effect
3. Force Plot: Explains a single prediction
4. Dependence Plots: Show how a feature affects predictions
5. Waterfall Plot: Detailed breakdown of a single prediction
""")

except ImportError as e:
    print(f"\nError: SHAP dependencies not found. Please install required packages: {e}")
    print("Run: pip install shap scikit-learn")
except Exception as e:
    print(f"\nError during SHAP analysis: {e}")

# =============================================
# 5. FINAL REPORT
# =============================================

print("\n=== ANALYSIS COMPLETE ===")
print(f"Original data shape: {df.shape}")
print(f"Cleaned data shape: {df_clean.shape}")
print(f"\nNumber of EDA visualizations generated: {len(os.listdir('eda_visualizations')) if os.path.exists('eda_visualizations') else 0}")

# Check if SHAP visualizations were created
shap_viz_count = len(os.listdir('shap_visualizations')) if os.path.exists('shap_visualizations') else 0
print(f"Number of SHAP visualizations generated: {shap_viz_count}")

print("\nSample of cleaned data:")
print(df_clean.head())

print("\nSaved outputs:")
print("- Cleaned dataset: 'cleaned_banktransaction.csv'")
print("- EDA visualizations: 'eda_visualizations' folder")
if shap_viz_count > 0:
    print("- SHAP explanations: 'shap_visualizations' folder")
else:
    print("- SHAP explanations: Not generated (see errors above)")