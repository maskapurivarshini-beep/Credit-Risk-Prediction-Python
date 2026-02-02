# =============================================================================
# GERMAN CREDIT RISK PREDICTION
# Dataset: https://www.kaggle.com/datasets/uciml/german-credit
# Target: Synthetically derived 'Risk' (0=Good, 1=Bad) based on heuristics
# Models: Logistic Regression, Random Forest, XGBoost + Deep Neural Network
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# Force permanent plots and styling
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

# ====================== 1. LOAD DATASET ======================
df = pd.read_csv("german_credit_data.csv", index_col=0)  # First column is index

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
df.head()

# ====================== 2. DERIVE SYNTHETIC TARGET 'Risk' ======================
# Heuristic: High credit/short duration OR risky purpose
# This is an approximation since no target exists

df['Credit_to_Duration_Ratio'] = df['Credit amount'] / (df['Duration'] + 1)  # +1 to avoid div/0

# Risky purposes (based on common credit patterns)
risky_purposes = ['business', 'education']
df['Risky_Purpose'] = df['Purpose'].isin(risky_purposes).astype(int)

# Synthetic Risk: 1 if high ratio (>1000) OR risky purpose
df['Risk'] = ((df['Credit_to_Duration_Ratio'] > 1000) | (df['Risky_Purpose'] == 1)).astype(int)

print("Synthetic 'Risk' column created (0=Good, 1=Bad)")
print("Risk distribution:")
print(df['Risk'].value_counts())
print(f"Bad credit rate: {df['Risk'].mean():.2%}")

# Drop temporary columns if not needed for modeling
df = df.drop(['Credit_to_Duration_Ratio', 'Risky_Purpose'], axis=1)

# ====================== 3. EXPLORATORY DATA ANALYSIS (EDA) ======================

print("=== DATA INFO ===")
print(df.info())
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Handle missing values
df['Saving accounts'] = df['Saving accounts'].fillna('none')
df['Checking account'] = df['Checking account'].fillna('none')

# Plot 1: Target Distribution (permanent plot)
plt.figure(figsize=(8,5))
risk_counts = df['Risk'].value_counts().sort_index()
ax = sns.countplot(data=df, x='Risk', palette=['#4CAF50', '#F44336'])
ax.set_title('Synthetic Credit Risk Distribution', fontsize=16, fontweight='bold')
ax.set_xticklabels(['Good Credit (0)', 'Bad Credit (1)'])
for i, v in enumerate(risk_counts):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=12)
plt.ylabel('Number of Applicants')
plt.xlabel('')
plt.show()

print(f"\nGood: {risk_counts[0]} ({risk_counts[0]/len(df)*100:.1f}%) | Bad: {risk_counts[1]} ({risk_counts[1]/len(df)*100:.1f}%)")

# Plot 2: Age Distribution by Risk
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='Age', hue='Risk', bins=20, kde=True, palette=['#4CAF50', '#F44336'], alpha=0.7)
plt.title('Age Distribution by Credit Risk', fontsize=16, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.legend(title='Risk', labels=['Good (0)', 'Bad (1)'])
plt.show()

# Plot 3: Purpose Distribution by Risk
plt.figure(figsize=(12,6))
sns.countplot(data=df, y='Purpose', hue='Risk', palette=['#4CAF50', '#F44336'])
plt.title('Loan Purpose by Credit Risk', fontsize=16, fontweight='bold')
plt.xlabel('Number of Applicants')
plt.ylabel('Purpose')
plt.legend(title='Risk', labels=['Good (0)', 'Bad (1)'])
plt.show()

# Plot 4: Correlation Heatmap (Numeric Features)
numeric_cols = ['Age', 'Job', 'Credit amount', 'Duration', 'Risk']
plt.figure(figsize=(8,6))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix of Numeric Features', fontsize=16, fontweight='bold')
plt.show()

# ====================== 4. DATA PREPROCESSING ======================
# Encode categorical columns
cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Features and target
X = df.drop('Risk', axis=1)
y = df['Risk']

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape[0]} samples")
print(f"Test set: {X_test_scaled.shape[0]} samples")
print(f"Bad risk in train: {y_train.mean():.2%}")

# ====================== 5. MODEL EVALUATION FUNCTION ======================
def evaluate_model(model, X_tr, X_te, y_tr, y_te, name, is_nn=False):
    model.fit(X_tr, y_tr)
    if is_nn:
        y_pred = (model.predict(X_te) > 0.5).astype(int).flatten()
        y_proba = model.predict(X_te).flatten()
    else:
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, y_proba)
    print(f"\n=== {name} ===")
    print(classification_report(y_te, y_pred, target_names=['Good', 'Bad']))
    print(f"AUC-ROC: {auc:.4f}")

    # Confusion Matrix (permanent plot)
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return auc

# ====================== 6. BASELINE MODELS ======================

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr_auc = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression")

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_auc = evaluate_model(rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest")

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_auc = evaluate_model(xgb_model, X_train_scaled, X_test_scaled, y_train, y_test, "XGBoost")

# ====================== 7. DEEP LEARNING MODEL ======================
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=20, restore_best_weights=True, verbose=1)

print("Training Deep Neural Network...")
history = nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Plot Training History (permanent)
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['auc'], label='Train AUC', linewidth=2)
plt.plot(history.history['val_auc'], label='Val AUC', linewidth=2)
plt.title('Neural Network AUC', fontweight='bold')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Neural Network Loss', fontweight='bold')
plt.legend()
plt.show()

# Evaluate Deep Learning
nn_auc = evaluate_model(nn_model, X_train_scaled, X_test_scaled, y_train, y_test, "Deep Neural Network", is_nn=True)

# ====================== 8. FINAL COMPARISON ======================
results_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Deep Neural Network'],
    'AUC-ROC': [lr_auc, rf_auc, xgb_auc, nn_auc]
}).sort_values('AUC-ROC', ascending=False)

print("\n" + "="*60)
print("                  FINAL MODEL PERFORMANCE")
print("="*60)
print(results_df.to_string(index=False, float_format='%.4f'))

# Comparison Bar Plot (permanent)
plt.figure(figsize=(10,6))
bars = plt.barh(results_df['Model'], results_df['AUC-ROC'], color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
plt.title('Model Comparison by AUC-ROC', fontsize=16, fontweight='bold')
plt.xlabel('AUC-ROC Score')
plt.xlim(0.5, 1.0)
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{results_df["AUC-ROC"].iloc[i]:.4f}',
             ha='left', va='center', fontweight='bold')
plt.gca().invert_yaxis()
plt.show()