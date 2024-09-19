# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Load dataset
data = pd.read_csv('creditcard.csv')

# Check class distribution (fraud vs non-fraud)
print(data['Class'].value_counts())

# Data Preprocessing
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable (fraud or not)

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling imbalanced dataset with SMOTE (Synthetic Minority Over-sampling Technique)
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Alternatively, you can use undersampling for non-fraud (majority class)
# Majority class (non-fraud)
X_train_majority = X_train[y_train == 0]
y_train_majority = y_train[y_train == 0]

# Minority class (fraud)
X_train_minority = X_train[y_train == 1]
y_train_minority = y_train[y_train == 1]

# Downsample the majority class
X_train_majority_downsampled, y_train_majority_downsampled = resample(
    X_train_majority, y_train_majority, 
    replace=False, n_samples=len(y_train_minority), random_state=42)

# Combine downsampled majority class with minority class
X_train_downsampled = pd.concat([X_train_majority_downsampled, X_train_minority])
y_train_downsampled = pd.concat([y_train_majority_downsampled, y_train_minority])

# Train a Random Forest classifier on SMOTE data
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_sm, y_train_sm)

# Train an XGBoost classifier on downsampled data
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train_downsampled, y_train_downsampled)

# Evaluate the models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

print("Random Forest on SMOTE data:")
evaluate_model(rf_classifier, X_test, y_test)

print("XGBoost on downsampled data:")
evaluate_model(xgb_classifier, X_test, y_test)
