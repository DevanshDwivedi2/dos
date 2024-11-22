import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = "C:/Users/devan/OneDrive/Desktop/RF/Cleaned Friday-WorkingHours-Afternoon-DDos.csv"
data = pd.read_csv(file_path)

# Encode the target variable
print("Encoding the target variable...")
data['Label'] = data['Label'].apply(lambda x: 0 if x == "BENIGN" else 1)

# Split into features (X) and target (y)
X = data.drop('Label', axis=1)  # Drop the target column
y = data['Label']

# Handle infinite and extremely large values
print("Replacing infinite and extremely large values...")
X = X.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
X = X.fillna(X.mean())  # Replace NaN with column means

# Train-test split
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
print("Training the Random Forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probability for ROC Curve

# Confusion Matrix
print("Generating the confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["BENIGN", "ATTACK"], yticklabels=["BENIGN", "ATTACK"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
print("Generating the ROC curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
