# To perform operations on dataset
# To perform operations on dataset
import pandas as pd
import numpy as np
# Machine learning model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# Visualization
# Visualization
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc
import joblib
from urllib.parse import urlparse

# Load the dataset
df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')
dot_file = 'tree.dot'
confusion_matrix_file = 'confusion_matrix.png'

# Display the first few rows of the dataset
print(df.head())

# --- DIAGNOSTIC CHECKS FOR DATA LEAKAGE AND DATA QUALITY ---

print("\n--- Checking for duplicate rows ---")
num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

print("\n--- Checking label distribution ---")
print(df['label'].value_counts())

print("\n--- Checking for highly correlated features with the label ---")
correlations = df.corr(numeric_only=True)['label'].sort_values(ascending=False)
print(correlations)

print("\n--- Checking for non-numeric columns in features ---")
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print("Non-numeric columns:", non_numeric_cols)

# Drop 'FILENAME' and 'URL' columns, keep only numeric features
X = df.select_dtypes(include=[np.number])
y = df['label']

features_to_drop = ['URLSimilarityIndex', 'HasSocialNet', 'HasCopyrightInfo']
X = X.drop(columns=[col for col in features_to_drop if col in X.columns])

# Drop all features with correlation > 0.5 (except the label itself and already dropped features)
correlations = df.corr(numeric_only=True)['label'].abs()
high_corr_features = correlations[correlations > 0.5].index.tolist()
high_corr_features = [f for f in high_corr_features if f != 'label' and f in X.columns]
print("Dropping highly correlated features:", high_corr_features)
X = X.drop(columns=high_corr_features)

# After loading your DataFrame and before splitting into train/test:
if 'label' in X.columns:
    X = X.drop(columns=['label'])

# Train a Decision Tree Classifier
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

# Create and fit the model
model = DecisionTreeClassifier()
model.fit(Xtrain, ytrain)

# Save the model and feature list
joblib.dump(model, 'phish_detector_model.pkl')
joblib.dump(list(X.columns), 'feature_list.pkl')

print("Model saved as phish_detector_model.pkl")

# Visualize the decision tree
ypred = model.predict(Xtest)
print(metrics.classification_report(ypred, ytest))
print("\n\nAccuracy Score:", round(metrics.accuracy_score(ytest, ypred)*100, 2), "%")

# Confusion matrix heatmap
mat = confusion_matrix(ytest, ypred)
plt.figure(figsize=(6, 5))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(confusion_matrix_file)
plt.close()

if len(np.unique(ytest)) == 2:  # Only plot ROC for binary classification
    yproba = model.predict_proba(Xtest)[:, 1]
    fpr, tpr, _ = roc_curve(ytest, yproba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()

# Feature importance bar plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Example prediction
example = Xtest.iloc[[0]]
example_pred = model.predict(example)[0]
print("\nExample prediction:")
print("Features:\n", example)
print("Predicted label:", example_pred)

# Save results table
results_df = Xtest.copy()
results_df['true_label'] = ytest.values
results_df['predicted_label'] = ypred
results_df.to_csv('test_results_table.csv', index=False)
print("\nTest results saved to test_results_table.csv")

export_graphviz(model, out_file=dot_file, feature_names=X.columns.values)

scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())

# Shuffle the labels
y_shuffled = y.sample(frac=1, random_state=42).reset_index(drop=True)
model.fit(Xtrain, y_shuffled.loc[ytrain.index])
print("Accuracy with shuffled labels:", model.score(Xtest, y_shuffled.loc[ytest.index]))

# --- Predict on new example websites ---

# Example: Manually create feature values for new URLs
# Replace these with real feature extraction for your pipeline!
example_sites = [
    {
        "URLLength": 14, "DomainLength": 10, "IsDomainIP": 0, "CharContinuationRate": 1.0,
        "TLDLegitimateProb": 0.99, "URLCharProb": 0.08, "TLDLength": 3, "NoOfSubDomain": 0,
        "HasObfuscation": 0, "HasDescription": 1, "IsHTTPS": 1, "DomainTitleMatchScore": 1.0,
        "HasSubmitButton": 0, "IsResponsive": 1, "URLTitleMatchScore": 1.0, "HasHiddenFields": 0,
        "HasFavicon": 1, "Robots": 1, "NoOfJS": 5, "Pay": 0, "Crypto": 0, "NoOfImage": 10,
        "NoOfCSS": 2, "NoOfJS": 5, "NoOfSelfRef": 2, "NoOfEmptyRef": 0, "NoOfExternalRef": 3,
        # ...add all other features used in X...
    },  # google.com (legitimate)
    {
        "URLLength": 45, "DomainLength": 25, "IsDomainIP": 0, "CharContinuationRate": 0.7,
        "TLDLegitimateProb": 0.2, "URLCharProb": 0.02, "TLDLength": 3, "NoOfSubDomain": 2,
        "HasObfuscation": 1, "HasDescription": 0, "IsHTTPS": 0, "DomainTitleMatchScore": 0.2,
        "HasSubmitButton": 1, "IsResponsive": 0, "URLTitleMatchScore": 0.1, "HasHiddenFields": 1,
        "HasFavicon": 0, "Robots": 0, "NoOfJS": 15, "Pay": 1, "Crypto": 0, "NoOfImage": 1,
        "NoOfCSS": 0, "NoOfJS": 15, "NoOfSelfRef": 0, "NoOfEmptyRef": 2, "NoOfExternalRef": 10,
        # ...add all other features used in X...
    },  # phishing example
    # Add more examples as needed
]

# Convert to DataFrame and align columns
new_X = pd.DataFrame(example_sites)
missing_cols = set(X.columns) - set(new_X.columns)
for col in missing_cols:
    new_X[col] = 0  # or a sensible default

new_X = new_X[X.columns]  # Ensure column order matches

# Predict
new_pred = model.predict(new_X)
print("\nPredictions on new example websites:")
for i, pred in enumerate(new_pred):
    print(f"Example {i+1}: {'Phishing' if pred == 1 else 'Legitimate'}")

# Remove 'label' if present
feature_columns = [col for col in X.columns if col != 'label']
joblib.dump(feature_columns, 'feature_list.pkl')
print("Feature list saved as feature_list.pkl")

# Load feature_list from file
try:
    feature_list = joblib.load('feature_list.pkl')
except Exception:
    feature_list = list(X.columns)

print("Feature list:", feature_list)

def extract_features(url):
    parsed = urlparse(url if url.startswith('http') else 'http://' + url)
    features = {}
    # Only add features that exist in feature_list
    if 'URLLength' in feature_list:
        features['URLLength'] = len(url)
    if 'DomainLength' in feature_list:
        features['DomainLength'] = len(parsed.netloc)
    if 'IsDomainIP' in feature_list:
        features['IsDomainIP'] = 1 if parsed.netloc.replace('.', '').isdigit() else 0
    # ...repeat for all features in your feature_list...
    # Fill missing features with 0
    for f in feature_list:
        if f not in features:
            features[f] = 0
    return features