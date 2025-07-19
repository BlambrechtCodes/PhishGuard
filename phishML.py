# To perform operations on dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from urllib.parse import urlparse

print("\n=== Loading Dataset ===")
df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')
print("First 5 rows:\n", df.head())

print("\n--- Legitimate Examples ---")
print(df[df['label'] == 1].head())
print("\n--- Phishing Examples ---")
print(df[df['label'] == 0].head())

print("\n=== Data Quality Checks ===")
print("--- Checking for duplicate rows ---")
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

print("\n=== Feature Selection ===")
X = df.select_dtypes(include=[np.number])
y = df['label']

features_to_drop = ['URLSimilarityIndex', 'HasSocialNet', 'HasCopyrightInfo']
X = X.drop(columns=[col for col in features_to_drop if col in X.columns])

correlations = df.corr(numeric_only=True)['label'].abs()
high_corr_features = correlations[correlations > 0.5].index.tolist()
high_corr_features = [f for f in high_corr_features if f != 'label' and f in X.columns]
print("Dropping highly correlated features:", high_corr_features)
X = X.drop(columns=high_corr_features)

if 'label' in X.columns:
    X = X.drop(columns=['label'])

print("\nFinal feature columns:", list(X.columns))

print("\n=== Splitting Data ===")
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
print(f"Train size: {Xtrain.shape[0]}, Test size: {Xtest.shape[0]}")

print("\n=== Training Model ===")
model = DecisionTreeClassifier(class_weight='balanced')
model.fit(Xtrain, ytrain)
print("Model trained.")

joblib.dump(model, 'phish_detector_model.pkl')
joblib.dump(list(X.columns), 'feature_list.pkl')
print("Model and feature list saved.")

print("\n=== Evaluating Model ===")
ypred = model.predict(Xtest)
print(metrics.classification_report(ypred, ytest))
print("Accuracy Score:", round(metrics.accuracy_score(ytest, ypred)*100, 2), "%")

mat = confusion_matrix(ytest, ypred)
plt.figure(figsize=(6, 5))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Confusion matrix saved as confusion_matrix.png")

if len(np.unique(ytest)) == 2:
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
    print("ROC curve saved as roc_curve.png")

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("Feature importance plot saved as feature_importance.png")

print("\n=== Example Prediction (from test set) ===")
example = Xtest.iloc[[0]]
example_pred = model.predict(example)[0]
print("Features:\n", example)
print("Predicted label:", example_pred)

results_df = Xtest.copy()
results_df['true_label'] = ytest.values
results_df['predicted_label'] = ypred
results_df.to_csv('test_results_table.csv', index=False)
print("Test results saved to test_results_table.csv")

export_graphviz(model, out_file='tree.dot', feature_names=X.columns.values)
print("Decision tree exported as tree.dot")

scores = cross_val_score(model, X, y, cv=5 )
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())

y_shuffled = y.sample(frac=1, random_state=42).reset_index(drop=True)
model.fit(Xtrain, y_shuffled.loc[ytrain.index])
print("Accuracy with shuffled labels:", model.score(Xtest, y_shuffled.loc[ytest.index]))

print("\n=== Predict on a Real Row ===")
row = df[df['URL'] == 'https://www.tourdatesearch.com']
X_row = row[X.columns]
print("Prediction:", model.predict(X_row))
print("True label:", row['label'].values)

print("\n=== Predict on a Custom Example ===")
example_features = {
    'URLLength': 29,
    'DomainLength': 22,
    'IsDomainIP': 0,
    'CharContinuationRate': 1.0,
    'TLDLegitimateProb': 0.52,
    'URLCharProb': 0.06,
    'TLDLength': 3,
    'NoOfSubDomain': 1,
    # ...add other features as needed
}
for col in X.columns:
    if col not in example_features:
        example_features[col] = 0
new_X = pd.DataFrame([example_features])[X.columns]
print("Prediction for custom example:", model.predict(new_X))

# Example: Use a stricter threshold (e.g., 0.3 instead of 0.5)
proba = model.predict_proba(new_X)[0][1]  # Probability of phishing (class 1)
threshold = 0.3  # Make this lower for stricter detection
prediction = int(proba > threshold)
print(f"Phishing probability: {proba:.2f} | Prediction: {'Phishing' if prediction else 'Legitimate'}")

print("\n=== Model Feature Alignment Check ===")
print("Model expects features (in order):")
for i, col in enumerate(X.columns):
    print(f"{i+1:2d}. {col}")
print("\nExample features provided (in order):")
for i, col in enumerate(new_X.columns):
    print(f"{i+1:2d}. {col}")

print("\n=== First 5 Test Predictions ===")
print("Predicted labels: ", model.predict(Xtest.iloc[:5]))
print("True labels     : ", ytest.iloc[:5].values)

print("\n=== Feature Statistics (Test Set) ===")
print(X.describe().T[['mean', 'std', 'min', 'max']])