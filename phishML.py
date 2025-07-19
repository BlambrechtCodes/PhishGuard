# To perform operations on dataset
# To perform operations on dataset
import pandas as pd
import numpy as np
# Machine learning model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# Visualization
# Visualization
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc

# Load the dataset
df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')
dot_file = 'tree.dot'
confusion_matrix_file = 'confusion_matrix.png'

# Display the first few rows of the dataset
print(df.head())

# Drop 'FILENAME' and 'URL' columns, keep only numeric features
X = df.select_dtypes(include=[np.number])
y = df['label']

# Train a Decision Tree Classifier
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

# Create and fit the model
model = DecisionTreeClassifier()
model.fit(Xtrain, ytrain)

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