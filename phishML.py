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

mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.savefig(confusion_matrix_file)

export_graphviz(model, out_file=dot_file, feature_names=X.columns.values)