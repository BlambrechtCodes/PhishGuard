# Imports | To perform operations on dataset
import pandas as pd
import numpy as np
# Machine learning model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# Visualization
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz

# Reading and Splitting the Dataset
df = pd.read_csv('.../dataset.csv')
dot_file = '.../tree.dot'
confusion_matrix_file = '.../confusion_matrix.png'

# Printing the Results of the Read Operation
print(df.head())
