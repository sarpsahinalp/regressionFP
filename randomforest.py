import pandas as pd
import requests
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
# import random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn import tree

# Create a data from .json file in the same directory
df = pd.read_json('fingerprints.botData.json')
df = df.drop(['_id'], axis=1)
df = df.drop(['_class'], axis=1)
df = df.drop('process', axis=1)
# Replace "None" with a placeholder value (e.g., 0) in the 'bot' column
df['bot'] = df['bot'].replace({'None': 0})

# Convert boolean columns to numeric
bool_columns = ['bot', 'android', 'documentFocus', 'notificationPermissions', 'pluginsArray', 'webDriver']
df[bool_columns] = df[bool_columns].astype('bool')

# Fill NaN values with a default value or use other imputation techniques
df = df.fillna(0)

# Convert the 'languages' column to a more usable format
df = df.drop(['languages'], axis=1)

# Handle the 'distinctiveProps' column containing JSON-like data
df = df.drop(['distinctiveProps'], axis=1)
df = df.drop(['documentElementKeys'], axis=1)
df = df.drop(['functionBind'], axis=1)
df = df.drop(['userAgent'], axis=1)
df = df.drop(['windowExternal'], axis=1)
# One-hot encode categorical columns
categorical_columns = ['appVersion', 'browserEngineKind', 'browserKind', 'webGlVendor', 'webGlRenderer']
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Feature matrix (X) and target variable (y)
X = df_encoded.drop('bot', axis=1)
y = df_encoded['bot']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize the Decision Tree model
clf = RandomForestClassifier()

# Train the model
clf.fit(X_train, y_train)

for i in range(0, 3):
    plt.figure(figsize=(10, 5))
    tree.plot_tree(clf.estimators_[i], filled=True)
    plt.title(f'Tree {i+1}')
    plt.show()

# Evaluate the model on the test set
print(confusion_matrix(y_test, clf.predict(X_test)))
report = classification_report(y_test, clf.predict(X_test), output_dict=True)
print('Accuracy: ', report['accuracy'])
print('Precision: ', report['False']['precision'])
print('Recall: ', report['False']['recall'])
print('F1 Score: ', report['False']['f1-score'])
print('Precision: ', report['True']['precision'])
print('Recall: ', report['True']['recall'])
print('F1 Score: ', report['True']['f1-score'])
