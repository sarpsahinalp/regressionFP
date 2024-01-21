import pandas as pd
import requests
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

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

# If you want to visualize the decision tree (optional)
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(32,32), dpi=300)

tree.plot_tree(clf);
fig.savefig('plottreedefault.png')
requests.put('https://fingerprint-server-czzzoqqzqa-ey.a.run.app/api/regression-data/659a06088f4c9270212e092b',
             json={'accuracy': report['accuracy'], 'precision': report['False']['precision'],
               'recall': report['False']['recall'], 'f1Score': report['False']['f1-score']})
