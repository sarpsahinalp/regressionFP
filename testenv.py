import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

response = requests.get('https://fingerprint-server-czzzoqqzqa-ey.a.run.app/api/bot-data')

df = pd.json_normalize(response.json())
df = df.drop(['id.timestamp'], axis=1)
df = df.drop(['id.date'], axis=1)
df = df.drop(0)
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
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# If you want to visualize the decision tree (optional)
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)

log = LogisticRegression()
log.fit(X_train, y_train)
cm = confusion_matrix(y_test, log.predict(X_test))
report = classification_report(y_test, log.predict(X_test), output_dict=True)
print('Accuracy: ', report['accuracy'])
print('Precision: ', report['False']['precision'])
print('Recall: ', report['False']['recall'])
print('F1 Score: ', report['False']['f1-score'])

requests.put('https://fingerprint-server-czzzoqqzqa-ey.a.run.app/api/regression-data/659a06088f4c9270212e092b',
             json={'accuracy': report['accuracy'], 'precision': report['False']['precision'],
                   'recall': report['False']['recall'], 'f1Score': report['False']['f1-score']})
