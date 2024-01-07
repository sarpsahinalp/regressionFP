import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

response = requests.get('https://fingerprint-server-czzzoqqzqa-ey.a.run.app/api/bot-data')

df = pd.json_normalize(response.json())
df = df.drop(['id.timestamp'], axis=1)
df = df.drop(['id.date'], axis=1)
df = df.drop(0)
# Replace "None" with a placeholder value (e.g., 0) in the 'bot' column
df['bot'] = df['bot'].replace({'None': 0})

# Convert boolean columns to numeric
bool_columns = ['bot', 'android', 'documentFocus', 'notificationPermissions', 'pluginsArray', 'webDriver']
df[bool_columns] = df[bool_columns].astype('int')

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

# Convert all columns to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

X_train = sm.add_constant(X_train)

# Fit logistic regression model
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()
