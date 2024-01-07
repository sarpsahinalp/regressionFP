import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create a DataFrame
df = pd.read_csv('FirstDB.csv', sep=';')

# Preprocess the data
# For simplicity, we'll encode 'BOT' as 0 for 'NOT' and 1 for 'BOT'
df['BOT'] = df['BOT'].map({'NOT': 0, 'BOT': 1})

# Select relevant features
features = ['Res']  # You might want to include more features based on your analysis

# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Platform', 'OS', 'Software', 'WebGL'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df['BOT'], test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report_str)