import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

response = requests.get('https://fingerprint-server-czzzoqqzqa-ey.a.run.app/api/bot-data')

df = pd.json_normalize(response.json())

# Feature matrix (X) and target variable (y)
features = ["android", "webDriver", "innerHeight", "innerWidth", "outerHeight", "outerWidth"]
target = "bot"

# Create the design matrix (X) and the target variable (y)
X = df[features]
y = df[target]

# Add a constant term to the design matrix (required for statsmodels)
X = sm.add_constant(X)

# Fit logistic regression model
model = sm.Logit(np.asarray(y), np.asarray(X))
result = model.fit()

# Display the summary of the logistic regression model
print(result.summary())