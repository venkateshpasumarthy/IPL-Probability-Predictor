import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv('data/final_dataset.csv')

# -------- Feature Engineering --------
df['runs_left'] = df['target'] - df['score']
df['balls_left'] = 120 - (df['overs'] * 6)
df['wickets_left'] = 10 - df['wickets']

df['current_rr'] = df['score'] / df['overs']
df['required_rr'] = (df['runs_left'] * 6) / df['balls_left']

# Drop invalid rows
df = df.replace([float('inf'), -float('inf')], 0)
df = df.dropna()

# Features
X = df[['runs_left','balls_left','wickets_left','current_rr','required_rr']]
y = df['result']  # 1 = win, 0 = lose

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------- Models --------
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)

    print(f"{name} Accuracy: {score}")

    if score > best_score:
        best_score = score
        best_model = model

# Save best model
pickle.dump(best_model, open('model.pkl', 'wb'))

print("Best model saved!")
