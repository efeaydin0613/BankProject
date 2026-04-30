import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score

# 1. Data Loading
df = pd.read_csv('bank-additional.csv', sep=';')

# 2. Data Cleaning — drop 'duration' (data leakage)
df = df.drop('duration', axis=1)

# 3. Feature Engineering — Real Interest Rate (Fisher equation)
df['real_interest_rate'] = df['euribor3m'] - df['cons.price.idx']

# 4. Prepare features and target
X = df.drop('y', axis=1)
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# 5. Preprocessing Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 6. Model Selection — compare 3 model families
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

param_grid = [
    {
        'classifier': [LogisticRegression(max_iter=1000)],
        'classifier__C': [0.01, 0.1, 1, 10]
    },
    {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20]
    },
    {
        'classifier': [MLPClassifier(max_iter=500, random_state=42)],
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)]
    }
]

# 7. Training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 8. Save best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'bank_model.pkl')

print("Best model saved as 'bank_model.pkl'.")
print("Best Parameters:", grid_search.best_params_)
print(f"F1 Score: {f1_score(y_test, best_model.predict(X_test)):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))
