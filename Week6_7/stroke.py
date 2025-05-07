import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Step 1: Load and Clean Data

# Assuming you already have a dataframe (e.g., 'stroke_data.csv' which contains the stroke dataset)
df = pd.read_csv("stroke_classification.csv")  # Replace with your actual file path

# Drop any columns not needed for prediction
df = df.drop(columns=['pat_id'], errors='ignore')  # Drop the patient ID if present

# Encode the target variable (stroke)
y = df['stroke']  # 1: has "stroke", 0: "no stroke" 

# Drop the target column from the features
X = df.drop(columns=['stroke'])

# Identify feature types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 3: Build Preprocessing Pipelines

# Pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Step 4: Define Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Step 5: Train and Evaluate Models
accuracy_scores = {}
roc_auc_scores = {}

for name, model in models.items():
    # Build full pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    accuracy_scores[name] = acc
    roc_auc_scores[name] = roc

# Step 6: Display Results
results_df = pd.DataFrame({
    'Accuracy': accuracy_scores,
    'ROC AUC': roc_auc_scores
}).sort_values(by='Accuracy', ascending=False)

print(results_df)

