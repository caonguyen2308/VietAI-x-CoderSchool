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
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

# Step 1: Load and Clean Data

df = pd.read_csv("csgo.csv")
df = df.drop(columns=['team_a_rounds', 'team_b_rounds', 'tmp'], errors='ignore') 

# Drop "Tie" games (binary classification only)
df = df[df['result'] != "Tie"]

# Encode target
y = df['result'].map({'Win': 1, 'Lost': 0})

# Drop target column from features
X = df.drop(columns=['result'])

# Identify feature types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 3: Build Preprocessing 

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
    'Logistic Regression': LogisticRegression(max_iter=1000),
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

# Step 7: Plot Result
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = range(len(results_df))

# Accuracy bars
ax.bar(index, results_df['Accuracy'], bar_width, label='Accuracy', color='skyblue')

# ROC AUC bars
ax.bar([i + bar_width for i in index], results_df['ROC AUC'], bar_width, label='ROC AUC', color='seagreen')

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Comparison: Accuracy vs ROC AUC')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(results_df.index, rotation=15)
ax.legend()

plt.tight_layout()
plt.show()