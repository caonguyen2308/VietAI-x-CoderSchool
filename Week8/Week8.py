import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = pd.read_excel('job_dataset.ods', dtype=str)
target = 'career_level'
x = data.drop(labels= target, axis=1)
y = data[target]
print(y.value_counts())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32, stratify=y)

# vectorizer = TfidfVectorizer()
# output = vectorizer.fit_transform(x_train['title'])
# print(output.shape)
# print(vectorizer.vocabulary_)

# def filter_location(location):
#     if ',' in location:
#         return location [-2:]
#     else:
#         return location
    
# data['location'] = data['location'].apply(filter_location)
# data['location']

# encoder = OneHotEncoder()
# output = encoder.fit_transform(x_train[['function']])
# print(output.shape)

x_train.dropna(subset=['description'], inplace=True)
preprocessor = ColumnTransformer(transformers=[
    ('title', TfidfVectorizer(max_features=5000), 'title'),
    ('location', OneHotEncoder(handle_unknown='ignore'), ['location']),
    ('description', TfidfVectorizer(ngram_range=(1,1), stop_words='english', max_features=5000), 'description'),
    ('industry', TfidfVectorizer(stop_words='english', max_features=5000), 'industry')
])

y_train = y_train[x_train.index]

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))