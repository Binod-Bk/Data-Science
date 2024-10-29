

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Replace 4 with the actual number of physical cores

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'great_customers.csv'
data = pd.read_csv(file_path)

data_cleaned = data.drop(columns=['user_id'])

categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
numerical_columns = data_cleaned.select_dtypes(include=[np.number]).columns

imputer = SimpleImputer(strategy='mean')
data_cleaned[numerical_columns] = imputer.fit_transform(data_cleaned[numerical_columns])

data_cleaned = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)

Other_data = data_cleaned.drop(columns=['great_customer_class'])
Target_variable = data_cleaned['great_customer_class']

X_train, X_test, y_train, y_test = train_test_split(Other_data, Target_variable, test_size=0.3, random_state=42, stratify=Target_variable)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
selector = RFECV(lr, step=1, cv=5, scoring='accuracy')
selector = selector.fit(X_train, y_train)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier()
}

model_accuracies = {}

for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")

voting_clf = VotingClassifier(estimators=[
    ('rf', models['Random Forest']),
    ('svm', models['SVM']),
    ('lr', models['Logistic Regression']),
    ('nb', models['Naive Bayes']),
    ('knn', models['KNN'])
], voting='soft')

voting_clf.fit(X_train_selected, y_train)
y_pred_voting = voting_clf.predict(X_test_selected)
voting_accuracy = accuracy_score(y_test, y_pred_voting)

model_accuracies['Voting Classifier'] = voting_accuracy

print("\nModel Accuracies:")
for model, acc in model_accuracies.items():
    print(f"{model}: {acc:.4f}")
