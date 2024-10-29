import os

os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('fifa19.csv')
Other_data = dataset.drop(columns=['Overall'])
target_variable=dataset['Overall']

for column in Other_data.columns:
    if Other_data[column].dtype == 'object':
        label_encoder = LabelEncoder()
        Other_data[column] = label_encoder.fit_transform(Other_data[column].astype(str))

if target_variable.dtype == 'object':
    label_encoder_y = LabelEncoder()
    target_variable = label_encoder_y.fit_transform(target_variable)


imputer = SimpleImputer(strategy='mean')
Other_data = imputer.fit_transform(Other_data)

Other_data = np.array(Other_data)
target_variable = np.array(target_variable)

X_train,X_test,y_train,y_test = train_test_split(Other_data,target_variable,test_size=0.3,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def time_calculation(model, name_of_model,X_train,y_train,X_test):
    start_time = time.time()
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    end_time = time.time()
    time_difference = end_time - start_time
    print(f"{name_of_model} took {time_difference:.4f} seconds")
    return time_difference


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

times = {}

for name_of_model, model in models.items():
    time_taken = time_calculation(model,name_of_model,X_train,y_train,X_test)
    times[name_of_model] = time_taken

sorted_times = sorted(times.items(), key=lambda x: x[1])

print("\nOrder of algorithms in terms of running times (in seconds):")
for algorithm, duration in sorted_times:
    print(f"{algorithm}: {duration:.4f} seconds")