import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def data_loader():
    df = pd.read_csv('great_customers.csv')
    return df

def clean_data(df):
    df.drop_duplicates(inplace=True)
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
    return df

def encode_categorical_features(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def split_features_and_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def select_features(X, y):
    k_best = SelectKBest(score_func=f_classif, k='all')
    X_selected = k_best.fit_transform(X, y)
    return X_selected

def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier()
    }
    # Trainiing and evaluating all models
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    return models

def ensemble_model(models, X_train, X_test, y_train, y_test):
    #ensembling using Voting techniques
    voting_clf = VotingClassifier(estimators=list(models.items()), voting='soft')
    voting_clf.fit(X_train, y_train)
    y_pred_ensemble = voting_clf.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    print(f"\nEnsemble Model (Voting) Accuracy: {ensemble_accuracy:.4f}")

    # Classification report after ensembling 
    print("\nClassification Report for Ensemble Model:")
    print(classification_report(y_test, y_pred_ensemble))

def main():
    df = data_loader()

    df = clean_data(df)

    df = encode_categorical_features(df)

    X, y = split_features_and_target(df, 'great_customer_class')

    X_selected = select_features(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    X_train, X_test = standardize_features(X_train, X_test)

    models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    ensemble_model(models, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
