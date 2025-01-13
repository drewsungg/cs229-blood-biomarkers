import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

def run_rand_forest_y_pred (filepath): 
    # Load the dataset with specified column names
    df = pd.read_csv(filepath, sep='\t')
    print(df.head())
    df = df[df['gender'] != 'http://7070652042092723530.owasp.org']

    # Check the unique values in the 'gender' column again
    print(df['gender'].unique())

    # One-hot encode the 'gender' column
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    # Separate the features and the target variable
    X = df.drop('group', axis=1)
    y = df['group']

    # Drop 'bmi' and 'age' if they are not to be used
    X = X.drop('bmi', axis=1)
    # X = X.drop('age', axis=1)

    # Define the columns that will be scaled - numerical ones, so all cols excluding gender which is categorical
    columns = X.columns.tolist()
    numerical_columns = [col for col in columns if col != 'gender_m']
    categorical_columns = ['gender_m']

    # Create a ColumnTransformer to impute and scale numerical features
    # and leave the binary/one-hot encoded columns as is
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', 'passthrough', categorical_columns)
        ])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline that preprocesses and then fits a Random Forest model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    return y_pred

def run_rand_forest_y_test (filepath): 
    # Load the dataset with specified column names
    df = pd.read_csv(filepath, sep='\t')
    print(df.head())
    df = df[df['gender'] != 'http://7070652042092723530.owasp.org']

    # Check the unique values in the 'gender' column again
    print(df['gender'].unique())

    # One-hot encode the 'gender' column
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    # Separate the features and the target variable
    X = df.drop('group', axis=1)
    y = df['group']

    # Drop 'bmi' and 'age' if they are not to be used
    X = X.drop('bmi', axis=1)
    # X = X.drop('age', axis=1)

    # Define the columns that will be scaled - numerical ones, so all cols excluding gender which is categorical
    columns = X.columns.tolist()
    numerical_columns = [col for col in columns if col != 'gender_m']
    categorical_columns = ['gender_m']

    # Create a ColumnTransformer to impute and scale numerical features
    # and leave the binary/one-hot encoded columns as is
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', 'passthrough', categorical_columns)
        ])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline that preprocesses and then fits a Random Forest model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    return y_test

def run_rand_forest_y_pred_proba (filepath): 
    # Load the dataset with specified column names
    df = pd.read_csv(filepath, sep='\t')
    print(df.head())
    df = df[df['gender'] != 'http://7070652042092723530.owasp.org']

    # Check the unique values in the 'gender' column again
    print(df['gender'].unique())

    # One-hot encode the 'gender' column
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    # Separate the features and the target variable
    X = df.drop('group', axis=1)
    y = df['group']

    # Drop 'bmi' and 'age' if they are not to be used
    X = X.drop('bmi', axis=1)
    # X = X.drop('age', axis=1)

    # Define the columns that will be scaled - numerical ones, so all cols excluding gender which is categorical
    columns = X.columns.tolist()
    numerical_columns = [col for col in columns if col != 'gender_m']
    categorical_columns = ['gender_m']

    # Create a ColumnTransformer to impute and scale numerical features
    # and leave the binary/one-hot encoded columns as is
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', 'passthrough', categorical_columns)
        ])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline that preprocesses and then fits a Random Forest model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    return y_pred_proba