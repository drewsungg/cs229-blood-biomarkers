import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

def run_adaboost_pred(filepath):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filepath, sep='\t')
    df = df[df['gender'] != 'http://7070652042092723530.owasp.org']

    # One-hot encode the 'gender' column
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    # Separate the features and the target variable
    X = df.drop('group', axis=1)
    y = df['group']

    # Drop 'bmi' and 'age' if they are not to be used
    X = X.drop('bmi', axis=1)
    # X = X.drop('age', axis=1)

    # Define the columns to be processed
    columns = X.columns.tolist()
    numerical_columns = [col for col in columns if col not in ['gender_m']]
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

    # Initialize the base estimator for AdaBoost
    base_estimator = DecisionTreeClassifier(max_depth=1)

    # Create a pipeline that preprocesses and then fits an AdaBoost model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k='all')), 
        ('classifier', AdaBoostClassifier(base_estimator=base_estimator,
                                        n_estimators=50,
                                        learning_rate=1.0,
                                        random_state=42))
    ])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the AdaBoost model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    return (y_pred)

def run_adaboost_test(filepath):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filepath, sep='\t')
    df = df[df['gender'] != 'http://7070652042092723530.owasp.org']

    # One-hot encode the 'gender' column
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    # Separate the features and the target variable
    X = df.drop('group', axis=1)
    y = df['group']

    # Drop 'bmi' and 'age' if they are not to be used
    X = X.drop('bmi', axis=1)
    # X = X.drop('age', axis=1)

    # Define the columns to be processed
    columns = X.columns.tolist()
    numerical_columns = [col for col in columns if col not in ['gender_m']]
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

    # Initialize the base estimator for AdaBoost
    base_estimator = DecisionTreeClassifier(max_depth=1)

    # Create a pipeline that preprocesses and then fits an AdaBoost model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k='all')), 
        ('classifier', AdaBoostClassifier(base_estimator=base_estimator,
                                        n_estimators=50,
                                        learning_rate=1.0,
                                        random_state=42))
    ])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the AdaBoost model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    return (y_test)
    
def run_adaboost_pred_proba(filepath):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filepath, sep='\t')
    df = df[df['gender'] != 'http://7070652042092723530.owasp.org']

    # One-hot encode the 'gender' column
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    # Separate the features and the target variable
    X = df.drop('group', axis=1)
    y = df['group']

    # Drop 'bmi' and 'age' if they are not to be used
    X = X.drop('bmi', axis=1)
    # X = X.drop('age', axis=1)

    # Define the columns to be processed
    columns = X.columns.tolist()
    numerical_columns = [col for col in columns if col not in ['gender_m']]
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

    # Initialize the base estimator for AdaBoost
    base_estimator = DecisionTreeClassifier(max_depth=1)

    # Create a pipeline that preprocesses and then fits an AdaBoost model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k='all')), 
        ('classifier', AdaBoostClassifier(base_estimator=base_estimator,
                                        n_estimators=50,
                                        learning_rate=1.0,
                                        random_state=42))
    ])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the AdaBoost model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    return (y_pred_proba)