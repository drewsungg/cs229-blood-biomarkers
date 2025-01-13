import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Define a custom scaler to scale features by their F-scores
def scale_by_fscore(X, f_scores):
    for feature in f_scores:
        if feature in X.columns:
            X[feature] *= f_scores[feature]
    return X

def lightGBM_y_pred_proba(filepath): 
    # Correcting the DtypeWarning by specifying low_memory=False
    df = pd.read_csv(filepath, sep='\t', low_memory=False)

    # Filter out unwanted rows and drop rows with NaN in 'group'
    df = df[df['gender'] != 'http://7070652042092723530.owasp.org']
    df.dropna(subset=['group'], inplace=True)

    # One-hot encode 'gender'
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    # Label encode 'group' (target)
    le = LabelEncoder()
    df['group'] = le.fit_transform(df['group'])
    y = df['group'].values

    # Extracting feature names for later use
    feature_names = df.columns.drop('group')

    X = df.drop('group', axis=1)

    # F-Scores for numerical features provided earlier
    f_scores = {
        'HDL': 199.163, 'gender_m': 163.194, 'Tg': 120.357, 'Tes': 95.449,
        'age': 89.918, 'LDL': 78.048, 'SHBG': 76.638, 'Fer': 72.747,
        'RBC': 56.805, 'MCH': 56.244, 'MCV': 55.928, 'AST': 55.057,
        'D': 50.465, 'hsCRP': 46.021, 'Fol': 42.753, 'WBC': 42.121,
        'HgbA1c': 41.601, 'NEUT': 40.046, 'TIBC': 39.820, 'RDW': 35.490,
        'MONOS': 34.640, 'TS': 29.818, 'Glu': 29.641, 'HCT': 28.743,
        'FE': 26.844, 'LYMPHS': 26.544, 'PLT': 25.861, 'Cor': 25.812,
        'GGT': 23.920, 'B12': 20.762, 'FT': 17.661, 'Chol': 16.352,
        'RBC_Mg': 14.471, 'K': 10.790, 'Ca': 9.493, 'CK': 9.452,
        'Mg': 9.197, 'ALT': 8.630, 'MPV': 8.182, 'NA': 7.731,
        'EOS_PCT': 7.363, 'BASOS_PCT': 6.254, 'Alb': 6.194, 'EOS': 5.731,
        'DHEAS': 5.346, 'Hb': 2.815, 'NEUT_PCT': 1.301, 'BASOS': 1.138,
        'MONOS_PCT': 0.996, 'LYMPHS_PCT': 0.759, 'MCHC': 0.734
    }

    # Scale numerical features by their F-scores
    X_scaled = scale_by_fscore(X.copy(), f_scores)

    # Define the preprocessing for numerical columns
    numerical_columns = X_scaled.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]), numerical_columns)
    ])

    X_preprocessed = preprocessor.fit_transform(X_scaled)
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=numerical_columns)

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Prepare the LightGBM datasets
    d_train = lgb.Dataset(X_train, label=y_train)

    # Define LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'num_class': len(np.unique(y)),
        'metric': 'multi_logloss'
    }

    # Train the model
    clf = lgb.train(params, d_train, 100)

    # Make predictions on the test set
    y_pred_proba = clf.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    return y_pred_proba

def lightGBM_y_pred (filepath): 
    # Correcting the DtypeWarning by specifying low_memory=False
    df = pd.read_csv(filepath, sep='\t', low_memory=False)

    # Filter out unwanted rows and drop rows with NaN in 'group'
    df = df[df['gender'] != 'http://7070652042092723530.owasp.org']
    df.dropna(subset=['group'], inplace=True)

    # One-hot encode 'gender'
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    # Label encode 'group' (target)
    le = LabelEncoder()
    df['group'] = le.fit_transform(df['group'])
    y = df['group'].values

    # Extracting feature names for later use
    feature_names = df.columns.drop('group')

    X = df.drop('group', axis=1)

    # F-Scores for numerical features provided earlier
    f_scores = {
        'HDL': 199.163, 'gender_m': 163.194, 'Tg': 120.357, 'Tes': 95.449,
        'age': 89.918, 'LDL': 78.048, 'SHBG': 76.638, 'Fer': 72.747,
        'RBC': 56.805, 'MCH': 56.244, 'MCV': 55.928, 'AST': 55.057,
        'D': 50.465, 'hsCRP': 46.021, 'Fol': 42.753, 'WBC': 42.121,
        'HgbA1c': 41.601, 'NEUT': 40.046, 'TIBC': 39.820, 'RDW': 35.490,
        'MONOS': 34.640, 'TS': 29.818, 'Glu': 29.641, 'HCT': 28.743,
        'FE': 26.844, 'LYMPHS': 26.544, 'PLT': 25.861, 'Cor': 25.812,
        'GGT': 23.920, 'B12': 20.762, 'FT': 17.661, 'Chol': 16.352,
        'RBC_Mg': 14.471, 'K': 10.790, 'Ca': 9.493, 'CK': 9.452,
        'Mg': 9.197, 'ALT': 8.630, 'MPV': 8.182, 'NA': 7.731,
        'EOS_PCT': 7.363, 'BASOS_PCT': 6.254, 'Alb': 6.194, 'EOS': 5.731,
        'DHEAS': 5.346, 'Hb': 2.815, 'NEUT_PCT': 1.301, 'BASOS': 1.138,
        'MONOS_PCT': 0.996, 'LYMPHS_PCT': 0.759, 'MCHC': 0.734
    }

    # Scale numerical features by their F-scores
    X_scaled = scale_by_fscore(X.copy(), f_scores)

    # Define the preprocessing for numerical columns
    numerical_columns = X_scaled.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]), numerical_columns)
    ])

    X_preprocessed = preprocessor.fit_transform(X_scaled)
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=numerical_columns)

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Prepare the LightGBM datasets
    d_train = lgb.Dataset(X_train, label=y_train)

    # Define LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'num_class': len(np.unique(y)),
        'metric': 'multi_logloss'
    }

    # Train the model
    clf = lgb.train(params, d_train, 100)

    # Make predictions on the test set
    y_pred_proba = clf.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    return y_pred