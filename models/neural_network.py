import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.metrics import classification_report


# Load the dataset
df = pd.read_csv('./bloodDataNAN2.csv', sep='\t')
df = df[df['gender'] != 'http://7070652042092723530.owasp.org']

# One-hot encode the 'gender' column
df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Separate the features and the target variable
X = df.drop('group', axis=1)
y = df['group']

# Define the columns to be processed
columns = X.columns.tolist()
numerical_columns = [col for col in columns if col not in ['gender_m']]
categorical_columns = ['gender_m']

# Define the preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', 'passthrough', categorical_columns)
    ])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)
y = y.values  # Ensure y is a numpy array


def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')  # num_classes is the number of exercise levels
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Change to 'categorical_crossentropy' for multi-class
                  metrics=['accuracy'])
    return model

# Get the number of features from the preprocessed data
input_shape = X_preprocessed.shape[1]
model = create_model(input_shape)


X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.1)


# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")  # Threshold for binary classification

print(classification_report(y_test, y_pred_classes))
