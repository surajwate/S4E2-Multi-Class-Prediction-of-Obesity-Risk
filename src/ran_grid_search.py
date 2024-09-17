import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engineering import FeatureEngineer
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import RandomizedSearchCV

import os
import argparse
import joblib

import config
import model_dispatcher

import time
import logging

# Set up logging
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

def run(fold, model):
    # Import the data
    df = pd.read_csv(config.TRAINING_FILE)

    # Split the data into training and testing
    train = df[df.kfold != fold].reset_index(drop=True)
    test = df[df.kfold == fold].reset_index(drop=True)

    # Split the data into features and target
    target_features = ['NObeyesdad']

    X_train = train.drop(['id', 'kfold'] + target_features, axis=1)
    X_test = test.drop(['id', 'kfold'] + target_features, axis=1)

    y_train = train[target_features].values
    y_test = test[target_features].values

    # Reshape the target arrays
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    # Initialize label encoder
    label_encoder = LabelEncoder()

    # Encode the target labels
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Define features
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

    # Create a column transformer for one-hot encoding and standard scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    # Define hyperparameters for random search for XGBoost or LightGBM
    if model in ['xgboost', 'lightgbm']:
        param_grid = {
        'model__n_estimators': np.arange(300, 400, 10),
        'model__learning_rate': np.linspace(0.09, 0.2, 10),
        'model__max_depth': [2, 3, 4],
        'model__subsample': np.linspace(0.7, 0.9, 5),
        'model__colsample_bytree': np.linspace(0.9, 1.0, 5),
        'model__gamma': np.linspace(0, 1, 10),  # Smaller range
        'model__reg_alpha': np.linspace(0, 1, 10),  # Reduced range
        'model__reg_lambda': np.linspace(0, 1, 10),  # Reduced range
        'model__min_child_weight': np.arange(1, 6),  # Reduced range
    }
    else:
        param_grid = {}

    # Create a pipeline with the preprocessor and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_dispatcher.models[model])
    ])

    # Use RandomizedSearchCV to find the best hyperparameters
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=50,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        cv=5
    )


    try:
        start = time.time()

        # logging.info(f"Fold={fold}, Model={model}")

        # Perform the random search
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        logging.info(f"Best parameters: {random_search.best_params_}")

        # make predictions
        preds = best_model.predict(X_test)

        end = time.time()
        time_taken = end - start

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, preds)

        logging.info(f"Fold={fold}, Accuracy = {accuracy:.4f}, Time Taken={time_taken:.2f}sec")
        print(f"Fold={fold}, Accuracy = {accuracy:.4f}, Time Taken={time_taken:.2f}sec")

        # Save the model
        joblib.dump(pipeline, os.path.join(config.MODEL_OUTPUT, f"model_{fold}.bin"))
    except Exception as e:
        logging.exception(f"Error occurred for Fold={fold}, Model={model}: {str(e)}")
    

if __name__ == '__main__':
    # Initialize the ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)

    # Read the arguments from the command line
    args = parser.parse_args()

    # Run the fold specified by the command line arguments
    for fold_ in range(5):
        run(fold=fold_, model=args.model)

