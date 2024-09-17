# Multi-Class Prediction of Obesity Risk (Kaggle Playground Series S4E2)

This repository contains the code, analysis, and results for the **Multi-Class Prediction of Obesity Risk** problem, part of the **Kaggle Playground Series S4E2** challenge.

The challenge focuses on building machine learning models to predict the risk of obesity using a synthetically generated dataset. The goal is to classify individuals into one of seven categories of obesity risk based on features like age, height, weight, and lifestyle habits.

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Models Implemented](#models-implemented)
3. [Feature Engineering](#feature-engineering)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Results](#results)
6. [How to Use](#how-to-use)
7. [References](#references)

## Dataset Overview

The dataset used for this challenge is synthetically generated based on obesity risk data. It consists of various features like:

- **Gender**: Gender of the individual
- **Age**: Age of the individual
- **Height**: Height of the individual
- **Weight**: Weight of the individual
- **family_history_with_overweight**: Whether there is a family history of being overweight
- **FAVC**: Frequent consumption of high-calorie food
- **CAEC**: Consumption of food between meals
- **CH2O**: Daily water intake
- ...and other lifestyle-related factors.

### Data Files
- **train.csv**: The training dataset with features and target variable (`NObeyesdad`).
- **test.csv**: The test dataset with features only.
- **sample_submission.csv**: A sample submission file showing the correct format.

## Models Implemented

A variety of machine learning models were implemented, including:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Gradient Boosting (XGBoost)**
- **LightGBM**
- **CatBoost**
- **K-Nearest Neighbors (KNN)**
- **Model Stacking**

## Feature Engineering

The dataset contains a mix of categorical and numerical features. The following preprocessing steps were applied:

- **One-Hot Encoding**: Applied to categorical features.
- **Standard Scaling**: Applied to numerical features.
- **Label Encoding**: Target variable (`NObeyesdad`) was label encoded for compatibility with models.

## Hyperparameter Tuning

Hyperparameter tuning was performed using **RandomizedSearchCV** and **GridSearchCV** to optimize the following models:

- **XGBoost**
- **LightGBM**
- **CatBoost**

Tuning parameters included learning rate, max depth, subsample, and more.

## Results

| **Model**        | **Accuracy** |
|------------------|--------------|
| **XGBoost**      | 0.9065       |
| **LightGBM**     | 0.9064       |
| **CatBoost**     | 0.9063       |
| **Logistic Regression** | 0.8630 |
| **Random Forest** | 0.8902 |
| **SVM**          | 0.8778       |
| **KNN**          | 0.7575       |
| **AdaBoost**     | 0.5280       |

Model stacking was also explored using logistic regression as the meta-model, which provided slight improvements.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/surajwate/S4E2-Multi-Class-Prediction-of-Obesity-Risk.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. To run the model training:
   ```bash
   python main.py --model xgboost
   ```

4. For hyperparameter tuning:
   ```bash
   python src/grid_search.py --model xgboost
   ```

5. Submit predictions:
   ```bash
   python src/predict.py
   ```

## References

- [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s4e2/data)
- [Blog Post](https://surajwate.com/blog/multi-class-prediction-of-obesity-risk/)
- [Kaggle Notebook](https://www.kaggle.com/code/surajwate/s4e2-prediction-of-obesity-risk)

