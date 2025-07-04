﻿# PRODIGY_ML_01
# House Price Prediction - Prodigy Infotech ML Internship Task 1

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-lightgrey?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-green?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-red?style=for-the-badge&logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-purple?style=for-the-badge&logo=seaborn)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

## Project Overview

This repository contains the solution for **Task 1: House Price Prediction** as part of the Prodigy Infotech Machine Learning Internship. The goal of this project is to develop a machine learning model that can accurately predict house sale prices based on various features of the houses.

A **Linear Regression** model has been implemented to establish the relationship between house characteristics and their selling prices. The project workflow includes data loading, essential preprocessing (missing value handling), feature engineering, data splitting, feature scaling, model training, evaluation, and generating a submission file.

## Problem Statement

The objective is to predict the `SalePrice` of houses given a set of features describing various aspects of residential homes in Ames, Iowa. This is a classic regression problem in machine learning.

## Dataset

The dataset used for this project is typically sourced from Kaggle's "House Prices - Advanced Regression Techniques" competition. It consists of:

* **`train.csv`**: Contains the training data with various features and the `SalePrice` (target variable).
* **`test.csv`**: Contains the test data with the same features as `train.csv` but without the `SalePrice`. Predictions for these houses need to be made.
* **`sample_submission.csv`**: Provides the format for the submission file.

**Note**: Please ensure these CSV files (`train.csv`, `test.csv`, `sample_submission.csv`) are placed in the root directory of this project alongside the `main.py` script for the code to run correctly.

## Features Used

Based on the problem statement in the provided code, the following features are primarily used for training the model:

* `GrLivArea`: Above grade (ground) living area square feet
* `BedroomAbvGr`: Number of bedrooms above grade (excluding basement bedrooms)
* `FullBath`: Full bathrooms above grade
* `HalfBath`: Half baths above grade
* **Engineered Feature**: `TotalBath` (calculated as `FullBath + 0.5 * HalfBath`)

The target variable is `SalePrice`.

## Project Structure
├── main.py                     # Main script for the house price prediction pipeline
├── train.csv                   # Training dataset (download from Kaggle)
├── test.csv                    # Test dataset (download from Kaggle)
├── sample_submission.csv       # Sample submission format (download from Kaggle)
├── linear_regression_model.joblib # Saved trained Linear Regression model
├── feature_scaler.joblib      # Saved StandardScaler object
├── submission.csv              # Generated submission file with predictions
└── README.md                   # This README file


## Setup and Installation

To run this project, you need Python installed (preferably Python 3.8+). You can set up a virtual environment and install the required libraries:

```bash
# Clone the repository (if you haven't already)
git clone [https://github.com/your-username/your-repo-name.git]
https://github.com/cheekati-Saicharan/PRODIGY_ML_01#prodigy_ml_01
# (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required Python packages
pip install pandas scikit-learn matplotlib seaborn numpy joblib
Important: Download train.csv, test.csv, and sample_submission.csv from the Kaggle House Prices Competition and place them in the root directory of this project.

How to Run
Execute the main.py script from your terminal:

Bash

python main.py
The script will:

Load the datasets.
Handle missing values and engineer the TotalBath feature.
Split the training data into training and validation sets.
Scale the selected features using StandardScaler.
Train a Linear Regression model.
Evaluate the model on the validation set, displaying MAE and R2 score.
Save the trained model (linear_regression_model.joblib) and the scaler (feature_scaler.joblib).
Make predictions on the test.csv dataset.
Generate a submission.csv file in the required Kaggle format.
Display a visualization of actual vs. predicted prices on the validation set.
All steps are logged to the console for clear tracking.

Model Details
Model Type: Linear Regression
Features: GrLivArea, BedroomAbvGr, TotalBath (derived from FullBath, HalfBath)
Preprocessing:
Missing numerical values (in selected features) are imputed with the median from the training set.
MSZoning in the test set is filled with its mode if missing.
Features are scaled using StandardScaler to ensure all features contribute equally to the distance calculations (though for Linear Regression, it primarily helps with gradient descent-based solvers and regularization, here it prepares for potential future complex models).
Evaluation Metrics: Mean Absolute Error (MAE) and R-squared (R2 Score).
Performance Metrics (on Validation Set)
The script will output metrics similar to:

--- Model Performance Metrics (on Validation Set) ---
Mean Absolute Error (MAE): $27,000.00 (Example value)
R-squared (R2) Score: 0.7500 (Example value)
(These values are examples and will vary based on data split and model performance)

Visualization
A scatter plot comparing Actual Sale Price vs. Predicted Sale Price on the validation set will be displayed, along with a detailed explanation of how to interpret the plot. This helps in visually assessing the model's prediction accuracy and identifying potential biases or outliers.

Future Enhancements
More Robust Feature Engineering: Explore creating additional features from existing ones (e.g., AreaPerRoom, AgeOfHouse).
Handling Categorical Features: Implement One-Hot Encoding for categorical features like MSZoning, Neighborhood, etc., to potentially improve model accuracy.
Advanced Missing Value Imputation: Explore more sophisticated imputation techniques (e.g., K-Nearest Neighbors imputation).
Outlier Detection and Handling: Identify and treat outliers in SalePrice or feature distributions.
Regularization: Implement Lasso or Ridge Regression to prevent overfitting, especially if more features are added.
Hyperparameter Tuning: Use techniques like GridSearchCV or RandomizedSearchCV to find optimal parameters for the model (though Linear Regression has few hyperparameters).
Other Regression Models: Experiment with more powerful models like RandomForest Regressor, Gradient Boosting Regressor (e.g., XGBoost, LightGBM).
License
This project is licensed under the MIT License - see the LICENSE file (if you add one) for details.

