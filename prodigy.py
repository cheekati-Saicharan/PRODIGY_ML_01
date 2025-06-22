import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler # For feature scaling
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib # For saving and loading the model
import logging
import os # For path operations and file existence checks

# --- Configuration ---
# Set up logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUBMISSION_FILE = 'sample_submission.csv'
SUBMISSION_OUTPUT_FILE = 'submission.csv'
MODEL_SAVE_PATH = 'linear_regression_model.joblib'
SCALER_SAVE_PATH = 'feature_scaler.joblib'

# Model parameters
TEST_SIZE_RATIO = 0.2
RANDOM_STATE_SEED = 42

# Features to be used for the model
FEATURES_BASE = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
TARGET_VARIABLE = 'SalePrice'


# --- Functions for Modularity ---

def load_data(train_path, test_path, sample_submission_path):
    """
    Loads training, testing, and sample submission datasets.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.
        sample_submission_path (str): Path to the sample submission CSV file.

    Returns:
        tuple: A tuple containing df_train, df_test, df_sample_submission DataFrames.
               Returns (None, None, None) if files are not found.
    """
    logging.info(f"Attempting to load '{train_path}', '{test_path}', and '{sample_submission_path}'...")
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df_sample_submission = pd.read_csv(sample_submission_path)
        logging.info(f"✅ Train dataset loaded successfully! Shape: {df_train.shape}")
        logging.info(f"✅ Test dataset loaded successfully! Shape: {df_test.shape}")
        logging.info(f"✅ Sample submission loaded successfully! (Used for final output format)")
        return df_train, df_test, df_sample_submission
    except FileNotFoundError as e:
        logging.error(f"❌ Error loading data: {e}. Please ensure all required CSV files are in the correct directory.")
        return None, None, None

def handle_missing_values(df_train, df_test, features_to_check):
    """
    Handles missing values in specified numerical features using the median from the training set.
    Also handles a specific categorical feature 'MSZoning' in the test set if missing.

    Args:
        df_train (pd.DataFrame): Training DataFrame.
        df_test (pd.DataFrame): Testing DataFrame.
        features_to_check (list): List of numerical features to check for NaNs.
    """
    logging.info("Checking and handling missing values in selected features...")
    for feature in features_to_check:
        if df_train[feature].isnull().any():
            median_val = df_train[feature].median()
            df_train[feature].fillna(median_val, inplace=True)
            logging.warning(f"  Warning: Missing values in '{feature}' (train set) filled with median: {median_val}.")
        if df_test[feature].isnull().any():
            # Use median from training data to prevent data leakage from test set
            median_val = df_train[feature].median() # Use train median for test set
            df_test[feature].fillna(median_val, inplace=True)
            logging.warning(f"  Warning: Missing values in '{feature}' (test set) filled with train median: {median_val}.")

    # Specific handling for 'MSZoning' in test set if it's present and has NaNs
    if 'MSZoning' in df_test.columns and df_test['MSZoning'].isnull().any():
        mode_val = df_test['MSZoning'].mode()[0]
        df_test['MSZoning'].fillna(mode_val, inplace=True)
        logging.warning(f"  Warning: Missing values in 'MSZoning' (test set) filled with mode: '{mode_val}'.")

def feature_engineer(df_train, df_test):
    """
    Performs feature engineering, specifically creating 'TotalBath'.

    Args:
        df_train (pd.DataFrame): Training DataFrame.
        df_test (pd.DataFrame): Testing DataFrame.

    Returns:
        tuple: Modified df_train and df_test DataFrames with new features.
    """
    logging.info("Creating a combined 'TotalBath' feature (FullBath + 0.5 * HalfBath) for consistency...")
    df_train['TotalBath'] = df_train['FullBath'] + (df_train['HalfBath'] * 0.5)
    df_test['TotalBath'] = df_test['FullBath'] + (df_test['HalfBath'] * 0.5)
    return df_train, df_test

def scale_features(X_train, X_val, X_test, features_to_scale):
    """
    Scales numerical features using StandardScaler. Fits on X_train and transforms all sets.

    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.
        features_to_scale (list): List of features to apply scaling to.

    Returns:
        tuple: Scaled X_train, X_val, X_test DataFrames, and the fitted StandardScaler.
    """
    logging.info("Applying StandardScaler to numerical features for optimal model performance...")
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_val_scaled[features_to_scale] = scaler.transform(X_val[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

    logging.info("✅ Features scaled successfully.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    """
    Trains the Linear Regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.

    Returns:
        sklearn.linear_model.LinearRegression: Trained Linear Regression model.
    """
    logging.info("Initializing and training the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    logging.info("✅ Linear Regression Model trained successfully!")
    return model

def evaluate_model(model, X_val, y_val, features_for_model):
    """
    Evaluates the model's performance on the validation set.

    Args:
        model (sklearn.linear_model.LinearRegression): Trained model.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target variable.
        features_for_model (list): List of features used in the model.
    """
    logging.info("Making predictions on the validation set to check model accuracy...")
    y_val_pred = model.predict(X_val)

    mae_val = mean_absolute_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)

    logging.info(f"\n--- Model Performance Metrics (on Validation Set) ---")
    logging.info(f"Mean Absolute Error (MAE): ${mae_val:,.2f}")
    logging.info(f"R-squared (R2) Score: {r2_val:.4f} (Closer to 1.0 indicates a better fit)")

    logging.info("\n--- Model Coefficients (Insights into Feature Impact) ---")
    logging.info("These values show how much the 'SalePrice' is expected to change for each unit increase in a feature, holding others constant.")
    for feature, coef in zip(features_for_model, model.coef_):
        logging.info(f"{feature}: {coef:,.2f} (A positive value indicates price increases with this feature, negative indicates decrease)")
    logging.info(f"Intercept: ${model.intercept_:,.2f} (Predicted price when all features are zero - often not directly interpretable)")
    return y_val, y_val_pred # Return for visualization

def make_final_predictions(model, X_test_predict):
    """
    Generates final predictions on the actual test dataset.

    Args:
        model (sklearn.linear_model.LinearRegression): Trained model.
        X_test_predict (pd.DataFrame): Features for the test dataset.

    Returns:
        np.array: Array of predicted prices.
    """
    logging.info("Applying the trained model to the 'test.csv' dataset to predict house prices.")
    final_predictions = model.predict(X_test_predict)

    initial_negative_count = np.sum(final_predictions < 0)
    if initial_negative_count > 0:
        logging.warning(f"  Warning: {initial_negative_count} negative predictions were found. Clipping them to 0.")
    final_predictions[final_predictions < 0] = 0
    return final_predictions

def generate_submission_file(df_test, final_predictions, output_file_name):
    """
    Creates and saves the submission file in Kaggle format.

    Args:
        df_test (pd.DataFrame): Original test DataFrame (to get 'Id' column).
        final_predictions (np.array): Array of predicted prices.
        output_file_name (str): Name of the CSV file to save the submission.
    """
    logging.info(f"Formatting predictions into '{output_file_name}' as required by Kaggle.")
    submission_df = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': final_predictions})
    submission_df['SalePrice'] = submission_df['SalePrice'].astype(int) # Convert to integer

    submission_df.to_csv(output_file_name, index=False)
    logging.info(f"✅ Submission file '{output_file_name}' created successfully!")
    logging.info("\nFirst 5 rows of the generated submission file:")
    logging.info(submission_df.head().to_string()) # Use to_string() for better logging of DataFrames
    logging.info(f"File saved at: {os.path.abspath(output_file_name)}")


def visualize_performance(y_val, y_val_pred):
    """
    Generates a scatter plot of actual vs. predicted prices for visualization.

    Args:
        y_val (pd.Series): Actual values from the validation set.
        y_val_pred (np.array): Predicted values for the validation set.
    """
    logging.info("\n--- Visualizing Model Performance (on Validation Set) ---")
    logging.info("Generating a scatter plot to visually assess how well our model's predictions align with actual house prices.")

    plt.figure(figsize=(12, 7))
    sns.scatterplot(x=y_val, y=y_val_pred, alpha=0.7, color='skyblue', edgecolor='black', s=50)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--', lw=2, label='Perfect Prediction Line (Actual = Predicted)')
    plt.title('Actual vs. Predicted House Prices (Validation Set)', fontsize=18, fontweight='bold')
    plt.xlabel('Actual Sale Price ($)', fontsize=14)
    plt.ylabel('Predicted Sale Price ($)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper left')
    plt.ticklabel_format(style='plain', axis='both') # Prevent scientific notation on axes
    plt.tight_layout()
    plt.show()

    logging.info("\n--- Detailed Visualization Description ---")
    logging.info("This scatter plot is a crucial tool for understanding our model's accuracy.")
    logging.info("Here's how to interpret what you see:")
    logging.info("")
    logging.info("  •  **X-axis (Actual Sale Price):** Represents the true, recorded sale prices of houses in our validation dataset.")
    logging.info("  •  **Y-axis (Predicted Sale Price):** Shows the prices that our Linear Regression model predicted for those same houses.")
    logging.info("")
    logging.info("  •  **The Red Dashed Line:** This line signifies a 'perfect prediction' scenario, where the model's predicted price is exactly equal to the actual price (Y = X).")
    logging.info("     If all points fell directly on this line, our model would be 100% accurate!")
    logging.info("")
    logging.info("  •  **Interpreting the Data Points:**")
    logging.info("     •  **Points very close to the red line:** These indicate highly accurate predictions. The model did a great job for these houses.")
    logging.info("     •  **Points above the red line:** For these houses, the model is **over-predicting** the price (predicted > actual).")
    logging.info("     •  **Points below the red line:** For these houses, the model is **under-predicting** the price (predicted < actual).")
    logging.info("")
    logging.info("  •  **What to Look For:**")
    logging.info("     •  **Tight Clustering:** A good model will have its data points tightly clustered around the red dashed line.")
    logging.info("     •  **No Obvious Patterns:** Ideally, you shouldn't see any clear patterns in the spread of points (e.g., all points below the line at higher prices), which might suggest systematic errors or limitations of the linear model.")
    logging.info("     •  **Outliers:** Points far away from the line are outliers where the model performed particularly poorly. Investigating these can reveal insights or data issues.")
    logging.info("")
    logging.info("This plot helps us visually confirm that, while not perfect, our linear model generally follows the trend of actual prices, with a reasonable spread of predictions.")


def main():
    """
    Main function to run the house price prediction pipeline.
    """
    logging.info("--- Starting House Price Prediction Pipeline ---")

    # 1. Load Data
    df_train, df_test, df_sample_submission = load_data(TRAIN_FILE, TEST_FILE, SAMPLE_SUBMISSION_FILE)
    if df_train is None:
        logging.error("Failed to load data. Exiting.")
        return

    # 2. Data Preprocessing and Feature Engineering
    logging.info("\n--- Step 2: Data Preprocessing and Feature Engineering ---")
    logging.info("Defining features based on problem statement: 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath'.")

    # Handle missing values first for the base features
    handle_missing_values(df_train, df_test, FEATURES_BASE)

    # Perform feature engineering
    df_train, df_test = feature_engineer(df_train, df_test)

    # Update features list to include the engineered feature
    features_for_model = FEATURES_BASE + ['TotalBath']
    # Filter out original FullBath and HalfBath as they are combined into TotalBath for the model
    features_for_model = [f for f in features_for_model if f not in ['FullBath', 'HalfBath']]

    logging.info(f"Final features selected for the model training: {features_for_model}")
    logging.info(f"Target variable for prediction: '{TARGET_VARIABLE}'")

    # Define X and y for training
    X_train_full = df_train[features_for_model]
    y_train_full = df_train[TARGET_VARIABLE]

    # Define X for final test prediction
    X_test_predict_raw = df_test[features_for_model]

    # Display summary of missing values after handling
    logging.info("\nMissing values after preprocessing (Train dataset, selected columns):")
    logging.info(df_train[features_for_model + [TARGET_VARIABLE]].isnull().sum().to_string())
    logging.info("\nMissing values after preprocessing (Test dataset, selected columns):")
    logging.info(df_test[features_for_model].isnull().sum().to_string())

    # 3. Split Training Data for Local Evaluation
    logging.info("\n--- Step 3: Splitting Data for Local Evaluation ---")
    X_train_eval, X_val, y_train_eval, y_val = train_test_split(
        X_train_full, y_train_full, test_size=TEST_SIZE_RATIO, random_state=RANDOM_STATE_SEED
    )
    logging.info(f"Training set size (for model learning): {X_train_eval.shape[0]} samples")
    logging.info(f"Validation set size (for performance check): {X_val.shape[0]} samples")
    logging.info(f"Test set size (for final submission prediction): {X_test_predict_raw.shape[0]} samples")

    # 4. Feature Scaling
    # Identify numerical features for scaling (all our current features are numerical)
    numerical_features_to_scale = features_for_model
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_eval, X_val, X_test_predict_raw, numerical_features_to_scale
    )

    # Save the scaler for future use (e.g., if deploying the model for new, unseen data)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    logging.info(f"✅ Feature scaler saved to '{SCALER_SAVE_PATH}'")

    # 5. Train the Linear Regression Model
    model = train_model(X_train_scaled, y_train_eval)

    # Save the trained model
    joblib.dump(model, MODEL_SAVE_PATH)
    logging.info(f"✅ Trained model saved to '{MODEL_SAVE_PATH}'")

    # Example of loading the model (for demonstration)
    # loaded_model = joblib.load(MODEL_SAVE_PATH)
    # loaded_scaler = joblib.load(SCALER_SAVE_PATH)

    # 6. Model Evaluation
    y_val_actual, y_val_predicted = evaluate_model(model, X_val_scaled, y_val, features_for_model)

    # 7. Make Final Predictions on the Actual Test Dataset
    final_predictions = make_final_predictions(model, X_test_scaled)

    # 8. Create Submission File
    generate_submission_file(df_test, final_predictions, SUBMISSION_OUTPUT_FILE)

    # 9. Visualize Performance
    visualize_performance(y_val_actual, y_val_predicted)

    logging.info("\n--- Pipeline Execution Finished ---")
    logging.info(f"Model saved to: {MODEL_SAVE_PATH}")
    logging.info(f"Submission file saved to: {SUBMISSION_OUTPUT_FILE}")
    logging.info("You can now use 'submission.csv' for Kaggle submission or further analysis.")


if __name__ == "__main__":
    main()
