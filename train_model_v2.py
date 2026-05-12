import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, classification_report

# Configure basic logging for production tracking
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

def load_and_prepare_data(filepath):
    logging.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)

    # Separate features and target
    X_raw = df.drop(columns=['koi_disposition'])
    y_raw = df['koi_disposition']

    return X_raw, y_raw

def apply_knn_imputation(X):
    logging.info("Initializing KNN Imputation (k=5) for topological data reconstruction...")
    # KNN Imputer looks at the 5 most mathematically similar planets to fill any NaNs
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_imputed_array = imputer.fit_transform(X)

    # Reconstruct the DataFrame
    X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)
    logging.info("KNN Imputation complete. No missing values remain.")

    return X_imputed

def objective(trial, X_train, y_train, X_val, y_val):
    # Define the hyperparameter search space for Optuna
    param = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600)
    }

    # Train model for this specific trial
    model = lgb.LGBMClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    # Predict and calculate log loss (or use accuracy if preferred)
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)

    # Optuna maximizes the return value, so we return accuracy
    return accuracy

def main():
    # 1. Load Data
    data_file = 'grand_dAAAAAAaataset_final_scaled.csv'
    X_raw, y_raw = load_and_prepare_data(data_file)

    # 2. Advanced Imputation (KNN)
    X_clean = apply_knn_imputation(X_raw)

    # 3. Label Encoding
    logging.info("Encoding target variables...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    # 4. Train/Validation Split
    logging.info("Splitting data for training and validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 5. Optuna Hyperparameter Optimization
    logging.info("Launching Optuna Automated Architecture Search...")
    # Note: n_trials is set to 20 for time efficiency. Increase to 100 for maximum robustness.
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)

    best_params = study.best_params
    logging.info(f"Optuna Optimization complete. Best parameters found: {best_params}")

    # 6. Train Final Model with Best Parameters
    logging.info("Compiling final LightGBM core using optimal parameters...")
    final_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        boosting_type='gbdt',
        **best_params
    )

    final_model.fit(X_train, y_train)

    # 7. Evaluate Performance
    preds = final_model.predict(X_val)
    final_acc = accuracy_score(y_val, preds)
    logging.info(f"Final Model Validation Accuracy: {final_acc * 100:.2f}%")
    print(classification_report(y_val, preds, target_names=le.classes_))

    # 8. Artifact Export
    logging.info("Serializing algorithmic artifacts to disk...")
    joblib.dump(final_model, 'LightGBM_exoplanet_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(list(X_clean.columns), 'features.pkl')

    logging.info("Pipeline execution finished successfully. Ready for Docker deployment.")

if __name__ == "__main__":
    main()
