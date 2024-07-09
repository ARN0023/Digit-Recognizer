import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from data_preparation import load_and_preprocess_data

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(binary=False)

    print(f"Number of columns in X_train: {X_train.shape}")

    # Convert Pandas DataFrames to NumPy arrays and normalize
    X_train_processed = X_train.values 
    X_test_processed = X_test.values 

    # Parameters
    num_iters = 1500
    C = 20  # Inverse regularization strength in scikit-learn

    # Create and train the model
    model = LogisticRegression(max_iter=num_iters, C=C, solver='lbfgs')
    model.fit(X_train_processed, y_train)

    # Evaluate the model
    accuracy = model.score(X_test_processed, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Save the model
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)  # Create the models directory if it doesn't exist
    model_path = os.path.join(model_dir, 'logistic_regression.pkl')
    joblib.dump(model, model_path)

if __name__ == '__main__':
    train_and_save_model()
