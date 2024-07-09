import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from data_preparation import load_and_preprocess_data

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(binary=False)

    print(f"Number of columns in X_train: {X_train.shape[1]}")
    
    # Convert Pandas DataFrames to NumPy arrays
    X_train = X_train.values
    X_test = X_test.values

    # Ensure consistent preprocessing
    X_train_processed = X_train.reshape(-1, 28, 28) / 255.0
    X_test_processed = X_test.reshape(-1, 28, 28) / 255.0

    # Print the shapes (for debugging)
    # print(f"Shapeessed: {X_test_processed[0]}")
    # print(f"Shape of X_test_processed: {X_test_processed.shape}")

    # Parameters
    num_iters = 1500
    alpha = 0.05  # learning rate (not directly used in scikit-learn's LogisticRegression)
    
    # Inverse regularization strength in scikit-learn (higher values of C means less regularization)
    C = 1 / alpha

    # Create and train the model
    model = LogisticRegression(max_iter=num_iters, C=C, solver='lbfgs', multi_class='auto')
    model.fit(X_train_processed.reshape(len(X_train), -1), y_train)

    # Evaluate the model
    accuracy = model.score(X_test_processed.reshape(len(X_test), -1), y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Save the model
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)  # Create the models directory if it doesn't exist
    model_path = os.path.join(model_dir, 'logistic_regression.pkl')
    joblib.dump(model, model_path)

if __name__ == '__main__':
    train_and_save_model()
