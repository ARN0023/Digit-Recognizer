from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(binary=True):
    mnist = fetch_openml('mnist_784', version=1)
    
    # Print the first sample and its target
    # print(f"First sample data: {mnist.data.iloc[0].values}")
    # print(f"First sample target: {mnist.target.iloc[0]}")

    X, y = mnist['data'], mnist['target']
    y = y.astype(int)
    
    if binary:
        # Filter only the digits 0 and 1 for binary classification
        binary_filter = (y == 0) | (y == 1)
        X, y = X[binary_filter], y[binary_filter]
    
    # Normalize the data
    X = X / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
