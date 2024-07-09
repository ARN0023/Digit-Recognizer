from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import resample

def load_and_preprocess_data(binary=True):
    mnist = fetch_openml('mnist_784', version=1)
    
    X, y = mnist['data'], mnist['target']
    y = y.astype(int)
    
    if binary:
        # Filter only the digits 0 and 1 for binary classification
        binary_filter = (y == 0) | (y == 1)
        X, y = X[binary_filter], y[binary_filter]
    
    # Combine X and y into a single DataFrame for easier manipulation
    df = pd.DataFrame(X)
    df['target'] = y
    
    # Balance the dataset
    max_count = df['target'].value_counts().max()
    df_balanced = pd.concat([
        df[df['target'] == digit].sample(n=max_count, replace=True, random_state=42)
        for digit in df['target'].unique()
    ])
    
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate the balanced DataFrame back into X and y
    X_balanced = df_balanced.drop(columns=['target'])
    y_balanced = df_balanced['target']
    
    # Normalize the data
    X_balanced = X_balanced / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Example usage
X_train, X_test, y_train, y_test = load_and_preprocess_data(binary=True)
