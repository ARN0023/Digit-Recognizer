from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd
from data_preparation import load_and_preprocess_data
import os

def neural_model_train():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(binary=False)

    # Convert Pandas DataFrames to NumPy arrays if needed
    X_train_processed = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_processed = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    num_epochs = 100
    batch_size = 100

    # Reshape input data to match the model's expected input shape
    x_train_reshaped = X_train_processed.reshape(-1, 28, 28)  # Assuming X_train_processed has shape (num_samples, 28*28)
    x_test_reshaped = X_test_processed.reshape(-1, 28, 28)    # Assuming X_test_processed has shape (num_samples, 28*28)

    # Define the model architecture
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Flattening the 28x28 input
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # Assuming 10 classes (digits 0-9)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train_reshaped, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test_reshaped, y_test))

    # Evaluate the model
    _, accuracy = model.evaluate(x_test_reshaped, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Save the model
    keras_model_dir = './models'
    os.makedirs(keras_model_dir, exist_ok=True)  # Create the directory if it doesn't exist
    keras_model_path = os.path.join(keras_model_dir, 'neural_network_model.h5')
    model.save(keras_model_path)

if __name__ == '__main__':
    neural_model_train()
