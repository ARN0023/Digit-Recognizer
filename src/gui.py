import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import joblib
from tensorflow.keras.models import load_model

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.logistic_model_path = './models/logistic_regression.pkl'
        self.neural_model_path = './models/neural_network_model.h5'
        
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', self.paint)
        
        self.button_predict = tk.Button(root, text='Predict', command=self.predict_digit)
        self.button_predict.pack()

        self.image = Image.new('L', (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        r = 3  # Adjust the radius for a smaller line
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black')

    def predict_digit(self):
        # Resize the drawn image to 28x28
        img = self.image.resize((28, 28))
        
        # Convert image to grayscale
        img = img.convert('L')
        
        # Invert colors (assuming black background and white drawing)
        img = Image.eval(img, lambda x: 255 - x)
        
        # Convert image to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Flatten the image to match model input shape
        img_flat = img_array.flatten().reshape(1, -1)
        
        # Load logistic regression model
        logistic_model = joblib.load(self.logistic_model_path)
        
        # Predict using logistic regression model
        logistic_prediction = logistic_model.predict(img_flat)
        
        # Load neural network model
        neural_model = load_model(self.neural_model_path)
        
        # Predict using neural network model
        neural_prediction = np.argmax(neural_model.predict(img_flat.reshape(-1, 28, 28)), axis=-1)
        
        # Show prediction results
        messagebox.showinfo('Predictions', f'Logistic Regression Prediction: {logistic_prediction[0]}\nNeural Network Prediction: {neural_prediction[0]}')
        
        self.clear_canvas()

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

if __name__ == '__main__':
    # Example usage for testing
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
