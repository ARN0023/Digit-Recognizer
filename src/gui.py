import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import joblib

class DigitRecognizerApp:
    def __init__(self, root, model_path):
        self.root = root
        self.model = joblib.load(model_path)
        
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', self.paint)
        
        self.button_predict = tk.Button(root, text='Predict', command=self.predict_digit)
        self.button_predict.pack()

        self.image = Image.new('L', (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        r = 4  # Adjust the radius for a smaller line
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
        img = np.array(img) / 255.0
        
        # Flatten the image to match model input shape
        img = img.flatten().reshape(1, -1)
        
        # Print or log the input data for verification
        # print(f"Input image shape: {img.shape}")
        # print(f"Input image data:\n{img}")
        
        # Make prediction
        prediction = self.model.predict(img)
        
        # Show prediction result
        messagebox.showinfo('Prediction', f'Predicted Digit: {prediction[0]}')
        
        self.clear_canvas()

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root, './models/logistic_regression.pkl')
    root.mainloop()
