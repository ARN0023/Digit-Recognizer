from src.gui import DigitRecognizerApp
import tkinter as tk
import joblib

if __name__ == '__main__':
    model_path = 'models/logistic_regression.pkl'
    root = tk.Tk()
    app = DigitRecognizerApp(root, model_path)
    root.mainloop()
