import tkinter as tk
from src.gui import DigitRecognizerApp
import joblib

if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
