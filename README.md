<h1>Digit Recognition Project</h1>

<p>This project is a digit recognition system using neural networks and logistic regression. It includes a GUI for digit recognition, and scripts for training models on the MNIST dataset.</p>

<h2>Table of Contents</h2>
<ul>
    <li>Requirements</li>
    <li>Installation</li>
    <li>Usage</li>
    <li>Project Structure</li>
    <li>Training Models</li>
    <li>Data Preparation</li>
    <li>License</li>
</ul>

<h2>Requirements</h2>
<ul>
    <li>Python: 3.10.11</li>
    <li>scikit-learn: 1.5.1</li>
    <li>numpy: 1.26.4</li>
    <li>Pillow: 10.4.0</li>
    <li>joblib: 1.4.2</li>
    <li>pandas: 2.2.2</li>
    <li>tensorflow: 2.16.2</li>
    <li>keras: 3.4.1</li>
</ul>

<h2>Installation</h2>
<ol>
    <li>Clone the repository:
        <pre>
            git clone https://github.com/yourusername/digit-recognition.git
            cd digit-recognition
        </pre>
    </li>
    <li>Create and activate a virtual environment:
        <pre>
            python -m venv venv
            source venv/bin/activate  # On Windows: venv\Scripts\activate
        </pre>
    </li>
    <li>Install the required dependencies:
        <pre>
            pip install -r requirements.txt
        </pre>
    </li>
</ol>

<h2>Usage</h2>
<ol>
    <li><strong>Start the GUI Application:</strong>
        <pre>
            python main.py
        </pre>
    </li>
    <li><strong>Train Models:</strong>
        <ul>
            <li>Train the neural network model:
                <pre>
                    python model_nn.py
                </pre>
            </li>
            <li>Train the logistic regression model:
                <pre>
                    python train_model.py
                </pre>
            </li>
        </ul>
    </li>
</ol>

<h2>Project Structure</h2>
<pre>
digit-recognition/
│
├── src/
│   ├── gui.py
│   ├── model_nn.py
│   ├── train_model.py
│   ├── data_preparation.py
│   └── ...
│
├── models/
│   └── ... (trained models saved here)
│
├── main.py
├── requirements.txt
└── README.md
</pre>

<h2>Training Models</h2>

<h3>Neural Network Model</h3>
<p>To train the neural network model, run the <code>model_nn.py</code> script. This script will:</p>
<ul>
    <li>Load and preprocess the MNIST dataset.</li>
    <li>Train a neural network model using Keras.</li>
    <li>Evaluate and save the model.</li>
</ul>
<pre>
python model_nn.py
</pre>

<h3>Logistic Regression Model</h3>
<p>To train the logistic regression model, run the <code>train_model.py</code> script. This script will:</p>
<ul>
    <li>Load and preprocess the MNIST dataset.</li>
    <li>Train a logistic regression model using scikit-learn.</li>
    <li>Evaluate and save the model.</li>
</ul>
<pre>
python train_model.py
</pre>

<h2>Data Preparation</h2>
<p>The <code>data_preparation.py</code> script handles loading, preprocessing, and balancing the MNIST dataset. The dataset is normalized and split into training and testing sets.</p>

<h2>License</h2>
<p>This project is licensed under the MIT License. See the LICENSE file for details.</p>
