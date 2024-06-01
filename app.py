from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import cv2
import pandas as pd

# Define the model path
model_path = r'C:\Users\Deepesh sahu\Desktop\depoly\my_model.keras'

# Load the model
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_path}'")
except Exception as e:
    print("An error occurred while loading the model:", e)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", data="hey")

@app.route("/prediction", methods=["POST"])
def prediction():
    img = request.files['img']
    img.save("image.jpg")
    image = cv2.imread("image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.reshape(image, (1, 224, 224, 3))
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    # Read labels from file
    labels = []
    with open("labels.txt", "r") as file:
        for line in file:
            labels.append(line.strip())
    
    predicted_label = labels[predicted_class]

    return render_template("prediction.html", data=predicted_label)

if __name__ == "__main__":
    app.run(debug=True)

