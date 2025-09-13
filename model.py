from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
import re
import base64
from PIL import Image
import io

# Load your trained model
model = keras.models.load_model("mnist_model.keras")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image data from canvas
        data = request.get_json()["image"]
        # Remove the header "data:image/png;base64,"
        image_data = re.sub('^data:image/.+;base64,', '', data)
        # Decode
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")
        image = image.resize((28, 28))  # MNIST size
        img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

        # Predict
        pred = model.predict(img_array)
        digit = int(np.argmax(pred))

        return jsonify({"prediction": digit})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

