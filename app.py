from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained CNN model
model_path = os.path.join(os.path.dirname(__file__), "cat_dog_cnn.h5")
model = load_model(model_path)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get uploaded file
    file = request.files["image"]

    # Save file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Preprocess image (same as training)
    img = image.load_img(file_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    # Result text
    if class_index == 0:
        result = "😼 Meow! It's a CAT!"
    else:
        result = "🐶 Woof! It's a DOG!"

    # Fix path for browser
    image_url = "/" + file_path.replace("\\", "/")

    return render_template(
        "result.html",
        prediction=result,
        image_path=image_url
    )


if __name__ == "__main__":
    app.run(debug=True)
