from disease_info import disease_data
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("model/leaf_model.h5")

classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/signup")
def signup():
    return render_template("signup.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "leaf" not in request.files:
        return "No image uploaded."

    img = request.files["leaf"]

    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, img.filename)
    img.save(filepath)

    test_image = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(test_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    index = np.argmax(prediction)
    confidence = round(float(np.max(prediction)) * 100, 2)

    disease = classes[index]

    crop = disease.split("___")[0]
    disease_name = disease.split("___")[1]

    # Search disease information
    info = disease_data.get(disease_name)

    if info is None:
        info = {
            "description": f"{disease_name.replace('_',' ')} disease affects {crop}.",
            "symptoms": "Symptoms information not available.",
            "treatment": "Consult agricultural expert.",
            "prevention": "Maintain field hygiene and inspect crops regularly."
        }

    return render_template(
        "result.html",
        disease=disease_name.replace("_", " "),
        confidence=confidence,
        description=info["description"],
        symptoms=info["symptoms"],
        treatment=info["treatment"],
        prevention=info["prevention"],
        image_path=filepath.replace("\\", "/")
    )


if __name__ == "__main__":
    app.run(debug=True)