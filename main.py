from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import uvicorn
import io

app = FastAPI()
model = load_model("model/best_fish_classifier.h5")

class_names = ["Bulath_hapaya", "Dankuda_pethiya", "Depulliya", "Halamal_dandiya", "Lethiththaya", "Pathirana_salaya", "Thal_kossa"]

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # depends on your model input
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_tensor = preprocess_image(image_bytes)
    prediction = model.predict(img_tensor)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    return JSONResponse(content={"prediction": predicted_class})