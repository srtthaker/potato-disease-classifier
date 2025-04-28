from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = ["http://localhost:3000"] # frontend URL

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True, # Allows cookies/authorization headers
                            # NOTE: allow_origins cannot be ["*"] if allow_credentials is True
    allow_methods=["*"],    # Allows all methods (GET, POST, PUT, etc.)
    allow_headers=["*"],    # Allows all headers
)

MODEL = tf.keras.models.load_model("saved_models/1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping(): 
    return "Hello, World!"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {'class': predicted_class, 'confidence': float(confidence)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)