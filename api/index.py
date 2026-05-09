import os
import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(title="DR Classification API")

CLASSES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]

ONNX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_model_quantized.onnx")
# Fallback for Vercel
if not os.path.exists(ONNX_PATH):
    ONNX_PATH = "best_model_quantized.onnx"

session = None

@app.on_event("startup")
async def load_model():
    global session
    if os.path.exists(ONNX_PATH):
        session = ort.InferenceSession(ONNX_PATH)
    else:
        print(f"Error: ONNX model file ({ONNX_PATH}) not found.")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    img_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    if session is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_image(image)
        
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0]
        
        probs = softmax(logits)[0]
        predicted_idx = int(np.argmax(probs))
        conf_val = float(probs[predicted_idx])
            
        return JSONResponse({
            "class_index": predicted_idx,
            "label": CLASSES[predicted_idx],
            "confidence": round(conf_val, 4)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
