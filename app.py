import os
import io
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image

from train_pipeline import build_model, get_val_transforms

app = FastAPI(title="DR Classification API")

# Define severity classes
CLASSES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
MODEL_PATH = "best_model_kfold.pth"
model = None

@app.on_event("startup")
async def load_model():
    global model
    print("Loading model architecture...")
    model = build_model(model_name="convnext_small", num_classes=5, pretrained=False)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"Warning: Model weights file ({MODEL_PATH}) not found.")
        print("Using UNTRAINED initialized weights for demonstration purposes.")
    
    model = model.to(device)
    model.eval()

# Get validation transforms (224x224 as per kfold_train.py)
transform = get_val_transforms(image_size=224)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Apply transforms
        image_np = np.array(image)
        augmented = transform(image=image_np)
        tensor_image = augmented["image"].unsqueeze(0).to(device)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            outputs = model(tensor_image)
            # Calculate probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            
            idx = predicted_idx.item()
            conf_val = confidence.item()
            
        return JSONResponse({
            "class_index": idx,
            "label": CLASSES[idx],
            "confidence": round(conf_val, 4)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files to serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
