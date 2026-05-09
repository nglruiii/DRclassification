import os
import io
import torch
import timm
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(title="DR Classification API")

CLASSES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
device = torch.device("cpu")

model = None
MODEL_PATH = "best_model_kfold.pth"

@app.on_event("startup")
async def load_model():
    global model
    model = timm.create_model("convnext_small", pretrained=False, num_classes=5)
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            
    model = model.to(device)
    model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor_image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor_image)
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
        raise HTTPException(status_code=500, detail=str(e))
