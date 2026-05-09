import os
import torch
import timm

def export_to_onnx():
    device = torch.device("cpu")
    model_path = "best_model_kfold.pth"
    onnx_path = "best_model.onnx"

    print("Building model architecture...")
    model = timm.create_model("convnext_small", pretrained=False, num_classes=5)
    
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: {model_path} not found. Exporting UNTRAINED model for demonstration purposes.")
    
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export successful!")

if __name__ == "__main__":
    export_to_onnx()
