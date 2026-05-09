from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model():
    model_input = "best_model.onnx"
    model_output = "best_model_quantized.onnx"
    print(f"Quantizing {model_input} to {model_output}...")
    quantize_dynamic(model_input, model_output, weight_type=QuantType.QUInt8)
    print("Quantization complete!")

if __name__ == "__main__":
    quantize_model()
