# ONNX Model Support

This updated version of the sample generator now supports both PyTorch (.pt) and ONNX (.onnx) models.

## Usage

### With ONNX models:
```bash
python generate_samples.py "Your text here" --max-samples 5 --model path/to/model.onnx --output-dir output
```

### With PyTorch models (original):
```bash
python generate_samples.py "Your text here" --max-samples 5 --model path/to/model.pt --output-dir output
```

## Key Differences

### ONNX Models:
- **Single Speaker**: Most ONNX Piper models are single-speaker, so speaker interpolation (SLERP) is not supported
- **Faster Inference**: ONNX Runtime can provide faster inference, especially on CPU
- **Cross-Platform**: Better portability across different platforms
- **Limited Flexibility**: Some advanced features like speaker mixing may not be available

### PyTorch Models:
- **Multi-Speaker Support**: Full support for speaker interpolation and mixing
- **Complete Feature Set**: All original features are available
- **GPU Acceleration**: Native CUDA support when available
- **Model Introspection**: Access to internal model components

## Model Detection

The script automatically detects the model type based on the file extension:
- `.onnx` files are loaded with ONNX Runtime
- `.pt` files are loaded with PyTorch

## Requirements

Make sure you have installed:
```bash
pip install onnxruntime
```

## Limitations with ONNX Models

1. **Speaker Mixing**: The `slerp_weight` parameter is ignored for ONNX models
2. **Phoneme Timing**: Phoneme-level timing information is approximated
3. **Model-Specific**: Input/output formats may vary between different ONNX models

## Example

```bash
# Generate Portuguese samples with ONNX model
python generate_samples.py "Olá, como está?" --max-samples 3 --model ../models/pt_PT-tugao-medium.onnx --output-dir portuguese_samples
```
