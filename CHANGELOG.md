# Changelog

## 3.1.0

- Support MPS acceleration on Apple Silicon
- Add `--phoneme-input` flag

## 3.0.0

- Move phonemization to piper 1.3.0 (piper-phonemize is deprecated)
- Move to PyTorch 2
- Add support for using Piper voices (`.onnx`) directly
- Allow multiple `--model` for Piper voices (`.onnx`)
- Remove silence trimming
- Remove `min-phoneme-count`
