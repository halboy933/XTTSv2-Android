# XTTSv2 Android V1
Bootstrap/test project for Android ARM64 XTTS-v2 ONNX.

This V1 intentionally verifies Android build + ONNX Runtime + reference import before porting the Python/NumPy autoregressive XTTS pipeline to Kotlin. It does NOT yet synthesize speech.

Target model: pltobing/XTTSv2-Streaming-ONNX, INT8 GPT, Russian `ru`, short <=6s reference.
