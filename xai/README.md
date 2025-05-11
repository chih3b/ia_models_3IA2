# Explainable AI (XAI) Components

This directory contains implementations of explainable AI techniques for both Speech Emotion Recognition (SER) and Phone Detection models. These components help visualize and understand the model's decision-making process, making the AI systems more transparent and interpretable.

## Contents

- `ser/`: Explainable AI components for HuBERT-based Speech Emotion Recognition
- `phone_detection/`: Explainable AI components for YOLOv8-based Phone Detection

## Speech Emotion Recognition XAI

The `ser/hubert_xai.py` file implements a `HuBERTExplainer` class that uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of an audio waveform contribute most to the model's emotion prediction.

### Features
- Visualization of audio sections that influence the emotion prediction
- Emotion probability distribution display
- Grad-CAM implementation for audio-based models
- Command-line interface for easy usage

### Usage
```bash
python hubert_xai.py --model_path path/to/model.pt --audio_path path/to/audio.wav --method grad_cam --output optional/output/path.png
```

## Phone Detection XAI

The `phone_detection/yolo_xai.py` file implements a `YOLOv8Explainer` class that uses a modified Grad-CAM approach to visualize which parts of an image contribute most to phone detection.

### Features
- Visualization of image regions contributing to phone detection
- Bounding box display with confidence scores
- Heatmap overlay showing important regions
- Command-line interface for easy usage

### Usage
```bash
python yolo_xai.py --model_path path/to/model.pt --image_path path/to/image.jpg --method grad_cam --output optional/output/path.png
```

## Technical Background

Both implementations use Grad-CAM, a technique that uses the gradients flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the input (audio waveform or image) for prediction. This provides insight into what the model is "looking at" when making predictions.

The YOLOv8 implementation uses a modified approach since object detection models have a different architecture than classification models, focusing on activation magnitudes in detection layers. 