# AI Models for 3IA2 Course

This repository contains implementations of state-of-the-art AI models for:
1. Speech Emotion Recognition (SER) using HuBERT
2. Phone detection using YOLOv8

## Speech Emotion Recognition (SER)

The `ser/` directory contains the training code for a Tunisian Speech Emotion Recognition system based on the HuBERT (Hidden Unit BERT) pre-trained model.

### File Descriptions

- `train_hubert_standard.py`: Main training script for the HuBERT model.
- `hubert_model.py`: Contains the HuBERT model architecture for SER with three fine-tuning approaches: full, QKV, and classifier-only.
- `standard_data_processor.py`: Handles data preparation with standard train/val/test splits.
- `data_augmentation.py`: Implements data augmentation techniques for model robustness.

## Phone Detection with YOLOv8

The `yolov8/` directory contains the implementation for training and evaluating a YOLOv8 model for phone detection in various scenarios. The notebook `yolotraining.ipynb` was trained on Kaggle and detects mobile phones in video conferencing contexts.

## Research Papers

The `research/` directory contains research papers and notes related to the implemented models.