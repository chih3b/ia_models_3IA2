# Speech Emotion Recognition Models

Repository containing implementations of state-of-the-art speech emotion recognition models for the 3IA2 course.

## Models Included

### HuBERT-based Speech Emotion Recognition

The `hubert/` directory contains a complete implementation of a Speech Emotion Recognition (SER) system based on the HuBERT (Hidden Unit BERT) pre-trained model. The implementation supports three different fine-tuning approaches:

1. **Full Fine-tuning**: Fine-tune all parameters of the pre-trained HuBERT model
2. **QKV Fine-tuning**: Parameter-efficient fine-tuning using LoRA on query, key, and value projection layers only
3. **Classifier Fine-tuning**: Only fine-tune the classifier head, keeping the rest of the model frozen

See the [HuBERT README](hubert/README.md) for detailed usage instructions.

## Installation

Each model directory contains its own `requirements.txt` file with the necessary dependencies for that specific model.

## Contributing

Feel free to contribute additional models or improvements to existing implementations through pull requests. 