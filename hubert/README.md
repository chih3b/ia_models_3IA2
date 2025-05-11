# HuBERT Speech Emotion Recognition

This repository contains code for training and evaluating a HuBERT-based Speech Emotion Recognition (SER) model. The implementation supports three different fine-tuning approaches:

1. **Full Fine-tuning**: Fine-tune all parameters of the pre-trained HuBERT model
2. **QKV Fine-tuning**: Parameter-efficient fine-tuning using LoRA on query, key, and value projection layers only
3. **Classifier Fine-tuning**: Only fine-tune the classifier head, keeping the rest of the model frozen

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Organization

Organize your audio data in subdirectories named after emotions:

```
data_directory/
├── anger/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── happiness/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── ...
```

## Training

To train the model, you can either:

### 1. Run all three fine-tuning approaches in sequence:

```bash
python run_hubert_standard.py --data_dir /path/to/emotion/data --output_dir ./results
```

### 2. Run a specific fine-tuning approach:

```bash
python train_hubert_standard.py --data_dir /path/to/emotion/data --output_dir ./results/full --fine_tuning_type full
```

Available fine-tuning types are: `full`, `qkv`, and `classifier`.

## Arguments

- `--data_dir`: Directory containing emotion data (required)
- `--output_dir`: Output directory for model files and results (default: ./results_hubert_standard)
- `--test_size`: Ratio of test set (default: 0.2)
- `--val_size`: Ratio of validation set (default: 0.1)
- `--batch_size`: Batch size for training (default: 8)
- `--epochs`: Number of training epochs (default: 20)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--hidden_size`: Size of hidden layer (default: 256)
- `--weight_decay`: Weight decay for optimizer (default: 0.01)
- `--label_smoothing`: Label smoothing factor (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 42)
- `--fine_tuning_type`: Type of fine-tuning approach (`full`, `qkv`, or `classifier`)

## Results

After training, the following outputs will be generated in the specified output directory:

- Model weights (.pt file)
- Confusion matrix visualization
- Results summary with F1 score and accuracy 