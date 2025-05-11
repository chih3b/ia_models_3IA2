import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from transformers import Wav2Vec2FeatureExtractor

from standard_data_processor import prepare_standard_data
from data_augmentation import prepare_augmented_data
from hubert_model import HubertForSER

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_hubert(args):
    """
    Train HuBERT model for Speech Emotion Recognition using the approach 
    described in the research paper with standard train/val/test splits.
    """
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    # First get the paths and labels
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, emotion_to_idx = prepare_standard_data(
        args.data_dir, 
        test_size=args.test_size,
        val_size=args.val_size,
        return_paths_and_labels=True
    )
    
    # Then create data loaders with augmentation
    train_loader, val_loader, test_loader = prepare_augmented_data(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
        batch_size=args.batch_size
    )
    
    num_emotions = len(emotion_to_idx)
    print(f"Number of emotion classes: {num_emotions}")
    print(f"Emotion mapping: {emotion_to_idx}")
    
    # Initialize feature extractor for padding and normalization
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    
    # Initialize model
    model = HubertForSER(
        num_emotions=num_emotions,
        fine_tuning_type=args.fine_tuning_type,
        hidden_size=args.hidden_size
    )
    model.to(device)
    
    # Count trainable parameters
    trainable_params = model.count_parameters()
    print(f"Number of trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Initialize loss function
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (waveforms, labels) in enumerate(progress_bar):
            # Prepare input
            inputs = feature_extractor(
                waveforms.numpy(), 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            input_values = inputs.input_values.to(device)
            
            # Create attention mask manually (all 1s since we padded already)
            attention_mask = torch.ones(input_values.shape, device=device)
            
            labels = labels.to(device)
            
            # Forward pass
            logits = model(input_values, attention_mask)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({
                "loss": total_loss / (batch_idx + 1),
                "acc": 100 * correct / total
            })
        
        # Validate
        val_f1, val_acc = evaluate(model, val_loader, feature_extractor, device, emotion_to_idx)
        print(f"Epoch {epoch+1}, Validation F1: {val_f1:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"{args.output_dir}/hubert_{args.fine_tuning_type}_model.pt")
            print(f"Saved best model with validation F1: {best_val_f1:.4f}")
    
    print("\nTraining completed!")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    
    # Load best model
    model.load_state_dict(torch.load(f"{args.output_dir}/hubert_{args.fine_tuning_type}_model.pt"))
    test_f1, test_acc, all_preds, all_labels = evaluate(
        model, test_loader, feature_extractor, device, emotion_to_idx, return_predictions=True
    )
    
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Create and save confusion matrix
    create_confusion_matrix(
        all_labels, 
        all_preds, 
        emotion_to_idx, 
        f"{args.output_dir}/confusion_matrix_{args.fine_tuning_type}.png"
    )
    
    # Save results
    with open(f"{args.output_dir}/results_{args.fine_tuning_type}.txt", "w") as f:
        f.write(f"Fine-tuning type: {args.fine_tuning_type}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Number of trainable parameters: {trainable_params:,}\n")
    
    print(f"Results saved to {args.output_dir}/results_{args.fine_tuning_type}.txt")

def evaluate(model, dataloader, feature_extractor, device, emotion_to_idx, return_predictions=False):
    """Evaluate the model on the given dataloader"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="Evaluating"):
            # Prepare input
            inputs = feature_extractor(
                waveforms.numpy(), 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            input_values = inputs.input_values.to(device)
            
            # Create attention mask manually (all 1s since we padded already)
            attention_mask = torch.ones(input_values.shape, device=device)
            
            labels = labels.to(device)
            
            # Forward pass
            logits = model(input_values, attention_mask)
            
            # Get predictions
            _, predicted = torch.max(logits, 1)
            
            # Save predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = 100 * accuracy_score(all_labels, all_preds)
    
    if return_predictions:
        return f1, acc, all_preds, all_labels
    else:
        return f1, acc

def create_confusion_matrix(true_labels, pred_labels, emotion_to_idx, save_path):
    """Create and save a confusion matrix visualization"""
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Get emotion names
    idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
    emotion_names = [idx_to_emotion[i] for i in range(len(emotion_to_idx))]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues', 
        xticklabels=emotion_names, 
        yticklabels=emotion_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (%)')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuBERT for Tunisian Speech Emotion Recognition")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing emotion data")
    parser.add_argument("--output_dir", type=str, default="./results_hubert", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set ratio")
    
    # Model arguments
    parser.add_argument("--fine_tuning_type", type=str, default="full", 
                        choices=["full", "qkv", "classifier"], 
                        help="Type of fine-tuning to use")
    parser.add_argument("--hidden_size", type=int, default=256, help="Size of hidden layer")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train_hubert(args)
