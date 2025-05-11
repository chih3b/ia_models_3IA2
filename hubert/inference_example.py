import os
import argparse
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor

from hubert_model import HubertForSER

def predict_emotion(model_path, audio_path, emotion_map_path=None):
    """
    Predict emotion from an audio file using a trained HuBERT model
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load emotion mapping if provided
    if emotion_map_path and os.path.exists(emotion_map_path):
        import json
        with open(emotion_map_path, 'r') as f:
            emotion_to_idx = json.load(f)
        idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
    else:
        print("No emotion mapping file provided or file not found.")
        print("Will return numeric prediction only.")
        idx_to_emotion = None
    
    # Extract model parameters from filename
    model_name = os.path.basename(model_path)
    if "full" in model_name:
        fine_tuning_type = "full"
    elif "qkv" in model_name:
        fine_tuning_type = "qkv"
    elif "classifier" in model_name:
        fine_tuning_type = "classifier"
    else:
        fine_tuning_type = "full"  # Default
    
    # Initialize feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    
    # Initialize model
    num_emotions = len(idx_to_emotion) if idx_to_emotion else 5  # Default to 5 emotions if no mapping
    model = HubertForSER(num_emotions=num_emotions, fine_tuning_type=fine_tuning_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load and preprocess audio
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Prepare input
    inputs = feature_extractor(
        waveform, 
        sampling_rate=16000, 
        return_tensors="pt", 
        padding=True
    )
    input_values = inputs.input_values.to(device)
    attention_mask = torch.ones(input_values.shape, device=device)
    
    # Make prediction
    with torch.no_grad():
        logits = model(input_values, attention_mask)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
    
    # Get human-readable emotion label if available
    if idx_to_emotion:
        emotion = idx_to_emotion.get(str(prediction), f"Unknown ({prediction})")
    else:
        emotion = f"Class {prediction}"
    
    return emotion, probabilities.cpu().numpy()[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from audio using trained HuBERT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model file (.pt)")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file for prediction")
    parser.add_argument("--emotion_map", type=str, help="Path to emotion mapping JSON file")
    
    args = parser.parse_args()
    
    emotion, probs = predict_emotion(args.model_path, args.audio_path, args.emotion_map)
    
    print(f"Predicted emotion: {emotion}")
    print(f"Probabilities: {probs}") 