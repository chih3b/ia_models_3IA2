import os
import torch
import numpy as np
import librosa
import argparse
from transformers import Wav2Vec2FeatureExtractor
from hubert_model import HubertForSER

def load_audio(file_path, sample_rate=16000, max_length=250000):
    """Load and preprocess an audio file"""
    try:
        # Load audio file
        waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        
        # Pad or truncate to fixed length
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        else:
            padding = max_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), 'constant')
            
        # Convert to tensor
        waveform = torch.from_numpy(waveform).float()
        return waveform
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

# Define emotion mapping globally for consistent use throughout the script
EMOTION_NAMES = {
    0: "happy",
    1: "fear",
    2: "surprise",
    3: "sadness",
    4: "neutral",
    5: "anger",
    6: "disgust"
}

def predict_emotion(model_path, audio_path, fine_tuning_type="full", num_emotions=7, hidden_size=256):
    """Predict emotion from audio file using a trained HuBERT model"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Processing audio file: {audio_path}")
    print(f"Using model: {model_path} with {fine_tuning_type} fine-tuning")
    
    try:
        # Load model
        model = HubertForSER(
            num_emotions=num_emotions,
            fine_tuning_type=fine_tuning_type,
            hidden_size=hidden_size
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Load audio
        waveform = load_audio(audio_path)
        if waveform is None:
            print(f"Error: Could not load audio file {audio_path}")
            return None
        
        print(f"Audio loaded successfully, length: {len(waveform)} samples")
        
        # Initialize feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        
        # Prepare input
        inputs = feature_extractor(
            waveform.numpy().reshape(1, -1), 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs.input_values.to(device)
        
        # Create attention mask manually (all 1s since we padded already)
        attention_mask = torch.ones(input_values.shape, device=device)
        
        # Predict
        with torch.no_grad():
            logits = model(input_values, attention_mask)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        emotion_name = EMOTION_NAMES.get(predicted_class, f"Unknown ({predicted_class})")
        print(f"Predicted emotion index: {predicted_class}, name: {emotion_name}")
        print(f"Confidence: {confidence:.4f}")
        
        return predicted_class, confidence, emotion_name
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predict_emotion_batch(model_path, audio_dir, fine_tuning_type="full", num_emotions=7, hidden_size=256):
    """Predict emotions for all audio files in a directory"""
    results = []
    
    # Process each audio file
    for filename in os.listdir(audio_dir):
        if filename.endswith(('.wav', '.mp3', '.m4a')):
            audio_path = os.path.join(audio_dir, filename)
            prediction = predict_emotion(model_path, audio_path, fine_tuning_type, num_emotions, hidden_size)
            
            if prediction:
                emotion_idx, confidence, emotion_name = prediction
                
                print(f"File: {filename}")
                print(f"Predicted emotion: {emotion_name}")
                print(f"Confidence: {confidence:.2f}")
                print("-" * 40)
                
                results.append({
                    "file": filename,
                    "emotion": emotion_name,
                    "confidence": confidence
                })
    
    return results

def predict_emotion_live(model_path, fine_tuning_type="full", num_emotions=7, hidden_size=256, duration=5):
    """Record audio from microphone and predict emotion"""
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write
        
        # Map emotion indices to names (update this based on your dataset)
        emotion_names = {
            0: "happy",
            1: "fear",
            2: "surprise",
            3: "sadness",
            4: "neutral",
            5: "anger",
            6: "disgust"
        }
        
        print(f"Recording {duration} seconds of audio...")
        sample_rate = 16000
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        # Save recording temporarily
        temp_file = "temp_recording.wav"
        write(temp_file, sample_rate, recording)
        
        # Predict emotion
        prediction = predict_emotion(model_path, temp_file, fine_tuning_type, num_emotions, hidden_size)
        
        if prediction:
            emotion_idx, confidence = prediction
            emotion_name = emotion_names.get(emotion_idx, f"Unknown ({emotion_idx})")
            
            print(f"Predicted emotion: {emotion_name}")
            print(f"Confidence: {confidence:.2f}")
        
        # Clean up
        os.remove(temp_file)
        
    except ImportError:
        print("Error: sounddevice and/or scipy packages are required for live recording.")
        print("Install them with: pip install sounddevice scipy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotions using trained HuBERT model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--fine_tuning_type", type=str, default="full", choices=["full", "qkv", "classifier"], 
                        help="Type of fine-tuning used for the model")
    parser.add_argument("--num_emotions", type=int, default=7, help="Number of emotion classes")
    parser.add_argument("--hidden_size", type=int, default=256, help="Size of hidden layer")
    
    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio_file", type=str, help="Path to a single audio file for prediction")
    group.add_argument("--audio_dir", type=str, help="Directory containing audio files for batch prediction")
    group.add_argument("--live", action="store_true", help="Record audio from microphone for prediction")
    
    # Live recording arguments
    parser.add_argument("--duration", type=int, default=5, help="Duration of live recording in seconds")
    
    args = parser.parse_args()
    
    if args.audio_file:
        # Single file prediction
        prediction = predict_emotion(
            args.model_path, 
            args.audio_file, 
            args.fine_tuning_type, 
            args.num_emotions, 
            args.hidden_size
        )
        
        if prediction:
            emotion_idx, confidence, emotion_name = prediction
            
            print(f"Predicted emotion: {emotion_name}")
            print(f"Confidence: {confidence:.2f}")
    
    elif args.audio_dir:
        # Batch prediction
        predict_emotion_batch(
            args.model_path, 
            args.audio_dir, 
            args.fine_tuning_type, 
            args.num_emotions, 
            args.hidden_size
        )
    
    elif args.live:
        # Live recording and prediction
        predict_emotion_live(
            args.model_path, 
            args.fine_tuning_type, 
            args.num_emotions, 
            args.hidden_size, 
            args.duration
        )
