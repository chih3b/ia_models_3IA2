import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
import sys
import os
import warnings

# Add parent directory to path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ser.hubert_model import HubertForSER
from transformers import Wav2Vec2FeatureExtractor

class HuBERTExplainer:
    """Class for explaining HuBERT SER model predictions using various XAI techniques."""
    
    def __init__(self, model_path, num_emotions=7, device=None):
        """
        Initialize the explainer with a trained HuBERT model.
        
        Args:
            model_path: Path to the trained model weights (.pt file)
            num_emotions: Number of emotion classes
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize model
        try:
            self.model = HubertForSER(num_emotions=num_emotions, fine_tuning_type="full")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Initialize feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        
        # Mapping for emotion labels (customize based on your dataset)
        self.idx_to_emotion = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise"
        }
    
    def load_audio(self, audio_path, sample_rate=16000):
        """
        Load audio from file and preprocess it.
        
        First tries scipy.io.wavfile, then falls back to librosa if that fails.
        """
        try:
            # First try scipy.io.wavfile which is faster and more reliable for WAV files
            sr, waveform = wavfile.read(audio_path)
            # Convert to float in range [-1, 1]
            if waveform.dtype == np.int16:
                waveform = waveform.astype(np.float32) / 32768.0
            elif waveform.dtype == np.int32:
                waveform = waveform.astype(np.float32) / 2147483648.0
            
            # Resample if needed
            if sr != sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
            
            # Ensure mono
            if len(waveform.shape) > 1 and waveform.shape[1] > 1:
                waveform = np.mean(waveform, axis=1)
                
            return waveform, sample_rate
            
        except Exception as e:
            print(f"Error loading with scipy: {e}")
            # Fall back to librosa
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
                return waveform, sr
            except Exception as e2:
                print(f"Error loading with librosa: {e2}")
                raise RuntimeError(f"Could not load audio file: {audio_path}")
    
    def preprocess_audio(self, waveform):
        """Preprocess audio waveform for model input."""
        # Convert to tensor and add batch dimension
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        
        # Use feature extractor for padding and normalization
        inputs = self.feature_extractor(
            waveform_tensor.numpy(), 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = torch.ones(input_values.shape, device=self.device)
        
        return input_values, attention_mask
    
    def predict(self, audio_path):
        """Make a prediction on an audio file and return the emotion and probabilities."""
        waveform, _ = self.load_audio(audio_path)
        input_values, attention_mask = self.preprocess_audio(waveform)
        
        with torch.no_grad():
            # Get model output
            logits = self.model(input_values, attention_mask)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            
        emotion = self.idx_to_emotion.get(prediction, f"Unknown ({prediction})")
        return emotion, probabilities.cpu().numpy()[0], waveform
    
    def grad_cam(self, audio_path):
        """
        Apply Grad-CAM to visualize which parts of the audio contribute most to the prediction.
        """
        try:
            waveform, sr = self.load_audio(audio_path)
            input_values, attention_mask = self.preprocess_audio(waveform)
            input_values.requires_grad_(True)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model.hubert(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.last_hidden_state
            
            # Get projection and final prediction
            projected = self.model.projector(hidden_states)
            mean_pooled = torch.mean(projected, dim=1)
            std_pooled = torch.std(projected, dim=1)
            pooled = torch.cat([mean_pooled, std_pooled], dim=1)
            pooled = self.model.batch_norm(pooled)
            pooled = self.model.dropout(pooled)
            logits = self.model.classifier(pooled)
            
            # Get prediction
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            emotion = self.idx_to_emotion.get(prediction, f"Unknown ({prediction})")
            
            # Backprop for the predicted class
            logits[0, prediction].backward()
            
            # Get gradients and feature maps
            gradients = input_values.grad.detach().cpu().numpy()[0]
            
            # Calculate importance weights
            weights = np.mean(gradients, axis=0)
            
            # Create heatmap based on weights
            heatmap = np.abs(weights)
            heatmap = heatmap / np.max(heatmap)  # Normalize
            
            return emotion, probabilities.detach().cpu().numpy()[0], waveform, heatmap
        except Exception as e:
            print(f"Error in grad_cam: {e}")
            raise
    
    def visualize_explanation(self, audio_path, method='grad_cam', save_path=None):
        """
        Visualize the explanation for a given audio file using the specified method.
        
        Args:
            audio_path: Path to the audio file
            method: Explanation method ('grad_cam' or 'lime')
            save_path: Path to save the visualization (if None, the plot is shown)
        """
        if method == 'grad_cam':
            try:
                emotion, probs, waveform, heatmap = self.grad_cam(audio_path)
                
                # Create plot
                plt.figure(figsize=(12, 8))
                
                # Plot waveform
                plt.subplot(2, 1, 1)
                librosa.display.waveshow(waveform, sr=16000)
                plt.title(f"Waveform (Predicted: {emotion})")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                
                # Plot heatmap
                plt.subplot(2, 1, 2)
                plt.plot(waveform, alpha=0.5, color='blue', label='Waveform')
                plt.plot(heatmap * np.max(np.abs(waveform)), color='red', alpha=0.5, label='Attention')
                plt.title("Grad-CAM Visualization")
                plt.xlabel("Time (samples)")
                plt.ylabel("Importance")
                plt.legend()
                
                # Plot emotion probabilities
                plt.figure(figsize=(10, 4))
                emotion_names = list(self.idx_to_emotion.values())
                plt.bar(emotion_names, probs)
                plt.title("Emotion Probabilities")
                plt.xlabel("Emotion")
                plt.ylabel("Probability")
                plt.xticks(rotation=45)
                
                if save_path:
                    plt.tight_layout()
                    plt.savefig(save_path)
                    print(f"Visualization saved to {save_path}")
                else:
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"Error visualizing with grad_cam: {e}")
                raise
                
        elif method == 'lime':
            # LIME implementation would go here
            # This is a placeholder for now
            print("LIME method not yet implemented")
            
        else:
            raise ValueError(f"Unknown explanation method: {method}")

def main():
    """Test the explainer on a sample audio file."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Explain HuBERT SER model predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file for explanation")
    parser.add_argument("--num_emotions", type=int, default=7, help="Number of emotion classes")
    parser.add_argument("--method", type=str, default="grad_cam", choices=["grad_cam", "lime"], 
                        help="Explanation method")
    parser.add_argument("--output", type=str, default=None, help="Path to save visualization")
    
    args = parser.parse_args()
    
    try:
        explainer = HuBERTExplainer(args.model_path, args.num_emotions)
        explainer.visualize_explanation(args.audio_path, args.method, args.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 