import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import sys
import os
from PIL import Image

class YOLOv8Explainer:
    """Class for explaining YOLOv8 phone detection model predictions using various XAI techniques."""
    
    def __init__(self, model_path, device=None):
        """
        Initialize the explainer with a trained YOLOv8 model.
        
        Args:
            model_path: Path to the trained model weights (.pt file)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = YOLO(model_path)
        
        # YOLOv8 always outputs class 0 for phone in our single-class detector
        self.class_name = "phone"
    
    def predict(self, image_path):
        """Make a prediction on an image file and return the detections."""
        # Run inference
        results = self.model(image_path)
        return results[0]  # First image results
    
    def grad_cam(self, image_path, layer_name="model.22"):
        """
        Apply Grad-CAM to visualize which parts of the image contribute most to the phone detection.
        
        Args:
            image_path: Path to the image file
            layer_name: Name of the layer to use for Grad-CAM (default is the second-to-last layer)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get original predictions for comparison
        results = self.predict(image_path)
        
        # Extract model
        model = self.model.model
        
        # Create hook for activation
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hook for the selected layer
        target_layer = None
        for name, layer in model.named_modules():
            if layer_name in name:
                layer.register_forward_hook(get_activation(name))
                target_layer = layer
                break
                
        if target_layer is None:
            print(f"Layer {layer_name} not found. Using default layer.")
            # Try a default detection head layer
            for name, layer in model.named_modules():
                if isinstance(layer, torch.nn.Conv2d) and name.startswith('model') and int(name.split('.')[1]) > 15:
                    layer.register_forward_hook(get_activation(name))
                    target_layer = layer
                    layer_name = name
                    print(f"Using layer {name} instead")
                    break
        
        # Run inference to get activations
        _ = self.model(image_path)
        
        # Get activation maps
        if layer_name in activations:
            # Get activation from the selected layer
            activation_maps = activations[layer_name]
            
            # For simplicity, use the mean across channels as our importance map
            importance_map = np.mean(np.abs(activation_maps), axis=1)[0]  # take first batch item
            
            # Normalize to 0-1
            importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
            
            # Resize to match the original image size
            heatmap = cv2.resize(importance_map, (img.shape[1], img.shape[0]))
            
            return results, img_rgb, heatmap
        else:
            print(f"No activations found for layer {layer_name}")
            return results, img_rgb, None
    
    def visualize_explanation(self, image_path, method='grad_cam', save_path=None):
        """
        Visualize the explanation for a given image file using the specified method.
        
        Args:
            image_path: Path to the image file
            method: Explanation method ('grad_cam' or 'lime')
            save_path: Path to save the visualization (if None, the plot is shown)
        """
        if method == 'grad_cam':
            results, img, heatmap = self.grad_cam(image_path)
            
            if heatmap is None:
                print("Could not generate heatmap!")
                return
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Plot original image with detections
            plt.subplot(2, 1, 1)
            plt.imshow(img)
            
            # Draw bounding boxes
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    confidence = confidences[i]
                    
                    # Create rectangle patch
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      linewidth=2, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)
                    plt.text(x1, y1-5, f"{self.class_name}: {confidence:.2f}", 
                            color='white', fontsize=10, 
                            bbox=dict(facecolor='red', alpha=0.5))
            else:
                plt.title("Original Image (No Detections)")
            
            plt.title("Original Image with Detections")
            plt.axis('off')
            
            # Plot heatmap overlay
            plt.subplot(2, 1, 2)
            
            # Create colored heatmap
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Overlay heatmap with transparency
            overlay = img * 0.7 + heatmap_colored * 0.3
            overlay = overlay / np.max(overlay)  # Normalize
            
            plt.imshow(overlay)
            plt.title("Grad-CAM Visualization (Areas Contributing to Detection)")
            plt.axis('off')
            
            if save_path:
                plt.tight_layout()
                plt.savefig(save_path)
                print(f"Visualization saved to {save_path}")
            else:
                plt.tight_layout()
                plt.show()
                
        elif method == 'lime':
            # LIME implementation would go here
            # This is a placeholder for now
            print("LIME method not yet implemented")
            
        else:
            raise ValueError(f"Unknown explanation method: {method}")

def main():
    """Test the explainer on a sample image file."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Explain YOLOv8 phone detection model predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image file for explanation")
    parser.add_argument("--method", type=str, default="grad_cam", choices=["grad_cam", "lime"], 
                        help="Explanation method")
    parser.add_argument("--output", type=str, default=None, help="Path to save visualization")
    
    args = parser.parse_args()
    
    explainer = YOLOv8Explainer(args.model_path)
    explainer.visualize_explanation(args.image_path, args.method, args.output)

if __name__ == "__main__":
    main() 