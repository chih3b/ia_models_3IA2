import os
import subprocess
import argparse

def run_experiments(args):
    """Run all three HuBERT fine-tuning approaches in sequence using standard train/test splits"""
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fine-tuning approaches to try
    approaches = ["full", "qkv", "classifier"]
    
    for approach in approaches:
        print(f"\n{'='*80}")
        print(f"Running {approach} fine-tuning approach")
        print(f"{'='*80}\n")
        
        # Create approach-specific output directory
        approach_dir = os.path.join(args.output_dir, approach)
        os.makedirs(approach_dir, exist_ok=True)
        
        # Build command
        cmd = [
            "python", "train_hubert_standard.py",
            "--data_dir", args.data_dir,
            "--output_dir", approach_dir,
            "--test_size", str(args.test_size),
            "--val_size", str(args.val_size),
            "--fine_tuning_type", approach,
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--learning_rate", str(args.learning_rate),
            "--hidden_size", str(args.hidden_size)
        ]
        
        # Run command
        subprocess.run(cmd)
    
    print("\nAll experiments completed!")
    print(f"Results are saved in {args.output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all HuBERT fine-tuning approaches with standard splits")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing emotion data")
    parser.add_argument("--output_dir", type=str, default="./results_hubert_standard", help="Base output directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set ratio")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=256, help="Size of hidden layer")
    
    args = parser.parse_args()
    
    run_experiments(args)
