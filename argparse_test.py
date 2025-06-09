import argparse

# Create parser
parser = argparse.ArgumentParser(description="Train a model with given hyperparameters.")

# Add arguments
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--model', type=str, choices=['mlp', 'cnn'], default='mlp', help='Model type')

# Parse arguments
args = parser.parse_args()

# Use the arguments
print(f"Training {args.model} with lr={args.lr} for {args.epochs} epochs")