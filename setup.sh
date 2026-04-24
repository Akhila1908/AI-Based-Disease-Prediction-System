#!/bin/bash
# setup.sh - Runs on cloud startup

echo "Running setup script..."
python train_model.py
echo "Setup complete!"
