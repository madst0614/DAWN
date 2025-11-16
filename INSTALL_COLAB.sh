#!/bin/bash
# SPROUT Installation Script for Google Colab
# Run this in a Colab cell with: !bash INSTALL_COLAB.sh

echo "======================================================================="
echo "SPROUT - Installing Dependencies"
echo "======================================================================="

# Install core dependencies
pip install -q torch>=2.0.0
pip install -q transformers>=4.30.0
pip install -q datasets>=2.12.0
pip install -q tqdm>=4.65.0

echo ""
echo "✅ Installation Complete!"
echo ""
echo "Now you can run:"
echo "  python scripts/train_sprout_mlm.py --debug_mode"
echo ""
