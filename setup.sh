#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BTC Prediction Model - Environment Setup Script
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "=============================================="
echo " BTC Prediction Model - Environment Setup"
echo "=============================================="

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p data logs models

# Copy env template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Created .env from template. Edit .env with your credentials."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Add your BTC data files to data/ directory:"
echo "     - data/btc_1m.csv"
echo "     - data/btc_5m.csv"
echo "     - data/btc_1h.csv"
echo ""
echo "  2. Edit .env with your Telegram bot token and chat ID"
echo ""
echo "  3. Train models:"
echo "     python main.py --train"
echo ""
echo "  4. Run live system:"
echo "     python main.py --live"
echo ""
echo "  Or run full pipeline (train if needed, then live):"
echo "     python main.py"
