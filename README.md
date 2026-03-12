# 🔮 BTC Prediction Model v1.0.0

A production-grade BTC direction prediction system designed for **Polymarket** 5-minute candle markets. This system uses institutional-grade quantitative techniques to provide high-probability signals.

[![Railway Deployment](https://img.shields.io/badge/Deploy-Railway-blue?style=for-the-badge&logo=railway)](https://railway.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 🚀 Quick Start (Local)

To run this project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/satish05112003/BTC_Prediction_Model.git
cd BTC_Prediction_Model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and add your credentials:
```bash
cp .env.example .env
```
Edit `.env`:
* `TELEGRAM_BOT_TOKEN`: Your bot token from [@BotFather](https://t.me/BotFather)
* `TELEGRAM_CHAT_ID`: Your channel or personal chat ID

### 4. Run the System
```bash
# Start the live prediction system (auto-warms up from Binance API)
python main.py --live
```

---

## ☁️ Railway Deployment (24/7)

This project is fully optimized for **Railway** with a lightweight build configuration.

1.  **Connect your GitHub Repository** to Railway.
2.  **Setup Environment Variables** in the Railway Dashboard (copy them from `.env.example`).
3.  **Deployment**: Railway will automatically detect the `Procfile` and start the worker service.
4.  **Build Optimization**: The `requirements.txt` is configured to use **CPU-only PyTorch** and has been stripped of heavy visualization libraries (matplotlib, seaborn, etc.) to ensure fast builds and prevent timeouts.

---

## 🛠️ System Architecture

The model uses a **multi-layer ensemble architecture**:

1.  **Data Layer**: Real-time Binance WebSocket for price updates + Chainlink Oracle for settlement verification.
2.  **Feature Layer**: 120+ features including technical indicators, order flow microstructure, and market regime detection.
3.  **ML Layer**: An ensemble of **XGBoost**, **LightGBM**, and **Random Forest** models.
4.  **Meta-Validation**: A secondary "Meta-Model" validates every signal to ensure it meets the highest confidence threshold before sending it to Telegram.

---

## 📁 Project Structure

```text
.
├── config/              # Configuration & Env Var management
├── features/            # Feature Engineering (Indicators, Volatility, Regime)
├── live/                # Real-time WebSocket & Prediction loops
├── models/              # Saved .joblib model files & Training logic
├── tg_bot/              # Telegram bot interface
├── utils/               # Shared utilities (logging, data loading)
├── main.py              # Main Entry Point
├── requirements.txt     # Dependencies
├── Procfile             # Cloud deployment config
└── .env.example         # Environment template
```

---

## 🤖 Telegram Commands

Once the bot is running, you can use these commands:
*   `/health` - Check if the system is alive and connected.
*   `/stats` - View today's accuracy and performance.
*   `/last` - Show the most recent signals.

---

## ⚠️ Disclaimer
This system is for **informational and research purposes only**. It does NOT execute trades. Trading involves significant risk.
