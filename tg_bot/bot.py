import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / ".env.example"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=False)

from telegram.ext import Application, CommandHandler
from logs.prediction_logger import PredictionLogger
from tg_bot.commands import BotCommands, build_signal_message, get_volume_stats

# Load local .env if running locally (Railway variables stay intact)
load_dotenv()

class BTCPredictionBot:
    def __init__(self, prediction_engine, stream_manager):
        self.prediction_engine = prediction_engine
        self.stream_manager = stream_manager
        
        self.latest_signal = None
        self.bot_start_time = time.time()
        self.last_latency_seconds = None
        self.signal_count = 0
        
        self.logger = PredictionLogger()
        self._commands = BotCommands(self)
        
        # Read environment variables safely
        TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        self.chat_id = CHAT_ID
        
        if not TOKEN or not self.chat_id:
            logging.warning(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in environment variables."
            )
            
        if TOKEN:
            self._app = Application.builder().token(TOKEN).build()
            self._register_handlers()
        else:
            self._app = None

    def _register_handlers(self):
        self._app.add_handler(CommandHandler("help", self._commands.handle_help))
        self._app.add_handler(CommandHandler("signal", self._commands.handle_signal))
        self._app.add_handler(CommandHandler("price", self._commands.handle_price))
        self._app.add_handler(CommandHandler("next", self._commands.handle_next))
        self._app.add_handler(CommandHandler("stats", self._commands.handle_stats))
        self._app.add_handler(CommandHandler("stats_today", self._commands.handle_stats_today))
        self._app.add_handler(CommandHandler("last", self._commands.handle_last))
        self._app.add_handler(CommandHandler("volume", self._commands.handle_volume))
        self._app.add_handler(CommandHandler("health", self._commands.handle_health))
        self._app.add_handler(CommandHandler("status", self._commands.handle_status))
        self._app.add_handler(CommandHandler("prob", self._commands.handle_prob))
        self._app.add_handler(CommandHandler("regime", self._commands.handle_regime))
        self._app.add_handler(CommandHandler("latency", self._commands.handle_latency))
        self._app.add_handler(CommandHandler("feeds", self._commands.handle_feeds))
        self._app.add_handler(CommandHandler("model", self._commands.handle_model))

    async def on_new_signal(self, signal):
        """Called by prediction engine every 5 minutes."""
        
        # Resolve previous prediction
        if self.stream_manager and self.stream_manager.settlement_price:
            self.logger.resolve_last_prediction(self.stream_manager.settlement_price)
            
        # Save prediction
        self.logger.log_prediction(signal)
        
        # Store signal
        self.latest_signal = signal
        self.signal_count += 1
        
        # Latency calculation
        if self.stream_manager and self.stream_manager.settlement_ts:
            self.last_latency_seconds = time.time() - self.stream_manager.settlement_ts
            
        # Volume stats
        buy_vol, sell_vol, trades = await get_volume_stats(self.stream_manager)
        
        # Build Telegram message (YOUR ORIGINAL FORMAT)
        msg = build_signal_message(signal, buy_vol, sell_vol, trades)
        
        # Send message
        if self._app and self.chat_id:
            try:
                await self._app.bot.send_message(
                    chat_id=self.chat_id,
                    text=msg
                )
                logging.info("Telegram message sent successfully")
            except Exception as e:
                logging.error(f"Failed to send signal to Telegram: {e}")

    async def start(self):
        if not self._app:
            return
            
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        await self._send_startup_message()

    async def _send_startup_message(self):
        if not self._app or not self.chat_id:
            return
            
        msg = (
            "⚠️ BTC Prediction Bot Started\n\n"
            "This bot provides AI-generated BTC market predictions.\n\n"
            "Important Notice\n\n"
            "• Signals are probabilistic and not guaranteed.\n"
            "• This bot is for informational purposes only.\n"
            "• Not financial advice.\n"
            "• Always manage risk.\n\n"
            "Use /help to see commands."
        )
        try:
            await self._app.bot.send_message(chat_id=self.chat_id, text=msg)
        except Exception as e:
            logging.error(f"Failed to send startup message: {e}")

    async def stop(self):
        if not self._app:
            return
            
        if self._app.updater:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
