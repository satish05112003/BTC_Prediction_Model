import time
from datetime import datetime, timezone
import pandas as pd
import pytz

from telegram import Update
from telegram.ext import ContextTypes

IST = pytz.timezone("Asia/Kolkata")

def build_signal_message(signal, buy_vol=0.0, sell_vol=0.0, trades=0) -> str:
    """Standardized signal message for Telegram"""
    
    if not signal.meta_validated:
        dir_emoji = "⚪"
        dir_label = "SKIP (Low Confidence)"
    else:
        if signal.direction == "UP":
            dir_emoji = "🟢"
            dir_label = "LONG (UP)"
        else:
            dir_emoji = "🔴"
            dir_label = "SHORT (DOWN)"

    ist_open = signal.market_open.astimezone(IST)
    ist_close = signal.market_close.astimezone(IST)

    msg = (
        f"BTC SIGNAL (5 MIN)\n\n"
        f"{dir_emoji} Direction: {dir_label}\n\n"
        f"Price to Beat: ${signal.price:,.2f}\n\n"
        f"Probabilities\n"
        f"⬆️ UP: {signal.prob_up * 100:.1f}%\n"
        f"⬇️ DOWN: {signal.prob_down * 100:.1f}%\n\n"
        f"Confidence: {signal.confidence * 100:.1f}%\n\n"
        f"Market Window: {ist_open.strftime('%I:%M %p')} - {ist_close.strftime('%I:%M %p')} (IST)\n\n"
        f"Previous Candle\n"
        f"📈 BUY Volume: {buy_vol:.2f} BTC\n"
        f"📉 SELL Volume: {sell_vol:.2f} BTC\n"
        f"🔢 Total Trades: {trades:,}\n\n"
        f"Use /help for commands"
    )
    return msg

async def get_volume_stats(stream_manager) -> tuple[float, float, int]:
    if not stream_manager or not stream_manager.candle_buffer_5m:
        return 0.0, 0.0, 0
        
    candles = await stream_manager.candle_buffer_5m.get_candles(200)
    if len(candles) < 2:
        return 0.0, 0.0, 0

    # second-to-last = previous closed candle
    prev_candle = candles[-2]
    start_ts = prev_candle["ts"]
    end_ts = start_ts + 300

    trades = await stream_manager.trade_buffer.get_trades(2000)
    if not trades:
        return 0.0, 0.0, 0
        
    window_trades = [t for t in trades if start_ts <= t["ts"] < end_ts]

    buy_vol = sum(t["qty"] for t in window_trades if not t["is_buyer_maker"])
    sell_vol = sum(t["qty"] for t in window_trades if t["is_buyer_maker"])
    total = len(window_trades)

    return buy_vol, sell_vol, total

class BotCommands:
    def __init__(self, bot_instance):
        self.bot = bot_instance

    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "🤖 BTC Prediction Bot\n\n"
            "Available Commands\n\n"
            "/signal – Latest prediction signal\n"
            "/price – Current BTC price\n"
            "/next – Next prediction window\n"
            "/stats – Overall bot performance\n"
            "/stats_today – Today's performance\n"
            "/last – Last 5 signals\n"
            "/volume – Previous candle volume data\n"
            "/health – System health status\n"
            "/status – Bot runtime status\n"
            "/prob – Latest model probabilities\n"
            "/regime – Market regime\n"
            "/latency – Prediction speed\n"
            "/feeds – Data feed status\n"
            "/model – Model information\n"
            "/help – Show command list"
        )
        await update.message.reply_text(msg)

    async def handle_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        signal = self.bot.latest_signal
        if not signal:
            await update.message.reply_text("No signal yet. Waiting for next settlement...")
            return

        buy_vol, sell_vol, trades = await get_volume_stats(self.bot.stream_manager)
        msg = build_signal_message(signal, buy_vol, sell_vol, trades)
        await update.message.reply_text(msg)

    async def handle_price(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        price = None
        if self.bot.stream_manager and self.bot.stream_manager.binance_streamer:
            price = self.bot.stream_manager.binance_streamer.last_price
        
        if price is None:
            await update.message.reply_text("Price data not available yet.")
            return
            
        msg = (
            f"💰 BTC Price\n\n"
            f"Current Price: ${price:,.2f}\n\n"
            f"Source: Binance Live Feed"
        )
        await update.message.reply_text(msg)

    async def handle_next(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        now = time.time()
        next_boundary_ts = ((int(now) // 300) + 1) * 300
        next_open = datetime.fromtimestamp(next_boundary_ts, tz=timezone.utc).astimezone(IST)
        next_close = datetime.fromtimestamp(next_boundary_ts + 300, tz=timezone.utc).astimezone(IST)
        
        msg = (
            f"⏰ Next Prediction Window\n\n"
            f"{next_open.strftime('%I:%M %p')} - {next_close.strftime('%I:%M %p')} (IST)\n\n"
            f"Settlement price will be captured at {next_open.strftime('%I:%M %p')}"
        )
        await update.message.reply_text(msg)

    async def handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        preds = self.bot.logger.get_all_predictions()
        stats = self.bot.logger.get_stats(preds)
        
        msg = (
            f"📊 Bot Performance\n\n"
            f"Total Signals: {stats['total']:,}\n"
            f"✅ Wins: {stats['wins']:,}\n"
            f"❌ Losses: {stats['losses']:,}\n"
            f"⚪ Skipped: {stats['skipped']:,}\n\n"
            f"Accuracy: {stats['accuracy']:.2f}%"
        )
        await update.message.reply_text(msg)

    async def handle_stats_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        preds = self.bot.logger.get_today_predictions()
        stats = self.bot.logger.get_stats(preds)
        
        msg = (
            f"📅 Today's Performance\n\n"
            f"Signals: {stats['total']:,}\n"
            f"✅ Wins: {stats['wins']:,}\n"
            f"❌ Losses: {stats['losses']:,}\n"
            f"⚪ Skipped: {stats['skipped']:,}\n\n"
            f"Accuracy: {stats['accuracy']:.2f}%"
        )
        await update.message.reply_text(msg)

    async def handle_last(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        preds = self.bot.logger.get_last_n(5)
        if not preds:
            await update.message.reply_text("No signals yet.")
            return
            
        lines = ["🕐 Last 5 Signals\n"]
        for p in preds:
            dt_str = p.get("market_open") or p.get("timestamp")
            
            try:
                dt = datetime.fromisoformat(dt_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)
                
            time_str = dt.astimezone(IST).strftime("%I:%M %p")
            
            outcome = p.get("outcome")
            dir_str = p.get("direction", "UNKNOWN")
            
            if not p.get("meta_validated", True) or outcome == "skip":
                res_emoji = "⚪"
                dir_str = "SKIP"
            elif outcome == "win":
                res_emoji = "✅"
            elif outcome == "loss":
                res_emoji = "❌"
            else:
                res_emoji = "⚪" # pending
                
            lines.append(f"{time_str} – {dir_str} {res_emoji}")
            
        await update.message.reply_text("\n".join(lines))

    async def handle_volume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        buy_vol, sell_vol, total_trades = await get_volume_stats(self.bot.stream_manager)
        
        msg = (
            f"📦 Previous Candle Volume\n\n"
            f"📈 BUY Volume: {buy_vol:.2f} BTC\n"
            f"📉 SELL Volume: {sell_vol:.2f} BTC\n\n"
            f"🔢 Total Trades: {total_trades:,}"
        )
        await update.message.reply_text(msg)

    async def handle_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stream_manager = self.bot.stream_manager
        pred_engine = self.bot.prediction_engine
        
        binance_ok = False
        poly_ok = False
        if stream_manager:
            binance_ok = stream_manager.is_healthy()
            if stream_manager.settlement_ts:
                poly_ok = (time.time() - stream_manager.settlement_ts) < 600
                
        engine_ok = pred_engine._loaded if pred_engine else False
        
        binance_str = "🟢 Online" if binance_ok else "🔴 Offline"
        poly_str = "🟢 Online" if poly_ok else "🔴 Offline"
        engine_str = "🟢 Running" if engine_ok else "🔴 Stopped"
        
        system_healthy = binance_ok and poly_ok and engine_ok
        status_str = "✅ Healthy" if system_healthy else "⚠️ Degraded"
        
        msg = (
            f"🏥 System Health\n\n"
            f"Binance Feed: {binance_str}\n"
            f"Polymarket Oracle: {poly_str}\n"
            f"Prediction Engine: {engine_str}\n"
            f"Telegram Bot: 🟢 Active\n\n"
            f"System Status: {status_str}"
        )
        await update.message.reply_text(msg)

    async def handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uptime_seconds = int(time.time() - self.bot.bot_start_time)
        hours = uptime_seconds // 3600
        minutes = (uptime_seconds % 3600) // 60
        
        today_count = len(self.bot.logger.get_today_predictions())
        
        now = time.time()
        next_boundary_ts = ((int(now) // 300) + 1) * 300
        next_open = datetime.fromtimestamp(next_boundary_ts, tz=timezone.utc).astimezone(IST)
        
        msg = (
            f"⚙️ Bot Status\n\n"
            f"Model: Ensemble\n"
            f"Uptime: {hours}h {minutes}m\n"
            f"Signals Generated: {today_count}\n"
            f"Next Signal: {next_open.strftime('%I:%M %p')}"
        )
        await update.message.reply_text(msg)

    async def handle_prob(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        signal = self.bot.latest_signal
        if not signal:
            await update.message.reply_text("No signal yet.")
            return
            
        msg = (
            f"🎯 Latest Model Probabilities\n\n"
            f"⬆️ UP Probability: {signal.prob_up * 100:.1f}%\n"
            f"⬇️ DOWN Probability: {signal.prob_down * 100:.1f}%\n\n"
            f"Model Confidence: {signal.confidence * 100:.1f}%"
        )
        await update.message.reply_text(msg)

    async def handle_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        pred_engine = self.bot.prediction_engine
        stream_manager = self.bot.stream_manager
        
        if not pred_engine or not pred_engine.regime_detector or not stream_manager:
            await update.message.reply_text("Data not available yet.")
            return
            
        candles = await stream_manager.candle_buffer_5m.get_candles(50)
        if len(candles) < 50:
            await update.message.reply_text("Not enough candles to detect regime.")
            return
            
        df = pd.DataFrame(candles)
        
        # ATR calculation on last 20 candles
        df_atr = df.tail(20).copy()
        df_atr["high"] = pd.to_numeric(df_atr["high"])
        df_atr["low"] = pd.to_numeric(df_atr["low"])
        df_atr["tr"] = df_atr["high"] - df_atr["low"]
        atr = df_atr["tr"].mean()
        
        if atr < 100:
            vol_lvl = "Low"
        elif atr < 300:
            vol_lvl = "Medium"
        else:
            vol_lvl = "High"
                
        try:
            states = pred_engine.regime_detector.predict(df)
            state = states[-1] if hasattr(states, '__getitem__') else states
            
            # Map HMM state index
            state_map = {0: "Ranging", 1: "Trending", 2: "Volatile"}
            curr_state = state_map.get(int(state), f"State {state}")
        except Exception as e:
            curr_state = "Unknown"
            
        msg = (
            f"📡 Market Regime\n\n"
            f"Current State: {curr_state}\n"
            f"Volatility Level: {vol_lvl}"
        )
        await update.message.reply_text(msg)

    async def handle_latency(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        lat = self.bot.last_latency_seconds
        if lat is None:
            await update.message.reply_text("No latency data yet.")
            return
            
        msg = (
            f"⚡ Prediction Engine Latency\n\n"
            f"Signal Generated In: {lat:.1f} seconds"
        )
        await update.message.reply_text(msg)

    async def handle_feeds(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stream_manager = self.bot.stream_manager
        
        binance_str = "🔴 Disconnected"
        poly_str = "🔴 Disconnected"
        last_update = 0
        
        if stream_manager:
            if stream_manager.binance_streamer and stream_manager.binance_streamer.last_update:
                last_update = time.time() - stream_manager.binance_streamer.last_update
                b_ok = last_update < 60
                binance_str = "🟢 Connected" if b_ok else "🔴 Disconnected"
                
            p_ok = False
            if stream_manager.settlement_ts:
                p_ok = (time.time() - stream_manager.settlement_ts) < 600
            poly_str = "🟢 Connected" if p_ok else "🔴 Disconnected"
            
        msg = (
            f"📡 Data Feed Status\n\n"
            f"Binance Trade Stream: {binance_str}\n"
            f"Polymarket Settlement Feed: {poly_str}\n\n"
            f"Last Price Update: {int(last_update)} seconds ago"
        )
        await update.message.reply_text(msg)

    async def handle_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            f"🤖 Model Information\n\n"
            f"Primary Model: Ensemble\n\n"
            f"Components\n"
            f"• XGBoost\n"
            f"• LightGBM\n"
            f"• Random Forest\n"
            f"• Logistic Regression\n\n"
            f"Total Features Used: 420+"
        )
        await update.message.reply_text(msg)
