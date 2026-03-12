"""
Configuration loader - loads settings.yaml and environment variables.
"""

import os
import re
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
root_dir = Path(__file__).parent.parent
env_file = root_dir / ".env"
if not env_file.exists():
    env_file = root_dir / ".env.example"

load_dotenv(dotenv_path=env_file, override=True)

logger = logging.getLogger(__name__)

token = os.environ.get("TELEGRAM_BOT_TOKEN", "NOT_FOUND")
chat  = os.environ.get("TELEGRAM_CHAT_ID", "NOT_FOUND")
logger.info(f"Telegram token loaded: ...{token[-10:] if len(token) > 10 else token}")
logger.info(f"Telegram chat ID loaded: {chat}")

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
_config: dict = {}

def _resolve_env_vars(value):
    """Replace ${VAR_NAME} with actual environment variable value."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        resolved = os.environ.get(var_name, "")
        return resolved
    return value

def _resolve_config(obj):
    """Recursively resolve all env vars in config dict."""
    if isinstance(obj, dict):
        return {k: _resolve_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_config(i) for i in obj]
    else:
        return _resolve_env_vars(obj)


def load_config(path: str = None) -> dict:
    global _config
    config_path = path or _CONFIG_PATH
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    _config = _resolve_config(raw)
    return _config


def get_config() -> dict:
    global _config
    if not _config:
        load_config()
    return _config


def get(key_path: str, default=None):
    """
    Get a nested config value using dot notation.
    E.g., get('models.xgboost.n_estimators')
    """
    cfg = get_config()
    keys = key_path.split(".")
    val = cfg
    try:
        for k in keys:
            val = val[k]
        return val
    except (KeyError, TypeError):
        return default

if __name__ == "__main__":
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env.example", override=True)
    cfg = get_config()
    print("Token :", cfg.get("telegram", {}).get("bot_token", "NOT FOUND"))
    print("Chat  :", cfg.get("telegram", {}).get("chat_id",  "NOT FOUND"))
