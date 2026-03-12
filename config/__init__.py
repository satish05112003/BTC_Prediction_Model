import os
import re
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env.example only locally
_env_path = Path(__file__).parent.parent / ".env.example"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=False)

_config_cache = None


def _resolve_env_vars(value):
    if not isinstance(value, str):
        return value
    pattern = re.compile(r'\$\{([^}]+)\}')
    def replacer(match):
        var_name = match.group(1)
        result = os.environ.get(var_name, "")
        if not result:
            logger.warning(f"Environment variable '{var_name}' is not set.")
        return result
    return pattern.sub(replacer, value)


def _resolve_config(obj):
    if isinstance(obj, dict):
        return {k: _resolve_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_config(i) for i in obj]
    else:
        return _resolve_env_vars(obj)


def get_config() -> dict:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    settings_path = Path(__file__).parent / "settings.yaml"
    with open(settings_path, "r") as f:
        raw = yaml.safe_load(f)

    _config_cache = _resolve_config(raw)

    token = _config_cache.get("telegram", {}).get("bot_token", "")
    chat  = _config_cache.get("telegram", {}).get("chat_id", "")
    logger.info(f"[Config] Telegram token: ...{token[-8:] if len(token) > 8 else 'NOT SET'}")
    logger.info(f"[Config] Telegram chat_id: {chat or 'NOT SET'}")

    return _config_cache
