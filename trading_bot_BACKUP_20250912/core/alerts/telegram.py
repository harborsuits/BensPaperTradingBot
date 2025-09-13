import os
import requests

def notify(message: str) -> bool:
    """
    Send a notification to Telegram if TG_BOT_TOKEN and TG_CHAT_ID are set.
    Falls back to printing to stdout when not configured.
    """
    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")
    if not token or not chat_id:
        print(f"[ALERT] {message}")
        return False
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=5,
        )
        return True
    except Exception as exc:
        print(f"[ALERT/FAIL] {message} ({exc})")
        return False


