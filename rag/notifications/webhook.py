"""Webhook Notifier - Send progress notifications via webhooks"""

import json
import logging
import os
import re
from urllib.parse import urlparse
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from .models import ProgressEvent, IndexingStage

logger = logging.getLogger(__name__)

WEBHOOK_TEMPLATES = {
    "discord": {
        "content_type": "application/json",
        "start": lambda fp, _: {"embeds": [{"title": "📚 Indexing Started", "description": f"**File:** `{fp}`", "color": 3447003}]},
        "progress": lambda e: {"embeds": [{"title": f"{e.emoji} {e.stage_description}", "description": e.message, "color": 16776960}]},
        "finish_success": lambda msg, dur: {"embeds": [{"title": "✅ Indexing Complete", "description": msg or "Success", "color": 5763719, "footer": {"text": f"Duration: {dur:.1f}s"}}]},
        "finish_error": lambda msg, dur: {"embeds": [{"title": "❌ Indexing Failed", "description": msg or "Error", "color": 15548997, "footer": {"text": f"Duration: {dur:.1f}s"}}]},
    },
    "slack": {
        "content_type": "application/json",
        "start": lambda fp, _: {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": f"📚 *Indexing Started*\n`{fp}`"}}]},
        "progress": lambda e: {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": f"{e.emoji} *{e.stage_description}*\n{e.message}"}}]},
        "finish_success": lambda msg, dur: {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": f"✅ *Complete*\n{msg}\n_Duration: {dur:.1f}s_"}}]},
        "finish_error": lambda msg, dur: {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": f"❌ *Failed*\n{msg}\n_Duration: {dur:.1f}s_"}}]},
    },
    "teams": {
        "content_type": "application/json",
        "start": lambda fp, _: {"@type": "MessageCard", "themeColor": "0076D7", "summary": "Indexing Started", "sections": [{"activityTitle": "📚 Indexing Started", "facts": [{"name": "File", "value": fp}]}]},
        "progress": lambda e: {"@type": "MessageCard", "themeColor": "FFCC00", "summary": e.stage_description, "sections": [{"activityTitle": f"{e.emoji} {e.stage_description}", "text": e.message}]},
        "finish_success": lambda msg, dur: {"@type": "MessageCard", "themeColor": "00FF00", "summary": "Complete", "sections": [{"activityTitle": "✅ Complete", "text": f"{msg} ({dur:.1f}s)"}]},
        "finish_error": lambda msg, dur: {"@type": "MessageCard", "themeColor": "FF0000", "summary": "Failed", "sections": [{"activityTitle": "❌ Failed", "text": f"{msg} ({dur:.1f}s)"}]},
    },
    "generic": {
        "content_type": "application/json",
        "start": lambda fp, stages: {"event": "indexing_start", "file_path": fp, "total_stages": stages},
        "progress": lambda e: {"event": "indexing_progress", **e.to_dict()},
        "finish_success": lambda msg, dur: {"event": "indexing_complete", "success": True, "message": msg, "duration_seconds": dur},
        "finish_error": lambda msg, dur: {"event": "indexing_complete", "success": False, "message": msg, "duration_seconds": dur},
    },
}


class WebhookNotifier:
    """Webhook-based progress notifier."""

    def __init__(
        self,
        url: str,
        template: str = "discord",
        notify_stages: Optional[List[IndexingStage]] = None,
        min_interval: float = 2.0,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        self.url = re.sub(r'\$\{([^}]+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), url)
        # Validate the scheme AFTER env expansion, not before: the URL is built
        # by substituting ${ENV_VAR}, so the scheme itself can come from the
        # environment. Without this, a file:// or gopher:// value reaches
        # urlopen and reads local files instead of POSTing a webhook.
        _scheme = urlparse(self.url).scheme.lower()
        if _scheme not in ("http", "https"):
            raise ValueError(
                f"Webhook URL must be http or https, got {_scheme or 'no scheme'!r}"
            )
        self.template = WEBHOOK_TEMPLATES.get(template, WEBHOOK_TEMPLATES["generic"])
        self.notify_stages = notify_stages
        self.min_interval = min_interval
        self.headers = headers or {}
        self.timeout = timeout
        self._last_send: float = 0
        self._lock = threading.Lock()
        self._start_time: Optional[datetime] = None

    def _should_notify(self, stage: IndexingStage) -> bool:
        return self.notify_stages is None or stage in self.notify_stages

    def _send(self, payload: Dict[str, Any]) -> bool:
        try:
            data = json.dumps(payload).encode('utf-8')
            headers = {"Content-Type": self.template["content_type"], **self.headers}
            request = Request(self.url, data=data, headers=headers, method="POST")
            # Scheme is validated to http/https in __init__ (after env
            # expansion). The rule cannot see that guard from here.
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            with urlopen(request, timeout=self.timeout) as response:
                return response.status < 400
        except Exception as e:
            logger.warning(f"Webhook error: {e}")
            return False

    def _send_async(self, payload: Dict[str, Any]) -> None:
        threading.Thread(target=self._send, args=(payload,), daemon=True).start()

    def start(self, file_path: str, total_stages: int = 6) -> None:
        self._start_time = datetime.now()
        if self._should_notify(IndexingStage.LOADING):
            self._send_async(self.template["start"](file_path, total_stages))

    def notify(self, event: ProgressEvent) -> None:
        if not self._should_notify(event.stage) or event.stage in (IndexingStage.COMPLETE, IndexingStage.ERROR):
            return
        with self._lock:
            now = time.time()
            if now - self._last_send < self.min_interval:
                return
            self._last_send = now
        self._send_async(self.template["progress"](event))

    def finish(self, success: bool, message: str = "") -> None:
        duration = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        target = IndexingStage.COMPLETE if success else IndexingStage.ERROR
        if self._should_notify(target):
            key = "finish_success" if success else "finish_error"
            self._send(self.template[key](message, duration))
        self._start_time = None
