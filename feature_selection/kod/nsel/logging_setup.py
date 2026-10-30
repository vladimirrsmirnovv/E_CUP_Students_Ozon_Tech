"""Логирование прогона: подробный текст на диск и события в JSONL."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

_EVENTS_PATH = {"path": None}


def setup_logging(log_file, events_file, level=logging.INFO) -> logging.Logger:
    """Настраивает логгер прогона и путь журнала событий."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nsel")
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    _EVENTS_PATH["path"] = Path(events_file)
    Path(events_file).parent.mkdir(parents=True, exist_ok=True)
    return logger


def emit_event(kind: str, **fields) -> None:
    """Дописывает одно машиночитаемое событие в журнал."""
    path = _EVENTS_PATH["path"]
    if path is None:
        return
    record = {"time": time.time(), "kind": kind}
    record.update(fields)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
