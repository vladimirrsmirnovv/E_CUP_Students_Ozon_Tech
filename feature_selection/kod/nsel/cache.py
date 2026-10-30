"""Кэш прогона: итерации, промежуточное состояние стадии и итоги стадий."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS iterations (
    key TEXT PRIMARY KEY, stage TEXT NOT NULL,
    payload TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS iterations_stage ON iterations(stage);
CREATE TABLE IF NOT EXISTS stage_state (
    stage TEXT NOT NULL, checkpoint TEXT NOT NULL, payload TEXT NOT NULL,
    updated_at REAL NOT NULL, PRIMARY KEY (stage, checkpoint));
CREATE TABLE IF NOT EXISTS stage_results (
    stage TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
    payload TEXT NOT NULL, finished_at REAL NOT NULL);
"""


def make_key(*parts) -> str:
    """Ключ кэша по всем параметрам, влияющим на результат."""
    blob = json.dumps(parts, sort_keys=True, default=str,
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _connect(db_path) -> sqlite3.Connection:
    """Открывает короткоживущее соединение с журналом WAL."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=60.0, isolation_level=None)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def _with_retry(action, retries: int = 12, base_delay: float = 0.05):
    """Повторяет операцию при блокировке базы."""
    last = None
    for attempt in range(retries):
        try:
            return action()
        except sqlite3.OperationalError as error:
            last = error
            if "lock" not in str(error).lower() or attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise last


def init_db(db_path) -> None:
    """Создаёт таблицы кэша, если их ещё нет."""
    connection = _connect(db_path)
    try:
        connection.executescript(SCHEMA)
    finally:
        connection.close()


def _query(db_path, sql: str, parameters: tuple):
    """Выполняет чтение и возвращает первую строку."""
    def action():
        connection = _connect(db_path)
        try:
            return connection.execute(sql, parameters).fetchone()
        finally:
            connection.close()

    return _with_retry(action)


def _execute(db_path, sql: str, parameters: tuple) -> None:
    """Выполняет запись."""
    def action():
        connection = _connect(db_path)
        try:
            connection.execute(sql, parameters)
        finally:
            connection.close()

    _with_retry(action)


def get_iteration(db_path, key: str):
    """Результат одного обучения из кэша или None."""
    row = _query(db_path, "SELECT payload FROM iterations WHERE key = ?", (key,))
    if row is None:
        return None
    return json.loads(row[0])


def get_iterations(db_path, keys: list) -> dict:
    """Пакетное чтение результатов обучений по списку ключей."""
    if not keys:
        return {}

    def action():
        connection = _connect(db_path)
        try:
            found = {}
            for start in range(0, len(keys), 500):
                chunk = keys[start:start + 500]
                marks = ",".join("?" * len(chunk))
                rows = connection.execute(
                    f"SELECT key, payload FROM iterations WHERE key IN ({marks})",
                    tuple(chunk)).fetchall()
                found.update({key: json.loads(payload) for key, payload in rows})
            return found
        finally:
            connection.close()

    return _with_retry(action)


def put_iteration(db_path, key: str, stage: str, payload: dict) -> None:
    """Сохраняет результат обучения сразу по его готовности."""
    _execute(db_path,
             "INSERT OR REPLACE INTO iterations (key, stage, payload, created_at) "
             "VALUES (?, ?, ?, ?)",
             (key, stage, json.dumps(payload, ensure_ascii=False), time.time()))


def put_stage_state(db_path, stage: str, checkpoint: str, payload: dict) -> None:
    """Сохраняет промежуточное состояние стадии для возобновления."""
    _execute(db_path,
             "INSERT OR REPLACE INTO stage_state (stage, checkpoint, payload, "
             "updated_at) VALUES (?, ?, ?, ?)",
             (stage, checkpoint, json.dumps(payload, ensure_ascii=False), time.time()))


def get_stage_state(db_path, stage: str, checkpoint: str):
    """Промежуточное состояние стадии или None."""
    row = _query(db_path,
                 "SELECT payload FROM stage_state WHERE stage = ? AND checkpoint = ?",
                 (stage, checkpoint))
    if row is None:
        return None
    return json.loads(row[0])


def latest_stage_state(db_path, stage: str):
    """Самое свежее промежуточное состояние стадии или None."""
    row = _query(db_path,
                 "SELECT payload FROM stage_state WHERE stage = ? "
                 "ORDER BY updated_at DESC LIMIT 1", (stage,))
    if row is None:
        return None
    return json.loads(row[0])


def put_stage_result(db_path, stage: str, fingerprint: str, payload: dict) -> None:
    """Записывает итог стадии."""
    _execute(db_path,
             "INSERT OR REPLACE INTO stage_results (stage, fingerprint, payload, "
             "finished_at) VALUES (?, ?, ?, ?)",
             (stage, fingerprint, json.dumps(payload, ensure_ascii=False), time.time()))


def get_stage_result(db_path, stage: str, fingerprint: str):
    """Итог стадии, если он посчитан с тем же отпечатком настроек."""
    row = _query(db_path,
                 "SELECT payload, fingerprint FROM stage_results WHERE stage = ?",
                 (stage,))
    if row is None:
        return None
    payload, saved = row
    if saved != fingerprint:
        return None
    return json.loads(payload)


def count_iterations(db_path, stage: str) -> int:
    """Сколько обучений стадии лежит в кэше."""
    row = _query(db_path, "SELECT COUNT(*) FROM iterations WHERE stage = ?", (stage,))
    return int(row[0])
