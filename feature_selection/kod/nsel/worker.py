"""Сторона процесса-исполнителя: одно обучение по описанию задания."""
from __future__ import annotations

import os

# Потоки библиотеки ограничиваются до её импорта: иначе каждый воркер поднимет
# пул на все ядра машины и процессы начнут отбирать ядра друг у друга.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import resource
import sys
import time

import numpy as np

from . import data, encoding, model
from .config import EncodingConfig


def _apply_categoricals(matrices: dict, job: dict) -> dict:
    """Готовит категориальные столбцы под конкретное семейство моделей."""
    positions = matrices["categorical_positions"]
    if job["model_mode"] == "catboost":
        return {"X_train": matrices["X_train"], "X_valid": matrices["X_valid"],
                "categorical_positions": positions, "encoding": {}}
    config = EncodingConfig(**job["encoding"])
    train, valid, diagnostics = encoding.encode_matrices(
        matrices["X_train"], matrices["X_valid"], matrices["y_train"],
        matrices["block_ids"], positions, config)
    return {"X_train": train, "X_valid": valid, "categorical_positions": [],
            "encoding": diagnostics}


def peak_memory_mb() -> float:
    """Пик памяти процесса-исполнителя за всё его время жизни."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / 1e6
    return peak / 1024.0


def _layout_for(model_mode: str) -> str:
    """CatBoost получает таблицу с целыми категориями, остальные — матрицу."""
    if model_mode == "catboost":
        return "frame"
    return "array"


def _training_target(matrices: dict, job: dict) -> np.ndarray:
    """Таргет обучения; для null-модели он перемешан внутри временных блоков."""
    target = model.build_targets(matrices["y_train_raw"], matrices["anchor_train"],
                                 job["target_mode"])
    if job.get("shuffle_seed") is None:
        return target
    return data.temporal_shuffle(target, matrices["block_ids"], job["shuffle_seed"])


def _validation_target(matrices: dict, job: dict) -> np.ndarray:
    """Проверочный таргет; у null-модели он перемешан тоже.

    Ранняя остановка тогда смотрит на разрушенную цель и о настоящей не знает
    ничего — утечки нет, а обучение обрывается быстро, потому что улучшать
    нечего.
    """
    target = model.build_targets(matrices["y_valid_raw"], matrices["anchor_valid"],
                                 job["target_mode"])
    if job.get("shuffle_seed") is None:
        return target
    blocks = np.zeros(len(target), dtype=np.int32)
    return data.temporal_shuffle(target, blocks, job["shuffle_seed"] + 1)


class TrainingFailed(RuntimeError):
    """Обучение в процессе-исполнителе не удалось; текст пришёл из воркера."""


def run_training(job: dict) -> dict:
    """Обучает щуп на одном фолде, возвращая ошибку текстом, а не объектом.

    Исключения библиотек моделей распаковываются только там, где библиотека
    импортирована. Родитель импортирует её лениво и у него её нет, поэтому
    настоящая ошибка воркера превращалась в невнятное «процесс убит».
    """
    import traceback

    try:
        return _train(job)
    except BaseException as error:
        return {"error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()[-4000:]}


def _train(job: dict) -> dict:
    """Обучает щуп на одном фолде и возвращает метрики."""
    started = time.perf_counter()
    matrices = data.materialize(job["prepared_dir"], job["fold"],
                                job["feature_columns"], job["target_mode"],
                                layout=_layout_for(job["model_mode"]))
    matrices["y_train"] = _training_target(matrices, job)
    matrices["y_valid"] = _validation_target(matrices, job)

    prepared = _apply_categoricals(matrices, job)
    payload = dict(matrices)
    payload.update(prepared)
    # Второй словарь держал бы вторую ссылку на матрицы, и освободить их до
    # конца обучения было бы нечем.
    matrices.clear()
    prepared_encoding = prepared["encoding"]
    prepared.clear()

    metrics = model.fit_and_score(
        payload, job["model_mode"], job["target_mode"], job["probe"], job["seed"],
        job["threads"], job["iterations"], job["early_stopping"])
    metrics["fold"] = job["fold"]["name"]
    metrics["weight"] = job["fold"]["weight"]
    metrics["n_features"] = len(job["feature_columns"])
    metrics["seconds"] = time.perf_counter() - started
    metrics["peak_rss_mb"] = peak_memory_mb()
    metrics["encoding"] = prepared_encoding
    if not job.get("want_importance", False):
        metrics.pop("importance", None)
    return metrics
