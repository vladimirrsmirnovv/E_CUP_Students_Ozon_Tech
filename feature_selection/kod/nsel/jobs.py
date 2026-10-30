"""Сборка заданий обучения и взвешивание метрик по фолдам."""
from __future__ import annotations

import dataclasses

from . import cache


def build_probe(config) -> dict:
    """Параметры щупа в виде словаря для передачи в воркер."""
    return dataclasses.asdict(config.probe)


def build_jobs(context: dict, feature_columns: list, seed: int, iterations: int,
               early_stopping: int, shuffle_seed=None,
               want_importance: bool = False, folds=None, wide: bool = False) -> list:
    """Задания «обучить набор на каждом фолде» с ключами кэша."""
    config = context["config"]
    folds = folds or context["folds"]
    # Повтор имени столбца недопустим ни для одной библиотеки и всегда означает
    # ошибку вызывающего, а не намерение: набор признаков — это множество.
    columns = sorted(set(feature_columns))
    encoding = dataclasses.asdict(config.encoding)
    probe = build_probe(config)
    threads = config.parallel.threads_for(wide)
    jobs = []
    for fold in folds:
        job = {"prepared_dir": str(config.prepared_dir), "fold": fold.as_dict(),
               "feature_columns": columns, "model_mode": config.model_mode,
               "target_mode": config.target_mode, "probe": probe, "seed": seed,
               "threads": threads,
               "iterations": iterations, "early_stopping": early_stopping,
               "shuffle_seed": shuffle_seed, "encoding": encoding,
               "want_importance": want_importance}
        key = cache.make_key(context["fingerprint"], fold.key_parts(), columns,
                             config.model_mode, config.target_mode, probe, seed,
                             iterations, early_stopping, shuffle_seed, encoding,
                             want_importance, threads)
        jobs.append((key, job))
    return jobs


def aggregate(results: dict, jobs: list, folds: list) -> dict:
    """Взвешенная метрика набора и разбивка по фолдам."""
    by_fold = {}
    for key, job in jobs:
        payload = results.get(key)
        if payload is None:
            raise KeyError(f"нет результата для фолда {job['fold']['name']}")
        by_fold[job["fold"]["name"]] = payload
    weighted = 0.0
    for fold in folds:
        weighted += fold.weight * by_fold[fold.name]["rmsle"]
    return {"weighted": weighted,
            "per_fold": {name: {"rmsle": payload["rmsle"],
                                "rmsle_zero": payload["rmsle_zero"],
                                "rmsle_positive": payload["rmsle_positive"],
                                "trees": payload["trees"]}
                         for name, payload in by_fold.items()}}


def improvement_profile(baseline: dict, candidate: dict, folds: list) -> dict:
    """Выигрыш набора относительно базы: взвешенный и по каждому фолду."""
    per_fold = {}
    improved = 0
    worst_degradation = 0.0
    worst_zero = 0.0
    worst_positive = 0.0
    for fold in folds:
        base = baseline["per_fold"][fold.name]
        new = candidate["per_fold"][fold.name]
        delta = base["rmsle"] - new["rmsle"]
        per_fold[fold.name] = delta
        if delta > 0:
            improved += 1
        worst_degradation = max(worst_degradation, -delta)
        worst_zero = max(worst_zero, new["rmsle_zero"] - base["rmsle_zero"])
        worst_positive = max(worst_positive,
                             new["rmsle_positive"] - base["rmsle_positive"])
    return {"weighted_improvement": baseline["weighted"] - candidate["weighted"],
            "per_fold_improvement": per_fold, "improved_folds": improved,
            "worst_fold_degradation": worst_degradation,
            "worst_zero_degradation": worst_zero,
            "worst_positive_degradation": worst_positive}
