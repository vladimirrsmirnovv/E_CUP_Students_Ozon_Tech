"""Стадия 1: временная null importance и ранжирование признаков."""
from __future__ import annotations

import numpy as np

from .. import cache, jobs, pool
from ..logging_setup import emit_event

STAGE = "stage1_null_importance"


def _normalise(values: np.ndarray) -> np.ndarray:
    """Приводит важности одного обучения к сумме единица."""
    total = float(np.sum(np.abs(values)))
    if total <= 0:
        return np.zeros_like(values)
    return np.abs(values) / total


def _build_all_jobs(context: dict, features: list) -> tuple:
    """Задания настоящих и null-моделей для всех фолдов."""
    config = context["config"].stage1
    seed = context["config"].random_state
    actual, null = [], []
    for run in range(config.actual_runs):
        actual.extend(jobs.build_jobs(
            context, features, seed=seed + run, iterations=config.actual_iterations,
            early_stopping=config.actual_early_stopping, want_importance=True,
            wide=True))
    for run in range(config.null_runs):
        null.extend(jobs.build_jobs(
            context, features, seed=seed + 1000 + run,
            iterations=config.null_iterations,
            early_stopping=config.null_early_stopping,
            shuffle_seed=seed + 5000 + run, want_importance=True, wide=True))
    return actual, null


def order_by_fold(job_list: list) -> list:
    """Ставит подряд задания одного фолда, чтобы матрица квантовалась однажды."""
    return sorted(job_list, key=lambda item: item[1]["fold"]["name"])


def _collect(results: dict, job_list: list, folds: list, n_features: int) -> dict:
    """Собирает важности по фолдам: массив «прогон x признак» на каждый фолд."""
    per_fold = {fold.name: [] for fold in folds}
    for key, job in job_list:
        payload = results[key]
        importance = np.asarray(payload["importance"], dtype=np.float64)
        if len(importance) != n_features:
            raise ValueError(f"важностей {len(importance)}, признаков {n_features}")
        per_fold[job["fold"]["name"]].append(_normalise(importance))
    return {name: np.vstack(rows) for name, rows in per_fold.items()}


def _weighted_scores(actual: dict, null: dict, folds: list, quantile: float) -> tuple:
    """Взвешенные по фолдам важность признака и планка null-моделей."""
    score = None
    threshold = None
    for fold in folds:
        actual_mean = actual[fold.name].mean(axis=0) * fold.weight
        null_level = np.quantile(null[fold.name], quantile, axis=0) * fold.weight
        if score is None:
            score = actual_mean
            threshold = null_level
        else:
            score = score + actual_mean
            threshold = threshold + null_level
    return score, threshold


def _unique_keep_order(names: list) -> list:
    """Убирает повторы, сохраняя порядок первого появления."""
    seen = set()
    result = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def select_features(features: list, score: np.ndarray, threshold: np.ndarray,
                    config) -> tuple:
    """Отбирает признаки по планке null-моделей, как в его keep_mode.

    Правило у него — объединение прошедших планку с верхней долей ранжирования,
    а не замена одного другим: признак, чуть не дотянувший до планки, но стоящий
    в первой сотне по важности, остаётся кандидатом.
    """
    order = np.argsort(-score)
    ranking = [features[i] for i in order]
    passed = [features[i] for i in order if score[i] > threshold[i]]
    quantile_count = len(passed)
    top_count = max(int(config.min_keep),
                    int(np.ceil(len(features) * config.fallback_top_fraction)))
    passed = _unique_keep_order(passed + ranking[:top_count])
    if len(passed) < config.min_keep:
        passed = _unique_keep_order(passed + ranking[:config.min_keep])
    reason = "quantile_or_top_fraction"
    if quantile_count >= len(passed):
        reason = "quantile"
    if len(passed) > config.max_keep:
        passed = passed[:config.max_keep]
        reason = reason + "+max_keep"
    return passed, ranking, reason


def run(context: dict, features: list, executor, stop_check) -> dict:
    """Считает стадию 1 и возвращает прошедшие признаки в порядке важности."""
    logger = context["logger"]
    config = context["config"].stage1
    cached = cache.get_stage_result(context["cache_db"], STAGE, context["fingerprint"])
    if cached is not None:
        logger.info("[%s] итог стадии уже посчитан: прошло %d признаков",
                    STAGE, len(cached["passed"]))
        return cached
    if not config.enabled:
        return {"passed": list(features), "ranking": list(features),
                "reason": "disabled", "scores": {}}

    actual_jobs, null_jobs = _build_all_jobs(context, features)
    logger.info("[%s] настоящих обучений %d, null-обучений %d, признаков %d",
                STAGE, len(actual_jobs), len(null_jobs), len(features))
    results = pool.run_jobs_cached(executor, context["cache_db"], STAGE,
                                   order_by_fold(actual_jobs + null_jobs), logger,
                                   stop_check)

    folds = context["folds"]
    actual = _collect(results, actual_jobs, folds, len(features))
    null = _collect(results, null_jobs, folds, len(features))
    score, threshold = _weighted_scores(actual, null, folds, config.quantile)
    passed, ranking, reason = select_features(features, score, threshold, config)

    payload = {"passed": passed, "ranking": ranking, "reason": reason,
               "scores": {name: float(score[i]) for i, name in enumerate(features)},
               "thresholds": {name: float(threshold[i])
                              for i, name in enumerate(features)}}
    cache.put_stage_result(context["cache_db"], STAGE, context["fingerprint"], payload)
    logger.info("[%s] прошло %d из %d признаков, правило %s",
                STAGE, len(passed), len(features), reason)
    emit_event("stage_done", stage=STAGE, passed=len(passed), reason=reason)
    return payload
