"""Оптуна: разведка, затем сужение пространства и уточнение."""
from __future__ import annotations

import dataclasses

import numpy as np

from .. import cache, jobs, pool
from ..config import ProbeConfig
from ..logging_setup import emit_event

STAGE = "stage7_optuna"

SEARCH_SPACE = {
    "learning_rate": {"low": 0.01, "high": 0.30, "log": True, "kind": "float"},
    "depth": {"low": 4, "high": 10, "log": False, "kind": "int"},
    "l2_leaf_reg": {"low": 0.5, "high": 30.0, "log": True, "kind": "float"},
    "subsample": {"low": 0.5, "high": 1.0, "log": False, "kind": "float"},
    "colsample": {"low": 0.5, "high": 1.0, "log": False, "kind": "float"},
    "min_data_in_leaf": {"low": 1, "high": 200, "log": True, "kind": "int"},
}


def _sampler(config, startup_trials: int):
    """Создаёт TPE-сэмплер с настройками из промпта."""
    import optuna

    return optuna.samplers.TPESampler(
        n_startup_trials=startup_trials, multivariate=config.tpe_multivariate,
        group=config.tpe_group, seed=config.random_state)


def _suggest(trial, space: dict) -> dict:
    """Берёт у Оптуны одну точку пространства."""
    params = {}
    for name, bounds in space.items():
        if bounds["kind"] == "int":
            params[name] = trial.suggest_int(name, int(bounds["low"]),
                                             int(bounds["high"]), log=bounds["log"])
        else:
            params[name] = trial.suggest_float(name, bounds["low"], bounds["high"],
                                               log=bounds["log"])
    return params


def narrow_space(space: dict, records: list, config) -> dict:
    """Сужает пространство вокруг лучших точек разведки."""
    if len(records) < 2:
        return dict(space)
    ordered = sorted(records, key=lambda item: item["value"])[:config.adaptive_top_k]
    narrowed = {}
    for name, bounds in space.items():
        values = np.array([item["params"][name] for item in ordered], dtype=np.float64)
        low = float(np.quantile(values, config.adaptive_quantile_low))
        high = float(np.quantile(values, config.adaptive_quantile_high))
        span = bounds["high"] - bounds["low"]
        pad = config.adaptive_range_expand * max(high - low, 0.0)
        low, high = low - pad, high + pad
        minimum = config.adaptive_min_range_fraction * span
        if high - low < minimum:
            middle = (high + low) / 2.0
            low, high = middle - minimum / 2.0, middle + minimum / 2.0
        guard = config.adaptive_boundary_guard_fraction * span
        if low - bounds["low"] < guard:
            low = bounds["low"]
        if bounds["high"] - high < guard:
            high = bounds["high"]
        low = max(low, bounds["low"])
        high = min(high, bounds["high"])
        narrowed[name] = {**bounds, "low": low, "high": high}
        if bounds["kind"] == "int":
            narrowed[name]["low"] = int(np.floor(low))
            narrowed[name]["high"] = max(int(np.ceil(high)), int(np.floor(low)) + 1)
    return narrowed


def _probe_from_params(base: ProbeConfig, params: dict, config) -> ProbeConfig:
    """Собирает параметры щупа из точки пространства."""
    return dataclasses.replace(
        base, learning_rate=params["learning_rate"], depth=int(params["depth"]),
        l2_leaf_reg=params["l2_leaf_reg"], subsample=params["subsample"],
        colsample=params["colsample"],
        min_data_in_leaf=int(params["min_data_in_leaf"]),
        early_stopping_rounds=config.early_stopping_rounds)


def _evaluate_batch(context: dict, features: list, points: list, executor,
                    stop_check) -> list:
    """Считает пачку точек пространства целиком, без барьера внутри пачки."""
    config = context["config"]
    per_point = []
    all_jobs = []
    for params in points:
        probe = _probe_from_params(config.probe, params, config.optuna)
        trial_context = {**context, "config": dataclasses.replace(config, probe=probe)}
        job_list = jobs.build_jobs(trial_context, features,
                                   seed=config.optuna.random_state,
                                   iterations=probe.iterations,
                                   early_stopping=probe.early_stopping_rounds)
        per_point.append(job_list)
        all_jobs.extend(job_list)
    results = pool.run_jobs_cached(executor, context["cache_db"], STAGE, all_jobs,
                                   context["logger"], stop_check)
    return [jobs.aggregate(results, job_list, context["folds"])["weighted"]
            for job_list in per_point]


def _run_phase(context: dict, features: list, space: dict, trials: int,
               startup: int, executor, stop_check, phase: str) -> list:
    """Проводит одну фазу поиска пачками ask/tell."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger = context["logger"]
    config = context["config"].optuna
    study = optuna.create_study(direction="minimize",
                                sampler=_sampler(config, startup))
    batch = max(1, getattr(executor, "_max_workers", 1) // max(1, len(context["folds"])))
    records = []
    while len(records) < trials:
        size = min(batch, trials - len(records))
        asked = [study.ask() for _ in range(size)]
        points = [_suggest(trial, space) for trial in asked]
        values = _evaluate_batch(context, features, points, executor, stop_check)
        for trial, params, value in zip(asked, points, values):
            study.tell(trial, value)
            records.append({"params": params, "value": value, "phase": phase})
        best = min(record["value"] for record in records)
        logger.info("[%s] %s %d/%d лучший %.6f", STAGE, phase, len(records),
                    trials, best)
    return records


def run(context: dict, features: list, executor, stop_check) -> dict:
    """Ищет параметры модели на той же кросс-валидации, что и отбор."""
    logger = context["logger"]
    config = context["config"].optuna
    if not config.enabled:
        logger.info("[%s] выключено", STAGE)
        return {"enabled": False}
    cached = cache.get_stage_result(context["cache_db"], STAGE, context["fingerprint"])
    if cached is not None:
        logger.info("[%s] итог стадии уже посчитан", STAGE)
        return cached

    exploration = _run_phase(context, features, SEARCH_SPACE,
                             config.exploration_trials,
                             config.tpe_startup_trials_explore, executor, stop_check,
                             "разведка")
    focused = []
    space = dict(SEARCH_SPACE)
    if config.adaptive_search:
        space = narrow_space(SEARCH_SPACE, exploration, config)
        logger.info("[%s] пространство сужено: %s", STAGE,
                    {name: (round(bounds["low"], 4), round(bounds["high"], 4))
                     for name, bounds in space.items()})
        focused = _run_phase(context, features, space, config.focused_trials,
                             config.tpe_startup_trials_focus, executor, stop_check,
                             "уточнение")

    records = exploration + focused
    best = min(records, key=lambda item: item["value"])
    payload = {"enabled": True, "best_params": best["params"],
               "best_value": best["value"], "trials": len(records),
               "narrowed_space": space,
               "exploration_best": min(item["value"] for item in exploration)}
    cache.put_stage_result(context["cache_db"], STAGE, context["fingerprint"], payload)
    logger.info("[%s] лучший счёт %.6f на %d трайлах", STAGE, best["value"],
                len(records))
    emit_event("stage_done", stage=STAGE, best=best["value"], trials=len(records))
    return payload
