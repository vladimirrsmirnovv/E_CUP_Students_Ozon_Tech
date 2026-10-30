"""Обратная чистка: один проход по набору в самом конце."""
from __future__ import annotations

import numpy as np

from .. import cache, jobs, pool
from ..logging_setup import emit_event
from .stage3_ranked_forward import guards_pass

STAGE = "stage6_backward"
CHECKPOINT = "progress"


def removal_order(features: list) -> list:
    """Порядок проверки — обратный порядку добавления, как у коллеги."""
    return list(reversed(features))


def removal_allowed(profile: dict, config, guards) -> tuple:
    """Решение об изъятии признака: нейтральность плюс те же защиты, что и приём."""
    if profile["weighted_improvement"] < -config.neutral_tolerance:
        return False, "изъятие ухудшает метрику"
    allowed, reason = guards_pass(profile, guards)
    if not allowed:
        return False, reason
    return True, "нейтрально"


def _removal_units(context: dict, features: list, order: list) -> list:
    """Единицы работы «набор без признака» в порядке проверки."""
    config = context["config"]
    units = []
    for name in order:
        reduced = [item for item in features if item != name]
        units.append((name, jobs.build_jobs(
            context, reduced, seed=config.random_state,
            iterations=config.probe.iterations,
            early_stopping=config.probe.early_stopping_rounds)))
    return units


def _baseline(context: dict, features: list, executor, stop_check) -> dict:
    """Метрика набора, пришедшего в чистку."""
    config = context["config"]
    job_list = jobs.build_jobs(context, features, seed=config.random_state,
                               iterations=config.probe.iterations,
                               early_stopping=config.probe.early_stopping_rounds)
    results = pool.run_jobs_cached(executor, context["cache_db"], STAGE, job_list,
                                   context["logger"], stop_check)
    return jobs.aggregate(results, job_list, context["folds"])


def run(context: dict, features: list, executor, stop_check) -> dict:
    """Удаляет признаки, чьё изъятие нейтрально и не нарушает защиту."""
    logger = context["logger"]
    config = context["config"].backward
    if not config.enabled:
        baseline = _baseline(context, features, executor, stop_check)
        return {"features": list(features), "baseline": baseline, "removed": []}
    cached = cache.get_stage_result(context["cache_db"], STAGE, context["fingerprint"])
    if cached is not None:
        logger.info("[%s] итог стадии уже посчитан", STAGE)
        return cached

    current = list(features)
    # База считается заново: стадия 4 могла снять сомнительные признаки, и
    # метрика, пришедшая из отбора, относилась бы уже к другому набору.
    current_baseline = _baseline(context, current, executor, stop_check)
    removed = []
    guards = context["config"].stage3
    pending = removal_order(current)

    while pending and len(current) > 1:
        units = _removal_units(context, current, pending)
        workers = getattr(executor, "_max_workers", 1) or 1
        width = max(config.speculation_width,
                    max(1, workers // max(1, len(context["folds"]))))
        stream = pool.iterate_in_order(executor, context["cache_db"], STAGE, units,
                                       logger, stop_check, width)
        removed_name = None
        checked = []
        for name, unit_jobs, results in stream:
            checked.append(name)
            metrics = jobs.aggregate(results, unit_jobs, context["folds"])
            profile = jobs.improvement_profile(current_baseline, metrics,
                                               context["folds"])
            drop, reason = removal_allowed(profile, config, guards)
            logger.info("[%s] без %-42s метрика %+.6f %s -> %s", STAGE, name,
                        -profile["weighted_improvement"], reason, drop)
            if drop:
                current = [item for item in current if item != name]
                current_baseline = metrics
                removed.append(name)
                removed_name = name
                break
        stream.close()
        pending = [name for name in pending if name not in set(checked)]
        cache.put_stage_state(context["cache_db"], STAGE, CHECKPOINT,
                              {"fingerprint": context["fingerprint"],
                               "features": current, "removed": removed})
        if removed_name is None:
            break

    payload = {"features": current, "baseline": current_baseline, "removed": removed}
    cache.put_stage_result(context["cache_db"], STAGE, context["fingerprint"], payload)
    logger.info("[%s] удалено %d, осталось %d", STAGE, len(removed), len(current))
    emit_event("stage_done", stage=STAGE, removed=len(removed), kept=len(current))
    return payload
