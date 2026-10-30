"""Стадия 5: парное спасение почти прошедших кандидатов."""
from __future__ import annotations

import itertools
import types

from .. import cache, jobs, pool
from ..data import feature_family
from ..logging_setup import emit_event
from .stage3_ranked_forward import guards_pass

STAGE = "stage5_pair_rescue"
GUARD_REASONS = ("просадка фолда", "просадка сегмента нулей",
                 "просадка сегмента покупавших")


def near_miss_candidates(trials: list, stage3_config, stage5_config,
                         selected=()) -> list:
    """Отвергнутые кандидаты, чей выигрыш был близок к порогу приёмки.

    Уже отобранные исключаются: после перезапусков один и тот же признак мог
    попасть в историю и отвергнутым, и принятым, а пара с ним дала бы
    повторяющееся имя столбца.
    """
    taken = set(selected)
    floor = stage5_config.near_miss_fraction * stage3_config.min_improvement
    near = [trial for trial in trials
            if not trial.get("accepted")
            and trial.get("feature") not in taken
            and trial.get("reason") not in GUARD_REASONS
            and trial.get("weighted_improvement", 0.0) >= floor]
    near.sort(key=lambda trial: -trial["weighted_improvement"])
    seen = []
    for trial in near:
        if trial["feature"] not in seen:
            seen.append(trial["feature"])
        if len(seen) >= stage5_config.top_rejected:
            break
    return seen


def build_pairs(candidates: list, prefer_cross_family: bool) -> list:
    """Пары кандидатов; при возможности из разных семей признаков."""
    pairs = list(itertools.combinations(candidates, 2))
    if not prefer_cross_family:
        return pairs
    cross = [pair for pair in pairs
             if feature_family(pair[0]) != feature_family(pair[1])]
    if cross:
        return cross
    return pairs


def _evaluate_pair(context: dict, selected: list, pair: tuple, executor,
                   stop_check) -> dict:
    """Метрика набора вместе с парой кандидатов."""
    config = context["config"]
    job_list = jobs.build_jobs(context, selected + list(pair),
                               seed=config.random_state,
                               iterations=config.probe.iterations,
                               early_stopping=config.probe.early_stopping_rounds)
    results = pool.run_jobs_cached(executor, context["cache_db"], STAGE, job_list,
                                   context["logger"], stop_check)
    return jobs.aggregate(results, job_list, context["folds"])


def _pair_accepted(profile: dict, config) -> bool:
    """Правило приёма пары."""
    allowed, _reason = guards_pass(profile, config)
    if not allowed:
        return False
    return (profile["weighted_improvement"] >= config.min_improvement
            and profile["improved_folds"] >= config.min_improved_folds)


def run(context: dict, selected: list, baseline: dict, trials: list, executor,
        stop_check) -> dict:
    """Пробует добавить пары почти прошедших кандидатов."""
    logger = context["logger"]
    config = context["config"].stage5
    if not config.enabled:
        logger.info("[%s] выключено по умолчанию", STAGE)
        return {"selected": list(selected), "baseline": baseline, "added": [],
                "enabled": False}
    cached = cache.get_stage_result(context["cache_db"], STAGE, context["fingerprint"])
    if cached is not None:
        logger.info("[%s] итог стадии уже посчитан", STAGE)
        return cached

    candidates = near_miss_candidates(trials, context["config"].stage3, config,
                                      selected)
    pairs = build_pairs(candidates, config.prefer_cross_family)[:config.max_pairs]
    logger.info("[%s] почти прошедших %d, пар к проверке %d",
                STAGE, len(candidates), len(pairs))

    current = list(selected)
    current_baseline = baseline
    added = []
    guard_config = types.SimpleNamespace(
        min_improvement=config.min_improvement,
        min_improved_folds=config.min_improved_folds,
        max_allowed_fold_degradation=config.max_allowed_fold_degradation,
        max_zero_segment_degradation=float("inf"),
        max_positive_segment_degradation=float("inf"))
    for pair in pairs:
        if len(added) >= config.max_additions:
            break
        # Пары строятся один раз, поэтому один признак может встретиться в двух
        # парах. Если он уже взят, пара перестаёт быть парой: её порог
        # 0.000075 задан для двух признаков сразу, а не для одного.
        if any(name in current for name in pair):
            logger.info("[%s] %s + %s пропущена, признак уже выбран", STAGE,
                        pair[0], pair[1])
            continue
        metrics = _evaluate_pair(context, current, pair, executor, stop_check)
        profile = jobs.improvement_profile(current_baseline, metrics,
                                           context["folds"])
        accepted = _pair_accepted(profile, guard_config)
        logger.info("[%s] %s + %s выигрыш %+.6f фолдов %d -> %s", STAGE, pair[0],
                    pair[1], profile["weighted_improvement"],
                    profile["improved_folds"], accepted)
        if accepted:
            current = current + list(pair)
            current_baseline = metrics
            added.append(list(pair))

    payload = {"selected": current, "baseline": current_baseline, "added": added,
               "enabled": True, "candidates": candidates}
    cache.put_stage_result(context["cache_db"], STAGE, context["fingerprint"], payload)
    emit_event("stage_done", stage=STAGE, added=len(added))
    return payload
