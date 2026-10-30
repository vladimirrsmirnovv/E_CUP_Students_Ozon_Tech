"""Стадия 4: проверка НАБОРА на нескольких сидах и разбор сомнительных."""
from __future__ import annotations

import statistics

from .. import cache, jobs, pool
from ..logging_setup import emit_event

STAGE = "stage4_multiseed"


def _metrics_for_seed(context: dict, features: list, seed: int, executor,
                      stop_check) -> dict:
    """Взвешенная метрика набора на одном сиде."""
    config = context["config"]
    job_list = jobs.build_jobs(context, features, seed=seed,
                               iterations=config.probe.iterations,
                               early_stopping=config.probe.early_stopping_rounds)
    results = pool.run_jobs_cached(executor, context["cache_db"], STAGE, job_list,
                                   context["logger"], stop_check)
    return jobs.aggregate(results, job_list, context["folds"])


def _stability(values: dict) -> dict:
    """Разброс взвешенной метрики по сидам."""
    numbers = [item["weighted"] for item in values.values()]
    spread = 0.0
    if len(numbers) > 1:
        spread = statistics.stdev(numbers)
    best = min(values, key=lambda seed: values[seed]["weighted"])
    worst = max(values, key=lambda seed: values[seed]["weighted"])
    return {"mean": statistics.fmean(numbers), "median": statistics.median(numbers),
            "stdev": spread, "spread": max(numbers) - min(numbers),
            "best_seed": best, "worst_seed": worst,
            "best": values[best]["weighted"], "worst": values[worst]["weighted"]}


def is_doubtful(trial: dict, config, forward_config) -> bool:
    """Признак сомнителен, если прошёл впритык к порогу приёма."""
    if trial.get("path") == "stable":
        return config.confirm_stable_tiny_gains
    threshold = config.doubtful_improvement_factor * forward_config.min_improvement
    return trial.get("weighted_improvement", 0.0) < threshold


def _confirm_feature(context: dict, features: list, feature: str, executor,
                     stop_check) -> dict:
    """Подтверждает пользу признака: знак и устойчивость по сидам.

    Метрики сегментов на дополнительных сидах только записываются: права
    отклонить признак у них нет, если временные свидетельства его держат.
    """
    config = context["config"].stage4
    without = [name for name in features if name != feature]
    improvements = []
    segments = []
    for seed in config.seeds:
        full = _metrics_for_seed(context, features, seed, executor, stop_check)
        reduced = _metrics_for_seed(context, without, seed, executor, stop_check)
        profile = jobs.improvement_profile(reduced, full, context["folds"])
        improvements.append(profile["weighted_improvement"])
        segments.append({"seed": seed, "zero": profile["worst_zero_degradation"],
                         "positive": profile["worst_positive_degradation"]})
    positive_seeds = sum(1 for value in improvements if value > 0)
    median_improvement = statistics.median(improvements)
    keep = (positive_seeds >= config.confirmation_min_positive_seeds
            and median_improvement > config.confirmation_min_median_improvement)
    return {"feature": feature, "median_improvement": median_improvement,
            "mean_improvement": statistics.fmean(improvements),
            "improvements": improvements, "positive_seeds": positive_seeds,
            "segments": segments, "segment_mode": config.segment_mode,
            "keep": keep}


def run(context: dict, features: list, baseline_features: list, trials: list,
        executor, stop_check) -> dict:
    """Проверяет устойчивость набора к случайности модели."""
    logger = context["logger"]
    config = context["config"].stage4
    cached = cache.get_stage_result(context["cache_db"], STAGE, context["fingerprint"])
    if cached is not None:
        logger.info("[%s] итог стадии уже посчитан", STAGE)
        return cached
    if not config.enabled:
        return {"features": list(features), "per_seed": {}, "stability": {},
                "confirmations": [], "dropped": []}

    per_seed = {}
    per_seed_baseline = {}
    for seed in config.seeds:
        per_seed[seed] = _metrics_for_seed(context, features, seed, executor,
                                           stop_check)
        per_seed_baseline[seed] = _metrics_for_seed(context, baseline_features, seed,
                                                    executor, stop_check)
        logger.info("[%s] сид %d: метрика набора %.6f, затравки %.6f",
                    STAGE, seed, per_seed[seed]["weighted"],
                    per_seed_baseline[seed]["weighted"])

    stability = _stability(per_seed)
    improvement = statistics.fmean(
        [per_seed_baseline[seed]["weighted"] - per_seed[seed]["weighted"]
         for seed in config.seeds])
    logger.info("[%s] среднее %.6f медиана %.6f разброс %.6f выигрыш набора %+.6f",
                STAGE, stability["mean"], stability["median"], stability["spread"],
                improvement)

    confirmations = []
    dropped = []
    kept = list(features)
    if config.confirm_doubtful:
        forward = context["config"].stage3
        accepted = set(features)
        doubtful = [trial["feature"] for trial in trials
                    if trial.get("accepted") and trial["feature"] in accepted
                    and is_doubtful(trial, config, forward)]
        logger.info("[%s] сомнительных признаков %d", STAGE, len(doubtful))
        for feature in doubtful:
            verdict = _confirm_feature(context, kept, feature, executor, stop_check)
            confirmations.append(verdict)
            logger.info("[%s] %-42s медиана %+.6f, сидов в плюс %d/%d -> %s",
                        STAGE, feature, verdict["median_improvement"],
                        verdict["positive_seeds"], len(config.seeds),
                        verdict["keep"])
            if not verdict["keep"]:
                kept = [name for name in kept if name != feature]
                dropped.append(feature)

    payload = {"features": kept, "dropped": dropped,
               "per_seed": {str(seed): per_seed[seed] for seed in per_seed},
               "stability": {**stability, "set_improvement": improvement},
               "confirmations": confirmations}
    cache.put_stage_result(context["cache_db"], STAGE, context["fingerprint"], payload)
    emit_event("stage_done", stage=STAGE, features=len(kept), dropped=len(dropped))
    return payload
