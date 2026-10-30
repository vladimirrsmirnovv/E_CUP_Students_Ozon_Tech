"""Стадия 3: ranked forward selection с двухуровневым правилом приёма."""
from __future__ import annotations

import statistics

import numpy as np

from .. import cache, jobs, pool
from ..logging_setup import emit_event

STAGE = "stage3_ranked_forward"
CHECKPOINT = "progress"


def guards_pass(profile: dict, config) -> tuple:
    """Проверяет защиту по просадке фолда и сегментов."""
    if profile["worst_fold_degradation"] > config.max_allowed_fold_degradation:
        return False, "просадка фолда"
    if profile["worst_zero_degradation"] > config.max_zero_segment_degradation:
        return False, "просадка сегмента нулей"
    if profile["worst_positive_degradation"] > config.max_positive_segment_degradation:
        return False, "просадка сегмента покупавших"
    return True, ""


def decide(profile: dict, config) -> dict:
    """Решение о приёме кандидата по обоим путям правила."""
    standard = (profile["weighted_improvement"] >= config.min_improvement
                and profile["improved_folds"] >= config.min_improved_folds)
    stable = (profile["weighted_improvement"] >= config.stable_min_improvement
              and profile["improved_folds"] >= config.stable_min_improved_folds)
    # Порог проверяется раньше защиты: иначе кандидат, не дотянувший до порога,
    # попадал бы в отчёт как «нарушил защиту» и парное спасение его теряло.
    if not (standard or stable):
        return {"accepted": False, "path": "none", "reason": "выигрыш ниже порога"}
    allowed, guard_reason = guards_pass(profile, config)
    if not allowed:
        return {"accepted": False, "path": "none", "reason": guard_reason}
    if standard:
        return {"accepted": True, "path": "standard", "reason": "обычный путь"}
    return {"accepted": True, "path": "stable",
            "reason": "устойчивый малый выигрыш"}


def _evaluate_baseline(context: dict, selected: list, executor, stop_check) -> dict:
    """Метрика текущего набора на всех фолдах."""
    config = context["config"]
    job_list = jobs.build_jobs(context, selected, seed=config.random_state,
                               iterations=config.probe.iterations,
                               early_stopping=config.probe.early_stopping_rounds)
    results = pool.run_jobs_cached(executor, context["cache_db"], STAGE, job_list,
                                   context["logger"], stop_check)
    return jobs.aggregate(results, job_list, context["folds"])


def _candidate_units(context: dict, selected: list, candidates: list) -> list:
    """Единицы работы «набор плюс кандидат» в порядке ранга."""
    config = context["config"]
    units = []
    for name in candidates:
        units.append((name, jobs.build_jobs(
            context, selected + [name], seed=config.random_state,
            iterations=config.probe.iterations,
            early_stopping=config.probe.early_stopping_rounds)))
    return units


def _rerank(context: dict, selected: list, candidates: list, executor,
            stop_check) -> list:
    """Переупорядочивает остаток кандидатов по важности при текущем наборе."""
    if not candidates:
        return candidates
    config = context["config"]
    job_list = jobs.build_jobs(context, selected + candidates,
                               seed=config.random_state,
                               iterations=config.probe.iterations,
                               early_stopping=config.probe.early_stopping_rounds,
                               want_importance=True)
    results = pool.run_jobs_cached(executor, context["cache_db"], STAGE, job_list,
                                   context["logger"], stop_check)
    columns = sorted(selected + candidates)
    score = np.zeros(len(columns))
    for key, job in job_list:
        weight = job["fold"]["weight"]
        values = np.asarray(results[key]["importance"], dtype=np.float64)
        total = float(np.sum(np.abs(values)))
        if total > 0:
            score += weight * np.abs(values) / total
    ranking = {name: score[i] for i, name in enumerate(columns)}
    return sorted(candidates, key=lambda name: -ranking[name])


def _load_state(context: dict, candidates: list, baseline: dict, seed: str) -> dict:
    """Состояние стадии из кэша или начальное."""
    saved = cache.get_stage_state(context["cache_db"], STAGE, CHECKPOINT)
    if saved is not None and saved.get("fingerprint") == context["fingerprint"]:
        context["logger"].info("[%s] возобновляю: отобрано %d, кандидатов %d",
                               STAGE, len(saved["selected"]), len(saved["candidates"]))
        return saved
    return {"fingerprint": context["fingerprint"], "selected": [seed],
            "candidates": [name for name in candidates if name != seed],
            "baseline": baseline, "trials": [], "added": 0,
            "rejections_in_row": 0, "since_rerank": 0, "round_sizes": []}


def _save_state(context: dict, state: dict) -> None:
    """Сохраняет состояние стадии после каждого решения."""
    cache.put_stage_state(context["cache_db"], STAGE, CHECKPOINT, state)


def _speculation_width(executor, context: dict, config, history: list) -> int:
    """Потолок опережения и стартовая ширина раунда.

    Опережать имеет смысл ровно настолько, сколько кандидатов обычно успевает
    получить отказ до очередного приёма. Принятый кандидат меняет набор, и всё
    посчитанное дальше не просто выбрасывается — оно ещё и отнимает полосу
    памяти у той единственной оценки, которая раунд и завершает.
    """
    workers = getattr(executor, "_max_workers", 1) or 1
    by_pool = max(1, workers // max(1, len(context["folds"])))
    ceiling = min(by_pool, config.speculation_width)
    if not history:
        return ceiling, 1
    recent = history[-5:]
    return ceiling, max(1, min(ceiling, int(statistics.median(recent))))


def _log_decision(logger, name: str, profile: dict, verdict: dict, size: int) -> None:
    """Пишет в лог одно решение о кандидате."""
    logger.info(
        "[%s] %-42s выигрыш %+.6f фолдов %d/3 путь %-8s %s -> набор %d",
        STAGE, name, profile["weighted_improvement"], profile["improved_folds"],
        verdict["path"], verdict["reason"], size)


def run(context: dict, ranking: list, executor, stop_check) -> dict:
    """Наращивает набор признаков по рангу до исчерпания кандидатов."""
    logger = context["logger"]
    config = context["config"].stage3
    cached = cache.get_stage_result(context["cache_db"], STAGE, context["fingerprint"])
    if cached is not None:
        logger.info("[%s] итог стадии уже посчитан: %d признаков",
                    STAGE, len(cached["selected"]))
        return cached

    candidates = list(ranking)
    if config.top_k_candidates > 0:
        candidates = ranking[:config.top_k_candidates]
    if not candidates:
        raise ValueError("стадии 3 нечего отбирать: пустой список кандидатов")
    seed_feature = candidates[0]
    baseline = _evaluate_baseline(context, [seed_feature], executor, stop_check)
    logger.info("[%s] затравка %s, взвешенная метрика %.6f",
                STAGE, seed_feature, baseline["weighted"])
    state = _load_state(context, candidates, baseline, seed_feature)

    while state["candidates"]:
        if state["added"] >= config.max_features_to_add:
            logger.info("[%s] достигнут предел добавлений %d",
                        STAGE, config.max_features_to_add)
            break
        if state["rejections_in_row"] >= config.patience_no_improvement:
            logger.info("[%s] терпение исчерпано", STAGE)
            break
        accepted_name = None
        consumed = []
        units = _candidate_units(context, state["selected"], state["candidates"])
        ceiling, start = _speculation_width(executor, context, config,
                                            state.setdefault("round_sizes", []))
        stream = pool.iterate_in_order(executor, context["cache_db"], STAGE, units,
                                       logger, stop_check, ceiling, start)
        for name, unit_jobs, results in stream:
            metrics = jobs.aggregate(results, unit_jobs, context["folds"])
            profile = jobs.improvement_profile(state["baseline"], metrics,
                                               context["folds"])
            verdict = decide(profile, config)
            consumed.append(name)
            state["trials"].append({"feature": name, **profile, **verdict})
            _log_decision(logger, name, profile, verdict,
                          len(state["selected"]) + int(verdict["accepted"]))
            emit_event("candidate", stage=STAGE, feature=name, **profile, **verdict)
            if verdict["accepted"]:
                state["selected"].append(name)
                state["baseline"] = metrics
                state["added"] += 1
                state["since_rerank"] += 1
                state["rejections_in_row"] = 0
                accepted_name = name
                break
            state["rejections_in_row"] += 1
        stream.close()
        state["round_sizes"].append(len(consumed))
        # Проход по рангу односторонний: отвергнутый кандидат выбывает и к нему
        # не возвращаются, поэтому счёт растёт линейно, а не квадратично.
        state["candidates"] = [name for name in state["candidates"]
                               if name not in set(consumed)]
        if accepted_name is None:
            _save_state(context, state)
            break
        if state["since_rerank"] >= config.rerank_after_n_added:
            state["candidates"] = _rerank(context, state["selected"],
                                          state["candidates"], executor, stop_check)
            state["since_rerank"] = 0
            logger.info("[%s] остаток кандидатов переупорядочен", STAGE)
        _save_state(context, state)

    payload = {"selected": state["selected"], "baseline": state["baseline"],
               "trials": state["trials"], "seed_feature": seed_feature}
    cache.put_stage_result(context["cache_db"], STAGE, context["fingerprint"], payload)
    logger.info("[%s] отобрано %d признаков, метрика %.6f",
                STAGE, len(payload["selected"]), payload["baseline"]["weighted"])
    emit_event("stage_done", stage=STAGE, selected=len(payload["selected"]))
    return payload
