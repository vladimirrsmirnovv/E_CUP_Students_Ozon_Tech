"""Порядок стадий, возобновление и итоговый отчёт."""
from __future__ import annotations

import dataclasses
import json
import signal
import time
from pathlib import Path

import psutil

from . import cache, checks, data, pool
from .config import assert_fold_contract, build_folds
from .logging_setup import emit_event, setup_logging
from .stages import (
    stage1_null_importance,
    stage3_ranked_forward,
    stage4_multiseed,
    stage5_pair_rescue,
    stage6_backward,
    stage7_optuna,
)

_STOP = {"requested": False}


def request_stop(*_arguments) -> None:
    """Помечает прогон к остановке на ближайшей границе задания."""
    _STOP["requested"] = True


def install_signal_handlers() -> None:
    """Ставит обработчики, чтобы прерывание не теряло посчитанное."""
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def make_stop_check(run_dir):
    """Проверка остановки: сигнал или файл STOP в каталоге прогона."""
    stop_file = Path(run_dir) / "STOP"

    def check() -> bool:
        if _STOP["requested"]:
            return True
        return stop_file.exists()

    return check


def build_context(config, logger) -> dict:
    """Общее состояние прогона, передаваемое стадиям."""
    folds = build_folds()
    assert_fold_contract(folds)
    return {"config": config, "folds": folds, "logger": logger,
            "cache_db": config.cache_db, "fingerprint": config.fingerprint()}


def open_pool(config, logger, wide: bool):
    """Поднимает пул под конкретный класс стадий."""
    available = psutil.virtual_memory().available / 1024 ** 3
    workers = config.parallel.resolve_workers(psutil.cpu_count() or 4, available, wide)
    threads = config.parallel.threads_for(wide)
    kind = "узкий"
    if wide:
        kind = "широкий"
    logger.info("пул %s: процессов %d по %d потоков, свободно памяти %.1f ГБ",
                kind, workers, threads, available)
    return pool.make_pool(workers, threads)


def _write_report(config, report: dict) -> Path:
    """Пишет итог прогона в JSON рядом с логом."""
    path = Path(config.run_dir) / f"selected_features_{config.combination}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return path


def run_stage_with_retry(config, logger, wide: bool, executor, action):
    """Выполняет стадию, переподнимая пул, если процесс-исполнитель погиб.

    Гибель воркера обесценивает пул целиком, но не посчитанное: каждая оценка
    уже лежит в кэше. Поэтому вторая попытка стоит дёшево и спасает прогон,
    который иначе оборвался бы на ровном месте.
    """
    try:
        return action(executor), executor
    except pool.WorkerLost as error:
        logger.warning("пул погиб (%s), поднимаю заново и повторяю стадию", error)
        executor.shutdown(wait=False, cancel_futures=True)
        fresh = open_pool(config, logger, wide)
        return action(fresh), fresh


def _stage_banner(logger, name: str, order: str) -> float:
    """Отмечает начало стадии в логе и возвращает момент старта."""
    logger.info("=" * 70)
    logger.info("СТАДИЯ %s: %s", order, name)
    return time.perf_counter()


def _stage_done(logger, name: str, started: float) -> None:
    """Отмечает конец стадии в логе."""
    logger.info("стадия %s заняла %.1fс", name, time.perf_counter() - started)


def run_selection(config) -> dict:
    """Проводит отбор целиком в порядке, заданном новыми требованиями."""
    Path(config.run_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(config.log_file, config.events_file)
    logger.info("сочетание %s, отпечаток %s", config.combination,
                config.fingerprint())
    cache.init_db(config.cache_db)
    install_signal_handlers()
    stop_check = make_stop_check(config.run_dir)
    context = build_context(config, logger)

    manifest = data.prepare_snapshots(
        config.snapshot_dir, config.prepared_dir,
        data.required_cutoffs(context["folds"]), users_limit=config.users_limit,
        feature_limit=config.feature_limit, logger=logger,
        drop_count_bins=config.drop_count_bin_categories,
        anchor_dir=config.anchor_dir or None)
    features = manifest["features"]

    stages_run = []
    # Стадия 1 учит на всех признаках сразу и держит одиннадцать гигабайт на
    # процесс; остальные стадии работают с наборами по десятку признаков и
    # укладываются в гигабайт. Один пул на весь прогон заставил бы их всех
    # жить по мерке самой прожорливой стадии.
    executor = open_pool(config, logger, wide=True)
    try:
        started = _stage_banner(logger, "временная null importance", "1")
        first, executor = run_stage_with_retry(
            config, logger, True, executor,
            lambda pool_: stage1_null_importance.run(context, features, pool_,
                                                     stop_check))
        stages_run.append(stage1_null_importance.STAGE)
        _stage_done(logger, "1", started)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    executor = open_pool(config, logger, wide=False)
    try:
        started = _stage_banner(logger, "ranked forward selection", "3")
        forward, executor = run_stage_with_retry(
            config, logger, False, executor,
            lambda pool_: stage3_ranked_forward.run(context, first["passed"],
                                                    pool_, stop_check))
        stages_run.append(stage3_ranked_forward.STAGE)
        _stage_done(logger, "3", started)

        started = _stage_banner(logger, "парное спасение", "5")
        rescue, executor = run_stage_with_retry(
            config, logger, False, executor,
            lambda pool_: stage5_pair_rescue.run(context, forward["selected"],
                                                 forward["baseline"],
                                                 forward["trials"], pool_,
                                                 stop_check))
        stages_run.append(stage5_pair_rescue.STAGE)
        _stage_done(logger, "5", started)

        started = _stage_banner(logger, "проверка набора на сидах", "4")
        multiseed, executor = run_stage_with_retry(
            config, logger, False, executor,
            lambda pool_: stage4_multiseed.run(context, rescue["selected"],
                                               [forward["seed_feature"]],
                                               forward["trials"], pool_,
                                               stop_check))
        stages_run.append(stage4_multiseed.STAGE)
        _stage_done(logger, "4", started)

        started = _stage_banner(logger, "обратная чистка", "6")
        pruned, executor = run_stage_with_retry(
            config, logger, False, executor,
            lambda pool_: stage6_backward.run(context, multiseed["features"],
                                              pool_, stop_check))
        stages_run.append(stage6_backward.STAGE)
        _stage_done(logger, "6", started)

        optuna_result = stage7_optuna.run(context, pruned["features"], executor,
                                          stop_check)
        if optuna_result.get("enabled"):
            stages_run.append(stage7_optuna.STAGE)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    report = {"combination": config.combination,
              "fingerprint": config.fingerprint(),
              "folds": [fold.as_dict() for fold in context["folds"]],
              "probe_learning_rate": config.probe.learning_rate,
              "feature_pool": len(features),
              "stage1_passed": len(first["passed"]),
              "stage1_reason": first["reason"],
              "forward_selected": forward["selected"],
              "seed_feature": forward["seed_feature"],
              "pair_rescue": rescue,
              "multiseed": multiseed,
              "backward_removed": pruned["removed"],
              "selected_features": pruned["features"],
              "final_metric": pruned["baseline"]["weighted"],
              "per_fold": pruned["baseline"]["per_fold"],
              "optuna": optuna_result,
              "config": config.decision_payload()}
    report["checks"] = checks.run_all(report, config, context["folds"], stages_run)
    path = _write_report(config, report)
    logger.info("итог: признаков %d, метрика %.6f, файл %s",
                len(report["selected_features"]), report["final_metric"], path)
    emit_event("run_done", features=len(report["selected_features"]),
               metric=report["final_metric"])
    return report
