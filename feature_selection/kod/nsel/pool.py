"""Пул процессов: непрерывная очередь заданий и упорядоченная спекуляция."""
from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    BrokenExecutor,
    ProcessPoolExecutor,
    wait,
)

import psutil

from . import cache
from .worker import TrainingFailed, run_training

POLL_INTERVAL = 0.5
WATCHDOG_INTERVAL = 1.0


class StopRequested(Exception):
    """Пришёл сигнал остановки посреди стадии."""


class WorkerLost(RuntimeError):
    """Процесс-исполнитель убит извне, чаще всего нехваткой памяти."""


def _check_payload(payload: dict, stage: str) -> dict:
    """Поднимает ошибку воркера, пришедшую текстом."""
    if "error" in payload:
        raise TrainingFailed(f"[{stage}] {payload['error']}\n{payload['traceback']}")
    return payload


def _result_or_explain(future, stage: str):
    """Достаёт результат задания, объясняя гибель процесса понятным текстом."""
    try:
        return _check_payload(future.result(), stage)
    except BrokenExecutor as error:
        raise WorkerLost(
            f"[{stage}] процесс-исполнитель убит: {error}. Обычная причина — "
            "нехватка памяти: уменьшите число процессов или поднимите "
            "--memory-per-worker") from error


def _worker_init(parent_pid: int, parent_created: float, threads: int) -> None:
    """Задаёт воркеру число потоков и не даёт ему пережить родителя.

    XGBoost и LightGBM параллелятся через OpenMP, и `OMP_NUM_THREADS` перебивает
    их собственные параметры: замер показал, что при значении 1 аргумент
    nthread=10 не даёт ничего, а без ограничения то же обучение идёт в 6.4 раза
    быстрее. Значение ставится здесь, до первого импорта библиотеки в этом
    процессе, и равно числу потоков, выделенных одному обучению.
    """
    os.environ["OMP_NUM_THREADS"] = str(max(1, threads))
    os.environ["MKL_NUM_THREADS"] = str(max(1, threads))
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 9, 0, 0, 0)
        except Exception:
            pass

    def watch() -> None:
        while True:
            time.sleep(WATCHDOG_INTERVAL)
            try:
                parent = psutil.Process(parent_pid)
                if parent.create_time() != parent_created or not parent.is_running():
                    os._exit(1)
            except psutil.NoSuchProcess:
                os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def make_pool(n_workers: int, threads: int = 1) -> ProcessPoolExecutor:
    """Поднимает пул процессов способом spawn."""
    parent = psutil.Process(os.getpid())
    return ProcessPoolExecutor(
        max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        initargs=(parent.pid, parent.create_time(), threads))


def _child_memory(child) -> int:
    """Память дочернего процесса; исчезнувший и зомби считаются нулём."""
    try:
        return child.memory_info().rss
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return 0


def memory_summary() -> str:
    """Короткая сводка по памяти процесса и его детей."""
    process = psutil.Process()
    own = process.memory_info().rss
    children = sum(_child_memory(child) for child in process.children(recursive=True))
    return (f"память={own / 1e6:.0f}МБ дети={children / 1e6:.0f}МБ "
            f"занято={psutil.virtual_memory().percent:.0f}%")


def _split_cached(db_path, jobs: list) -> tuple:
    """Делит задания на посчитанные ранее и требующие счёта."""
    known = cache.get_iterations(db_path, [key for key, _ in jobs])
    pending = [(key, job) for key, job in jobs if key not in known]
    return known, pending


def run_jobs_cached(pool, db_path, stage: str, jobs: list, logger: logging.Logger,
                    stop_check) -> dict:
    """Считает все задания стадии, сохраняя каждый результат сразу."""
    results, pending = _split_cached(db_path, jobs)
    logger.info("[%s] заданий %d, из кэша %d, к счёту %d",
                stage, len(jobs), len(results), len(pending))
    if not pending:
        return results

    futures = {pool.submit(run_training, job): key for key, job in pending}
    waiting = set(futures)
    started = time.time()
    done_count = 0
    try:
        while waiting:
            if stop_check():
                raise StopRequested(stage)
            finished, waiting = wait(waiting, timeout=POLL_INTERVAL,
                                     return_when=FIRST_COMPLETED)
            for future in finished:
                key = futures[future]
                payload = _result_or_explain(future, stage)
                cache.put_iteration(db_path, key, stage, payload)
                results[key] = payload
                done_count += 1
                elapsed = time.time() - started
                remaining = elapsed / done_count * (len(pending) - done_count)
                logger.info("[%s] %d/%d прошло %.0fс осталось ~%.0fс пик %.0fМБ %s",
                            stage, done_count, len(pending), elapsed, remaining,
                            payload.get("peak_rss_mb", 0.0), memory_summary())
    finally:
        for future in waiting:
            future.cancel()
    return results


def iterate_in_order(pool, db_path, stage: str, units: list, logger: logging.Logger,
                     stop_check, width: int, start_width: int = 1):
    """Выдаёт результаты единиц работы строго по рангу, считая их с опережением.

    Единица — список заданий одного кандидата. Опережающий счёт не меняет
    решений: отвергнутый кандидат не меняет состояние отбора, а на принятом
    очередь обрывается ровно там же, где оборвался бы последовательный проход.

    Ширина начинается с предсказанной по прошлым раундам и удваивается внутри
    раунда на каждом отвергнутом кандидате, до потолка. Так покрыты оба случая:
    короткий раунд, который кончится приёмом на первом же кандидате, ничего не
    выбрасывает, а длинная полоса отказов быстро занимает весь пул.
    """
    in_flight = {}
    submitted = 0
    position = 0
    current = max(1, min(width, start_width))
    try:
        while position < len(units):
            while submitted < len(units) and len(in_flight) < current:
                name, jobs = units[submitted]
                known, pending = _split_cached(db_path, jobs)
                futures = {pool.submit(run_training, job): key for key, job in pending}
                in_flight[submitted] = (name, jobs, known, futures)
                submitted += 1
            name, jobs, known, futures = in_flight.pop(position)
            for future in list(futures):
                if stop_check():
                    raise StopRequested(stage)
                key = futures[future]
                payload = _result_or_explain(future, stage)
                cache.put_iteration(db_path, key, stage, payload)
                known[key] = payload
            logger.debug("[%s] готова единица %s (в полёте %d)",
                         stage, name, len(in_flight))
            yield name, jobs, known
            # Управление вернулось — кандидат отвергнут, состояние не менялось,
            # значит следующие можно считать смелее.
            position += 1
            current = min(width, current * 2)
    finally:
        for _name, _jobs, _known, futures in in_flight.values():
            for future in futures:
                future.cancel()
