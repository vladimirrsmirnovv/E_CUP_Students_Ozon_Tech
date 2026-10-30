"""Запуск отбора: боевой прогон или быстрый прогон на подвыборке."""
from __future__ import annotations

import argparse
import sys
import dataclasses
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nsel import config as config_module
from nsel.orchestrator import run_selection

DEFAULT_SNAPSHOTS = "/Volumes/ECupOzon/backups/server_cpu_final_20260828/ozon_snapshot_cache_v8"


def parse_arguments():
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="отбор признаков, схема трёх фолдов")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--model", default="catboost", choices=config_module.MODEL_MODES)
    parser.add_argument("--target", default="log_level",
                        choices=config_module.TARGET_MODES)
    parser.add_argument("--anchor-dir", default="",
                        help="каталог с якорем segment_adaptive")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--users", type=int, default=0)
    parser.add_argument("--features", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pair-rescue", action="store_true")
    parser.add_argument("--confirm-doubtful", action="store_true")
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--null-runs", type=int, default=0)
    parser.add_argument("--actual-runs", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=0)
    parser.add_argument("--probe-iterations", type=int, default=0)
    parser.add_argument("--probe-lr", type=float, default=0.0)
    parser.add_argument("--speculation", type=int, default=0)
    parser.add_argument("--memory-per-worker", type=float, default=0.0)
    parser.add_argument("--prepared-dir", default="")
    return parser.parse_args()


def apply_scale_overrides(overrides: dict, base, arguments) -> dict:
    """Точечно урезает объём счёта, не трогая правила отбора."""
    stage1 = dict(actual_runs=arguments.actual_runs or base.stage1.actual_runs,
                  null_runs=arguments.null_runs or base.stage1.null_runs)
    overrides["stage1"] = dataclasses.replace(base.stage1, **stage1)
    stage3 = dict(top_k_candidates=arguments.top_k or base.stage3.top_k_candidates,
                  max_features_to_add=(arguments.max_features
                                       or base.stage3.max_features_to_add),
                  speculation_width=(arguments.speculation
                                     or base.stage3.speculation_width))
    overrides["stage3"] = dataclasses.replace(base.stage3, **stage3)
    probe = dict(iterations=arguments.probe_iterations or base.probe.iterations,
                 learning_rate=arguments.probe_lr or base.probe.learning_rate)
    overrides["probe"] = dataclasses.replace(base.probe, **probe)
    return overrides


def build_config(arguments):
    """Собирает конфиг прогона по аргументам."""
    parallel = {"n_workers": arguments.workers,
                "threads_per_worker": arguments.threads}
    if arguments.memory_per_worker:
        parallel["memory_per_worker_gb"] = arguments.memory_per_worker
    overrides = {"model_mode": arguments.model, "target_mode": arguments.target,
                 "parallel": config_module.ParallelConfig(**parallel)}
    if arguments.pair_rescue:
        overrides["stage5"] = config_module.Stage5Config(enabled=True)
    if arguments.confirm_doubtful:
        overrides["stage4"] = config_module.Stage4Config(confirm_doubtful=True)
    if arguments.optuna:
        overrides["optuna"] = config_module.OptunaConfig(enabled=True)
    if arguments.smoke:
        base = config_module.smoke_config(arguments.run_dir, arguments.snapshot_dir)
    else:
        base = config_module.RunConfig(run_dir=Path(arguments.run_dir),
                                       snapshot_dir=Path(arguments.snapshot_dir))
    if arguments.anchor_dir:
        overrides["anchor_dir"] = arguments.anchor_dir
    if arguments.prepared_dir:
        overrides["shared_prepared_dir"] = arguments.prepared_dir
    if arguments.users:
        overrides["users_limit"] = arguments.users
    if arguments.features:
        overrides["feature_limit"] = arguments.features
    overrides = apply_scale_overrides(overrides, base, arguments)
    return dataclasses.replace(base, **overrides)


def main() -> int:
    """Точка входа."""
    arguments = parse_arguments()
    report = run_selection(build_config(arguments))
    print(f"признаков {len(report['selected_features'])}, "
          f"метрика {report['final_metric']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
