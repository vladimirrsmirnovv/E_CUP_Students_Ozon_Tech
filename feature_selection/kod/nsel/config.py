"""Конфигурация отбора: контракт фолдов, пороги стадий, параметры Оптуны."""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path

HORIZON_DAYS = 30

# Новое требование: три фолда вместо пяти, февраль и октябрь сняты.
FOLD_CONTRACT = (
    ("nov_2025", ("2025-09-16", "2025-10-16"), "2025-11-15", 0.25),
    ("dec_2025", ("2025-09-16", "2025-10-16", "2025-11-15"), "2025-12-15", 0.35),
    ("jan_2026", ("2025-09-16", "2025-10-16", "2025-11-15", "2025-12-15"),
     "2026-01-14", 0.40),
)
EXPECTED_FOLD_WEIGHTS = {"nov_2025": 0.25, "dec_2025": 0.35, "jan_2026": 0.40}

MODEL_MODES = ("catboost", "lightgbm", "xgboost")
TARGET_MODES = ("log_level", "log_diff", "segment_adaptive",
                "volatility_adaptive", "personal_hard", "segment_logspace",
                "log_ratio_ewma")
# У log_ratio_ewma якорь обычный, он уже лежит в срезах отдельным столбцом,
# считать его не нужно.
SNAPSHOT_ANCHORS = {"log_ratio_ewma": "gmv_decay30_hl30"}
# Адаптивные якоря не лежат в срезах, они считаются из сырой истории функцией
# коллеги и кладутся рядом файлом, см. tools/build_adaptive_anchor.py
ADAPTIVE_ANCHORS = {"segment_adaptive": "anchor_segment_adaptive",
                    "volatility_adaptive": "anchor_volatility_adaptive",
                    "personal_hard": "anchor_personal_hard",
                    "segment_logspace": "anchor_segment_logspace_log"}
# У segment_logspace якорь уже в логарифме, поэтому таргет считается как
# log1p(y) - anchor, без второго логарифма. Обычная формула дала бы молча
# другой таргет и неверную метрику.
LOG_ANCHOR_MODES = ("segment_logspace",)

TARGET_ANCHOR_COLUMN = "gmv_30d"
TARGET_COLUMN = "target_30d"
SNAPSHOT_INDEX_COLUMN = "__snapshot_index_user_id__"


class ContractError(AssertionError):
    """Нарушение контракта валидации."""


@dataclasses.dataclass(frozen=True)
class FoldSpec:
    """Один временной фолд единой кросс-валидации."""

    name: str
    train_cutoffs: tuple
    valid_cutoff: str
    weight: float

    def key_parts(self) -> tuple:
        """Части ключа кэша, описывающие фолд."""
        return (self.name, tuple(self.train_cutoffs), self.valid_cutoff,
                round(self.weight, 12))

    def as_dict(self) -> dict:
        """Представление фолда, пригодное для передачи в процесс-воркер."""
        return {"name": self.name, "train_cutoffs": list(self.train_cutoffs),
                "valid_cutoff": self.valid_cutoff, "weight": self.weight}


def build_folds(contract=FOLD_CONTRACT) -> list:
    """Строит список фолдов по контракту."""
    return [FoldSpec(name, tuple(cutoffs), valid, float(weight))
            for name, cutoffs, valid, weight in contract]


def assert_fold_contract(folds: list) -> None:
    """Проверяет состав, веса и отсутствие заглядывания в будущее."""
    if len(folds) != 3:
        raise ContractError(f"фолдов {len(folds)}, по новой схеме должно быть три")
    names = [fold.name for fold in folds]
    if names != list(EXPECTED_FOLD_WEIGHTS):
        raise ContractError(f"состав фолдов {names}")
    total = sum(fold.weight for fold in folds)
    if abs(total - 1.0) > 1e-12:
        raise ContractError(f"веса дают {total}, а должны единицу")
    for fold in folds:
        if abs(fold.weight - EXPECTED_FOLD_WEIGHTS[fold.name]) > 1e-12:
            raise ContractError(f"{fold.name}: вес {fold.weight}")
        valid = datetime.date.fromisoformat(fold.valid_cutoff)
        if valid.year == 2025 and valid.month in (2, 10):
            raise ContractError(f"{fold.name}: февраль и октябрь сняты с проверки")
        for cutoff in fold.train_cutoffs:
            gap = (valid - datetime.date.fromisoformat(cutoff)).days
            if gap < HORIZON_DAYS:
                raise ContractError(
                    f"{fold.name}: между {cutoff} и {fold.valid_cutoff} {gap} дней")


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    """Модель-щуп: сравнивает наборы между собой, а не строит финальную модель."""

    # 0.15 против боевых 0.03 по замеру research/REPORT.md: ранговая
    # корреляция 0.9998, десять лучших наборов совпали, счёт быстрее в 2.5 раза.
    learning_rate: float = 0.15
    depth: int = 5
    l2_leaf_reg: float = 3.0
    # Потолок деревьев и ранняя остановка — его числа: auto_extend выключен,
    # ceiling 8000, остановка через 30 итераций без улучшения вместо 50.
    iterations: int = 8000
    early_stopping_rounds: int = 30
    subsample: float = 0.8
    colsample: float = 0.8
    min_data_in_leaf: int = 20


@dataclasses.dataclass(frozen=True)
class Stage1Config:
    """Временная null importance."""

    enabled: bool = True
    actual_runs: int = 3
    null_runs: int = 15
    quantile: float = 0.90
    min_keep: int = 250
    max_keep: int = 450
    fallback_top_fraction: float = 0.15
    # У null-модели перемешан и проверочный таргет тоже, поэтому ранняя
    # остановка ничего не знает о настоящей цели и утечки не создаёт. Так же
    # устроено у коллеги, и именно поэтому его null-прогоны дёшевы: без сигнала
    # обучение обрывается через несколько десятков деревьев.
    null_iterations: int = 8000
    null_early_stopping: int = 30
    actual_iterations: int = 8000
    actual_early_stopping: int = 30


@dataclasses.dataclass(frozen=True)
class Stage3Config:
    """Ranked forward selection с двухуровневым правилом приёма."""

    # Его комментарий над аргументом: проверить надо все признаки, пережившие
    # null importance, дополнительным top-K фильтром их не резать. Ноль здесь
    # и означает «без ограничения».
    top_k_candidates: int = 0
    rerank_after_n_added: int = 10
    min_improvement: float = 0.00005
    min_improved_folds: int = 2
    stable_min_improvement: float = 0.00001
    stable_min_improved_folds: int = 3
    # Защиты ослаблены его правкой: не душить признаки, полезные в одном
    # режиме, guard остаётся только против аномального провала.
    max_allowed_fold_degradation: float = 0.005
    max_zero_segment_degradation: float = 0.02
    max_positive_segment_degradation: float = 0.02
    max_features_to_add: int = 10 ** 9
    patience_no_improvement: int = 10 ** 9
    # Потолок опережения. Фактическая ширина растёт от единицы вдвое на каждом
    # отказе: пока приёмы часты, считать далеко вперёд — выбрасывать работу.
    speculation_width: int = 16


@dataclasses.dataclass(frozen=True)
class Stage4Config:
    """Multiseed: сиды проверяют набор целиком, а не каждый признак."""

    enabled: bool = True
    seeds: tuple = (42, 52, 62)
    confirm_doubtful: bool = False
    doubtful_improvement_factor: float = 2.0
    # Признак, прошедший путём устойчивого малого выигрыша, на точечную
    # проверку не отправляется: он и так входит в проверку набора целиком.
    confirm_stable_tiny_gains: bool = False
    # Достаточно, чтобы польза подтвердилась на двух сидах из трёх и медиана
    # была положительной. Никакого дополнительного порога: задача проверки —
    # знак и устойчивость, а не повторное применение порога приёма.
    confirmation_min_positive_seeds: int = 2
    confirmation_min_median_improvement: float = 0.0
    # Метрики сегментов на дополнительных сидах только логируются и права
    # отклонить признак не имеют.
    segment_mode: str = "diagnostic"


@dataclasses.dataclass(frozen=True)
class Stage5Config:
    """Парное спасение почти прошедших кандидатов."""

    enabled: bool = True
    top_rejected: int = 6
    max_pairs: int = 15
    max_additions: int = 2
    # Порог пары полтора порога одиночного признака, а не четыре: спасение ищет
    # умеренные эффекты взаимодействия, требовать от них вчетверо больше нечем.
    min_improvement: float = 0.000075
    min_improved_folds: int = 2
    max_allowed_fold_degradation: float = 0.005
    near_miss_fraction: float = 0.2
    prefer_cross_family: bool = True


@dataclasses.dataclass(frozen=True)
class BackwardConfig:
    """Обратная чистка: ровно один раз, в конце."""

    enabled: bool = True
    neutral_tolerance: float = 0.000005
    speculation_width: int = 16


@dataclasses.dataclass(frozen=True)
class EncodingConfig:
    """Сглаживание ordered target encoding для XGBoost и LightGBM."""

    mode: str = "empirical_bayes"
    fixed_smoothing: float = 20.0
    min_count: int = 1
    first_block_value: str = "nan"
    unknown_value: str = "global"
    n_blocks: int = 20
    minimum_variance: float = 1e-9
    maximum_smoothing: float = 10000.0


@dataclasses.dataclass(frozen=True)
class OptunaConfig:
    """Параметры адаптивного поиска из второго промпта."""

    enabled: bool = False
    adaptive_search: bool = True
    exploration_trials: int = 210
    focused_trials: int = 500
    tpe_startup_trials_explore: int = 20
    tpe_startup_trials_focus: int = 15
    tpe_multivariate: bool = True
    tpe_group: bool = True
    # Раздел 3 промпта пересматривает эти числа: приоритет — не потерять
    # возможный оптимум, поэтому границы берутся по минимуму и максимуму
    # лучших двадцати точек, а не по децилям.
    adaptive_top_k: int = 20
    adaptive_range_expand: float = 0.25
    adaptive_min_range_fraction: float = 0.30
    adaptive_boundary_guard_fraction: float = 0.10
    adaptive_quantile_low: float = 0.00
    adaptive_quantile_high: float = 1.00
    adaptive_keep_categorical_neighbors: int = 1
    adaptive_narrow_binary_categoricals: bool = False
    adaptive_seed_focus_with_best: bool = True
    early_stopping_rounds: int = 50
    gpu: bool = False
    devices: str = "0"
    fallback_to_cpu: bool = False
    strict_ozon_fold_contract: bool = True
    random_state: int = 42


@dataclasses.dataclass(frozen=True)
class ParallelConfig:
    """Занятость машины: сколько процессов и сколько потоков каждому."""

    n_workers: int = 0
    threads_per_worker: int = 2
    memory_per_worker_gb: float = 1.5
    # Стадия 1 учит на всём наборе признаков: замер на сервере дал пик 11 ГБ на
    # процесс против гигабайта на остальных стадиях. Один пул на весь прогон
    # означал бы, что самая прожорливая стадия задаёт размер всем остальным.
    # Замер на пустой машине: одно обучение стадии 1 стоит 0.84 секунды на
    # дерево, а под десятью параллельными процессами — шестнадцать. Упирается
    # не в ядра, а в полосу памяти: каждый процесс прокачивает матрицу на
    # четыре гигабайта. Поэтому здесь мало процессов и много потоков каждому.
    wide_threads_per_worker: int = 10
    wide_memory_per_worker_gb: float = 14.0
    reserve_gb: float = 8.0

    def threads_for(self, wide: bool) -> int:
        """Сколько потоков даётся одному обучению."""
        if wide:
            return self.wide_threads_per_worker
        return self.threads_per_worker

    def memory_for(self, wide: bool) -> float:
        """Сколько памяти закладывается на один процесс."""
        if wide:
            return self.wide_memory_per_worker_gb
        return self.memory_per_worker_gb

    def resolve_workers(self, cpu_count: int, available_gb: float,
                        wide: bool = False) -> int:
        """Число процессов пула по ядрам и свободной памяти."""
        if self.n_workers > 0:
            return self.n_workers
        by_cores = max(1, (cpu_count - 2) // max(1, self.threads_for(wide)))
        by_memory = int((available_gb - self.reserve_gb) / self.memory_for(wide))
        return max(1, min(by_cores, max(1, by_memory)))


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Полное описание одного прогона отбора."""

    run_dir: Path
    snapshot_dir: Path
    model_mode: str = "catboost"
    target_mode: str = "log_level"
    random_state: int = 42
    users_limit: int = 0
    feature_limit: int = 0
    # Разложенные срезы зависят только от каталога данных и объёма выборки,
    # поэтому шесть сочетаний делят один каталог: иначе он копируется шесть раз
    # по четыре с половиной гигабайта и раскладывается шесть раз подряд.
    shared_prepared_dir: str = ""
    # Категории из биннинга счётчиков он просил отключить целиком.
    drop_count_bin_categories: bool = True
    # Каталог с посчитанным якорём segment_adaptive; для остальных таргетов пуст.
    anchor_dir: str = ""
    probe: ProbeConfig = dataclasses.field(default_factory=ProbeConfig)
    stage1: Stage1Config = dataclasses.field(default_factory=Stage1Config)
    stage3: Stage3Config = dataclasses.field(default_factory=Stage3Config)
    stage4: Stage4Config = dataclasses.field(default_factory=Stage4Config)
    stage5: Stage5Config = dataclasses.field(default_factory=Stage5Config)
    backward: BackwardConfig = dataclasses.field(default_factory=BackwardConfig)
    encoding: EncodingConfig = dataclasses.field(default_factory=EncodingConfig)
    optuna: OptunaConfig = dataclasses.field(default_factory=OptunaConfig)
    parallel: ParallelConfig = dataclasses.field(default_factory=ParallelConfig)

    def __post_init__(self) -> None:
        """Отклоняет неизвестную модель или таргет до начала счёта."""
        if self.model_mode not in MODEL_MODES:
            raise ValueError(f"model_mode={self.model_mode}")
        if self.target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode={self.target_mode}")
        if self.target_mode in ADAPTIVE_ANCHORS and not self.anchor_dir:
            raise ValueError(f"{self.target_mode} требует anchor_dir с якорем")

    @property
    def combination(self) -> str:
        """Имя сочетания модели и таргета."""
        return f"{self.model_mode}_{self.target_mode}"

    @property
    def cache_db(self) -> Path:
        """Путь до базы кэша итераций и стадий."""
        return Path(self.run_dir) / "cache.sqlite3"

    @property
    def log_file(self) -> Path:
        """Путь до подробного текстового лога."""
        return Path(self.run_dir) / "run.log"

    @property
    def events_file(self) -> Path:
        """Путь до машиночитаемого журнала событий."""
        return Path(self.run_dir) / "events.jsonl"

    @property
    def prepared_dir(self) -> Path:
        """Каталог разложенных срезов, общий для всех сочетаний."""
        if self.shared_prepared_dir:
            return Path(self.shared_prepared_dir)
        return Path(self.run_dir) / "prepared"

    def decision_payload(self) -> dict:
        """Всё, что влияет на решения отбора."""
        payload = {
            "folds": [fold.key_parts() for fold in build_folds()],
            "model_mode": self.model_mode,
            "target_mode": self.target_mode,
            "random_state": self.random_state,
            "users_limit": self.users_limit,
            "feature_limit": self.feature_limit,
            "probe": dataclasses.asdict(self.probe),
            "stage1": dataclasses.asdict(self.stage1),
            "stage3": dataclasses.asdict(self.stage3),
            "stage4": dataclasses.asdict(self.stage4),
            "stage5": dataclasses.asdict(self.stage5),
            "backward": dataclasses.asdict(self.backward),
            "encoding": dataclasses.asdict(self.encoding),
        }
        # Ширина спекуляции меняет только порядок счёта, поэтому в отпечаток не
        # входит: прогон должен продолжаться на машине с другим числом ядер.
        payload["stage3"].pop("speculation_width")
        payload["backward"].pop("speculation_width")
        return payload

    def fingerprint(self) -> str:
        """Отпечаток решающих настроек прогона."""
        blob = json.dumps(self.decision_payload(), sort_keys=True,
                          default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def smoke_config(run_dir, snapshot_dir, **overrides) -> RunConfig:
    """Быстрый прогон на настоящих данных: меньше объём, те же правила."""
    base = dict(
        run_dir=Path(run_dir),
        snapshot_dir=Path(snapshot_dir),
        users_limit=8000,
        feature_limit=90,
        probe=ProbeConfig(iterations=300, early_stopping_rounds=25),
        stage1=Stage1Config(actual_runs=2, null_runs=6, min_keep=20, max_keep=60,
                            null_iterations=60, actual_iterations=200,
                            actual_early_stopping=25),
        stage3=Stage3Config(top_k_candidates=24, max_features_to_add=8,
                            speculation_width=6),
        parallel=ParallelConfig(threads_per_worker=2),
    )
    base.update(overrides)
    return RunConfig(**base)


def config_from_env(run_dir, snapshot_dir) -> RunConfig:
    """Боевой конфиг с точечными правками из переменных окружения."""
    kwargs = {"run_dir": Path(run_dir), "snapshot_dir": Path(snapshot_dir),
              "model_mode": os.environ.get("NSEL_MODEL", "catboost"),
              "target_mode": os.environ.get("NSEL_TARGET", "log_level")}
    if os.environ.get("NSEL_USERS"):
        kwargs["users_limit"] = int(os.environ["NSEL_USERS"])
    if os.environ.get("NSEL_FEATURES"):
        kwargs["feature_limit"] = int(os.environ["NSEL_FEATURES"])
    if os.environ.get("NSEL_WORKERS") or os.environ.get("NSEL_THREADS"):
        kwargs["parallel"] = ParallelConfig(
            n_workers=int(os.environ.get("NSEL_WORKERS", "0")),
            threads_per_worker=int(os.environ.get("NSEL_THREADS", "2")))
    if os.environ.get("NSEL_PROBE_LR"):
        kwargs["probe"] = ProbeConfig(learning_rate=float(os.environ["NSEL_PROBE_LR"]))
    if os.environ.get("NSEL_PAIR_RESCUE") == "1":
        kwargs["stage5"] = Stage5Config(enabled=True)
    if os.environ.get("NSEL_CONFIRM_DOUBTFUL") == "1":
        kwargs["stage4"] = Stage4Config(confirm_doubtful=True)
    if os.environ.get("NSEL_OPTUNA") == "1":
        kwargs["optuna"] = OptunaConfig(
            enabled=True, gpu=os.environ.get("NSEL_OPTUNA_GPU", "0") == "1")
    return RunConfig(**kwargs)
