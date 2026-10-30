"""Раскладка срезов в numpy и материализация фолдов."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .config import (
    ADAPTIVE_ANCHORS,
    SNAPSHOT_ANCHORS,
    SNAPSHOT_INDEX_COLUMN,
    TARGET_ANCHOR_COLUMN,
    TARGET_COLUMN,
)

SERVICE_COLUMNS = {SNAPSHOT_INDEX_COLUMN, TARGET_COLUMN}
NUMERIC_PREFIXES = ("int", "float", "double", "uint")
CALENDAR_CATEGORY_PREFIX = "cutoff_"
CATEGORY_SUFFIX = "_cat"


def is_count_bin_category(name: str) -> bool:
    """Категория, построенная биннингом счётчика активности."""
    if not name.endswith(CATEGORY_SUFFIX):
        return False
    return not name.startswith(CALENDAR_CATEGORY_PREFIX)


def snapshot_path(snapshot_dir, cutoff: str) -> Path:
    """Путь до parquet-среза на дату отсечения."""
    return Path(snapshot_dir) / f"snapshot_{cutoff.replace('-', '')}.parquet"


def required_cutoffs(folds: list) -> list:
    """Все даты отсечения, нужные набору фолдов."""
    dates = set()
    for fold in folds:
        dates.update(fold.train_cutoffs)
        dates.add(fold.valid_cutoff)
    return sorted(dates)


def _hash_users(user_ids) -> np.ndarray:
    """Устойчивый хэш идентификаторов пользователей."""
    return np.array(
        [int(hashlib.blake2b(str(value).encode(), digest_size=8).hexdigest(), 16)
         for value in user_ids],
        dtype=np.uint64,
    )


def stable_user_mask(user_ids, limit: int) -> np.ndarray:
    """Детерминированная подвыборка пользователей по хэшу."""
    if limit <= 0 or limit >= len(user_ids):
        return np.ones(len(user_ids), dtype=bool)
    digest = _hash_users(user_ids)
    order = np.argsort(digest, kind="stable")
    mask = np.zeros(len(user_ids), dtype=bool)
    mask[order[:limit]] = True
    return mask


MINIMUM_CATEGORICAL_IN_SUBSET = 3


def _sample_names(pool: list, count: int, seed: int) -> set:
    """Детерминированная выборка заданного размера из списка имён."""
    if count <= 0 or not pool:
        return set()
    generator = np.random.RandomState(seed)
    chosen = generator.choice(len(pool), size=min(count, len(pool)), replace=False)
    return {pool[index] for index in chosen}


def choose_feature_subset(names: list, categorical: list, limit: int,
                          seed: int = 42) -> list:
    """Подмножество признаков для укороченного прогона с долей категориальных."""
    if limit <= 0 or limit >= len(names):
        return list(names)
    keep = set()
    if TARGET_ANCHOR_COLUMN in names:
        keep.add(TARGET_ANCHOR_COLUMN)
    # Категориальные держим в выборке принудительно: иначе быстрый прогон не
    # проверял бы кодирование, а именно оно ломается тише всего.
    categorical_pool = [name for name in categorical if name not in keep]
    keep |= _sample_names(categorical_pool,
                          min(MINIMUM_CATEGORICAL_IN_SUBSET, max(0, limit - 1)), seed)
    numeric_pool = [name for name in names
                    if name not in keep and name not in set(categorical)]
    keep |= _sample_names(numeric_pool, limit - len(keep), seed + 1)
    return [name for name in names if name in keep]


def _column_kinds(schema, feature_names: list) -> tuple:
    """Делит признаки на числовые и категориальные по схеме parquet."""
    types = dict(zip(schema.names, schema.types))
    categorical = [name for name in feature_names
                   if not str(types[name]).startswith(NUMERIC_PREFIXES)]
    numeric = [name for name in feature_names if name not in set(categorical)]
    return numeric, categorical


def _shared_feature_names(schemas: dict, cutoffs: list, feature_limit: int,
                          drop_count_bins: bool = True) -> list:
    """Признаки, присутствующие во всех срезах, в порядке первого среза."""
    common = set(schemas[cutoffs[0]].names)
    for cutoff in cutoffs[1:]:
        common &= set(schemas[cutoff].names)
    names = [name for name in schemas[cutoffs[0]].names
             if name in common and name not in SERVICE_COLUMNS]
    # Категории из биннинга счётчиков он просил отключить: не показали пользы.
    # Календарные категории остаются, их он оставил включёнными.
    if drop_count_bins:
        names = [name for name in names if not is_count_bin_category(name)]
    _numeric, categorical = _column_kinds(schemas[cutoffs[0]], names)
    return choose_feature_subset(names, categorical, feature_limit)


def _encode_categoricals(frame, categorical: list, tables: dict) -> np.ndarray:
    """Кодирует категориальные столбцы общим для всех срезов словарём."""
    import polars as pl

    codes = np.empty((len(categorical), frame.height), dtype=np.int32)
    for row, name in enumerate(categorical):
        values = frame[name].cast(pl.Utf8).fill_null("__nan__").to_list()
        table = tables[name]
        for position, value in enumerate(values):
            code = table.get(value)
            if code is None:
                code = len(table)
                table[value] = code
            codes[row, position] = code
    return codes


def _write_numeric(frame, numeric: list, destination: Path) -> None:
    """Пишет числовые признаки матрицей «признак x строка»."""
    matrix = np.empty((len(numeric), frame.height), dtype=np.float32)
    for row, name in enumerate(numeric):
        column = frame[name].to_numpy().astype(np.float32, copy=False)
        matrix[row, :] = np.where(np.isfinite(column), column, np.nan)
    np.save(destination, matrix)


def _write_adaptive_anchor(frame, cutoff: str, prepared_dir: Path,
                           anchor_dir) -> None:
    """Кладёт адаптивный якорь в порядке строк среза.

    Порядок задаётся самим срезом, поэтому якорь не приклеивается по позиции, а
    сопоставляется по пользователю: молчаливый сдвиг на одну строку испортил бы
    весь таргет и не проявился бы ни в одной проверке.
    """
    import polars as pl

    source = Path(anchor_dir) / (
        f"anchor_segment_adaptive_{cutoff.replace('-', '')}.parquet")
    if not source.exists():
        raise FileNotFoundError(f"нет файла якоря {source}")
    anchors = pl.read_parquet(source).rename(
        {"user_id": SNAPSHOT_INDEX_COLUMN})
    joined = frame.select(SNAPSHOT_INDEX_COLUMN).join(
        anchors, on=SNAPSHOT_INDEX_COLUMN, how="left")
    for column in sorted(set(ADAPTIVE_ANCHORS.values())):
        if column not in joined.columns:
            raise KeyError(f"в файле якоря {source.name} нет столбца {column}")
        values = joined[column].to_numpy().astype(np.float64)
        missing = int(np.isnan(values).sum())
        if missing:
            raise ValueError(f"якорь {column} за {cutoff}: не нашлось "
                             f"{missing} пользователей из {len(values)}")
        np.save(prepared_dir / f"{column}_{cutoff}.npy", values)


def _prepare_one_cutoff(source: Path, cutoff: str, prepared_dir: Path,
                        columns: list, numeric: list, categorical: list,
                        tables: dict, users_limit: int, anchor_dir=None) -> int:
    """Раскладывает один срез в файлы numpy и возвращает число строк."""
    import polars as pl

    frame = pl.read_parquet(source, columns=columns).sort(SNAPSHOT_INDEX_COLUMN)
    mask = stable_user_mask(frame[SNAPSHOT_INDEX_COLUMN].to_numpy(), users_limit)
    if not mask.all():
        frame = frame.filter(pl.Series(mask))
    _write_numeric(frame, numeric, prepared_dir / f"num_{cutoff}.npy")
    np.save(prepared_dir / f"cat_{cutoff}.npy",
            _encode_categoricals(frame, categorical, tables))
    target = np.nan_to_num(frame[TARGET_COLUMN].to_numpy().astype(np.float64), nan=0.0)
    np.save(prepared_dir / f"y_{cutoff}.npy", target)
    anchor = np.nan_to_num(
        frame[TARGET_ANCHOR_COLUMN].to_numpy().astype(np.float64), nan=0.0)
    np.save(prepared_dir / f"anchor_{cutoff}.npy", anchor)
    if anchor_dir:
        _write_adaptive_anchor(frame, cutoff, prepared_dir, anchor_dir)
    for column in sorted(set(SNAPSHOT_ANCHORS.values())):
        if column not in frame.columns:
            continue
        values = np.nan_to_num(
            frame[column].to_numpy().astype(np.float64), nan=0.0)
        np.save(prepared_dir / f"{column}_{cutoff}.npy", values)
    return frame.height


def prepare_snapshots(snapshot_dir, prepared_dir, cutoffs, users_limit=0,
                      feature_limit=0, logger=None, force=False,
                      drop_count_bins=True, anchor_dir=None) -> dict:
    """Раскладывает срезы в numpy один раз и возвращает манифест."""
    import pyarrow.parquet as pq

    log = logger or logging.getLogger("nsel.data")
    prepared_dir = Path(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    cutoffs = list(cutoffs)
    signature = hashlib.sha256(json.dumps(
        {"cutoffs": cutoffs, "users": users_limit, "features": feature_limit,
         "dir": str(snapshot_dir), "drop_count_bins": drop_count_bins,
         "anchor_dir": str(anchor_dir or "")},
        sort_keys=True).encode()).hexdigest()[:16]

    manifest_path = prepared_dir / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("signature") == signature:
            log.info("[подготовка] срезы уже разложены, подпись %s", signature)
            return manifest

    schemas = {}
    for cutoff in cutoffs:
        path = snapshot_path(snapshot_dir, cutoff)
        if not path.exists():
            raise FileNotFoundError(f"нет среза {path}")
        schemas[cutoff] = pq.ParquetFile(path).schema_arrow

    feature_names = _shared_feature_names(schemas, cutoffs, feature_limit,
                                          drop_count_bins)
    numeric, categorical = _column_kinds(schemas[cutoffs[0]], feature_names)
    columns = [SNAPSHOT_INDEX_COLUMN, TARGET_COLUMN] + feature_names
    if TARGET_ANCHOR_COLUMN not in columns:
        columns.append(TARGET_ANCHOR_COLUMN)
    for extra in sorted(set(SNAPSHOT_ANCHORS.values())):
        if extra not in columns:
            columns.append(extra)
    log.info("[подготовка] признаков %d: числовых %d, категориальных %d",
             len(feature_names), len(numeric), len(categorical))

    tables = {name: {} for name in categorical}
    rows = {}
    for cutoff in cutoffs:
        log.info("[подготовка] раскладываю срез %s", cutoff)
        rows[cutoff] = _prepare_one_cutoff(
            snapshot_path(snapshot_dir, cutoff), cutoff, prepared_dir, columns,
            numeric, categorical, tables, users_limit, anchor_dir)

    manifest = {"signature": signature, "cutoffs": cutoffs, "numeric": numeric,
                "categorical": categorical, "features": feature_names, "rows": rows,
                "cardinality": {name: len(table) for name, table in tables.items()},
                "prepared_dir": str(prepared_dir)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log.info("[подготовка] готово: срезов %d, строк на срез %s",
             len(cutoffs), sorted(set(rows.values())))
    return manifest


_MEMORY_MAPS: dict = {}
_MANIFESTS: dict = {}


def load_manifest(prepared_dir: str) -> dict:
    """Читает манифест подготовки, кэшируя его в процессе."""
    if prepared_dir not in _MANIFESTS:
        _MANIFESTS[prepared_dir] = json.loads(
            (Path(prepared_dir) / "manifest.json").read_text())
    return _MANIFESTS[prepared_dir]


def _memory_map(prepared_dir: str, name: str) -> np.ndarray:
    """Открывает файл numpy отображением в память, разделяемым между процессами."""
    key = f"{prepared_dir}/{name}"
    if key not in _MEMORY_MAPS:
        _MEMORY_MAPS[key] = np.load(Path(prepared_dir) / name, mmap_mode="r")
    return _MEMORY_MAPS[key]


def _column_values(prepared_dir: str, cutoffs: list, prefix: str, row: int,
                   dtype) -> np.ndarray:
    """Значения одного признака, склеенные по срезам обучения."""
    parts = [np.asarray(_memory_map(prepared_dir, f"{prefix}_{cutoff}.npy")[row, :],
                        dtype=dtype) for cutoff in cutoffs]
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


def _build_frame(prepared_dir: str, cutoffs: list, numeric_names: list,
                 numeric_rows: list, categorical_names: list,
                 categorical_rows: list):
    """Собирает таблицу по одному столбцу за раз, без промежуточной матрицы.

    Промежуточная матрица «строка x признак» удваивала бы память: на миллионе
    строк и девятистах признаках это лишние четыре гигабайта на процесс, и
    пятнадцать процессов клали машину по памяти.
    """
    import pandas as pd

    columns = {}
    for name, row in zip(numeric_names, numeric_rows):
        columns[name] = _column_values(prepared_dir, cutoffs, "num", row, np.float32)
    for name, row in zip(categorical_names, categorical_rows):
        columns[name] = _column_values(prepared_dir, cutoffs, "cat", row, np.int32)
    return pd.DataFrame(columns, columns=numeric_names + categorical_names,
                        copy=False)


def _build_matrix(prepared_dir: str, cutoffs: list, numeric_rows: list,
                  categorical_rows: list) -> np.ndarray:
    """Собирает матрицу «строка x признак» сразу, без промежуточных блоков.

    Раньше блоки по срезам собирались отдельно и потом склеивались, и матрица
    жила в памяти дважды: на миллионе строк и девятистах признаках это лишние
    три с половиной гигабайта на процесс, из-за которых в память влезало
    вдвое меньше процессов.
    """
    manifest = load_manifest(prepared_dir)
    heights = [manifest["rows"][cutoff] for cutoff in cutoffs]
    width = len(numeric_rows) + len(categorical_rows)
    matrix = np.empty((sum(heights), width), dtype=np.float32)
    offset = 0
    for cutoff, height in zip(cutoffs, heights):
        column = 0
        for row in numeric_rows:
            source = _memory_map(prepared_dir, f"num_{cutoff}.npy")[row, :]
            matrix[offset:offset + height, column] = source
            column += 1
        for row in categorical_rows:
            source = _memory_map(prepared_dir, f"cat_{cutoff}.npy")[row, :]
            matrix[offset:offset + height, column] = source
            column += 1
        offset += height
    return matrix


def _stack_features(prepared_dir: str, cutoff: str, numeric_rows: list,
                    categorical_rows: list) -> np.ndarray:
    """Собирает матрицу «строка x признак» из отображённых в память срезов."""
    parts = []
    if numeric_rows:
        block = _memory_map(prepared_dir, f"num_{cutoff}.npy")[numeric_rows, :]
        parts.append(np.asarray(block, dtype=np.float32).T)
    if categorical_rows:
        block = _memory_map(prepared_dir, f"cat_{cutoff}.npy")[categorical_rows, :]
        parts.append(np.asarray(block, dtype=np.float32).T)
    if not parts:
        return np.zeros((0, 0), dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    return np.hstack(parts)


def anchor_file(target_mode: str, cutoff: str) -> str:
    """Имя файла якоря: у адаптивных и снимочных таргетов он свой."""
    column = ADAPTIVE_ANCHORS.get(target_mode) or SNAPSHOT_ANCHORS.get(target_mode)
    if column:
        return f"{column}_{cutoff}.npy"
    return f"anchor_{cutoff}.npy"


def materialize(prepared_dir: str, fold: dict, feature_columns: list,
                target_mode: str, layout: str = "array") -> dict:
    """Собирает обучающую и проверочную матрицы одного фолда."""
    manifest = load_manifest(prepared_dir)
    numeric_index = {name: row for row, name in enumerate(manifest["numeric"])}
    categorical_index = {name: row for row, name in enumerate(manifest["categorical"])}

    numeric_names = [name for name in feature_columns if name in numeric_index]
    categorical_names = [name for name in feature_columns if name in categorical_index]
    numeric_rows = [numeric_index[name] for name in numeric_names]
    categorical_rows = [categorical_index[name] for name in categorical_names]

    train_targets, train_anchors, block_ids = [], [], []
    for block, cutoff in enumerate(fold["train_cutoffs"]):
        target = np.asarray(_memory_map(prepared_dir, f"y_{cutoff}.npy"))
        train_targets.append(target)
        train_anchors.append(np.asarray(
            _memory_map(prepared_dir, anchor_file(target_mode, cutoff))))
        block_ids.append(np.full(len(target), block, dtype=np.int32))

    valid_cutoff = fold["valid_cutoff"]
    columns = numeric_names + categorical_names
    if layout == "frame":
        train_features = _build_frame(prepared_dir, list(fold["train_cutoffs"]),
                                      numeric_names, numeric_rows,
                                      categorical_names, categorical_rows)
        valid_features = _build_frame(prepared_dir, [valid_cutoff], numeric_names,
                                      numeric_rows, categorical_names,
                                      categorical_rows)
    else:
        train_features = _build_matrix(prepared_dir, list(fold["train_cutoffs"]),
                                       numeric_rows, categorical_rows)
        valid_features = _build_matrix(prepared_dir, [valid_cutoff], numeric_rows,
                                       categorical_rows)
    return {
        "X_train": train_features,
        "y_train_raw": np.concatenate(train_targets),
        "anchor_train": np.concatenate(train_anchors),
        "block_ids": np.concatenate(block_ids),
        "X_valid": valid_features,
        "y_valid_raw": np.asarray(_memory_map(prepared_dir, f"y_{valid_cutoff}.npy")),
        "anchor_valid": np.asarray(
            _memory_map(prepared_dir, anchor_file(target_mode, valid_cutoff))),
        "columns": columns,
        "categorical_positions": list(range(len(numeric_names), len(columns))),
    }


def materialize_targets(prepared_dir: str, fold: dict,
                        target_mode: str = "log_level") -> dict:
    """Только таргеты и якоря фолда: признаки уже лежат проквантованными."""
    targets, anchors, block_ids = [], [], []
    for block, cutoff in enumerate(fold["train_cutoffs"]):
        target = np.asarray(_memory_map(prepared_dir, f"y_{cutoff}.npy"))
        targets.append(target)
        anchors.append(np.asarray(
            _memory_map(prepared_dir, anchor_file(target_mode, cutoff))))
        block_ids.append(np.full(len(target), block, dtype=np.int32))
    valid_cutoff = fold["valid_cutoff"]
    return {"X_train": None, "X_valid": None,
            "y_train_raw": np.concatenate(targets),
            "anchor_train": np.concatenate(anchors),
            "block_ids": np.concatenate(block_ids),
            "y_valid_raw": np.asarray(_memory_map(prepared_dir, f"y_{valid_cutoff}.npy")),
            "anchor_valid": np.asarray(
                _memory_map(prepared_dir, anchor_file(target_mode, valid_cutoff))),
            "columns": [], "categorical_positions": []}


def temporal_shuffle(target: np.ndarray, block_ids: np.ndarray, seed: int) -> np.ndarray:
    """Перемешивает таргет внутри временных блоков, а не по всей выборке."""
    generator = np.random.RandomState(seed)
    shuffled = target.copy()
    for block in np.unique(block_ids):
        positions = np.where(block_ids == block)[0]
        if len(positions) < 2:
            continue
        shuffled[positions] = target[generator.permutation(positions)]
    return shuffled


_WINDOW_TOKENS = ("7d", "14d", "30d", "60d", "90d", "180d", "365d", "lifetime",
                  "hl30", "hl60", "cat")


def feature_family(name: str) -> str:
    """Семья признака: общий префикс без хвостового токена-окна."""
    tokens = name.split("__", 1)[0].split("_")
    while tokens and (tokens[-1] in _WINDOW_TOKENS or tokens[-1].isdigit()):
        tokens.pop()
    if not tokens:
        return name
    return "_".join(tokens)
