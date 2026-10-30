"""Расчёт якоря segment_adaptive для каждого среза.

Якоря нет в готовых срезах, он строится по сырой истории: двенадцать блоков по
тридцать дней, псевдобэктест по каждому масштабу и выбор для каждого сегмента
пользователей того масштаба, что исторически ошибался меньше.

Логика не переписывается, а берётся из модуля коллеги: любое переписывание
такого объёма это способ незаметно разойтись с его числами.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

# Функция коллеги считает все адаптивные якоря разом, поэтому сохраняем их
# все: пересчёт ради одного столбца стоил бы столько же.
ANCHOR_COLUMNS = ("anchor_segment_adaptive", "anchor_volatility_adaptive",
                  "anchor_segment_logspace_log", "anchor_segment_logspace",
                  "anchor_personal_hard", "anchor_frequency_monetary_adaptive")
SEGMENT_COLUMNS = ("gmv_30d", "gmv_60d", "purchase_days_90d",
                   "days_since_last_purchase")
INDEX_COLUMN = "user_id"
# to_ord нужен соседнему якорю: обёртка коллеги считает все адаптивные
# якоря разом, и без этого столбца падает целиком.
RAW_COLUMNS = ("user_id", "event_date", "gmv", "to_ord")


def load_colleague_module(path):
    """Загружает модуль коллеги как обычный модуль по пути к файлу."""
    spec = importlib.util.spec_from_file_location("ozon_colleague", str(path))
    module = importlib.util.module_from_spec(spec)
    # Регистрация до выполнения обязательна: иначе dataclass внутри модуля не
    # находит собственное пространство имён и падает.
    sys.modules["ozon_colleague"] = module
    spec.loader.exec_module(module)
    return module


def cutoff_from_name(path) -> str:
    """Дата среза из имени файла snapshot_YYYYMMDD.parquet."""
    stem = Path(path).stem.split("_")[-1]
    return f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"


def read_snapshot(path, users_limit: int):
    """Срез с полями, нужными для сегментов, индексированный пользователем."""
    import pandas as pd

    frame = pd.read_parquet(path, columns=[INDEX_COLUMN, *SEGMENT_COLUMNS])
    # В срезах пользователь записан категорией из строк, в сырой истории целым
    # числом. Без приведения соединение не находит никого и якорь выходит
    # нулевым у всех, причём молча.
    frame[INDEX_COLUMN] = frame[INDEX_COLUMN].astype(str).astype("int64")
    frame = frame.sort_values(INDEX_COLUMN).set_index(INDEX_COLUMN)
    if users_limit:
        frame = frame.iloc[:users_limit]
    return frame


def build_one(module, raw, snapshot, cutoff):
    """Все адаптивные якоря одного среза и краткая сводка о них."""
    anchors, diagnostics = module._build_adaptive_target_anchors(
        raw, snapshot, cutoff)
    missing = [name for name in ANCHOR_COLUMNS if name not in anchors.columns]
    if missing:
        raise KeyError(f"в результате нет якорей {missing}")
    return anchors[list(ANCHOR_COLUMNS)], diagnostics


def main(argv=None) -> int:
    """Считает якорь для всех срезов каталога и кладёт рядом отдельным файлом."""
    import pandas as pd

    parser = argparse.ArgumentParser(description="якорь segment_adaptive")
    parser.add_argument("--raw", required=True, help="сырая история с gmv")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--module", required=True, help="файл конвейера коллеги")
    parser.add_argument("--users", type=int, default=0)
    parser.add_argument("--cutoffs", default="", help="через запятую, пусто это все")
    arguments = parser.parse_args(argv)

    module = load_colleague_module(arguments.module)
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    raw = pd.read_parquet(arguments.raw, columns=list(RAW_COLUMNS))
    print(f"[сырьё] строк {len(raw)}, прочитано за {time.time() - started:.0f} с")

    wanted = {c for c in arguments.cutoffs.split(",") if c}
    files = sorted(Path(arguments.snapshot_dir).glob("snapshot_*.parquet"))
    written = 0
    for path in files:
        cutoff = cutoff_from_name(path)
        if wanted and cutoff not in wanted:
            continue
        moment = time.time()
        snapshot = read_snapshot(path, arguments.users)
        anchors, _ = build_one(module, raw, snapshot, cutoff)
        target = out_dir / f"anchor_segment_adaptive_{cutoff.replace('-', '')}.parquet"
        anchors.reset_index().to_parquet(target, index=False)
        parts = []
        for name in ANCHOR_COLUMNS:
            values = anchors[name]
            # Логарифмический якорь законно бывает нулевым у неактивных, а вот
            # нулевой у всех означает, что идентификаторы не совпали.
            if int((values <= 0).sum()) == len(values):
                raise ValueError(
                    f"[{cutoff}] якорь {name} нулевой у всех: почти наверняка "
                    "не совпали идентификаторы среза и сырой истории")
            parts.append(f"{name.replace('anchor_', '')} {values.mean():.1f}")
        print(f"[{cutoff}] пользователей {len(anchors)}, "
              f"{', '.join(parts)}, за {time.time() - moment:.0f} с")
        written += 1
    print(f"готово: файлов {written}, всего {time.time() - started:.0f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
