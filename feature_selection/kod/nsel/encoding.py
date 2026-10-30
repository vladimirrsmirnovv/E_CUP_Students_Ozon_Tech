"""Ordered target encoding с эмпирико-байесовским сглаживанием.

Действует только для XGBoost и LightGBM. CatBoost сюда не заходит: у него своя
обработка категорий, и трогать её запрещено.

Все статистики берутся исключительно из обучающей части своего фолда —
средние по категориям, глобальное априорное, оценка дисперсии и само
сглаживание.
"""
from __future__ import annotations

import numpy as np


def _global_mean(target: np.ndarray) -> float:
    """Априорное среднее по обучающей части фолда."""
    if len(target) == 0:
        return 0.0
    return float(np.mean(target))


def _category_statistics(codes: np.ndarray, target: np.ndarray,
                         cardinality: int) -> tuple:
    """Счётчики, суммы и суммы квадратов таргета по категориям."""
    counts = np.bincount(codes, minlength=cardinality).astype(np.float64)
    sums = np.bincount(codes, weights=target, minlength=cardinality)
    squares = np.bincount(codes, weights=target * target, minlength=cardinality)
    return counts, sums, squares


def estimate_smoothing(counts: np.ndarray, sums: np.ndarray, squares: np.ndarray,
                       prior_mean: float, config) -> float:
    """Оценивает силу сглаживания методом моментов."""
    if config.mode == "fixed":
        return float(config.fixed_smoothing)
    present = counts > 0
    if present.sum() < 2:
        return float(config.fixed_smoothing)
    means = np.zeros_like(sums)
    means[present] = sums[present] / counts[present]
    total = counts.sum()
    within = squares.sum() - (sums[present] ** 2 / counts[present]).sum()
    degrees = max(1.0, total - present.sum())
    within_variance = within / degrees
    weights = counts[present] / total
    between_variance = float(np.sum(weights * (means[present] - prior_mean) ** 2))
    harmonic = present.sum() / np.sum(1.0 / counts[present])
    between_variance = between_variance - within_variance / max(1.0, harmonic)
    if between_variance < config.minimum_variance:
        return float(config.maximum_smoothing)
    return float(min(config.maximum_smoothing, within_variance / between_variance))


def _smoothed_values(counts: np.ndarray, sums: np.ndarray, prior_mean: float,
                     smoothing: float, config) -> np.ndarray:
    """Сглаженное среднее таргета по каждой категории."""
    values = np.full(len(counts), np.nan, dtype=np.float64)
    eligible = counts >= max(1, config.min_count)
    values[eligible] = (
        (sums[eligible] + smoothing * prior_mean) / (counts[eligible] + smoothing))
    if config.unknown_value == "global":
        values[~eligible] = prior_mean
    return values


def _fill_unseen(encoded: np.ndarray, prior_mean: float, config) -> np.ndarray:
    """Подставляет значение для категорий, не встреченных в обучении."""
    if config.unknown_value == "global":
        return np.where(np.isnan(encoded), prior_mean, encoded)
    return encoded


def encode_column(train_codes: np.ndarray, train_target: np.ndarray,
                  valid_codes: np.ndarray, block_ids: np.ndarray,
                  config) -> tuple:
    """Кодирует один категориальный столбец по обучающей части фолда."""
    cardinality = int(max(train_codes.max(initial=-1), valid_codes.max(initial=-1))) + 1
    prior_mean = _global_mean(train_target)
    order = np.argsort(block_ids, kind="stable")
    blocks = np.array_split(order, max(1, config.n_blocks))

    encoded_train = np.full(len(train_codes), np.nan, dtype=np.float64)
    counts = np.zeros(cardinality, dtype=np.float64)
    sums = np.zeros(cardinality, dtype=np.float64)
    squares = np.zeros(cardinality, dtype=np.float64)

    for position, rows in enumerate(blocks):
        if len(rows) == 0:
            continue
        if position == 0:
            if config.first_block_value == "global":
                encoded_train[rows] = prior_mean
        else:
            smoothing = estimate_smoothing(counts, sums, squares, prior_mean, config)
            values = _smoothed_values(counts, sums, prior_mean, smoothing, config)
            encoded_train[rows] = _fill_unseen(values[train_codes[rows]],
                                               prior_mean, config)
        part_counts, part_sums, part_squares = _category_statistics(
            train_codes[rows], train_target[rows], cardinality)
        counts += part_counts
        sums += part_sums
        squares += part_squares

    smoothing = estimate_smoothing(counts, sums, squares, prior_mean, config)
    values = _smoothed_values(counts, sums, prior_mean, smoothing, config)
    encoded_valid = _fill_unseen(values[valid_codes], prior_mean, config)
    diagnostics = {"prior_mean": prior_mean, "smoothing": smoothing,
                   "cardinality": cardinality,
                   "known_categories": int((counts > 0).sum())}
    return encoded_train, encoded_valid, diagnostics


def encode_matrices(X_train: np.ndarray, X_valid: np.ndarray, target: np.ndarray,
                    block_ids: np.ndarray, categorical_positions: list,
                    config) -> tuple:
    """Заменяет коды категорий сглаженным средним таргета в обеих матрицах."""
    if not categorical_positions:
        return X_train, X_valid, {}
    # Столбцы заменяются на месте: копия матрицы на миллион строк и девятьсот
    # признаков стоит четыре гигабайта на процесс, а матрицы всё равно наши.
    diagnostics = {}
    for position in categorical_positions:
        encoded_train, encoded_valid, info = encode_column(
            X_train[:, position].astype(np.int64), target,
            X_valid[:, position].astype(np.int64), block_ids, config)
        X_train[:, position] = encoded_train.astype(np.float32)
        X_valid[:, position] = encoded_valid.astype(np.float32)
        diagnostics[str(position)] = info
    return X_train, X_valid, diagnostics
