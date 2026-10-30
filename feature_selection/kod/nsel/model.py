"""Обучение щупа и метрики фолда для трёх семейств моделей."""
from __future__ import annotations

import numpy as np


def _anchor_in_log(anchor: np.ndarray, target_mode: str) -> np.ndarray:
    """Якорь в логарифмическом пространстве.

    У части таргетов якорь уже логарифмический, и второй логарифм дал бы совсем
    другую величину, не подав виду.
    """
    from .config import LOG_ANCHOR_MODES

    if target_mode in LOG_ANCHOR_MODES:
        return np.clip(anchor, 0.0, None)
    return np.log1p(np.clip(anchor, 0.0, None))


def build_targets(y_raw: np.ndarray, anchor: np.ndarray, target_mode: str) -> np.ndarray:
    """Переводит сырой таргет в пространство, в котором учится модель."""
    level = np.log1p(np.clip(y_raw, 0.0, None))
    if target_mode == "log_level":
        return level
    return level - _anchor_in_log(anchor, target_mode)


def restore_level(prediction: np.ndarray, anchor: np.ndarray,
                  target_mode: str) -> np.ndarray:
    """Восстанавливает предсказание в логарифме уровня."""
    if target_mode == "log_level":
        return np.clip(prediction, 0.0, None)
    return np.clip(prediction + _anchor_in_log(anchor, target_mode), 0.0, None)


def _root_mean_square(values: np.ndarray) -> float:
    """Корень из среднего квадрата, пустой вход даёт NaN."""
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values ** 2)))


def score_predictions(prediction_level: np.ndarray, truth_raw: np.ndarray) -> dict:
    """RMSLE целиком и по сегментам нулевых и покупавших пользователей."""
    truth_level = np.log1p(np.clip(truth_raw, 0.0, None))
    residual = prediction_level - truth_level
    zero_mask = truth_raw <= 0
    return {"rmsle": _root_mean_square(residual),
            "rmsle_zero": _root_mean_square(residual[zero_mask]),
            "rmsle_positive": _root_mean_square(residual[~zero_mask])}


def _catboost_model(probe: dict, seed: int, threads: int, iterations: int):
    """Создаёт CatBoost-щуп."""
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=iterations, learning_rate=probe["learning_rate"],
        depth=probe["depth"], l2_leaf_reg=probe["l2_leaf_reg"],
        loss_function="RMSE", random_seed=seed, thread_count=threads,
        min_data_in_leaf=probe["min_data_in_leaf"], verbose=False,
        allow_writing_files=False)


def _fit_catboost(matrices: dict, probe: dict, seed: int, threads: int,
                  iterations: int, early_stopping: int) -> tuple:
    """Обучает CatBoost с родной обработкой категорий."""
    from catboost import Pool

    positions = matrices["categorical_positions"]
    train_pool = Pool(matrices["X_train"], matrices["y_train"], cat_features=positions)
    valid_pool = Pool(matrices["X_valid"], matrices["y_valid"], cat_features=positions)
    # Матрицы уже проквантованы внутри Pool, исходные таблицы держать незачем.
    matrices["X_train"] = None
    matrices["X_valid"] = None
    model = _catboost_model(probe, seed, threads, iterations)
    fit_arguments = {}
    if early_stopping > 0:
        fit_arguments["eval_set"] = valid_pool
        fit_arguments["use_best_model"] = True
        fit_arguments["early_stopping_rounds"] = early_stopping
    model.fit(train_pool, **fit_arguments)
    importance = np.asarray(model.get_feature_importance(train_pool), dtype=np.float64)
    return model.predict(valid_pool), importance, int(model.tree_count_)


def _fit_lightgbm(matrices: dict, probe: dict, seed: int, threads: int,
                  iterations: int, early_stopping: int) -> tuple:
    """Обучает LightGBM на матрицах с уже закодированными категориями."""
    import lightgbm as lgb

    parameters = {"objective": "regression", "metric": "rmse",
                  "learning_rate": probe["learning_rate"],
                  "max_depth": probe["depth"], "num_leaves": 2 ** probe["depth"],
                  "lambda_l2": probe["l2_leaf_reg"], "seed": seed,
                  "num_threads": threads, "verbosity": -1,
                  "bagging_fraction": probe["subsample"], "bagging_freq": 1,
                  "feature_fraction": probe["colsample"],
                  "min_data_in_leaf": probe["min_data_in_leaf"]}
    train_set = lgb.Dataset(matrices["X_train"], label=matrices["y_train"],
                            free_raw_data=True)
    callbacks = []
    valid_sets = []
    if early_stopping > 0:
        valid_sets.append(lgb.Dataset(matrices["X_valid"], label=matrices["y_valid"],
                                      reference=train_set, free_raw_data=True))
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
    booster = lgb.train(parameters, train_set, num_boost_round=iterations,
                        valid_sets=valid_sets, callbacks=callbacks)
    importance = np.asarray(booster.feature_importance("gain"), dtype=np.float64)
    return (booster.predict(matrices["X_valid"]), importance,
            int(booster.current_iteration()))


def _fit_xgboost(matrices: dict, probe: dict, seed: int, threads: int,
                 iterations: int, early_stopping: int) -> tuple:
    """Обучает XGBoost на матрицах с уже закодированными категориями."""
    import xgboost as xgb

    parameters = {"objective": "reg:squarederror", "eval_metric": "rmse",
                  "tree_method": "hist",
                  "learning_rate": probe["learning_rate"], "max_depth": probe["depth"],
                  "reg_lambda": probe["l2_leaf_reg"], "seed": seed,
                  "nthread": threads, "subsample": probe["subsample"],
                  "colsample_bytree": probe["colsample"],
                  "min_child_weight": probe["min_data_in_leaf"]}
    # QuantileDMatrix бинует один раз при сборке и не держит исходную матрицу
    # рядом с гистограммами: это штатный способ кормить hist, и он снимает
    # лишние гигабайты на процесс.
    n_features = matrices["X_train"].shape[1]
    train_matrix = xgb.QuantileDMatrix(matrices["X_train"], label=matrices["y_train"],
                                       nthread=threads)
    valid_matrix = xgb.QuantileDMatrix(matrices["X_valid"], label=matrices["y_valid"],
                                       nthread=threads, ref=train_matrix)
    matrices["X_train"] = None
    matrices["X_valid"] = None
    arguments = {"num_boost_round": iterations, "verbose_eval": False}
    if early_stopping > 0:
        arguments["evals"] = [(valid_matrix, "valid")]
        arguments["early_stopping_rounds"] = early_stopping
    booster = xgb.train(parameters, train_matrix, **arguments)
    scores = booster.get_score(importance_type="gain")
    importance = np.array(
        [scores.get(f"f{index}", 0.0) for index in range(n_features)],
        dtype=np.float64)
    return (booster.predict(valid_matrix), importance,
            int(booster.num_boosted_rounds()))


_FITTERS = {"catboost": _fit_catboost, "lightgbm": _fit_lightgbm,
            "xgboost": _fit_xgboost}


def fit_and_score(matrices: dict, model_mode: str, target_mode: str, probe: dict,
                  seed: int, threads: int, iterations: int,
                  early_stopping: int) -> dict:
    """Обучает щуп на одном фолде и возвращает метрики с важностями."""
    prediction, importance, trees = _FITTERS[model_mode](
        matrices, probe, seed, threads, iterations, early_stopping)
    level = restore_level(prediction, matrices["anchor_valid"], target_mode)
    metrics = score_predictions(level, matrices["y_valid_raw"])
    metrics["trees"] = trees
    metrics["importance"] = importance.tolist()
    return metrics
