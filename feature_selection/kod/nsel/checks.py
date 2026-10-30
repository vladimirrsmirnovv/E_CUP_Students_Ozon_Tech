"""Обязательные проверки итога прогона."""
from __future__ import annotations

from .config import ContractError, assert_fold_contract

FORBIDDEN_STAGES = ("group_lofo", "stage2_group_lofo", "lofo")


def check_no_group_lofo(stages_run: list) -> None:
    """Group LOFO вырезан: стадии с таким именем в прогоне быть не может."""
    for name in stages_run:
        for forbidden in FORBIDDEN_STAGES:
            if forbidden in name:
                raise ContractError(f"в прогоне встретилась стадия LOFO: {name}")


def check_single_backward(stages_run: list) -> None:
    """Обратная чистка проходит ровно один раз."""
    count = sum(1 for name in stages_run if name.startswith("stage6_backward"))
    if count != 1:
        raise ContractError(f"обратных чисток {count}, должна быть одна")


def check_multiseed(report: dict, config) -> None:
    """Сиды проверяют набор целиком, а не каждый признак по отдельности."""
    if not config.stage4.enabled:
        return
    seeds = set(report["multiseed"]["per_seed"])
    if seeds != {str(seed) for seed in config.stage4.seeds}:
        raise ContractError(f"проверка на сидах {sorted(seeds)}")


def check_pair_rescue(report: dict, config) -> None:
    """Состояние парного спасения записано и совпадает с настройкой."""
    if report["pair_rescue"]["enabled"] != config.stage5.enabled:
        raise ContractError("состояние парного спасения не совпадает с настройкой")


def check_selection(report: dict) -> None:
    """Итоговый набор непустой и без повторов.

    Повтор не меняет метрику, но ломает обучение у того, кто возьмёт список
    как есть: CatBoost отказывается работать с одинаковыми именами признаков.
    """
    features = report["selected_features"]
    if not features:
        raise ContractError("отбор вернул пустой набор")
    if len(features) != len(set(features)):
        seen, repeated = set(), []
        for name in features:
            if name in seen:
                repeated.append(name)
            seen.add(name)
        raise ContractError(f"в наборе повторяются признаки: {sorted(set(repeated))}")


def run_all(report: dict, config, folds: list, stages_run: list) -> list:
    """Прогоняет все обязательные проверки и возвращает их список."""
    assert_fold_contract(folds)
    check_no_group_lofo(stages_run)
    check_single_backward(stages_run)
    check_multiseed(report, config)
    check_pair_rescue(report, config)
    check_selection(report)
    return ["контракт трёх фолдов", "Group LOFO отсутствует",
            "одна обратная чистка", "multiseed на наборе",
            "состояние парного спасения", "набор непустой и без повторов"]
