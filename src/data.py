import os
from typing import Tuple

import numpy as np
import pandas as pd

from .config import CSV_PATH


def read_table_smart(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv", ".txt"]:
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1251", "latin1"]
        last_err = None
        for enc in encodings_to_try:
            try:
                print(f"Пробую читать как CSV с encoding='{enc}'")
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError as e:
                last_err = e
                continue
        raise RuntimeError(
            f"Не удалось прочитать CSV ни с одной кодировкой. "
            f"Последняя ошибка: {last_err}"
        )

    elif ext in [".xlsx", ".xls"]:
        print("Читаю Excel-файл (read_excel, header=0)...")
        return pd.read_excel(path, sheet_name=0, header=0)

    else:
        raise ValueError(
            f"Неизвестное расширение файла '{ext}'. Ожидаю .csv/.txt или .xlsx/.xls"
        )


def load_data(path: str | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает:
        X_train, y_train, X_test, y_test

    Ожидается формат:
        - столбцы sample_1..sample_10000  (или первые 10000 столбцов сигнала)
        - столбец 'class'      (значения 1, 2, 3)
        - столбец 'train_set'  (1 = train, 2 = test)
    """
    if path is None:
        path = CSV_PATH

    df = read_table_smart(path)

    # сигнал: по именам sample_* или первые 10000 столбцов
    signal_cols = [col for col in df.columns if str(col).startswith("sample_")]
    if len(signal_cols) == 0:
        signal_cols = df.columns[:10000]

    X = df[signal_cols].values.astype(np.float32)

    if "class" not in df.columns:
        raise ValueError(f"В таблице нет столбца 'class'. Есть столбцы: {list(df.columns[-10:])}")

    if "train_set" not in df.columns:
        raise ValueError(f"В таблице нет столбца 'train_set'. Есть столбцы: {list(df.columns[-10:])}")

    y = df["class"].values.astype(np.int64)  # 1/2/3
    y = y - 1  # → 0/1/2

    train_mask = df["train_set"].values == 1
    test_mask = df["train_set"].values == 2

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    # Нормализация по train
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test
