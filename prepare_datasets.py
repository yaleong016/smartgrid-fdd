# file: prepare_datasets.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import os
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

SEQ_LEN = 32
DATA_PATH = "data/smart_grid_dataset.csv"
OUT_DIR = "prepared_data"
os.makedirs(OUT_DIR, exist_ok=True)

def load_and_scale_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Sort by time column if present
    time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if time_cols:
        df = df.sort_values(by=time_cols[0])
    # print(df.head())
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.isnull().any().any():
        numeric_df = numeric_df.fillna(method="ffill").fillna(method="bfill")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(numeric_df.values.astype(np.float32))
    print("X_scaled ", X_scaled)
    print(X_scaled.shape)
    print("scaler_min ", scaler.data_min_)
    print("scaler_max ", scaler.data_max_)
    return X_scaled, scaler

def generate_physicsish_attack(context: np.ndarray) -> np.ndarray:
    x_last = context[-1].copy()
    D = x_last.shape[0]

    mode = np.random.choice(["scale", "offset", "noise"])

    if mode == "scale":
        alpha = np.random.uniform(0.2, 0.5) * np.random.choice([-1, 1])
        x_fake = x_last * (1.0 + alpha)
    elif mode == "offset":
        delta = np.random.uniform(0.1, 0.3) * np.random.choice([-1, 1])
        x_fake = x_last + delta
    else:
        noise = np.random.normal(loc=0.0, scale=0.05, size=D)
        x_fake = x_last + noise

    x_fake = np.clip(x_fake, 0.0, 1.0)
    return x_fake.astype(np.float32)

def create_clean_and_attacked_sequences(X: np.ndarray, seq_len: int):
    T, D = X.shape
    usable_T = (T // seq_len) * seq_len
    X_trimmed = X[:usable_T]
    windows = X_trimmed.reshape(-1, seq_len, D)  # (W, seq_len, D)

    W = windows.shape[0]
    half_W = W // 2

    clean_windows = windows[:half_W]

    attacked_windows = []
    for w in windows[half_W: 2 * half_W]:
        pos = np.random.randint(0, seq_len)
        context = np.delete(w, pos, axis=0)  # (seq_len-1, D)
        fake_point = generate_physicsish_attack(context)
        attacked = np.insert(context, pos, fake_point, axis=0)
        attacked_windows.append(attacked)

    attacked_windows = np.stack(attacked_windows, axis=0)

    X_all = np.concatenate([clean_windows, attacked_windows], axis=0)
    y_all = np.concatenate([
        np.zeros(half_W, dtype=np.float32),
        np.ones(half_W, dtype=np.float32)
    ], axis=0)

    return X_all, y_all

def train_val_test_split_sequences(
    X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        shuffle=True,
        stratify=y,
    )

    val_frac_of_temp = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_frac_of_temp),
        random_state=seed,
        shuffle=True,
        stratify=y_temp,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    X_scaled, scaler = load_and_scale_data(DATA_PATH)
    print("Scaled data shape:", X_scaled.shape)

    X_seq, y_seq = create_clean_and_attacked_sequences(X_scaled, SEQ_LEN)
    print("Sequences:", X_seq.shape, "Labels:", y_seq.shape)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_sequences(
        X_seq, y_seq, 0.8, 0.1, 0.1, seed=RANDOM_SEED
    )

    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    # Save everything as npz
    np.savez_compressed(
        os.path.join(OUT_DIR, f"smartgrid_fdi_seq{SEQ_LEN}.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
    )

    # Optionally save the scaler too (for later use)
    import joblib
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

    print("Saved prepared dataset to", OUT_DIR)

if __name__ == "__main__":
    main()

