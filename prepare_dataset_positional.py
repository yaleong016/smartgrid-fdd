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

SEQ_LEN = 30
ATTACK_DIFFICULTY = 0
# Indices in numeric_df / X_scaled for the features we want to perturb
ATTACK_FEATURE_INDICES = [0, 1, 2, 3, 4, 7, 14]
# 0: Voltage (V)
# 1: Current (A)
# 2: Power Consumption (kW)
# 3: Reactive Power (kVAR)
# 4: Power Factor
# 7: Grid Supply (kW)
# 14: Predicted Load (kW)


DATA_PATH = "data/smart_grid_dataset.csv"
OUT_DIR = "prepared_data"
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_scale_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Sort by time column if present
    time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if time_cols:
        df = df.sort_values(by=time_cols[0])

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


def generate_physicsish_attack(
    context: np.ndarray,
    difficulty: float = ATTACK_DIFFICULTY,
    feature_indices=ATTACK_FEATURE_INDICES
) -> np.ndarray:
    """
    Generate a fake point in [0, 1]^D given a context of shape (seq_len-1, D),
    with difficulty in [0, 1]:

      difficulty = 0.0  -> very obvious attack (extreme spikes to 0 or 1)
      difficulty = 1.0  -> almost identical to the original last point
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    if feature_indices is None:
        feature_indices = ATTACK_FEATURE_INDICES

    # Baseline: last real point in the context
    x_last = context[-1].copy()  # shape (D,)
    D = x_last.shape[0]
    x_fake = x_last.copy()

    # How far we move toward the extreme
    alpha = 1.0 - difficulty
    # Small noise amplitude shrinks as difficulty ↑
    noise_scale = 0.05 * alpha  # 0.05 at diff=0, 0 at diff=1

    for j in feature_indices:
        if j >= D:
            continue

        v = x_last[j]

        # Define an extreme target: flip to opposite side of [0,1]
        if v < 0.5:
            extreme = 1.0
        else:
            extreme = 0.0

        # Interpolate between original v and extreme
        x_fake[j] = (1.0 - alpha) * v + alpha * extreme

    # Optional noise on all features
    if noise_scale > 0:
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=D)
        x_fake = x_fake + noise

    # Clip back to [0,1]
    x_fake = np.clip(x_fake, 0.0, 1.0)
    return x_fake.astype(np.float32)


def create_clean_and_attacked_sequences(X: np.ndarray, seq_len: int):
    """
    Create sequences and positional labels:

      - Clean sequences: label = 0  (no false data)
      - Attacked sequences: label = position of false data (1..seq_len)
    """
    T, D = X.shape
    usable_T = (T // seq_len) * seq_len
    X_trimmed = X[:usable_T]
    windows = X_trimmed.reshape(-1, seq_len, D)  # (W, seq_len, D)

    W = windows.shape[0]
    half_W = W // 2

    # First half: clean windows -> label = 0
    clean_windows = windows[:half_W]

    # Second half: attacked windows -> label = attack position (1..seq_len)
    attacked_windows = []
    attacked_labels = []  # NEW: store attack positions

    for w in windows[half_W: 2 * half_W]:
        pos = np.random.randint(0, seq_len)  # 0-based index inside sequence
        context = np.delete(w, pos, axis=0)  # (seq_len-1, D)
        fake_point = generate_physicsish_attack(context)
        attacked = np.insert(context, pos, fake_point, axis=0)
        attacked_windows.append(attacked)
        attacked_labels.append(pos + 1)  # 1-based position for label

    attacked_windows = np.stack(attacked_windows, axis=0)
    attacked_labels = np.array(attacked_labels, dtype=np.int64)

    X_all = np.concatenate([clean_windows, attacked_windows], axis=0)
    # clean labels = 0, attacked labels = position 1..seq_len
    y_all = np.concatenate([
        np.zeros(half_W, dtype=np.int64),
        attacked_labels
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
    print("Label values (0 = clean, 1..SEQ_LEN = attack position):",
          np.unique(y_seq))

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_sequences(
        X_seq, y_seq, 0.8, 0.1, 0.1, seed=RANDOM_SEED
    )

    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    # Save everything as npz
    np.savez_compressed(
        os.path.join(
            OUT_DIR,
            f"smartgrid_fdi_positional_seq{SEQ_LEN}_diff{ATTACK_DIFFICULTY}.npz"
        ),
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
