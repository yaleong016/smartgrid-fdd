#ensures the prepared data is accurate

import joblib
import pickle

scaler = joblib.load("prepared_data/scaler.joblib")

# print(scaler.data_min_)
# print(scaler.n_features_in_)

import numpy as np

data = np.load("prepared_data/smartgrid_fdi_seq10.npz", allow_pickle=True)
print(data.files)


X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:  ", X_val.shape)
print("y_val shape:  ", y_val.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

# Quick label counts
print("Train label counts:", np.bincount(y_train.astype(int)))
print("Val label counts:  ", np.bincount(y_val.astype(int)))
print("Test label counts: ", np.bincount(y_test.astype(int)))


print(X_train[0])
print(y_train[0])