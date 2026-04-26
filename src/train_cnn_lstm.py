import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = Path("data/processed/model_dataset.csv")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

target = "hairtail_catch"
feature_cols = [col for col in df.columns if col not in ["date", target]]

data = df[feature_cols + [target]].dropna()

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(data[feature_cols])
y_scaled = target_scaler.fit_transform(data[[target]])

def make_sequences(X, y, time_steps=12):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 12

X_seq, y_seq = make_sequences(X_scaled, y_scaled, time_steps)

split = int(len(X_seq) * 0.8)

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

model = Sequential([
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(filters=32, kernel_size=3, activation="relu"),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

pred_scaled = model.predict(X_test)

pred = target_scaler.inverse_transform(pred_scaled).flatten()
actual = target_scaler.inverse_transform(y_test).flatten()

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2 = r2_score(actual, pred)
mape = np.mean(np.abs((actual - pred) / actual)) * 100

print("CNN-LSTM Result")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R2:", r2)

Path("outputs").mkdir(exist_ok=True)

result = pd.DataFrame({
    "actual": actual,
    "predicted": pred
})

result.to_csv("outputs/cnn_lstm_prediction.csv", index=False, encoding="utf-8-sig")

print(result.head())

## fasdfafdfdf