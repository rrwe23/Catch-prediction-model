"""
제주 갈치 어획량 예측 - CNN-LSTM 모델
[변경] 갈치 서식 수심대 (1.5~55m) 평균 사용
       근거: 채낚기 + 근해연승 통합 어획 수심대

[CNN-LSTM 구조]
  Input (12, 6, 25, 30)  ← 12개월 시퀀스
    ↓ (각 시점마다 CNN)
  CNN Encoder: Conv → Pool → Conv → Pool → GAP → Vector(64)
    ↓
  12개 벡터 시퀀스 (12, 64)
    ↓
  LSTM (hidden=64)
    ↓
  Dense(64→32→1)
    ↓
  Output (어획량)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
from tkinter import filedialog, messagebox

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 한글 폰트 ────────────────────────────────────
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

# ── 설정 ─────────────────────────────────────────
TRAIN_END = "2020-12-01"
VAL_END   = "2022-12-01"

# ⭐ 갈치 서식 수심대 (1.5~55m)
DEPTH_SLICE  = slice(1, 19)  # depth[0] = 0.49m NaN 제외

SEQ_LENGTH   = 12       # 과거 12개월
CNN_OUT_DIM  = 64
LSTM_HIDDEN  = 64
LSTM_LAYERS  = 1
DROPOUT      = 0.3
BATCH_SIZE   = 8
EPOCHS       = 200
LR           = 0.0005
PATIENCE     = 30
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"학습 디바이스: {DEVICE}")

# ── 파일 선택 ────────────────────────────────────
root = tk.Tk()
root.withdraw()

print("=" * 60)
print("CNN-LSTM 모델 학습 (갈치 서식 수심대 1.5~55m 평균)")
print("=" * 60)

nc_path = filedialog.askopenfilename(
    title="병합 NC 파일 선택 (jeju_merged.nc)",
    filetypes=[("NetCDF files", "*.nc")]
)
if not nc_path:
    print("파일 선택 취소")
    exit()

# ── 데이터 로드 ──────────────────────────────────
import xarray as xr

print(f"\nNC 파일 로드: {os.path.basename(nc_path)}")
ds = xr.open_dataset(nc_path)

# ⭐ 갈치 서식 수심대 (1.5~55m) 평균
print(f"\n[깊이 처리] 갈치 서식 수심대 평균 (1.5~55m)")
print("  근거: 채낚기(표층~50m) + 근해연승(16~64m) 통합")
ds_fishing = ds.isel(depth=DEPTH_SLICE).mean(dim="depth", skipna=True)

env_vars = ["thetao", "so", "uo", "vo", "chl", "o2"]

# 각 시점별로 (channels, lat, lon) 격자 생성
X_full = np.stack([
    ds_fishing[var].values  # (time, lat, lon)
    for var in env_vars
], axis=1)  # (time, channels, lat, lon)

y_full = ds["catch"].values
times  = pd.to_datetime(ds["time"].values)

# 결측치 처리 (변수별 평균값으로 대체)
for i in range(X_full.shape[1]):
    channel  = X_full[:, i, :, :]
    mean_val = np.nanmean(channel)
    X_full[:, i, :, :] = np.where(np.isnan(channel), mean_val, channel)

print(f"  입력 형태: {X_full.shape}  (시간, 채널, 위도, 경도)")

# ── 정규화 (Train 기준) ──────────────────────────
train_mask_initial = times <= TRAIN_END

x_scalers = {}
for i, var in enumerate(env_vars):
    v_min = X_full[train_mask_initial, i].min()
    v_max = X_full[train_mask_initial, i].max()
    rng   = v_max - v_min if v_max != v_min else 1.0
    X_full[:, i] = (X_full[:, i] - v_min) / rng
    x_scalers[var] = {"min": float(v_min), "max": float(v_max)}

y_scaler = MinMaxScaler()
y_scaler.fit(y_full[train_mask_initial].reshape(-1, 1))
y_scaled = y_scaler.transform(y_full.reshape(-1, 1)).flatten()

# ── 시퀀스 생성 ──────────────────────────────────
def make_sequences(X, y, dates, seq_len):
    Xs, ys, ds_list = [], [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
        ds_list.append(dates[i+seq_len])
    return (
        np.array(Xs, dtype=np.float32),
        np.array(ys, dtype=np.float32),
        np.array(ds_list),
    )

X_seq, y_seq, date_seq = make_sequences(X_full, y_scaled, times.values, SEQ_LENGTH)
print(f"  시퀀스 형태: {X_seq.shape}")

# ── 분할 ─────────────────────────────────────────
train_idx = date_seq <= np.datetime64(TRAIN_END)
val_idx   = (date_seq > np.datetime64(TRAIN_END)) & (date_seq <= np.datetime64(VAL_END))
test_idx  = date_seq > np.datetime64(VAL_END)

X_train, y_train = X_seq[train_idx], y_seq[train_idx]
X_val,   y_val   = X_seq[val_idx],   y_seq[val_idx]
X_test,  y_test  = X_seq[test_idx],  y_seq[test_idx]
date_test        = date_seq[test_idx]

print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

X_train_t = torch.FloatTensor(X_train).to(DEVICE)
y_train_t = torch.FloatTensor(y_train).to(DEVICE)
X_val_t   = torch.FloatTensor(X_val).to(DEVICE)
y_val_t   = torch.FloatTensor(y_val).to(DEVICE)
X_test_t  = torch.FloatTensor(X_test).to(DEVICE)
y_test_t  = torch.FloatTensor(y_test).to(DEVICE)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True
)

# ── CNN-LSTM 모델 정의 ───────────────────────────
class CNNEncoder(nn.Module):
    """각 시점을 압축하는 CNN 인코더"""
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc(x))
        return x


class CNNLSTMModel(nn.Module):
    def __init__(self, in_channels, cnn_out, lstm_hidden, lstm_layers, dropout):
        super().__init__()
        self.cnn = CNNEncoder(in_channels, cnn_out)
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_features)
        lstm_out = lstm_out[:, -1, :]
        return self.fc(lstm_out).squeeze(-1)


model = CNNLSTMModel(
    in_channels=len(env_vars),
    cnn_out=CNN_OUT_DIM,
    lstm_hidden=LSTM_HIDDEN,
    lstm_layers=LSTM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

print(f"\n[모델 구조]")
print(model)
print(f"\n학습 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# ── 학습 ─────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

best_val_loss = float("inf")
best_state = None
patience_counter = 0
train_losses, val_losses = [], []

print(f"\n{'='*60}\n학습 시작\n{'='*60}")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_train)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")

model.load_state_dict(best_state)

# ── 평가 ─────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred_n = model(X_test_t).cpu().numpy()

y_pred = y_scaler.inverse_transform(y_pred_n.reshape(-1, 1)).flatten()
y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}

metrics = evaluate(y_true, y_pred)

print(f"\n{'='*60}")
print(f"CNN-LSTM 평가 결과 (Test)")
print(f"{'='*60}")
print(f"  RMSE: {metrics['RMSE']:.2f} 톤")
print(f"  MAE:  {metrics['MAE']:.2f} 톤")
print(f"  MAPE: {metrics['MAPE']:.2f} %")
print(f"  R²:   {metrics['R2']:.4f}")

# ── 저장 ─────────────────────────────────────────
save_dir = os.path.join(os.path.dirname(nc_path), "..", "..", "outputs", "cnn_lstm")
os.makedirs(save_dir, exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "in_channels":  len(env_vars),
        "cnn_out":      CNN_OUT_DIM,
        "lstm_hidden":  LSTM_HIDDEN,
        "lstm_layers":  LSTM_LAYERS,
        "dropout":      DROPOUT,
        "seq_length":   SEQ_LENGTH,
        "depth_slice":  "slice(1, 19) = 1.5~55m",
    },
    "x_scalers": x_scalers,
    "y_scaler": {
        "min": float(y_scaler.data_min_[0]),
        "max": float(y_scaler.data_max_[0]),
    },
}, os.path.join(save_dir, "cnn_lstm_model.pt"))

metrics_json = {k: float(v) for k, v in metrics.items()}
with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_json, f, indent=2, ensure_ascii=False)

pred_df = pd.DataFrame({
    "date":      pd.to_datetime(date_test),
    "actual":    y_true,
    "predicted": y_pred,
})
pred_df.to_csv(os.path.join(save_dir, "predictions.csv"),
               index=False, encoding="utf-8-sig")

# ── 시각화 ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={"height_ratios": [1.2, 1]})

ax = axes[0, 0]
ax.plot(pd.to_datetime(date_test), y_true, "o-", color="#0D2444",
        label="실제값", linewidth=2, markersize=5)
ax.plot(pd.to_datetime(date_test), y_pred, "d--", color="#C13C2A",
        label="CNN-LSTM 예측", linewidth=1.8, markersize=5, alpha=0.85)
ax.set_title("CNN-LSTM 예측 결과 (Test)", fontsize=12, fontweight="bold")
ax.set_ylabel("어획량 (톤)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=30)

ax = axes[0, 1]
ax.plot(train_losses, label="Train Loss", color="#185FA5", linewidth=2)
ax.plot(val_losses,   label="Val Loss",   color="#C13C2A", linewidth=2)
ax.set_title("학습 곡선", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 0]
metric_names = ["RMSE", "MAE", "MAPE"]
vals = [metrics[m] for m in metric_names]
bars = ax.bar(metric_names, vals, color=["#0F6E56", "#185FA5", "#EF9F27"], alpha=0.85)
ax.set_title("CNN-LSTM 성능 지표", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3, axis="y")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.02,
            f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")

ax = axes[1, 1]
ax.scatter(y_true, y_pred, alpha=0.65, color="#185FA5", s=50, edgecolor="white")
lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
ax.plot(lim, lim, "k--", linewidth=1.5, alpha=0.6, label="이상 예측선")
ax.set_xlabel("실제 어획량 (톤)")
ax.set_ylabel("예측 어획량 (톤)")
ax.set_title(f"실제 vs 예측 (R² = {metrics['R2']:.3f})", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "cnn_lstm_results.png"), dpi=120, bbox_inches="tight")

print(f"\n[저장 위치]\n  {save_dir}")

plt.show()
messagebox.showinfo(
    "완료",
    f"CNN-LSTM 학습 완료!\n\n"
    f"R²: {metrics['R2']:.4f}\n"
    f"MAPE: {metrics['MAPE']:.2f}%\n"
    f"RMSE: {metrics['RMSE']:.2f}톤"
)