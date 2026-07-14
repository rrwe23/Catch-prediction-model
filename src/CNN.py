"""
제주 갈치 어획량 예측 - CNN 모델

[CNN이란?]
  Convolutional Neural Network - 공간 패턴 학습 (이미지/지도 분석)
  각 시점의 25×30 격자에서 공간적 패턴을 추출해 어획량 예측

[입력 형태]
  (배치, 6채널, 25위도, 30경도)
  → 환경변수 6개를 RGB처럼 채널로 사용
  → 위경도 격자는 이미지의 H×W처럼 처리

[모델 구조]
  Input (6, 25, 30)
    ↓
  Conv2d(6→32) + BatchNorm + ReLU + MaxPool
    ↓
  Conv2d(32→64) + BatchNorm + ReLU + MaxPool
    ↓
  Conv2d(64→128) + BatchNorm + ReLU
    ↓
  Global Average Pooling
    ↓
  Dense(128→64→1)
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

DEPTH_LAYERS = 10     # 표층~30m
BATCH_SIZE   = 16
EPOCHS       = 200
LR           = 0.001
DROPOUT      = 0.3
SEED         = 42
PATIENCE     = 30

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"학습 디바이스: {DEVICE}")

# ── 파일 선택 ────────────────────────────────────
root = tk.Tk()
root.withdraw()

print("=" * 60)
print("CNN 모델 학습 (공간 격자 활용)")
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

# 깊이 슬라이스 후 깊이 평균 (격자 정보 유지)
ds = ds.isel(depth=slice(0, DEPTH_LAYERS))

env_vars = ["thetao", "so", "uo", "vo", "chl", "o2"]

# (time, depth, lat, lon) → (time, lat, lon) - 깊이 평균
X_full = np.stack([
    ds[var].mean(dim="depth", skipna=True).values
    for var in env_vars
], axis=1)  # (time, channels, lat, lon)

y_full = ds["catch"].values
times  = pd.to_datetime(ds["time"].values)

print(f"  입력 형태: {X_full.shape}  (time, channels, lat, lon)")
print(f"  출력 형태: {y_full.shape}")
print(f"  기간: {times.min().strftime('%Y-%m')} ~ {times.max().strftime('%Y-%m')}")

# ── 결측치 처리 ──────────────────────────────────
# NaN을 채널별 평균으로 채우기
for i in range(X_full.shape[1]):
    channel = X_full[:, i, :, :]
    mean_val = np.nanmean(channel)
    X_full[:, i, :, :] = np.where(np.isnan(channel), mean_val, channel)

print(f"  결측치 처리 완료")

# ── 시간 기준 분할 ───────────────────────────────
train_mask = times <= TRAIN_END
val_mask   = (times > TRAIN_END) & (times <= VAL_END)
test_mask  = times > VAL_END

X_train = X_full[train_mask]
y_train = y_full[train_mask]
X_val   = X_full[val_mask]
y_val   = y_full[val_mask]
X_test  = X_full[test_mask]
y_test  = y_full[test_mask]
date_test = times[test_mask]

print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ── 정규화 (Train 기준, 채널별) ──────────────────
x_scalers = {}
X_train_n = np.zeros_like(X_train, dtype=np.float32)
X_val_n   = np.zeros_like(X_val,   dtype=np.float32)
X_test_n  = np.zeros_like(X_test,  dtype=np.float32)

for i, var in enumerate(env_vars):
    v_min = X_train[:, i].min()
    v_max = X_train[:, i].max()
    rng = v_max - v_min if v_max != v_min else 1.0

    X_train_n[:, i] = (X_train[:, i] - v_min) / rng
    X_val_n[:, i]   = (X_val[:, i]   - v_min) / rng
    X_test_n[:, i]  = (X_test[:, i]  - v_min) / rng
    x_scalers[var] = {"min": float(v_min), "max": float(v_max)}

# y 정규화
y_scaler = MinMaxScaler()
y_train_n = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
y_val_n   = y_scaler.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)
y_test_n  = y_scaler.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)

print(f"  정규화 완료 (Train 기준)")

# Tensor 변환
X_train_t = torch.FloatTensor(X_train_n).to(DEVICE)
y_train_t = torch.FloatTensor(y_train_n).to(DEVICE)
X_val_t   = torch.FloatTensor(X_val_n).to(DEVICE)
y_val_t   = torch.FloatTensor(y_val_n).to(DEVICE)
X_test_t  = torch.FloatTensor(X_test_n).to(DEVICE)
y_test_t  = torch.FloatTensor(y_test_n).to(DEVICE)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True
)

# ── CNN 모델 정의 ────────────────────────────────
class CNNModel(nn.Module):
    def __init__(self, in_channels, dropout=0.3):
        super().__init__()

        # Block 1: 6 → 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Block 3: 64 → 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.gap  = nn.AdaptiveAvgPool2d(1)  # Global Average Pool

        # FC
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 6, 25, 30)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # (B, 32, 12, 15)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # (B, 64, 6, 7)
        x = F.relu(self.bn3(self.conv3(x)))              # (B, 128, 6, 7)
        x = self.gap(x).flatten(1)                       # (B, 128)
        x = self.dropout(F.relu(self.fc1(x)))            # (B, 64)
        x = self.fc2(x).squeeze(-1)                      # (B,)
        return x


model = CNNModel(in_channels=len(env_vars), dropout=DROPOUT).to(DEVICE)

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

print(f"\n{'='*60}\n학습 시작 (최대 {EPOCHS} epochs)\n{'='*60}")

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_train_n)

    # Validation
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

# 최적 모델 복원
model.load_state_dict(best_state)

# ── 평가 ─────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred_n = model(X_test_t).cpu().numpy()

y_pred = y_scaler.inverse_transform(y_pred_n.reshape(-1, 1)).flatten()
y_true = y_test

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}

metrics = evaluate(y_true, y_pred)

print(f"\n{'='*60}")
print(f"CNN 평가 결과 (Test)")
print(f"{'='*60}")
print(f"  RMSE: {metrics['RMSE']:.2f} 톤")
print(f"  MAE:  {metrics['MAE']:.2f} 톤")
print(f"  MAPE: {metrics['MAPE']:.2f} %")
print(f"  R²:   {metrics['R2']:.4f}")

# ── 베이스라인/LSTM 비교 ─────────────────────────
print(f"\n[모델 비교]")
print(f"  베이스라인 XGBoost:   R² ~0.76, MAPE ~47%")
print(f"  LSTM (시계열 단독):   R² ~0.55, MAPE ~55%")
print(f"  CNN  (공간 단독):    R² {metrics['R2']:.3f}, MAPE {metrics['MAPE']:.1f}%")

# ── 저장 ─────────────────────────────────────────
save_dir = os.path.join(os.path.dirname(nc_path), "..", "..", "outputs", "cnn")
os.makedirs(save_dir, exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "in_channels": len(env_vars),
        "dropout":     DROPOUT,
    },
    "x_scalers": x_scalers,
    "y_scaler": {
        "min": float(y_scaler.data_min_[0]),
        "max": float(y_scaler.data_max_[0]),
    },
}, os.path.join(save_dir, "cnn_model.pt"))

with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

pred_df = pd.DataFrame({
    "date":      date_test,
    "actual":    y_true,
    "predicted": y_pred,
})
pred_df.to_csv(os.path.join(save_dir, "predictions.csv"),
               index=False, encoding="utf-8-sig")

loss_df = pd.DataFrame({
    "epoch":      range(1, len(train_losses) + 1),
    "train_loss": train_losses,
    "val_loss":   val_losses,
})
loss_df.to_csv(os.path.join(save_dir, "training_history.csv"),
               index=False, encoding="utf-8-sig")

# ── 시각화 ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={"height_ratios": [1.2, 1]})

# (1) 예측 비교
ax = axes[0, 0]
ax.plot(date_test, y_true, "o-", color="#0D2444", label="실제값", linewidth=2, markersize=5)
ax.plot(date_test, y_pred, "d--", color="#C13C2A", label="CNN 예측",
        linewidth=1.8, markersize=5, alpha=0.85)
ax.set_title("CNN 예측 결과 (Test: 2023~2025)", fontsize=12, fontweight="bold")
ax.set_ylabel("어획량 (톤)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=30)

# (2) 학습 곡선
ax = axes[0, 1]
ax.plot(train_losses, label="Train Loss", color="#185FA5", linewidth=2)
ax.plot(val_losses,   label="Val Loss",   color="#C13C2A", linewidth=2)
ax.set_title("학습 곡선", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss (정규화)")
ax.legend()
ax.grid(alpha=0.3)

# (3) 메트릭
ax = axes[1, 0]
metric_names = ["RMSE", "MAE", "MAPE"]
vals = [metrics[m] for m in metric_names]
bars = ax.bar(metric_names, vals, color=["#0F6E56", "#185FA5", "#EF9F27"], alpha=0.85)
ax.set_title("CNN 성능 지표", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3, axis="y")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.02,
            f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")

# (4) 산점도
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
plt.savefig(os.path.join(save_dir, "cnn_results.png"),
            dpi=120, bbox_inches="tight")

print(f"\n[저장 위치]")
print(f"  {save_dir}")

plt.show()
messagebox.showinfo(
    "완료",
    f"CNN 학습 완료!\n\n"
    f"R²: {metrics['R2']:.4f}\n"
    f"MAPE: {metrics['MAPE']:.2f}%\n"
    f"RMSE: {metrics['RMSE']:.2f}톤"
)