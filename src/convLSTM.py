"""
제주 갈치 어획량 예측 - ConvLSTM 모델

[ConvLSTM이란?]
  Convolutional LSTM - 시공간 통합 학습의 최강 모델
  - LSTM 내부 연산을 모두 Convolution으로 대체
  - 공간 구조를 유지하면서 시계열 학습
  - 해양/기상 예측 분야의 State-of-the-Art

[CNN-LSTM과의 차이]
  CNN-LSTM:
    각 시점 → CNN으로 압축(1D 벡터) → LSTM
    문제: 공간 구조 손실
    
  ConvLSTM:
    공간 구조 유지하면서 시간 학습
    Gate 연산 자체가 Convolution
    문제 없음 → 더 강력

[참고 논문]
  Shi et al. (2015) "Convolutional LSTM Network"
  → 강수량 예측, 해양 SST 예측에 표준
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

DEPTH_LAYERS   = 10
SEQ_LENGTH     = 12
HIDDEN_CHANNELS = 32     # ConvLSTM 은닉 채널
KERNEL_SIZE    = 3       # Conv 커널 크기
NUM_LAYERS     = 2       # ConvLSTM 층 수
DROPOUT        = 0.3
BATCH_SIZE     = 4       # ConvLSTM은 메모리 많이 씀
EPOCHS         = 200
LR             = 0.0005
PATIENCE       = 30
SEED           = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"학습 디바이스: {DEVICE}")

# ── 파일 선택 ────────────────────────────────────
root = tk.Tk()
root.withdraw()

print("=" * 60)
print("ConvLSTM 모델 학습 (시공간 통합)")
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
ds = ds.isel(depth=slice(0, DEPTH_LAYERS))

env_vars = ["thetao", "so", "uo", "vo", "chl", "o2"]

X_full = np.stack([
    ds[var].mean(dim="depth", skipna=True).values
    for var in env_vars
], axis=1)  # (time, channels, lat, lon)

y_full = ds["catch"].values
times  = pd.to_datetime(ds["time"].values)

# 결측치 처리
for i in range(X_full.shape[1]):
    channel  = X_full[:, i, :, :]
    mean_val = np.nanmean(channel)
    X_full[:, i, :, :] = np.where(np.isnan(channel), mean_val, channel)

print(f"  원본 형태: {X_full.shape}")

# 정규화
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

# 시퀀스 생성
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

# 분할
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


# ── ConvLSTM Cell 정의 ───────────────────────────
class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 단일 셀
    - 일반 LSTM의 행렬곱을 Conv 연산으로 교체
    - 공간 구조를 유지하면서 시간 정보 학습
    """
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        # 4개 gate (i, f, g, o)를 한 번에 계산
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h_prev, c_prev):
        # x: (B, C_in, H, W), h_prev/c_prev: (B, C_hidden, H, W)
        combined = torch.cat([x, h_prev], dim=1)  # 채널 방향 결합
        gates = self.conv(combined)

        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        g = torch.tanh(g)      # candidate
        o = torch.sigmoid(o)   # output gate

        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new

    def init_hidden(self, batch_size, height, width, device):
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """다층 ConvLSTM"""
    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        layers = []
        for i in range(num_layers):
            ic = in_channels if i == 0 else hidden_channels
            layers.append(ConvLSTMCell(ic, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(layers)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # 각 레이어의 hidden state 초기화
        h, c = [], []
        for cell in self.cells:
            h_l, c_l = cell.init_hidden(B, H, W, x.device)
            h.append(h_l); c.append(c_l)

        # 시간 순회
        for t in range(T):
            x_t = x[:, t]
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell(x_t, h[l], c[l])
                x_t = h[l]

        # 마지막 시점의 마지막 레이어 hidden state 반환
        return h[-1]  # (B, hidden_channels, H, W)


class ConvLSTMModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size,
                 num_layers, dropout):
        super().__init__()
        self.conv_lstm = ConvLSTM(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
        )

        # 공간 정보를 단일 값으로 압축
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        out = self.conv_lstm(x)            # (B, hidden, H, W)
        out = self.gap(out).flatten(1)      # (B, hidden)
        return self.fc(out).squeeze(-1)


model = ConvLSTMModel(
    in_channels=len(env_vars),
    hidden_channels=HIDDEN_CHANNELS,
    kernel_size=KERNEL_SIZE,
    num_layers=NUM_LAYERS,
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

print(f"\n{'='*60}\n학습 시작 (최대 {EPOCHS} epochs)\n{'='*60}")
print("⚠ ConvLSTM은 학습이 오래 걸려요 (CPU 기준 10~20분)\n")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        # Gradient clipping (안정성)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    if (epoch + 1) % 5 == 0:
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
print(f"ConvLSTM 평가 결과 (Test)")
print(f"{'='*60}")
print(f"  RMSE: {metrics['RMSE']:.2f} 톤")
print(f"  MAE:  {metrics['MAE']:.2f} 톤")
print(f"  MAPE: {metrics['MAPE']:.2f} %")
print(f"  R²:   {metrics['R2']:.4f}")

# ── 전체 모델 비교 ───────────────────────────────
print(f"\n[전체 모델 비교]")
print(f"  베이스라인 XGBoost:   R² ~0.76,  MAPE ~47%")
print(f"  LSTM     (시간만):   R² ~0.55,  MAPE ~55%")
print(f"  CNN      (공간만):   R² ~0.67,  MAPE ~39%")
print(f"  CNN-LSTM (시공간):   R² ~0.74,  MAPE ~40%")
print(f"  ConvLSTM (시공통합): R² {metrics['R2']:.3f},  MAPE {metrics['MAPE']:.1f}%")

# ── 저장 ─────────────────────────────────────────
save_dir = os.path.join(os.path.dirname(nc_path), "..", "..", "outputs", "convlstm")
os.makedirs(save_dir, exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "in_channels":     len(env_vars),
        "hidden_channels": HIDDEN_CHANNELS,
        "kernel_size":     KERNEL_SIZE,
        "num_layers":      NUM_LAYERS,
        "dropout":         DROPOUT,
        "seq_length":      SEQ_LENGTH,
    },
    "x_scalers": x_scalers,
    "y_scaler": {
        "min": float(y_scaler.data_min_[0]),
        "max": float(y_scaler.data_max_[0]),
    },
}, os.path.join(save_dir, "convlstm_model.pt"))

# JSON 저장 시 numpy float 변환
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
ax.plot(pd.to_datetime(date_test), y_true, "o-", color="#0D2444",
        label="실제값", linewidth=2, markersize=5)
ax.plot(pd.to_datetime(date_test), y_pred, "d--", color="#C13C2A",
        label="ConvLSTM 예측", linewidth=1.8, markersize=5, alpha=0.85)
ax.set_title("ConvLSTM 예측 결과 (Test: 2023~2025)", fontsize=12, fontweight="bold")
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
ax.set_title("ConvLSTM 성능 지표", fontsize=12, fontweight="bold")
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
plt.savefig(os.path.join(save_dir, "convlstm_results.png"),
            dpi=120, bbox_inches="tight")

print(f"\n[저장 위치]")
print(f"  {save_dir}")

plt.show()
messagebox.showinfo(
    "완료",
    f"ConvLSTM 학습 완료!\n\n"
    f"R²: {metrics['R2']:.4f}\n"
    f"MAPE: {metrics['MAPE']:.2f}%\n"
    f"RMSE: {metrics['RMSE']:.2f}톤"
)