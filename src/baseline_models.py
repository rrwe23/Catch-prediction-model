"""
제주 갈치 어획량 예측 - 베이스라인 모델 (시차 없음)

[모델]
  1. Naive       - 전월값 그대로
  2. 선형회귀     - 환경변수 → 어획량 1차식
  3. XGBoost     - 트리 기반 머신러닝

[입력]
  - jeju_merged.nc (병합된 NC 파일)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
from tkinter import filedialog, messagebox

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ xgboost가 설치되어 있지 않습니다. 'pip install xgboost' 후 다시 실행하세요.")

# ── 한글 폰트 ────────────────────────────────────
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

# ── 설정 ─────────────────────────────────────────
TRAIN_END = "2020-12-01"
VAL_END   = "2022-12-01"

# ── 파일 선택 ────────────────────────────────────
root = tk.Tk()
root.withdraw()

print("=" * 60)
print("베이스라인 학습 시작")
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
ds_surf = ds.isel(depth=slice(0, 10))

env_vars = ["thetao", "so", "uo", "vo", "chl", "o2"]
df = pd.DataFrame({
    var: ds_surf[var].mean(dim=["depth", "latitude", "longitude"], skipna=True).values
    for var in env_vars
})
df["catch"] = ds["catch"].values
df["date"]  = pd.to_datetime(ds["time"].values)

# 시간 변수
df["month"] = df["date"].dt.month
df["year"]  = df["date"].dt.year
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# 어획량 시차만 추가 (Naive용 + 기본 시계열 정보)
df["catch_lag1"]  = df["catch"].shift(1)
df["catch_lag12"] = df["catch"].shift(12)

# 결측치 처리
for col in env_vars:
    df[col] = df[col].interpolate(method="linear", limit_direction="both")

df = df.dropna().reset_index(drop=True)

print(f"\n[데이터] {len(df)}개월 ({df['date'].min().strftime('%Y-%m')} ~ {df['date'].max().strftime('%Y-%m')})")

# 병합 데이터 저장
save_dir_data = os.path.dirname(nc_path)
df.to_csv(os.path.join(save_dir_data, "merged_dataset.csv"),
          index=False, encoding="utf-8-sig")

# 데이터 분할
train = df[df["date"] <= TRAIN_END].copy()
val   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
test  = df[df["date"] > VAL_END].copy()

print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

train_full = pd.concat([train, val], ignore_index=True)

# 평가 함수
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


# 피처 (시차 없음, 기본 변수만)
feature_cols = env_vars + ["month_sin", "month_cos", "catch_lag1", "catch_lag12"]

print(f"\n[피처: {len(feature_cols)}개]")
for c in feature_cols:
    print(f"  - {c}")

X_train = train_full[feature_cols].values
y_train = train_full["catch"].values
X_test  = test[feature_cols].values
y_test  = test["catch"].values

# ── 1. Naive ─────────────────────────────────────
print(f"\n{'='*60}\n[1/3] Naive\n{'='*60}")
y_pred_naive = test["catch_lag1"].values
m_naive = evaluate(y_test, y_pred_naive)
print(f"  RMSE: {m_naive['RMSE']:.2f}, MAE: {m_naive['MAE']:.2f}, MAPE: {m_naive['MAPE']:.2f}%, R²: {m_naive['R2']:.4f}")

# ── 2. 선형회귀 ──────────────────────────────────
print(f"\n{'='*60}\n[2/3] 선형회귀\n{'='*60}")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
m_lr = evaluate(y_test, y_pred_lr)
print(f"  RMSE: {m_lr['RMSE']:.2f}, MAE: {m_lr['MAE']:.2f}, MAPE: {m_lr['MAPE']:.2f}%, R²: {m_lr['R2']:.4f}")

# ── 3. XGBoost ───────────────────────────────────
print(f"\n{'='*60}\n[3/3] {'XGBoost' if HAS_XGB else 'RandomForest'}\n{'='*60}")

if HAS_XGB:
    model3 = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)
    model_name = "XGBoost"
else:
    model3 = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    model_name = "RandomForest"

model3.fit(X_train, y_train)
y_pred_3 = model3.predict(X_test)
m_3 = evaluate(y_test, y_pred_3)
print(f"  RMSE: {m_3['RMSE']:.2f}, MAE: {m_3['MAE']:.2f}, MAPE: {m_3['MAPE']:.2f}%, R²: {m_3['R2']:.4f}")

print(f"\n  [변수 중요도]")
imp_df = pd.DataFrame({"변수": feature_cols, "중요도": model3.feature_importances_})
imp_df = imp_df.sort_values("중요도", ascending=False)
print(imp_df.to_string(index=False))

# ── 저장 ─────────────────────────────────────────
save_dir = os.path.join(os.path.dirname(nc_path), "..", "..", "outputs", "baseline")
os.makedirs(save_dir, exist_ok=True)

all_m = {"Naive": m_naive, "LinearRegression": m_lr, model_name: m_3}
m_df = pd.DataFrame(all_m).T
m_df.to_csv(os.path.join(save_dir, "metrics.csv"), encoding="utf-8-sig")

with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(all_m, f, indent=2, ensure_ascii=False)

pred_df = pd.DataFrame({
    "date": test["date"].values, "actual": y_test,
    "naive_pred": y_pred_naive, "linreg_pred": y_pred_lr,
    f"{model_name.lower()}_pred": y_pred_3,
})
pred_df.to_csv(os.path.join(save_dir, "predictions.csv"),
               index=False, encoding="utf-8-sig")

imp_df.to_csv(os.path.join(save_dir, "feature_importance.csv"),
              index=False, encoding="utf-8-sig")

# ── 시각화 ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={"height_ratios": [1.2, 1]})

ax = axes[0, 0]
ax.plot(test["date"], y_test,       "o-",  color="#0D2444", label="실제값",   linewidth=2, markersize=5)
ax.plot(test["date"], y_pred_naive,  "s--", color="#888888", label="Naive",       alpha=0.7)
ax.plot(test["date"], y_pred_lr,     "^--", color="#185FA5", label="선형회귀",   alpha=0.85)
ax.plot(test["date"], y_pred_3,      "d--", color="#C13C2A", label=model_name,    alpha=0.85)
ax.set_title("Test 구간 예측 비교 (2023~2025)", fontsize=12, fontweight="bold")
ax.set_ylabel("어획량 (톤)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=30)

ax = axes[0, 1]
metric_names = ["RMSE", "MAE", "MAPE"]
x = np.arange(len(metric_names))
w = 0.25
vn = [m_naive[m] for m in metric_names]
vl = [m_lr[m]    for m in metric_names]
v3 = [m_3[m]     for m in metric_names]
ax.bar(x - w, vn, w, label="Naive",     color="#888888")
ax.bar(x,     vl, w, label="선형회귀",   color="#185FA5")
ax.bar(x + w, v3, w, label=model_name,  color="#C13C2A")
ax.set_xticks(x); ax.set_xticklabels(metric_names)
ax.set_title("모델별 오차 비교", fontsize=12, fontweight="bold")
ax.legend(); ax.grid(alpha=0.3, axis="y")
for i, (a, b, c) in enumerate(zip(vn, vl, v3)):
    ax.text(i-w, a, f"{a:.1f}", ha="center", va="bottom", fontsize=9)
    ax.text(i,   b, f"{b:.1f}", ha="center", va="bottom", fontsize=9)
    ax.text(i+w, c, f"{c:.1f}", ha="center", va="bottom", fontsize=9)

ax = axes[1, 0]
r2_vals = [m_naive["R2"], m_lr["R2"], m_3["R2"]]
bars = ax.bar(["Naive", "선형회귀", model_name], r2_vals,
              color=["#888888", "#185FA5", "#C13C2A"], alpha=0.85)
ax.set_title("R² 점수", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1); ax.grid(alpha=0.3, axis="y")
for bar, v in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}",
            ha="center", fontsize=11, fontweight="bold")

ax = axes[1, 1]
top10 = imp_df.head(10).iloc[::-1]
ax.barh(top10["변수"], top10["중요도"], color="#0F6E56", alpha=0.85)
ax.set_title(f"{model_name} 변수 중요도", fontsize=12, fontweight="bold")
ax.set_xlabel("중요도")
ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=120, bbox_inches="tight")

# ── 최종 ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"베이스라인 학습 완료!")
print(f"{'='*60}")
print(f"\n[성능 비교]")
print(m_df.round(2).to_string())

print(f"\n[저장 위치]")
print(f"  {save_dir}")

plt.show()