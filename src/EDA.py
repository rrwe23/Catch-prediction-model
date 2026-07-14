"""
제주 갈치 어획량 예측 모델 - EDA (탐색적 데이터 분석)
[변경] 갈치 서식 수심대(1.5~55m) 전체 평균 사용
       근거: 채낚기(0~50m) + 근해연승(16~64m) 통합 어획 수심대
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
from tkinter import filedialog
import os

# ── 한글 폰트 설정 (Windows) ─────────────────────
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

# ── 파일 선택 ────────────────────────────────────
root = tk.Tk()
root.withdraw()

print("병합된 NC 파일(jeju_merged.nc)을 선택하세요")
file_path = filedialog.askopenfilename(
    title="jeju_merged.nc 선택",
    filetypes=[("NetCDF files", "*.nc")]
)
if not file_path:
    print("파일을 선택하지 않았습니다.")
    exit()

# ── 데이터 로드 ──────────────────────────────────
ds = xr.open_dataset(file_path)
print("\n[데이터 구조]")
print(ds)

# ── 갈치 서식 수심대 평균 (1.5~55m) ⭐ ──────────
# depth[0] = 0.49m는 CMEMS BGC 원본에서 값 없음 (NaN)
# depth[1:19] = 1.5~55m 범위 = 갈치 채낚기+연승 어획 수심대
print("\n[깊이 처리] 갈치 서식 수심대 평균 (1.5~55m)")
print("  근거: 채낚기(표층~50m) + 근해연승(16~64m) 통합")
ds_fishing = ds.isel(depth=slice(1, 19)).mean(dim="depth", skipna=True)

env_vars = ["thetao", "so", "uo", "vo", "chl", "o2"]
var_labels = {
    "thetao": "수온 (°C)",
    "so":     "염분 (PSU)",
    "uo":     "동서 해류 (m/s)",
    "vo":     "남북 해류 (m/s)",
    "chl":    "클로로필 (mg/m³)",
    "o2":     "용존산소 (mmol/m³)",
}

# 변수별 공간 평균 시계열
df = pd.DataFrame({
    var: ds_fishing[var].mean(dim=["latitude", "longitude"], skipna=True).values
    for var in env_vars
})
df["catch"] = ds["catch"].values
df["time"]  = pd.to_datetime(ds["time"].values)
df["month"] = df["time"].dt.month
df["year"]  = df["time"].dt.year
df = df.set_index("time")

print(f"\n[데이터 요약]")
print(f"  기간: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")
print(f"  총 {len(df)}개월")

print(f"\n[결측치 점검]")
print(df.isnull().sum())

nan_rows = df[df[env_vars].isnull().any(axis=1)]
if len(nan_rows) > 0:
    print(f"\n결측치 {len(nan_rows)}개월 → 선형 보간")
    df[env_vars] = df[env_vars].interpolate(method="linear", limit_direction="both")

print(f"\n[변수별 통계]")
print(df[env_vars + ["catch"]].describe().round(3))

# ── 저장 폴더 ────────────────────────────────────
save_dir = os.path.join(
    os.path.dirname(os.path.abspath(file_path)),
    "..", "..", "outputs", "eda"
)
os.makedirs(save_dir, exist_ok=True)

def safe_hist(ax, values, **kwargs):
    clean = values[~np.isnan(values)]
    if len(clean) > 0:
        ax.hist(clean, **kwargs)

period_str = f"{df.index[0].strftime('%Y')}~{df.index[-1].strftime('%Y')}"

# ── 1. 어획량 시계열 ─────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df.index, df["catch"], color="#185FA5", linewidth=1.5, marker="o", markersize=3)
ax.set_title(f"제주 갈치 월별 어획량 ({period_str})", fontsize=13, fontweight="bold")
ax.set_xlabel("연도")
ax.set_ylabel("어획량 (톤)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "01_catch_timeseries.png"), dpi=120)
print(f"\n저장: 01_catch_timeseries.png")

# ── 2. 환경변수 시계열 ──────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 9))
for ax, var in zip(axes.flatten(), env_vars):
    ax.plot(df.index, df[var], color="#0F6E56", linewidth=1.2)
    ax.set_title(var_labels[var], fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)
plt.suptitle("환경변수 시계열 (갈치 서식 수심대 1.5~55m 평균)",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "02_env_timeseries.png"), dpi=120)
print(f"저장: 02_env_timeseries.png")

# ── 3. 변수별 분포 ──────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
for ax, var in zip(axes.flatten(), env_vars):
    safe_hist(ax, df[var].values, bins=30, color="#378ADD", edgecolor="white", alpha=0.85)
    ax.set_title(var_labels[var], fontsize=11)
    ax.set_ylabel("빈도")
    ax.grid(alpha=0.3, axis="y")
plt.suptitle("환경변수 분포", fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "03_distributions.png"), dpi=120)
print(f"저장: 03_distributions.png")

# ── 4. 변수-어획량 상관관계 ─────────────────────
corr = df[env_vars + ["catch"]].corr()
print(f"\n[어획량과의 상관계수]")
catch_corr = corr["catch"].drop("catch").sort_values(key=abs, ascending=False)
print(catch_corr.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center", color="black", fontsize=10)

ax.set_title("변수 간 상관관계", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.045)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "04_correlation.png"), dpi=120)
print(f"저장: 04_correlation.png")

# ── 5. 산점도 ───────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 9))
for ax, var in zip(axes.flatten(), env_vars):
    valid = df[[var, "catch"]].dropna()
    if len(valid) > 0:
        ax.scatter(valid[var], valid["catch"], alpha=0.6, color="#EF9F27", s=30, edgecolor="white")
        z = np.polyfit(valid[var], valid["catch"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[var].min(), valid[var].max(), 50)
        ax.plot(x_line, p(x_line), color="#C13C2A", linewidth=2, linestyle="--")
        r = corr.loc[var, "catch"]
        ax.set_title(f"{var_labels[var]}  (r = {r:.3f})", fontsize=11)
        ax.set_xlabel(var_labels[var])
        ax.set_ylabel("어획량 (톤)")
        ax.grid(alpha=0.3)
plt.suptitle("환경변수 vs 어획량", fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "05_scatter.png"), dpi=120)
print(f"저장: 05_scatter.png")

# ── 6. 월별 박스플롯 ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
data_by_month = [df[df["month"] == m]["catch"].dropna().values for m in range(1, 13)]
bp = ax.boxplot(data_by_month, patch_artist=True,
                boxprops=dict(facecolor="#9FE1CB", edgecolor="#0F6E56"),
                medianprops=dict(color="#C13C2A", linewidth=2))
ax.set_xticklabels([f"{m}월" for m in range(1, 13)])
ax.set_ylabel("어획량 (톤)")
ax.set_title("월별 어획량 분포 (계절성 확인)", fontsize=13, fontweight="bold")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "06_monthly_boxplot.png"), dpi=120)
print(f"저장: 06_monthly_boxplot.png")

# ── 7. 연도별 트렌드 ────────────────────────────
yearly = df.groupby("year")["catch"].agg(["mean", "sum"])
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(yearly.index, yearly["sum"], color="#185FA5", alpha=0.8, edgecolor="white")
ax.set_xlabel("연도")
ax.set_ylabel("연간 총 어획량 (톤)")
ax.set_title("연도별 갈치 어획량 변화 (장기 트렌드)", fontsize=13, fontweight="bold")
ax.grid(alpha=0.3, axis="y")
z = np.polyfit(yearly.index, yearly["sum"], 1)
p = np.poly1d(z)
ax.plot(yearly.index, p(yearly.index), color="#C13C2A", linewidth=2, linestyle="--", label="추세선")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "07_yearly_trend.png"), dpi=120)
print(f"저장: 07_yearly_trend.png")

# ── 결과 요약 ───────────────────────────────────
print(f"\n{'='*50}")
print(f"EDA 완료! 저장 위치: {save_dir}")
print(f"{'='*50}")
print(f"\n[핵심 인사이트]")
print(f"  - 분석 기간: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")
print(f"  - 깊이 처리: 갈치 서식 수심대 1.5~55m 평균")
print(f"  - 어획량 평균: {df['catch'].mean():.1f}톤 / 월")
print(f"  - 어획량 변동: 최저 {df['catch'].min():.1f}톤 ~ 최고 {df['catch'].max():.1f}톤")
print(f"  - 가장 관련 큰 변수: {catch_corr.index[0]} (r={catch_corr.iloc[0]:.3f})")

plt.show()