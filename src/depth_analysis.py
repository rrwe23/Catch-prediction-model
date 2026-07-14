"""
제주 갈치 어획량 예측 모델 - 깊이별 EDA 비교 분석

[목적]
  갈치가 실제로 회유하는 수심대를 학술적 근거로 결정
  6개 환경변수 × 5개 수심대 = 30개 상관계수 비교

[수심대 구분 (갈치 어법 근거)]
  - 표층대   1.5~5m   : 채낚기 야간 유인 (depth[0] NaN 제외)
  - 천층대   6~14m    : 채낚기 + 연승 일부
  - 중층대   16~32m   : 근해연승 주력 ⭐
  - 중하층   38~55m   : 연승 일부
  - 통합    1.5~55m   : 갈치 서식 수심대 전체 (depth[0] 제외)

[출력]
  01_depth_correlation_table.png   : 5×6 히트맵
  02_depth_variable_bars.png       : 변수별 깊이 비교 막대
  03_depth_recommendation.png      : 최적 깊이 시각화
  depth_analysis_results.csv       : 원본 결과 저장
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
from tkinter import filedialog
import os

# ── 한글 폰트 ────────────────────────────────────
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

# ── 파일 선택 ────────────────────────────────────
root = tk.Tk()
root.withdraw()

print("=" * 60)
print("깊이별 EDA 비교 분석")
print("=" * 60)
print("\nNC 파일 선택 (jeju_merged.nc)")

file_path = filedialog.askopenfilename(
    title="병합 NC 파일 선택",
    filetypes=[("NetCDF files", "*.nc")]
)
if not file_path:
    print("파일을 선택하지 않았습니다.")
    exit()

# ── 데이터 로드 ──────────────────────────────────
ds = xr.open_dataset(file_path)

print(f"\n[데이터 정보]")
print(f"  기간: {str(ds.time.values[0])[:10]} ~ {str(ds.time.values[-1])[:10]}")
print(f"  개월: {ds.time.size}")
print(f"  깊이: {ds.depth.size}개  ({float(ds.depth.min()):.2f}m ~ {float(ds.depth.max()):.2f}m)")

# ── 수심대 정의 (갈치 어법 근거) ─────────────────
# 통합 구역을 slice(1, 19)로 변경 (depth[0] NaN 제외)
depth_ranges = {
    "표층대\n(1.5~5m)":  {"slice": slice(1, 5),   "note": "채낚기 야간 유인"},
    "천층대\n(6~14m)":   {"slice": slice(5, 10),  "note": "채낚기+연승"},
    "중층대\n(16~32m)":  {"slice": slice(10, 15), "note": "근해연승 주력"},
    "중하층\n(38~55m)":  {"slice": slice(15, 19), "note": "연승 일부"},
    "통합\n(1.5~55m)":   {"slice": slice(1, 19),  "note": "갈치 서식 수심대"},
}

env_vars = ["thetao", "so", "uo", "vo", "chl", "o2"]
var_labels = {
    "thetao": "수온",
    "so":     "염분",
    "uo":     "동서해류",
    "vo":     "남북해류",
    "chl":    "클로로필",
    "o2":     "용존산소",
}

# ── 상관계수 계산 ────────────────────────────────
print(f"\n[분석 중...]")
results = {}
for name, info in depth_ranges.items():
    ds_slice = ds.isel(depth=info["slice"]).mean(dim="depth", skipna=True)
    df = pd.DataFrame({
        var: ds_slice[var].mean(dim=["latitude", "longitude"], skipna=True).values
        for var in env_vars
    })
    df["catch"] = ds["catch"].values
    corr = df.corr()["catch"].drop("catch")
    results[name] = corr

corr_df = pd.DataFrame(results)

print(f"\n[깊이별 상관계수 표]")
print(corr_df.round(3).to_string())

# ── 저장 폴더 ────────────────────────────────────
save_dir = os.path.join(
    os.path.dirname(os.path.abspath(file_path)),
    "..", "..", "outputs", "depth_analysis"
)
os.makedirs(save_dir, exist_ok=True)

corr_df.to_csv(os.path.join(save_dir, "depth_analysis_results.csv"),
               encoding="utf-8-sig")

# ── 1. 히트맵 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

data_matrix = corr_df.values
im = ax.imshow(data_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(range(len(depth_ranges)))
ax.set_yticks(range(len(env_vars)))
ax.set_xticklabels(list(depth_ranges.keys()), fontsize=10)
ax.set_yticklabels([f"{var}\n({var_labels[var]})" for var in env_vars], fontsize=10)

for i in range(len(env_vars)):
    for j in range(len(depth_ranges)):
        val = data_matrix[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        text = f"{val:.3f}" if not pd.isna(val) else "NaN"
        ax.text(j, i, text, ha="center", va="center",
                color=color, fontsize=10, fontweight="bold")

ax.set_title("수심대별 환경변수와 어획량 상관계수", fontsize=14, fontweight="bold", pad=15)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="상관계수 r")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "01_depth_correlation_table.png"), dpi=120)
print(f"\n저장: 01_depth_correlation_table.png")

# ── 2. 변수별 깊이 비교 (그룹 막대) ──────────────
fig, ax = plt.subplots(figsize=(13, 6))

x = np.arange(len(env_vars))
width = 0.16
colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]

for i, (name, corr) in enumerate(results.items()):
    offset = (i - 2) * width
    values = [corr[var] if not pd.isna(corr[var]) else 0 for var in env_vars]
    bars = ax.bar(x + offset, values, width, label=name.replace("\n", " "),
                  color=colors[i], alpha=0.85, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels([f"{var}\n({var_labels[var]})" for var in env_vars], fontsize=10)
ax.set_ylabel("어획량과 상관계수 r", fontsize=11)
ax.set_title("수심대별 변수 상관계수 비교", fontsize=14, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(0.5, color="green", linewidth=0.5, linestyle="--", alpha=0.5, label="강한 양의 상관 (r=0.5)")
ax.axhline(-0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5, label="강한 음의 상관 (r=-0.5)")
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="upper left", ncol=2, fontsize=9)
ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "02_depth_variable_bars.png"), dpi=120)
print(f"저장: 02_depth_variable_bars.png")

# ── 3. 최적 깊이 시각화 ─────────────────────────
depth_strength = {}
for name, corr in results.items():
    valid = corr.dropna()
    if len(valid) > 0:
        depth_strength[name] = valid.abs().mean()
    else:
        depth_strength[name] = 0

strength_df = pd.DataFrame({
    "수심대": list(depth_strength.keys()),
    "평균 |r|": list(depth_strength.values()),
})
strength_df = strength_df.sort_values("평균 |r|", ascending=False)

fig, ax = plt.subplots(figsize=(11, 6))
bar_colors = ["#0F6E56", "#185FA5", "#EF9F27", "#7F77DD", "#B4B2A9"]
bars = ax.bar(
    range(len(strength_df)),
    strength_df["평균 |r|"].values,
    color=bar_colors,
    alpha=0.85, edgecolor="white", linewidth=2,
)

ax.set_xticks(range(len(strength_df)))
ax.set_xticklabels(strength_df["수심대"], fontsize=10)
ax.set_ylabel("평균 |r| (상관관계 강도)", fontsize=11)
ax.set_title("수심대별 종합 상관 강도 (6개 변수 |r| 평균)",
             fontsize=14, fontweight="bold")
ax.grid(alpha=0.3, axis="y")

for bar, val in zip(bars, strength_df["평균 |r|"].values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

for i, (bar, name) in enumerate(zip(bars, strength_df["수심대"])):
    rank_text = ["1위 ⭐", "2위", "3위", "4위", "5위"][i]
    ax.text(bar.get_x() + bar.get_width()/2, 0.02,
            rank_text, ha="center", fontsize=10,
            color="white", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "03_depth_recommendation.png"), dpi=120)
print(f"저장: 03_depth_recommendation.png")

# ── 결과 요약 ───────────────────────────────────
print(f"\n{'='*60}")
print(f"분석 완료!")
print(f"{'='*60}")
print(f"\n[수심대별 종합 순위 (평균 |r|)]")
for i, (_, row) in enumerate(strength_df.iterrows()):
    marker = " ⭐ 추천" if i == 0 else ""
    name = row["수심대"].replace("\n", " ")
    print(f"  {i+1}위. {name}:  {row['평균 |r|']:.3f}{marker}")

print(f"\n[변수별 최강 상관 수심대]")
for var in env_vars:
    row = corr_df.loc[var].dropna()
    if len(row) > 0:
        best_depth = row.abs().idxmax()
        best_val = row[best_depth]
        print(f"  {var} ({var_labels[var]}): {best_depth.replace(chr(10), ' ')} → r = {best_val:.3f}")

print(f"\n[저장 위치] {save_dir}")

plt.show()