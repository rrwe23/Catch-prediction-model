"""
제주 갈치 어획량 예측 모델 - 데이터 전처리 스크립트

[입력]
  - jeju_phy.nc                : 수온, 염분, 해류 (0.083° 격자)
  - jeju_bgc.nc                : 클로로필, 용존산소 (0.25° 격자)
  - jeju_hairtail_catch.csv    : 갈치 월별 어획량

[처리]
  1. PHY와 BGC 격자 통일 (BGC를 PHY 격자로 보간)
  2. 두 NC 파일 병합
  3. 어획량 CSV를 time 차원으로 추가
  4. 단일 NC 파일로 저장

[출력]
  - jeju_merged.nc : CNN 학습용 통합 파일
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# ── 파일 선택 ────────────────────────────────────
root = tk.Tk()
root.withdraw()

print("[1/3] PHY 파일 선택")
phy_path = filedialog.askopenfilename(
    title="PHY NC 파일 선택 (수온·염분·해류)",
    filetypes=[("NetCDF files", "*.nc")]
)
if not phy_path:
    print("PHY 파일을 선택하지 않았습니다.")
    exit()

print("[2/3] BGC 파일 선택")
bgc_path = filedialog.askopenfilename(
    title="BGC NC 파일 선택 (클로로필·용존산소)",
    filetypes=[("NetCDF files", "*.nc")]
)
if not bgc_path:
    print("BGC 파일을 선택하지 않았습니다.")
    exit()

print("[3/3] 어획량 CSV 파일 선택")
csv_path = filedialog.askopenfilename(
    title="어획량 CSV 파일 선택",
    filetypes=[("CSV files", "*.csv")]
)
if not csv_path:
    print("CSV 파일을 선택하지 않았습니다.")
    exit()

# ── NC 파일 로드 ─────────────────────────────────
print("\n[데이터 로드 및 처리 중...]")
ds_phy = xr.open_dataset(phy_path)
ds_bgc = xr.open_dataset(bgc_path)

print(f"  PHY: {dict(ds_phy.sizes)}")
print(f"  BGC: {dict(ds_bgc.sizes)}")

# ── BGC를 PHY 격자에 맞게 보간 ───────────────────
print("  BGC를 PHY 격자로 보간 중...")
ds_bgc_interp = ds_bgc.interp(
    latitude=ds_phy.latitude,
    longitude=ds_phy.longitude,
    depth=ds_phy.depth,
    method="linear",
)
print(f"  BGC 보간 완료: {dict(ds_bgc_interp.sizes)}")

# ── PHY + BGC 병합 ───────────────────────────────
print("  PHY + BGC 병합 중...")
ds_env = xr.merge([ds_phy, ds_bgc_interp])
print(f"  병합 완료. 변수: {list(ds_env.data_vars)}")

# ── 어획량 CSV 처리 ──────────────────────────────
print("  어획량 데이터 통합 중...")
df = pd.read_csv(csv_path, encoding="utf-8-sig")

# "2016년 1월" → "2016-01-01" 형식으로 변환
def parse_korean_date(s):
    s = s.replace("년 ", "-").replace("월", "")
    year, month = s.split("-")
    return pd.Timestamp(year=int(year), month=int(month), day=1)

df["time"] = df["기간"].apply(parse_korean_date)
df = df[["time", "어획량(톤)"]].rename(columns={"어획량(톤)": "catch"})
df = df.set_index("time").sort_index()

# xarray DataArray로 변환
catch_da = xr.DataArray(
    df["catch"].values,
    coords={"time": df.index.values},
    dims=["time"],
    name="catch",
    attrs={
        "units": "ton",
        "long_name": "Jeju Hairtail Monthly Catch",
        "source": "KOSIS 어업생산동향조사",
    },
)

# 시간축 정렬 (NC와 CSV 시간 길이 맞추기)
common_time = np.intersect1d(ds_env.time.values, catch_da.time.values)
print(f"  공통 시간 포인트: {len(common_time)}개")

ds_env = ds_env.sel(time=common_time)
catch_da = catch_da.sel(time=common_time)

# 어획량을 데이터셋에 추가
ds_final = ds_env.assign(catch=catch_da)

# ── 메타데이터 추가 ──────────────────────────────
ds_final.attrs.update({
    "title": "Jeju Hairtail CNN Model Input Data",
    "description": "Merged ocean environmental data (PHY + BGC) with monthly catch",
    "source_phy":  os.path.basename(phy_path),
    "source_bgc":  os.path.basename(bgc_path),
    "source_catch": os.path.basename(csv_path),
    "spatial_range": "32.0~34.0°N, 125.0~127.5°E (제주 갈치 어장)",
    "temporal_range": f"{str(common_time[0])[:10]} ~ {str(common_time[-1])[:10]}",
    "preprocessing": "BGC interpolated to PHY grid, monthly catch added as time-series",
})

# ── 저장 ─────────────────────────────────────────
print("\nNC 파일 저장 중...")
save_dir = os.path.dirname(os.path.abspath(phy_path))
save_path = os.path.join(save_dir, "jeju_merged.nc")

# 압축 옵션 (용량 줄이기)
encoding = {var: {"zlib": True, "complevel": 4} for var in ds_final.data_vars}
ds_final.to_netcdf(save_path, encoding=encoding)

# ── 결과 출력 ────────────────────────────────────
print(f"\n저장 완료: {save_path}")
print(f"파일 크기: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
print("\n[최종 데이터셋 구조]")
print(ds_final)

print("\n[변수 요약]")
for var in ds_final.data_vars:
    da = ds_final[var]
    valid = da.values[~np.isnan(da.values)]
    if len(valid) > 0:
        print(f"  {var:10s}: shape={da.shape}, min={valid.min():.4f}, max={valid.max():.4f}, mean={valid.mean():.4f}")

messagebox.showinfo("완료", f"전처리 완료!\n\n{save_path}")