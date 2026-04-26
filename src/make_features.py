import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed/jeju_hairtail_monthly.csv")
OUT_PATH = Path("data/processed/model_dataset.csv")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

regions = ["north", "south", "east", "west"]

variables = [
    "sst",
    "salinity",
    "do",
    "chla",
    "rainfall",
    "wind_speed",
    "wave_height",
    "nitrate",
    "phosphate",
]

target = "hairtail_catch"

# 변화량 생성
for region in regions:
    for var in variables:
        col = f"{region}_{var}"
        if col in df.columns:
            df[f"{col}_diff"] = df[col].diff()

# 해구별 가중치 계산
region_scores = {}

for region in regions:
    corrs = []

    for var in variables:
        diff_col = f"{region}_{var}_diff"
        if diff_col in df.columns:
            corr = df[diff_col].corr(df[target])
            if pd.notna(corr):
                corrs.append(abs(corr))

    region_scores[region] = np.mean(corrs) if corrs else 0

total = sum(region_scores.values())

region_weights = {
    region: score / total if total != 0 else 1 / len(regions)
    for region, score in region_scores.items()
}

print("Region weights:", region_weights)

# 해구 가중 통합 변수 생성
weighted_cols = []

for var in variables:
    available = [f"{region}_{var}" for region in regions if f"{region}_{var}" in df.columns]

    if not available:
        continue

    weighted_col = f"weighted_{var}"
    df[weighted_col] = 0

    for region in regions:
        col = f"{region}_{var}"
        if col in df.columns:
            df[weighted_col] += df[col] * region_weights[region]

    weighted_cols.append(weighted_col)

# 변수별 변화량
for col in weighted_cols:
    df[f"{col}_diff"] = df[col].diff()

# 변수별 가중치 계산
variable_scores = {}

for col in weighted_cols:
    diff_col = f"{col}_diff"
    corr = df[diff_col].corr(df[target])
    variable_scores[col] = abs(corr) if pd.notna(corr) else 0

total_var_score = sum(variable_scores.values())

variable_weights = {
    col: score / total_var_score if total_var_score != 0 else 1 / len(weighted_cols)
    for col, score in variable_scores.items()
}

print("Variable weights:", variable_weights)

# 변수별 가중치 반영 feature 생성
final_features = []

for col in weighted_cols:
    new_col = f"{col}_vw"
    df[new_col] = df[col] * variable_weights[col]
    final_features.append(new_col)

    diff_col = f"{col}_diff"
    new_diff_col = f"{diff_col}_vw"
    df[new_diff_col] = df[diff_col] * variable_weights[col]
    final_features.append(new_diff_col)

final_df = df[["date"] + final_features + [target]].dropna()

final_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("Saved:", OUT_PATH)
print(final_df.head())