import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_excel(file_name, date_col, value_col, new_col):
    df = pd.read_excel(RAW_DIR / file_name)
    df[date_col] = pd.to_datetime(df[date_col])
    df["date"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("date")[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: new_col})
    )
    return monthly

# 예시: 파일명/컬럼명은 실제 엑셀에 맞게 수정
north_sst = load_excel("north_sst.xlsx", "측정일자", "수온", "north_sst")
south_sst = load_excel("south_sst.xlsx", "측정일자", "수온", "south_sst")
east_sst = load_excel("east_sst.xlsx", "측정일자", "수온", "east_sst")
west_sst = load_excel("west_sst.xlsx", "측정일자", "수온", "west_sst")

catch = pd.read_excel(RAW_DIR / "hairtail_catch.xlsx")
catch["date"] = pd.to_datetime(catch["date"]).dt.to_period("M").dt.to_timestamp()
catch = catch[["date", "hairtail_catch"]]

dfs = [catch, north_sst, south_sst, east_sst, west_sst]

merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on="date", how="outer")

merged = merged.sort_values("date")
merged.to_csv(OUT_DIR / "jeju_hairtail_monthly.csv", index=False, encoding="utf-8-sig")

print("Saved:", OUT_DIR / "jeju_hairtail_monthly.csv")
print(merged.head())
## dfdfdf