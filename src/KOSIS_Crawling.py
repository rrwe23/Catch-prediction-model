import requests
import pandas as pd
import time
import os
from api_key import KOSIS_API_KEY

BASE_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
ORG_ID   = "101"
TBL_ID   = "DT_1EW0005"

# 확정된 파라미터
OBJ_L1 = "0"        # 어업종류 합계
OBJ_L2 = "110009"   # 갈치
OBJ_L3 = "0000000"  # 어법별 합계
OBJ_L4 = "39"       # 제주도
OBJ_L5 = "0"        # 어업구분 합계

YEARS = [str(y) for y in range(2000, 2026)]


def fetch_year(year: str) -> list:
    params = {
        "method":     "getList",
        "apiKey":     KOSIS_API_KEY,
        "orgId":      ORG_ID,
        "tblId":      TBL_ID,
        "itmId":      "T01",
        "objL1":      OBJ_L1,
        "objL2":      OBJ_L2,
        "objL3":      OBJ_L3,
        "objL4":      OBJ_L4,
        "objL5":      OBJ_L5,
        "objL6":      "",
        "objL7":      "",
        "objL8":      "",
        "format":     "json",
        "jsonVD":     "Y",
        "prdSe":      "M",
        "startPrdDe": f"{year}01",
        "endPrdDe":   f"{year}12",
    }

    r = requests.get(BASE_URL, params=params, timeout=60)

    if r.status_code != 200:
        print(f"  [{year}] HTTP 오류: {r.status_code}")
        return []

    data = r.json()

    if isinstance(data, dict) and "err" in data:
        print(f"  [{year}] API 오류: {data}")
        return []

    print(f"  [{year}] {len(data)}건 수신")
    return data


def parse(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    print("\n[컬럼]", df.columns.tolist())
    print(df.head(3).to_string())

    rename = {}
    if "PRD_DE" in df.columns: rename["PRD_DE"] = "기간"
    if "DT"     in df.columns: rename["DT"]     = "어획량(톤)"
    if "C1_NM"  in df.columns: rename["C1_NM"]  = "어업종류"
    if "C2_NM"  in df.columns: rename["C2_NM"]  = "품종"
    if "C3_NM"  in df.columns: rename["C3_NM"]  = "어법"
    if "C4_NM"  in df.columns: rename["C4_NM"]  = "지역"
    if "C5_NM"  in df.columns: rename["C5_NM"]  = "어업구분"
    if "ITM_NM" in df.columns: rename["ITM_NM"] = "항목"

    df = df.rename(columns=rename)

    keep = [c for c in ["기간", "어업종류", "품종", "어법", "지역", "어업구분", "항목", "어획량(톤)"] if c in df.columns]
    df = df[keep]

    if "기간" in df.columns:
        df["기간"] = pd.to_datetime(df["기간"], format="%Y%m").dt.strftime("%Y-%m")

    if "어획량(톤)" in df.columns:
        df["어획량(톤)"] = pd.to_numeric(df["어획량(톤)"], errors="coerce")

    return df


if __name__ == "__main__":
    all_data = []

    print(f"제주 갈치 월별 어획량 수집 시작 ({YEARS[0]}~{YEARS[-1]})\n")

    for year in YEARS:
        raw = fetch_year(year)
        if raw:
            df_year = parse(raw)
            if not df_year.empty:
                all_data.append(df_year)
        time.sleep(0.5)

    if not all_data:
        print("\n수집된 데이터 없음.")
        exit()

    df_final = pd.concat(all_data, ignore_index=True)
    df_final = df_final.sort_values("기간").reset_index(drop=True)

    print(f"\n[최종 결과] {len(df_final)}행")
    print(df_final.to_string())

    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "kosis"
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "jeju_hairtail_catch.csv")
    df_final.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {save_path}")