# 🐟 Catch Prediction Model

> 제주도 갈치 어획량 변화 예측 CNN 딥러닝 모델

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green?style=flat-square) ![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square) ![Data](https://img.shields.io/badge/Data-CMEMS%20%7C%20KOSIS-blue?style=flat-square)

------

## 📌 개요

지구온난화로 인한 해양 환경 변화가 제주도 갈치 어획량에 미치는 영향을 분석하고,
 해양 환경 데이터를 입력으로 하는 **CNN(Convolutional Neural Network) 기반 어획량 예측 모델**을 구축하고자 한다.

제주도는 국내 최대 갈치 어장으로, 수온 변화에 민감하게 반응하는 갈치의 특성상
 기후변화에 따른 어획량 변동이 두드러지게 나타나는 지역이다.

------

## 🎯 목적

- 해양 환경 변수(수온·염분·해류·클로로필·용존산소)와 갈치 어획량 간의 **공간적 상관관계 분석**
- CNN 모델을 활용한 **월별 갈치 어획량 예측**
- 지구온난화가 제주 갈치 어장에 미치는 **장기적 영향 시각화**

------

## 🗺️ 분석 해역

제주도 갈치 주어장을 기준으로 아래 범위를 분석 해역으로 설정하였다. 기준은 KOSIS 어업생산동향조사 기준을 참고하였으며 HTML로 나타냈다.

| 구분 | 범위                                      |
| ---- | ----------------------------------------- |
| 위도 | 32.0°N ~ 34.0°N                           |
| 경도 | 125.0°E ~ 127.5°E                         |
| 면적 | 약 53,280 km²                             |
| 기준 | 갈치 주어장 (KOSIS 어업생산동향조사 기준) |

```
        34.0°N ┌─────────────────────┐
               │                     │
               │    제주 갈치 어장    │
               │       ⊙ 제주도      │
               │                     │
        32.0°N └─────────────────────┘
             125.0°E             127.5°E
```

------



![image-20260428204645594](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20260428204645594.png)











## 🗂️ 데이터

### 해양 환경 데이터 (CMEMS)

- **출처:** [Copernicus Marine Service](https://marine.copernicus.eu/)
- **기간:** 2016년 1월 ~ 2025년 12월 (10년, 월별)
- **해상도:** 0.083° (약 9km) 격자

| 변수명   | 설명                                          | 단위    | 데이터셋                     |
| -------- | --------------------------------------------- | ------- | ---------------------------- |
| `thetao` | 수온 (Sea Water Potential Temperature)        | °C      | GLOBAL_MULTIYEAR_PHY_001_030 |
| `so`     | 염분 (Sea Water Salinity)                     | PSU     | GLOBAL_MULTIYEAR_PHY_001_030 |
| `uo`     | 동서 해류 속도 (Eastward Sea Water Velocity)  | m/s     | GLOBAL_MULTIYEAR_PHY_001_030 |
| `vo`     | 남북 해류 속도 (Northward Sea Water Velocity) | m/s     | GLOBAL_MULTIYEAR_PHY_001_030 |
| `chl`    | 클로로필 a (Chlorophyll-a)                    | mg/m³   | GLOBAL_MULTIYEAR_BGC_001_029 |
| `o2`     | 용존산소 (Dissolved Oxygen)                   | mmol/m³ | GLOBAL_MULTIYEAR_BGC_001_029 |

### 어획량 데이터 (KOSIS)

- **출처:** [국가통계포털 KOSIS](https://kosis.kr/) — 어업생산동향조사
- **기간:** 2016년 1월 ~ 2025년 12월 (월별)
- **지역:** 제주도
- **어종:** 갈치 단일 어종
- **단위:** 톤(ton)

------

## ⚙️ 환경 변수 선정 근거

| 변수               | 선정 근거                                                    |
| ------------------ | ------------------------------------------------------------ |
| 수온 (`thetao`)    | 수심 20m 수온 21~23℃에서 어획량 증가, 27℃ 이상에서 어장 분산·감소 (국립수산과학원, 2025) |
| 염분 (`so`)        | 어업 자원량의 주요 환경 드라이버로 확인                      |
| 해류 (`uo`, `vo`)  | 갈치 회유 경로 및 어장 형성에 직접적 영향                    |
| 클로로필 a (`chl`) | 먹이사슬 하단 지표, 갈치 먹잇감(멸치 등) 분포에 영향         |
| 용존산소 (`o2`)    | 저층 용존산소 농도가 어획량 주요 환경 드라이버로 확인        |

------

## 🛠️ 준비물 (Requirements)

### Python 버전

```
Python 3.12+
```

### 패키지

```bash
pip install xarray netCDF4 numpy pandas matplotlib cartopy
pip install openpyxl
pip install copernicusmarine
pip install torch torchvision  # 또는 tensorflow
```

------

## 📁 프로젝트 구조

```
Catch-prediction-model/
│
├── data/
│   ├── raw/
│   │   ├── jeju_phy.nc          # CMEMS 물리 데이터 (수온·염분·해류)
│   │   └── jeju_bgc.nc          # CMEMS 생지화학 데이터 (클로로필·용존산소)
│   ├── processed/
│   │   └── jeju_zone.xlsx       # 전처리된 좌표별 환경 데이터
│   └── kosis/
│       └── hairtail_catch.csv   # 갈치 월별 어획량 데이터
│
├── src/
│   ├── open_nc.py               # NC 파일 열기 및 시각화
│   └── nc_to_excel.py           # NC → 엑셀 변환 (제주 해역 필터링)
│
├── notebooks/
│   └── eda.ipynb                # 탐색적 데이터 분석 (예정)
│
├── model/
│   └── cnn_model.py             # CNN 모델 정의 (예정)
│
├── requirements.txt
└── README.md
```

------

## 🔄 진행 상황

- [✅] 프로젝트 방향 설정 (어종: 갈치, 지역: 제주)

- [✅] 분석 해역 범위 확정 (32.0~34.0°N / 125.0~127.5°E)

- [✅] CMEMS 환경 변수 선정 (6개 변수)

  - 

  - | 변수               | 선정 근거                                                    |
    | ------------------ | ------------------------------------------------------------ |
    | 수온 (`thetao`)    | 수심 20m 수온 21~23℃에서 어획량 증가, 27℃ 이상에서 어장 분산·감소 (국립수산과학원, 2025) |
    | 염분 (`so`)        | 어업 자원량의 주요 환경 드라이버로 확인                      |
    | 해류 (`uo`, `vo`)  | 갈치 회유 경로 및 어장 형성에 직접적 영향                    |
    | 클로로필 a (`chl`) | 먹이사슬 하단 지표, 갈치 먹잇감(멸치 등) 분포에 영향         |
    | 용존산소 (`o2`)    | 저층 용존산소 농도가 어획량 주요 환경 드라이버로 확인        |

- [✅] NC 파일 열기 및 시각화 코드 작성

  - tkinter 라이브러리를 통해 파일을 첨부하는 방식
  - xarray  라이브러리를 통해 구조 확인 및 시각화
  - 해당 지점 데이터를 엑셀화하여 출력

  <img src="C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20260428212207874.png" alt="image-20260428212207874" style="zoom:50%;" />

- [✅] NC 파일 → 엑셀 변환 코드 작성 (제주 해역 필터링 포함)

- [❌] 획득한 데이터 시각화

- [❌] KOSIS 갈치 어획량 데이터 수집

- [❌ ] 탐색적 데이터 분석 (EDA)

- [❌] 데이터 전처리 및 정규화

- [❌ ] CNN 모델 설계 및 학습

- [❌] 모델 평가 및 시각화

------

## 📅 TODO

```
1.  KOSIS에서 갈치 월별 어획량 CSV 다운로드
2.  두 데이터 시계열 병합 및 전처리
3.  어획량 데이터와 환경 데이터 변화에 따른 연관성 및 변수 획득
4.  CNN 입력 형태로 데이터 재구성 (공간 격자 → 텐서)
5.  모델 학습 및 검증
```



------

## 📬 참고 자료

- [Copernicus Marine Service](https://marine.copernicus.eu/)
- [국가통계포털 KOSIS 어업생산동향조사](https://kosis.kr/)
- 국립수산과학원 (2025). 고수온에 갈치 어장 약화 연구
- 
