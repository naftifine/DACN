import pandas as pd
import os

INPUT_FILE = "../data/traffic/traffic_hcm_20251017_1828_mapbox.csv"  # file bạn vừa đưa
OUTPUT_FILE = "../data/traffic/traffic_mapbox_cleaned.csv"

# đọc CSV
df = pd.read_csv(INPUT_FILE)

# các cột trùng ý nghĩa cần bỏ
columns_to_drop = [
    "lengthKm",
    "crossTime",
    "laneCount",
    "surface",
    "currentSpeed",
]

# kiểm tra nếu cột tồn tại trong DataFrame
columns_to_drop = [c for c in columns_to_drop if c in df.columns]

# drop các cột
df_clean = df.drop(columns=columns_to_drop)

# lưu file mới
df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Saved cleaned CSV with {len(df_clean.columns)} columns to {OUTPUT_FILE}")
