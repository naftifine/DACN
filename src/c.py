import pandas as pd

# Đọc 2 file CSV
weather_df = pd.read_csv("../data/traffic/weather_hcm_1107.csv")
traffic_df = pd.read_csv("../data/traffic/traffic_hcm_1107.csv")

# Merge theo segmentId
merged_df = pd.merge(traffic_df, weather_df, on="segmentId", how="left")

# Xuất ra file CSV mới
merged_df.to_csv("merged.csv", index=False)

print("✅ Đã merge xong! File: merged.csv")
