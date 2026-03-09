import pandas as pd

# Đọc file CSV
df = pd.read_csv("../data/traffic/data_1107.csv")

# Danh sách cột cần loại bỏ (GIỮ NGUYÊN NHƯ BẠN VIẾT)
cols_to_drop = [
    "name_x",
    "name_y",
    "weather_date",
    "mb_start_coords",
    "mb_end_coords",
    "mb_matched_geometry",
    "mb_weight_name",
    "weather_date",  # Lặp cũng được, không sao
    "mb_surface",
    "mb_congestion",
    "mb_avg_speed",
    "mb_speed_profile",
    "laneCount",
    "lengthKm",
    "mb_durations_per_step",
    "latitude_lookup",
    "longitude_lookup",
]

# Loại bỏ nếu cột tồn tại
cols_to_drop = [col for col in cols_to_drop if col in df.columns]

# === LOẠI BỎ CỘT TRƯỚC ===
df_clean = df.drop(columns=cols_to_drop)

# === ĐỔI TÊN SAU, TRÊN df_clean (SỬA LỖI) ===
mb_rename = {
    "mb_distance": "dist_m",
    "mb_duration": "duration_sec",
    "mb_weight": "route_weight",
    "mb_geometry": "geometry",
    "mb_distances_per_step": "step_distances_m",
    "mb_confidence": "match_confidence",
    "mb_lane_count": "lane_count",  # Đổi tên
    "mb_maneuvers": "maneuvers",  # Đổi tên
    "mb_road_classes": "turn_directions",
}

df_clean = df_clean.rename(columns=mb_rename)  # SỬA: đổi trên df_clean

# Lưu file mới
df_clean.to_csv("../data/traffic/data_1107_edited.csv", index=False)

print(f"Đã loại bỏ {len(cols_to_drop)} cột: {cols_to_drop}")
print(f"Số cột còn lại: {df_clean.shape[1]}")
print("Đã lưu: ../data/traffic/data_11071111.csv")
