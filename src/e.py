import pandas as pd

df = pd.read_csv("../data/traffic/data_11071111.csv")

# Chỉ xóa các cột thừa
columns_to_drop = [
    "lengthKm",
    "crossTime",
    "mb_lane_count",
    "mb_surface",
    "mb_start_coords",
    "mb_end_coords",
    "mb_matched_geometry",
    "mb_road_classes",
    "mb_speed_profile",
    "mb_durations_per_step",
    "mb_distances_per_step",
]

df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

df.to_csv("../data/traffic/data_110711112222.csv", index=False)
print(f"Đã xóa {len(columns_to_drop)} cột. Còn lại: {df.shape[1]} cột.")
