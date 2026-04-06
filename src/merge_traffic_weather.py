import pandas as pd
import os
from datetime import datetime

# === CẤU HÌNH ===
# Lấy thư mục gốc của project (thư mục cha của src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATN_DIR = os.path.join(BASE_DIR, 'DATN')

FILE_TIME = '20260406_1143'  # Cập nhật thời gian file traffic và weather mới nhất ở đây

TRAFFIC_FILE = os.path.join(DATN_DIR, f'traffic_hcm_{FILE_TIME}.csv')
WEATHER_FILE = os.path.join(DATN_DIR, 'weather', f'weather_hcm_{FILE_TIME}.csv')

OUTPUT_DIR = os.path.join(DATA_DIR, 'traffic_weather')

def merge_traffic_weather(traffic_path, weather_path, output_dir):
    """Gộp file traffic và weather theo segmentId."""

    print(f"Đọc file traffic: {traffic_path}")
    df_traffic = pd.read_csv(traffic_path)

    print(f"Đọc file weather: {weather_path}")
    df_weather = pd.read_csv(weather_path)

    print(f"Traffic: {df_traffic.shape[0]} dòng, {df_traffic.shape[1]} cột")
    print(f"Weather: {df_weather.shape[0]} dòng, {df_weather.shape[1]} cột")

    # Loại bỏ cột trùng lặp từ weather (giữ lại segmentId để merge)
    common_cols = set(df_traffic.columns) & set(df_weather.columns) - {'segmentId'}
    weather_cols_to_use = [c for c in df_weather.columns if c not in common_cols]
    print(f"Cột chung (bỏ từ weather): {common_cols}")

    # Merge theo segmentId
    df_merged = pd.merge(
        df_traffic,
        df_weather[weather_cols_to_use],
        on='segmentId',
        how='inner'
    )

    print(f"Kết quả merge: {df_merged.shape[0]} dòng, {df_merged.shape[1]} cột")

    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # Lưu file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'traffic_weather_{timestamp}.csv')
    df_merged.to_csv(output_file, index=False)
    print(f"Đã lưu file: {output_file}")

    return df_merged

if __name__ == '__main__':
    df = merge_traffic_weather(TRAFFIC_FILE, WEATHER_FILE, OUTPUT_DIR)
    print("\nCác cột trong file kết quả:")
    for col in df.columns:
        print(f"  - {col}")
