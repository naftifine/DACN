import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'traffic_weather')

def merge_all_traffic_weather():
    """Gộp tất cả file CSV trong thư mục traffic_weather thành 1 file traffic_weather_latest.csv."""
    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith('.csv') and f != 'traffic_weather_latest.csv'
    ])

    if not files:
        print("Không tìm thấy file CSV nào trong thư mục traffic_weather.")
        return

    print(f"Tìm thấy {len(files)} file:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(inplace=True)

    output_path = os.path.join(DATA_DIR, 'traffic_weather_latest.csv')
    df.to_csv(output_path, index=False)
    print(f"\nKết quả: {df.shape[0]} dòng, {df.shape[1]} cột")
    print(f"Đã lưu: {output_path}")

if __name__ == '__main__':
    merge_all_traffic_weather()
