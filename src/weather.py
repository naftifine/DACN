import os
import csv

def print_weather_csv():
    project_root = os.path.dirname(os.path.dirname(__file__))

    csv_path = os.path.join(project_root, "data", "weather.csv")

    if not os.path.isfile(csv_path):
        print("Không tìm thấy file weather.csv trong thư mục 'data' hoặc weather_data.csv trong thư mục 'assets'/'datas'.")
        return

    try:
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                print(", ".join(row))
    except Exception as e:
        print(f"Đã có lỗi khi đọc file CSV: {e}")

if __name__ == "__main__":
    print_weather_csv()
else:
    print_weather_csv()
