import requests
import pandas as pd
import time


def get_full_geometry(lat1, lon1, lat2, lon2):
    # API OSRM: lấy full đường đi chi tiết nhất giữa 2 điểm
    # geometries=geojson trả về danh sách tọa độ chi tiết
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"

    try:
        response = requests.get(url, timeout=10).json()
        if response["code"] == "Ok":
            # Lấy danh sách tọa độ từ kết quả
            # OSRM trả về [long, lat], mình cần đổi lại thành [lat, long] cho Android
            coords = response["routes"][0]["geometry"]["coordinates"]

            # Nối thành chuỗi: "lat,long;lat,long;..."
            path_string = ";".join([f"{c[1]},{c[0]}" for c in coords])
            return path_string
    except Exception as e:
        print(f"Lỗi tại tọa độ {lat1},{lon1}: {e}")
    return ""


def process_csv_to_geometry(input_file, output_file):
    df = pd.read_csv(input_file)
    print(f"Đang bắt đầu lấy đường cong cho {len(df)} đoạn đường. Chờ xíu nhé Huy...")

    full_geometries = []
    for index, row in df.iterrows():
        # Lấy danh sách tọa độ uốn lượn
        geom = get_full_geometry(
            row["lat_snode"], row["long_snode"], row["lat_enode"], row["long_enode"]
        )
        full_geometries.append(geom)

        # Tránh bị server OSRM chặn vì gọi quá nhanh
        time.sleep(0.2)
        if (index + 1) % 10 == 0:
            print(f"Đã xong {index + 1} đoạn...")

    # Thêm cột mới vào file của Huy
    df["full_geometry"] = full_geometries

    # Lưu lại file CSV mới
    df.to_csv(output_file, index=False)
    print(
        f"\nNGON LÀNH! File mới '{output_file}' đã có cột full_geometry chứa hàng chục tọa độ uốn lượn."
    )


# Chạy lệnh (Huy đổi tên file input của mình vào đây)
process_csv_to_geometry("streets_merged.csv", "test.csv")
