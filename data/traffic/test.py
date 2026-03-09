import pandas as pd
import sys


def remove_empty_columns(input_file, output_file=None):
    """
    Đọc file CSV, loại bỏ các cột được liệt kê trong danh sách,
    và lưu lại kết quả.
    """
    if output_file is None:
        output_file = input_file  # ghi đè file gốc

    # Đọc dữ liệu
    df = pd.read_csv(input_file)

    # Danh sách các cột luôn trống cần loại bỏ
    cols_always_empty = [
        "width",
        "busRoute",
        "bicycleLane",
        "sidewalk",
        "bridgeTunnel",
        "maxHeight",
        "maxWeight",
        "barrier",
        "lit",
        "crossing",
        "route_distance_m",
        "route_duration_sec",
        "route_weight",
        "congestion_levels",
        "map_match_confidence",
    ]

    # Lọc ra những cột thực sự tồn tại trong DataFrame
    existing_cols = [c for c in cols_always_empty if c in df.columns]

    if existing_cols:
        df.drop(columns=existing_cols, inplace=True)
        print(f"✅ Đã loại bỏ các cột: {existing_cols}")
    else:
        print("⚠️  Không có cột nào trong danh sách tồn tại trong file.")

    # Lưu lại
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"💾 Đã lưu file: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "🔧 Cách dùng: python script.py <đường_dẫn_file_input> [đường_dẫn_file_output]"
        )
        print("   Nếu không chỉ định output, file gốc sẽ bị ghi đè.")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    remove_empty_columns(input_path, output_path)
