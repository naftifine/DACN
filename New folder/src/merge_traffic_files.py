import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def merge_two_traffic_files(
    traffic_path_1: Path, traffic_path_2: Path, output_path: Path
) -> None:
    if not traffic_path_1.exists():
        raise FileNotFoundError(f"Không tìm thấy file traffic 1: {traffic_path_1}")
    if not traffic_path_2.exists():
        raise FileNotFoundError(f"Không tìm thấy file traffic 2: {traffic_path_2}")

    rows1 = read_csv(traffic_path_1)
    rows2 = read_csv(traffic_path_2)

    if not rows1 and not rows2:
        print("⚠️  Cả hai file traffic đều không có dữ liệu.")
        return

    # Xác định danh sách cột: ưu tiên thứ tự cột của file 1, sau đó thêm cột mới từ file 2 nếu có
    fieldnames: List[str] = []
    if rows1:
        fieldnames.extend(list(rows1[0].keys()))

    if rows2:
        for k in rows2[0].keys():
            if k not in fieldnames:
                fieldnames.append(k)

    merged_rows: List[Dict[str, str]] = []
    merged_rows.extend(rows1)
    merged_rows.extend(rows2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    print(
        f"✅ Đã merge {len(rows1)} dòng từ {traffic_path_1.name} và {len(rows2)} dòng từ {traffic_path_2.name} vào {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge 2 file traffic (đã merge weather) lại với nhau."
    )
    parser.add_argument(
        "--traffic-file-1",
        type=Path,
        required=True,
        help="Đường dẫn tới file traffic thứ nhất (CSV).",
    )
    parser.add_argument(
        "--traffic-file-2",
        type=Path,
        required=True,
        help="Đường dẫn tới file traffic thứ hai (CSV).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Đường dẫn file CSV merge kết quả.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_two_traffic_files(args.traffic_file_1, args.traffic_file_2, args.output_file)


if __name__ == "__main__":
    main()
