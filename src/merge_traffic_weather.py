import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def merge_traffic_and_weather(
    traffic_path: Path, weather_path: Path, output_path: Path
) -> None:
    if not traffic_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file traffic: {traffic_path}")
    if not weather_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file weather: {weather_path}")

    traffic_rows = read_csv(traffic_path)
    weather_rows = read_csv(weather_path)

    if not traffic_rows:
        print("⚠️  File traffic không có dữ liệu.")
        return
    if not weather_rows:
        print("⚠️  File weather không có dữ liệu.")
        return

    # Index weather theo segmentId (nếu trùng, lấy bản ghi cuối cùng)
    weather_by_segment: Dict[str, Dict[str, str]] = {}
    for row in weather_rows:
        seg_id = row.get("segmentId")
        if seg_id:
            weather_by_segment[seg_id] = row

    merged_rows: List[Dict[str, str]] = []

    for t_row in traffic_rows:
        seg_id = t_row.get("segmentId")
        w_row = weather_by_segment.get(seg_id) if seg_id is not None else None

        merged: Dict[str, str] = dict(t_row)
        if w_row:
            for k, v in w_row.items():
                # Không ghi đè segmentId; giữ segmentId từ traffic
                if k == "segmentId":
                    continue
                # Nếu weather cũng có cột name, ưu tiên name từ traffic
                if k == "name" and "name" in t_row:
                    continue
                merged[k] = v

        merged_rows.append(merged)

    # Xác định danh sách cột: ưu tiên thứ tự cột traffic, sau đó thêm cột mới từ weather
    traffic_fieldnames = list(traffic_rows[0].keys())
    extra_fields = []
    for w_row in weather_rows:
        for k in w_row.keys():
            if k not in traffic_fieldnames and k not in extra_fields:
                extra_fields.append(k)

    fieldnames = traffic_fieldnames + extra_fields

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    print(f"✅ Đã merge {len(merged_rows)} dòng vào {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge file traffic và weather theo segmentId."
    )
    parser.add_argument(
        "--traffic-file",
        type=Path,
        required=True,
        help="Đường dẫn tới file traffic (CSV).",
    )
    parser.add_argument(
        "--weather-file",
        type=Path,
        required=True,
        help="Đường dẫn tới file weather (CSV).",
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
    merge_traffic_and_weather(args.traffic_file, args.weather_file, args.output_file)


if __name__ == "__main__":
    main()
