import argparse
import csv
import datetime as dt
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry

# Dictionary để dịch mã thời tiết WMO (World Meteorological Organization)
WMO_CODES = {
    0: "Trời quang, không mây",
    1: "Trời quang, ít mây thay đổi",
    2: "Trời quang, có mây rải rác",
    3: "Trời nhiều mây",
    45: "Sương mù",
    48: "Sương mù trắng xóa",
    51: "Mưa phùn nhẹ",
    53: "Mưa phùn vừa",
    55: "Mưa phùn dày",
    61: "Mưa nhẹ",
    63: "Mưa vừa",
    65: "Mưa to",
    71: "Tuyết rơi nhẹ",
    73: "Tuyết rơi vừa",
    75: "Tuyết rơi dày",
    80: "Mưa rào nhẹ",
    81: "Mưa rào vừa",
    82: "Mưa rào rất to",
    95: "Dông, có sấm sét",
    96: "Dông kèm mưa đá nhỏ",
    99: "Dông kèm mưa đá lớn",
}

DAILY_PARAMS = [
    "weathercode",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "sunrise",
    "sunset",
    "uv_index_max",
    "precipitation_sum",
    "precipitation_probability_max",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "winddirection_10m_dominant",
]

FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
DEFAULT_TRAFFIC_FILE = DATA_DIR / "traffic" / "traffic_hcm_updated.csv"
DEFAULT_OUTPUT_FILE = DATA_DIR / "weather" / "weather_updated.csv"

WEATHER_OUTPUT_COLUMNS: List[str] = [
    "name",
    "segmentId",
    "weather_date",
    "latitude_lookup",
    "longitude_lookup",
    "weathercode",
    "weathercode_unit",
    "weather_description",
    "temperature_2m_max",
    "temperature_2m_max_unit",
    "temperature_2m_min",
    "temperature_2m_min_unit",
    "apparent_temperature_max",
    "apparent_temperature_max_unit",
    "apparent_temperature_min",
    "apparent_temperature_min_unit",
    "sunrise",
    "sunrise_unit",
    "sunset",
    "sunset_unit",
    "uv_index_max",
    "precipitation_sum",
    "precipitation_sum_unit",
    "precipitation_probability_max",
    "precipitation_probability_max_unit",
    "windspeed_10m_max",
    "windspeed_10m_max_unit",
    "windgusts_10m_max",
    "windgusts_10m_max_unit",
    "winddirection_10m_dominant",
    "winddirection_10m_dominant_unit",
    "time",
    "time_unit",
    "weather_fetch_failed",
]


def get_wmo_description(code: Optional[int]) -> str:
    """Trả về mô tả thời tiết từ mã WMO."""
    return WMO_CODES.get(code, "Không có dữ liệu")


def build_http_session() -> requests.Session:
    """Tạo session kèm retry để gọi API ổn định hơn."""
    session = requests.Session()
    retries = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def parse_traffic_timestamp(value: object) -> Optional[dt.datetime]:
    """Chuyển cột timeStamp (định dạng yyMMddHHmm) sang datetime."""
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    if text.endswith(".0"):
        text = text[:-2]

    text = text.split(".")[0]
    # ensure at least 10 chars (pad with leading zeros if needed)
    text = text.zfill(10)

    # Try common formats. Prefer yy-mm-dd order (yymmddhhmm) because many files encode
    # timestamps as yymmddhhmm (e.g. 2511070819 -> 2025-11-07 08:19). Fall back to
    # ddmmyyhhmm if that better matches the input.
    for fmt in ("%y%m%d%H%M", "%d%m%y%H%M"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue

    # If none matched, return None
    return None


def safe_float(value: Optional[str]) -> Optional[float]:
    """Chuyển chuỗi sang float, trả về None nếu không hợp lệ."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def has_meaningful_value(value: object) -> bool:
    """Kiểm tra giá trị có thông tin hay không."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def fetch_daily_weather(
    session: requests.Session, latitude: float, longitude: float, date_str: str
) -> Optional[Dict]:
    """Lấy dữ liệu thời tiết hàng ngày từ Open-Meteo."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(DAILY_PARAMS),
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "auto",
    }

    try:
        response = session.get(FORECAST_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        print(
            f"⚠️  Không lấy được dữ liệu thời tiết cho ({latitude}, {longitude}) ngày {date_str}: {exc}"
        )
        return None


def extract_daily_values(payload: Dict) -> Tuple[Dict[str, Optional[object]], Dict[str, str]]:
    """Trích dữ liệu thời tiết cần thiết từ phản hồi API."""
    daily_info = payload.get("daily", {})
    units = payload.get("daily_units", {})

    values: Dict[str, Optional[object]] = {}
    for field in DAILY_PARAMS:
        raw = daily_info.get(field)
        if isinstance(raw, list) and raw:
            values[field] = raw[0]
        else:
            values[field] = None

    time_list = daily_info.get("time")
    values["time"] = time_list[0] if isinstance(time_list, list) and time_list else None
    weather_code = values.get("weathercode")
    values["weather_description"] = (
        get_wmo_description(int(weather_code)) if weather_code is not None else None
    )

    return values, units


def collect_weather_from_traffic(traffic_path: Path, output_path: Path) -> None:
    """Ghép thông tin thời tiết cho từng dòng dữ liệu traffic."""
    if not traffic_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file traffic: {traffic_path}")

    with traffic_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        rows: List[Dict[str, str]] = list(reader)

    if not rows:
        print("⚠️  File traffic không có dữ liệu.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = build_http_session()
    cache: Dict[Tuple[float, float, str], Tuple[Optional[Dict], Dict[str, str]]] = {}
    records: List[Dict[str, object]] = []
    total = len(rows)

    for idx, row in enumerate(rows, start=1):
        # Try several common coordinate column names found in traffic CSVs
        lat_candidates = ["lat", "latitude", "lat_start", "latitude_start"]
        lon_candidates = ["lon", "longitude", "lon_start", "longitude_start"]

        lat = None
        lon = None
        used_coord = None
        for la in lat_candidates:
            for lo in lon_candidates:
                lat_val = safe_float(row.get(la))
                lon_val = safe_float(row.get(lo))
                if lat_val is not None and lon_val is not None:
                    lat = lat_val
                    lon = lon_val
                    used_coord = (la, lo)
                    break
            if lat is not None:
                break

        if lat is None or lon is None:
            # also try end coordinates if start not present
            lat = safe_float(row.get("lat_end"))
            lon = safe_float(row.get("lon_end"))
            if lat is not None and lon is not None:
                used_coord = ("lat_end", "lon_end")

        if lat is None or lon is None:
            print(f"⚠️  Bỏ qua hàng {idx} vì thiếu tọa độ cho đường {row.get('name')}")
            continue

        timestamp_raw = row.get("timeStamp")
        dt_value = parse_traffic_timestamp(timestamp_raw)
        if dt_value is None:
            dt_value = dt.datetime.now()
        date_str = dt_value.strftime("%Y-%m-%d")

        rounded_lat = round(lat, 4)
        rounded_lon = round(lon, 4)
        cache_key = (rounded_lat, rounded_lon, date_str)

        weather_values: Optional[Dict]
        units: Dict[str, str]

        cached_entry = cache.get(cache_key)
        if cached_entry is None:
            payload = fetch_daily_weather(session, rounded_lat, rounded_lon, date_str)
            if payload:
                weather_values, units = extract_daily_values(payload)
            else:
                weather_values, units = None, {}
            cache[cache_key] = (weather_values, units)
            time.sleep(0.2)  # tránh spam API quá nhanh
        else:
            weather_values, units = cached_entry

        row_data: Dict[str, object] = dict(row)
        row_data["weather_date"] = date_str
        row_data["latitude_lookup"] = rounded_lat
        row_data["longitude_lookup"] = rounded_lon

        if weather_values:
            for field, value in weather_values.items():
                row_data[field] = value
                unit = units.get(field)
                if unit:
                    row_data[f"{field}_unit"] = unit
        else:
            row_data["weather_fetch_failed"] = True

        records.append(row_data)

        if idx % 50 == 0 or idx == total:
            print(f"Đã xử lý {idx}/{total} dòng...")

    session.close()

    if not records:
        print("⚠️  Không có bản ghi hợp lệ để ghi.")
        return

    existing_keys = {key for record in records for key in record.keys()}

    fieldnames: List[str] = []
    for col in WEATHER_OUTPUT_COLUMNS:
        if col not in existing_keys:
            continue
        if any(has_meaningful_value(record.get(col)) for record in records):
            fieldnames.append(col)

    with output_path.open("w", encoding="utf-8-sig", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            filtered_row = {field: record.get(field) for field in fieldnames}
            writer.writerow(filtered_row)

    print(f"✅ Đã lưu {len(records)} dòng dữ liệu vào {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lấy dữ liệu thời tiết từ Open-Meteo cho toàn bộ file traffic và xuất CSV."
    )
    parser.add_argument(
        "--traffic-file",
        type=Path,
        default=DEFAULT_TRAFFIC_FILE,
        help="Đường dẫn tới file traffic nguồn.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Đường dẫn file CSV sẽ xuất dữ liệu thời tiết.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_weather_from_traffic(args.traffic_file, args.output_file)


if __name__ == "__main__":
    main()