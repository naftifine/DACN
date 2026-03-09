import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from geopy.distance import geodesic
import hashlib

load_dotenv()
TOMTOM_KEY = os.getenv("TOMTOM_KEY")

MAX_WORKERS = 30
SEGMENT_LENGTH_KM = 0.5
POINT_DISTANCE_KM = 0.5  # Khoảng cách tạo điểm dọc đường

TRAFFIC_URL = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
)
REVERSE_URL = "https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json"
SNAP_URL = "https://api.tomtom.com/snapToRoads/1/snapToRoads"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
INCIDENT_URL = "https://api.tomtom.com/traffic/services/5/incidentDetails"


# --- API Helpers ---
def get_traffic(lat, lon):
    try:
        params = {"point": f"{lat},{lon}", "unit": "KMPH", "key": TOMTOM_KEY}
        r = requests.get(TRAFFIC_URL, params=params, timeout=5)
        if r.status_code == 200:
            return r.json().get("flowSegmentData", {})
    except:
        return None


def reverse_geocode(lat, lon):
    try:
        r = requests.get(
            REVERSE_URL.format(lat=lat, lon=lon),
            params={"key": TOMTOM_KEY, "language": "vi-VN"},
            timeout=5,
        )
        addr = r.json()["addresses"][0]["address"]
        return addr.get("streetName")
    except:
        return None


import requests
from time import sleep

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# --- (2) get_lane_count_osm: lấy tag 'lanes' từ OSM quanh điểm ---
def get_lane_count_osm(lat, lon, radius=30, tries=2):
    """
    Tra cứu tag 'lanes' của way OSM quanh điểm (lat, lon).
    Trả về int nếu tìm thấy, else None.
    radius: bán kính tìm kiếm (m)
    tries: số lần thử khi Overpass timeouts
    """
    query = f"""
[out:json][timeout:25];
way(around:{radius},{lat},{lon})[highway];
out tags;
"""
    for attempt in range(tries):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=15)
            r.raise_for_status()
            data = r.json()
            if "elements" in data and data["elements"]:
                for elem in data["elements"]:
                    tags = elem.get("tags", {})
                    # ưu tiên tag 'lanes'
                    if "lanes" in tags:
                        val = tags["lanes"].strip()
                        # có thể là '2;2' hoặc '2' hoặc '2 lanes' -> lấy số đầu
                        num = ''.join(ch for ch in val if (ch.isdigit() or ch in "/,;"))
                        if num:
                            # lấy chữ số đầu tiên của chuỗi (ví dụ '2/1' -> 2)
                            try:
                                return int(num.split('/')[0].split(',')[0].split(';')[0])
                            except:
                                continue
                    # nếu chỉ có lanes:forward/backward -> cộng lại
                    if "lanes:forward" in tags or "lanes:backward" in tags:
                        try:
                            f = int(tags.get("lanes:forward", 0))
                            b = int(tags.get("lanes:backward", 0))
                            tot = f + b
                            if tot > 0:
                                return tot
                        except:
                            pass
            return None
        except Exception as e:
            # retry nhẹ nếu lỗi mạng hoặc timeout
            print(f"[OSM lanes] attempt {attempt+1} error: {e}")
            sleep(1)
    return None


# --- (3) Cải thiện get_road_attributes: thử TomTom trước, fallback OSM ---
def get_road_attributes(lat, lon):
    """
    Thử lấy thông tin từ TomTom (snapToRoads / road attributes).
    Nếu không lấy được 'numberOfLanes', fallback sang OSM (get_lane_count_osm).
    Trả về: (segment_id, lane_count (int|None), speed_limit (int|None))
    """
    # small helper để parse response an toàn
    def safe_get(dct, *keys):
        cur = dct
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
            if cur is None:
                return None
        return cur

    try:
        # Gọi TomTom Snap (hoặc thay bằng endpoint Road Attributes nếu bạn có)
        # lưu ý: snapToRoads có thể không trả các thuộc tính bạn mong -> phải kiểm tra
        points_str = f"{lon},{lat};{lon + 0.0005},{lat + 0.0005}"
        params = {
            "key": TOMTOM_KEY,
            "points": points_str,
        }
        r = requests.get(SNAP_URL, params=params, timeout=8)
        if r.status_code != 200:
            print("[TomTom] snapToRoads status:", r.status_code, r.text[:200])
        else:
            data = r.json()
            # debug: nếu cần in kết cấu trả về để xác định trường chứa laneInfo
            # print(json.dumps(data, indent=2))
            # Nhiều SDK trả 'route' hoặc 'snapPoints' - xử lý an toàn:
            # Thử lấy ở nhiều vị trí có thể:
            lane_count = None
            speed_limit = None
            segment_id = None

            # cố gắng lấy laneInfo từ route.properties (nếu có)
            props = safe_get(data, "route", "properties") or safe_get(data, "route")
            if isinstance(props, dict):
                lane_count = safe_get(props, "laneInfo", "numberOfLanes")
                # parse to int nếu cần
                if lane_count is not None:
                    try:
                        lane_count = int(lane_count)
                    except:
                        lane_count = None
                # speedLimits có thể là dict hoặc list
                sl = props.get("speedLimits")
                if isinstance(sl, dict):
                    speed_limit = sl.get("value")
                elif isinstance(sl, list) and sl:
                    speed_limit = sl[0].get("value")
                segment_id = props.get("id") or props.get("segmentId")

            # nếu không có lane_count, thử tìm trong snapPoints (tùy response)
            if lane_count is None:
                snapped = data.get("snappedPoints") or data.get("snapped_points") or []
                for sp in snapped:
                    sp_props = sp.get("properties") or sp.get("attributes") or {}
                    lc = sp_props.get("numberOfLanes") or sp_props.get("lanes")
                    if lc:
                        try:
                            lane_count = int(lc)
                            break
                        except:
                            continue

            # Nếu TomTom không trả lane_count -> fallback OSM
            if lane_count is None:
                lane_count = get_lane_count_osm(lat, lon)

            # Nếu TomTom không trả speed limit -> fallback OSM
            if speed_limit is None:
                speed_limit = get_speed_limit_osm(lat, lon)

            return segment_id, lane_count, speed_limit

    except Exception as e:
        print("[get_road_attributes] error calling TomTom:", e)

    # cuối cùng: fallback hoàn toàn sang OSM nếu TomTom bị lỗi
    lane_osm = get_lane_count_osm(lat, lon)
    speed_osm = get_speed_limit_osm(lat, lon)
    return None, lane_osm, speed_osm


def get_speed_limit_osm(lat, lon):
    try:
        query = f"""
[out:json][timeout:25];
way(around:50,{lat},{lon})[highway][maxspeed];
out body;
"""
        r = requests.post(OVERPASS_URL, data={"data": query}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if "elements" in data and data["elements"]:
                for elem in data["elements"]:
                    if "tags" in elem and "maxspeed" in elem["tags"]:
                        maxspeed_str = elem["tags"]["maxspeed"]
                        if maxspeed_str.isdigit():
                            return int(maxspeed_str)
                        elif "km/h" in maxspeed_str or "kph" in maxspeed_str:
                            return int(
                                maxspeed_str.replace(" km/h", "").replace(" kph", "")
                            )
                        elif "mph" in maxspeed_str:
                            return int(maxspeed_str.replace(" mph", "")) * 1.60934
        return None
    except:
        return None


def get_incidents():
    try:
        # Vùng HCM
        HCM_LAT_MIN, HCM_LAT_MAX = 10.3, 11.2
        HCM_LON_MIN, HCM_LON_MAX = 106.3, 107.1
        params = {
            "key": TOMTOM_KEY,
            "bbox": f"{HCM_LON_MIN},{HCM_LAT_MIN},{HCM_LON_MAX},{HCM_LAT_MAX}",
            "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code},startTime,endTime,from,to,length,delay,roadNumbers,timeValidity}}}",
            "language": "vi-VN",
            "timeValidityFilter": "present",
        }
        r = requests.get(INCIDENT_URL, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("incidents", [])
    except:
        pass
    return []


def is_incident_near(lat, lon, incidents, threshold_km=0.5):
    point = (lat, lon)
    for inc in incidents:
        geom = inc.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") == "Point":
            inc_point = (coords[1], coords[0])
            if geodesic(point, inc_point).km < threshold_km:
                return 1
        elif geom.get("type") == "LineString":
            for coord in coords:
                inc_point = (coord[1], coord[0])
                if geodesic(point, inc_point).km < threshold_km:
                    return 1
    return 0


# --- Helper functions ---
def generate_points_along_line(
    start_lat, start_lon, end_lat, end_lon, distance_km=POINT_DISTANCE_KM
):
    points = [(start_lat, start_lon)]
    start = (start_lat, start_lon)
    end = (end_lat, end_lon)
    total_distance = geodesic(start, end).km
    if total_distance <= distance_km:
        return [start, end]
    steps = int(total_distance // distance_km)
    lat_step = (end_lat - start_lat) / (steps + 1)
    lon_step = (end_lon - start_lon) / (steps + 1)
    for i in range(1, steps + 1):
        points.append((start_lat + lat_step * i, start_lon + lon_step * i))
    points.append(end)
    return points


def get_base_capacity_by_frc(frc):
    if frc is None:
        return 1800
    capacity_map = {
        0: 2400,
        1: 2200,
        2: 2000,
        3: 1700,
        4: 1200,
        5: 1000,
        6: 1000,
        7: 1000,
        8: 1000,
    }
    if isinstance(frc, str):
        if frc.upper().startswith("FRC"):
            try:
                frc_num = int(frc[3:])
                return capacity_map.get(frc_num, 1800)
            except:
                return 1800
    return capacity_map.get(frc, 1800)


def estimate_traffic_volume(current_speed_kmph, freeflow_kmph, lane_count, frc=None):
    if current_speed_kmph is None or freeflow_kmph is None:
        utilization = 0.5
    else:
        congestion_index = (
            current_speed_kmph / freeflow_kmph if freeflow_kmph > 0 else 1.0
        )
        utilization = max(0.05, min(1.0, 1.2 - congestion_index))
    lane_count = max(1, lane_count or 1)
    base_capacity = get_base_capacity_by_frc(frc)
    return round(base_capacity * lane_count * utilization, 2)


# --- Process each road ---
def process_road(row, incidents):
    start_lat = row["lat_snode"]
    start_lon = row["long_snode"]
    end_lat = row["lat_enode"]
    end_lon = row["long_enode"]

    points = generate_points_along_line(start_lat, start_lon, end_lat, end_lon)

    speeds, free_speeds, jams = [], [], []
    name_vn_list = []
    frc = None
    lane_counts = []
    speed_limits = []
    segment_ids = []
    incident_flags = []

    for lat, lon in points:
        traffic = get_traffic(lat, lon)
        if traffic:
            if traffic.get("currentSpeed") is not None:
                speeds.append(traffic.get("currentSpeed"))
            if traffic.get("freeFlowSpeed") is not None:
                free_speeds.append(traffic.get("freeFlowSpeed"))
            if traffic.get("jamFactor") is not None:
                jams.append(traffic.get("jamFactor"))
            frc = traffic.get("frc") or frc
            street_name = reverse_geocode(lat, lon)
            if street_name:
                name_vn_list.append(street_name)

            segment_id, lane_count, speed_limit_tom = get_road_attributes(lat, lon)
            if segment_id:
                segment_ids.append(segment_id)
            if lane_count is not None:
                lane_counts.append(lane_count)
            if speed_limit_tom is not None:
                speed_limits.append(speed_limit_tom)
            else:
                speed_limit_osm = get_speed_limit_osm(lat, lon)
                if speed_limit_osm is not None:
                    speed_limits.append(speed_limit_osm)

            incident_flags.append(is_incident_near(lat, lon, incidents))

    if not speeds:
        return None

    current_speed_avg = sum(speeds) / len(speeds)
    free_flow_speed_avg = (
        sum(free_speeds) / len(free_speeds) if free_speeds else current_speed_avg
    )
    valid_jams = [j for j in jams if j is not None]
    congestion_index = (
        current_speed_avg / free_flow_speed_avg if free_flow_speed_avg else 1.0
    )
    cross_time = (
        SEGMENT_LENGTH_KM / current_speed_avg * 3600 if current_speed_avg else None
    )

    name_vn = (
        max(set(name_vn_list), key=name_vn_list.count)
        if name_vn_list
        else row["street_name"]
    )
    lane_count_avg = sum(lane_counts) / len(lane_counts) if lane_counts else 1
    speed_limit_avg = sum(speed_limits) / len(speed_limits) if speed_limits else 50
    incident_flag = 1 if any(incident_flags) else 0
    segment_id = (
        max(set(segment_ids), key=segment_ids.count)
        if segment_ids
        else hashlib.md5(row["street_name"].encode()).hexdigest()
    )

    # LOS
    if congestion_index >= 0.9:
        los = "A"
    elif congestion_index >= 0.7:
        los = "B"
    elif congestion_index >= 0.5:
        los = "C"
    elif congestion_index >= 0.3:
        los = "D"
    elif congestion_index >= 0.1:
        los = "E"
    else:
        los = "F"

    # --- Traffic volume & occupancy ---
    traffic_volume = estimate_traffic_volume(
        current_speed_avg, free_flow_speed_avg, lane_count_avg, frc=frc
    )
    occupancy = (
        100 * (1 - current_speed_avg / free_flow_speed_avg)
        if free_flow_speed_avg
        else 0
    )

    timestamp = datetime.now().strftime("%y%m%d%H%M")
    day_of_week = datetime.now().weekday()

    return {
        "segmentId": segment_id,
        "name": row["street_name"],
        "name_vn": name_vn,
        "lat": start_lat,
        "lon": start_lon,
        "roadType": frc,
        "laneCount": lane_count_avg,
        "frc": frc,
        "currentSpeed": current_speed_avg,
        "freeFlowSpeed": free_flow_speed_avg,
        "jamFactor": sum(valid_jams) / len(valid_jams) if valid_jams else 0,
        "congestionIndex": congestion_index,
        "crossTime": cross_time,
        "trafficVolume": traffic_volume,
        "occupancy": occupancy,
        "speedLimit": speed_limit_avg,
        "incidentFlag": incident_flag,
        "LOS": los,
        "timeStamp": timestamp,
        "dayOfWeek": day_of_week,
    }


# --- Main ---
if __name__ == "__main__":
    input_file = "../data/traffic/streets_merged.csv"  
    output_file = "../data/traffic/traffic_hcm.csv"

    df_roads = pd.read_csv(input_file)
    if df_roads.empty:
        print("No roads in CSV")
        exit()

    incidents = get_incidents()
    traffic_data = []

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_road, row, incidents)
            for _, row in df_roads.iterrows()
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                traffic_data.append(result)

    df = pd.DataFrame(traffic_data)
    if df.empty:
        print("No traffic data collected")
        exit()

    # Nếu file cũ tồn tại, ghép dữ liệu và loại trùng
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file, encoding="utf-8-sig")
        df = pd.concat([df_existing, df], ignore_index=True)
        df.drop_duplicates(subset=["segmentId", "timeStamp"], keep="last", inplace=True)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(
        f"Traffic data saved/appended to {output_file} in {time.time() - start_time:.2f}s"
    )
