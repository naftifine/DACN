import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from geopy.distance import geodesic
import hashlib

# -----------------------
# Load API keys
# -----------------------
load_dotenv()
TOMTOM_KEY = os.getenv("TOMTOM_KEY")
HERE_KEY = os.getenv("HERE_KEY")
MAPBOX_KEY = os.getenv("MAPBOX_KEY")

MAX_WORKERS = 15
SEGMENT_LENGTH_KM = 0.5
POINT_DISTANCE_KM = 1

# -----------------------
# URLs
# -----------------------
TRAFFIC_URL_TOMTOM = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
)
REVERSE_URL_TOMTOM = "https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json"
SNAP_URL_TOMTOM = "https://api.tomtom.com/snapToRoads/1/snapToRoads"
INCIDENT_URL_TOMTOM = "https://api.tomtom.com/traffic/services/5/incidentDetails"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# -----------------------
# Utilities
# -----------------------
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


def safe_int(val):
    try:
        return int(val)
    except:
        return None


# -----------------------
# TomTom
# -----------------------
def get_traffic_tomtom(lat, lon):
    try:
        r = requests.get(
            TRAFFIC_URL_TOMTOM,
            params={"point": f"{lat},{lon}", "unit": "KMPH", "key": TOMTOM_KEY},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json().get("flowSegmentData", {})
    except:
        return {}


def reverse_geocode_tomtom(lat, lon):
    try:
        r = requests.get(
            REVERSE_URL_TOMTOM.format(lat=lat, lon=lon),
            params={"key": TOMTOM_KEY, "language": "vi-VN"},
            timeout=5,
        )
        addr = r.json()["addresses"][0]["address"]
        return addr.get("streetName")
    except:
        return None


def snap_to_road_tomtom(lat, lon):
    try:
        points_str = f"{lon},{lat}"
        r = requests.get(
            SNAP_URL_TOMTOM, params={"key": TOMTOM_KEY, "points": points_str}, timeout=8
        )
        data = r.json()
        props = data.get("route", {}).get("properties", {}) if "route" in data else {}
        lane_count = safe_int(props.get("laneInfo", {}).get("numberOfLanes"))
        speed_limit = None
        sl = props.get("speedLimits")
        if isinstance(sl, dict):
            speed_limit = safe_int(sl.get("value"))
        elif isinstance(sl, list) and sl:
            speed_limit = safe_int(sl[0].get("value"))
        segment_id = props.get("id") or props.get("segmentId")
        return segment_id, lane_count, speed_limit
    except:
        return None, None, None


def get_incidents_tomtom():
    try:
        HCM_LAT_MIN, HCM_LAT_MAX = 10.3, 11.2
        HCM_LON_MIN, HCM_LON_MAX = 106.3, 107.1
        params = {
            "key": TOMTOM_KEY,
            "bbox": f"{HCM_LON_MIN},{HCM_LAT_MIN},{HCM_LON_MAX},{HCM_LAT_MAX}",
            "language": "vi-VN",
            "timeValidityFilter": "present",
        }
        r = requests.get(INCIDENT_URL_TOMTOM, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("incidents", [])
    except:
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


# -----------------------
# OSM
# -----------------------
def get_osm_attributes(lat, lon, radius=50):
    try:
        query = f"""
[out:json][timeout:25];
way(around:{radius},{lat},{lon})[highway];
out tags;
"""
        r = requests.post(OVERPASS_URL, data={"data": query}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "elements" in data and data["elements"]:
            tags = data["elements"][0].get("tags", {})
            return {
                "roadType": tags.get("highway"),
                "lanes": safe_int(tags.get("lanes")),
                "speedLimit": safe_int(tags.get("maxspeed")),
                "oneway": tags.get("oneway"),
                "surface": tags.get("surface"),
                "width": tags.get("width"),
                "ref": tags.get("ref"),
            }
    except:
        return {}
    return {}


# -----------------------
# HERE
# -----------------------
def get_here_attributes(lat, lon):
    try:
        url = f"https://router.hereapi.com/v8/routes?transportMode=car&origin={lat},{lon}&destination={lat+0.0001},{lon+0.0001}&return=summary,polyline&apikey={HERE_KEY}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            lane_count = safe_int(
                data.get("routes", [{}])[0]
                .get("sections", [{}])[0]
                .get("summary", {})
                .get("laneCount")
            )
            return {"lanes": lane_count}
    except:
        return {}


# -----------------------
# Mapbox
# -----------------------
def get_mapbox_attributes(lat, lon):
    try:
        url = f"https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/tilequery/{lon},{lat}.json?layers=road&access_token={MAPBOX_KEY}"
        r = requests.get(url, timeout=8)
        data = r.json()
        props = data.get("features", [{}])[0].get("properties", {})
        return props
    except:
        return {}


# -----------------------
# Traffic volume & LOS
# -----------------------
def estimate_traffic_volume(
    current_speed_kmph, freeflow_kmph, lane_count, base_capacity=1800
):
    if current_speed_kmph is None or freeflow_kmph is None:
        utilization = 0.5
    else:
        congestion_index = (
            current_speed_kmph / freeflow_kmph if freeflow_kmph > 0 else 1.0
        )
        utilization = max(0.05, min(1.0, 1.2 - congestion_index))
    lane_count = max(1, lane_count or 1)
    return round(base_capacity * lane_count * utilization, 2)


def get_los(congestion_index):
    if congestion_index >= 0.9:
        return "A"
    elif congestion_index >= 0.7:
        return "B"
    elif congestion_index >= 0.5:
        return "C"
    elif congestion_index >= 0.3:
        return "D"
    elif congestion_index >= 0.1:
        return "E"
    else:
        return "F"


# -----------------------
# Process each road
# -----------------------
def process_road(row, incidents):
    start_lat = row["lat_snode"]
    start_lon = row["long_snode"]
    end_lat = row["lat_enode"]
    end_lon = row["long_enode"]
    points = generate_points_along_line(start_lat, start_lon, end_lat, end_lon)

    lane_counts, speed_limits, segment_ids, incident_flags, names = [], [], [], [], []
    current_speeds, free_speeds, jam_factors = [], [], []

    for lat, lon in points:
        # TomTom traffic & snap
        traffic = get_traffic_tomtom(lat, lon)
        current_speeds.append(traffic.get("currentSpeed") or 0)
        free_speeds.append(traffic.get("freeFlowSpeed") or 0)
        jam_factors.append(traffic.get("jamFactor") or 0)
        frc = traffic.get("frc")
        seg_id_tt, lane_tt, speed_tt = snap_to_road_tomtom(lat, lon)
        street_name = reverse_geocode_tomtom(lat, lon) or row["street_name"]

        # OSM
        osm_attrs = get_osm_attributes(lat, lon)
        lane_osm = osm_attrs.get("lanes")
        speed_osm = osm_attrs.get("speedLimit")
        roadType = osm_attrs.get("roadType")
        surface = osm_attrs.get("surface")
        oneway = osm_attrs.get("oneway")
        width = osm_attrs.get("width")
        ref = osm_attrs.get("ref")

        # HERE
        here_attrs = get_here_attributes(lat, lon)
        lane_here = here_attrs.get("lanes")

        # Mapbox
        mapbox_props = get_mapbox_attributes(lat, lon)
        lane_mapbox = safe_int(mapbox_props.get("lanes"))
        speed_mapbox = safe_int(mapbox_props.get("maxspeed"))

        # Merge lane & speed
        lane_count = lane_tt or lane_here or lane_mapbox or lane_osm or 1
        speed_limit = speed_tt or speed_osm or speed_mapbox or 50
        segment_id = seg_id_tt or hashlib.md5(f"{lat},{lon}".encode()).hexdigest()
        incident_flags.append(is_incident_near(lat, lon, incidents))

        lane_counts.append(lane_count)
        speed_limits.append(speed_limit)
        segment_ids.append(segment_id)
        names.append(street_name)

    # Aggregation
    lane_avg = sum(lane_counts) / len(lane_counts) if lane_counts else 1
    speed_avg = sum(speed_limits) / len(speed_limits) if speed_limits else 50
    incident_flag = 1 if any(incident_flags) else 0
    segment_id = max(set(segment_ids), key=segment_ids.count)
    name_final = max(set(names), key=names.count)
    current_speed_avg = (
        sum(current_speeds) / len(current_speeds) if current_speeds else 0
    )
    free_flow_avg = (
        sum(free_speeds) / len(free_speeds) if free_speeds else current_speed_avg
    )
    congestion_index = current_speed_avg / free_flow_avg if free_flow_avg else 1.0
    traffic_volume = estimate_traffic_volume(current_speed_avg, free_flow_avg, lane_avg)
    los = get_los(congestion_index)
    occupancy = 100 * (1 - current_speed_avg / free_flow_avg) if free_flow_avg else 0
    cross_time = (
        SEGMENT_LENGTH_KM / current_speed_avg * 3600 if current_speed_avg else None
    )

    timestamp = datetime.now().strftime("%y%m%d%H%M")
    day_of_week = datetime.now().weekday()

    return {
        "segmentId": segment_id,
        "name": row["street_name"],
        "name_vn": name_final,
        "lat": start_lat,
        "lon": start_lon,
        "laneCount": lane_avg,
        "speedLimit": speed_avg,
        "frc": frc,
        "roadType": roadType,
        "surface": surface,
        "oneway": oneway,
        "width": width,
        "ref": ref,
        "currentSpeed": current_speed_avg,
        "freeFlowSpeed": free_flow_avg,
        "jamFactor": sum(jam_factors) / len(jam_factors) if jam_factors else 0,
        "congestionIndex": congestion_index,
        "trafficVolume": traffic_volume,
        "occupancy": occupancy,
        "incidentFlag": incident_flag,
        "LOS": los,
        "crossTime": cross_time,
        "timeStamp": timestamp,
        "dayOfWeek": day_of_week,
    }


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    input_file = "../data/traffic/streets_merged.csv"
    output_file = "../data/traffic/traffic_hcm_full.csv"

    df_roads = pd.read_csv(input_file)
    if df_roads.empty:
        print("No roads")
        exit()

    incidents = get_incidents_tomtom()
    traffic_data = []

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_road, row, incidents)
            for _, row in df_roads.iterrows()
        ]
        for f in as_completed(futures):
            res = f.result()
            if res:
                traffic_data.append(res)

    df_out = pd.DataFrame(traffic_data)
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df_out = pd.concat([df_existing, df_out], ignore_index=True)
        df_out.drop_duplicates(
            subset=["segmentId", "timeStamp"], keep="last", inplace=True
        )

    df_out.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df_out)} records in {time.time()-start_time:.2f}s")
