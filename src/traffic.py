import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from geopy.distance import geodesic

load_dotenv()
TOMTOM_KEY = os.getenv("TOMTOM_KEY")

HCM_LAT_MIN, HCM_LAT_MAX = 10.3, 11.2
HCM_LON_MIN, HCM_LON_MAX = 106.3, 107.1

MAX_WORKERS = 30
SEGMENT_LENGTH_KM = 0.5  
POINT_DISTANCE_KM = 5     

TRAFFIC_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
REVERSE_URL = "https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json"

def get_main_roads_in_hcm(limit=200):
    url = "https://api.tomtom.com/search/2/categorySearch/majorRoad.json"
    params = {
        "key": TOMTOM_KEY,
        "limit": limit,
        "topLeft": f"{HCM_LAT_MAX},{HCM_LON_MIN}",
        "btmRight": f"{HCM_LAT_MIN},{HCM_LON_MAX}"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json().get("results", [])
        roads = []
        for res in data:
            address = res.get("address", {})
            street = address.get("streetName")
            pos = res.get("position", {})
            if street and pos:
                roads.append({
                    "name": street,
                    "lat": pos["lat"],
                    "lon": pos["lon"]
                })
        if roads:
            return roads
    except Exception as e:
        print("Error categorySearch:", e)

    url = "https://api.tomtom.com/search/2/search/Đường.json"
    params["limit"] = limit
    try:
        r = requests.get(url, params=params, timeout=10)
        results = r.json().get("results", [])
        roads = []
        for res in results:
            address = res.get("address", {})
            street = address.get("streetName")
            pos = res.get("position", {})
            if street and pos:
                roads.append({
                    "name": street,
                    "lat": pos["lat"],
                    "lon": pos["lon"]
                })
        return roads
    except Exception as e:
        print("Error fallback search:", e)
        return []


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
        r = requests.get(REVERSE_URL.format(lat=lat, lon=lon),
                         params={"key": TOMTOM_KEY, "language": "vi-VN"}, timeout=5)
        addr = r.json()["addresses"][0]["address"]
        return addr.get("streetName")
    except:
        return None


def generate_points_along_line(start_lat, start_lon, end_lat, end_lon, distance_km=POINT_DISTANCE_KM):
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


def process_road(road):
    
    points = generate_points_along_line(road["lat"], road["lon"],
                                        road["lat"]+0.01, road["lon"]+0.01) 
    
    speeds, free_speeds, jams = [], [], []
    name_vn_list = []
    frc = None
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
    if not speeds:
        return None
    current_speed_avg = sum(speeds)/len(speeds)
    free_flow_speed_avg = sum(free_speeds)/len(free_speeds) if free_speeds else None
    valid_jams = [j for j in jams if j is not None]
    congestion_index = current_speed_avg / free_flow_speed_avg if free_flow_speed_avg else None
    cross_time = SEGMENT_LENGTH_KM / current_speed_avg * 3600 if current_speed_avg else None
    
    
    name_vn = max(set(name_vn_list), key=name_vn_list.count) if name_vn_list else road["name"]
    return {
        "name": road["name"],
        "name_vn": name_vn,
        "lat": road["lat"],
        "lon": road["lon"],
        "frc": frc,
        "currentSpeed": current_speed_avg,
        "freeFlowSpeed": free_flow_speed_avg,
        "jamFactor": sum(valid_jams)/len(valid_jams) if valid_jams else None,
        "congestionIndex": congestion_index,
        "crossTime": cross_time,
        "trafficVolume": "NA",
        "occupancy": "NA"
    }



def get_los(congestion_index):
    if congestion_index is None: return "NA"
    if congestion_index >= 0.9: return "A"
    if congestion_index >= 0.7: return "B"
    if congestion_index >= 0.5: return "C"
    if congestion_index >= 0.3: return "D"
    if congestion_index >= 0.1: return "E"
    return "F"



if __name__ == "__main__":
    roads = get_main_roads_in_hcm(limit=200)
    if not roads:
        exit()

    traffic_data = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_road, road) for road in roads]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                traffic_data.append(result)

    df = pd.DataFrame(traffic_data)
    if df.empty:
        exit()

    df = df.groupby("name", as_index=False).agg({
        "name_vn": "first",
        "lat": "first",
        "lon": "first",
        "frc": "first",
        "currentSpeed": "mean",
        "freeFlowSpeed": "mean",
        "jamFactor": "mean",
        "congestionIndex": "mean",
        "crossTime": "mean",
        "trafficVolume": "first",
        "occupancy": "first"
    })

    df = df[df["frc"].isin(["FRC0","FRC1","FRC2","FRC3","FRC4","FRC5"])]
    df["LOS"] = df["congestionIndex"].apply(get_los)

    timestamp = datetime.now().strftime("%y%m%d%H%M")
    output_file = f"../data/traffic/traffic_{timestamp}.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

