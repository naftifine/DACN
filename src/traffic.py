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

HCM_LAT_MIN, HCM_LAT_MAX = 10.3, 11.2
HCM_LON_MIN, HCM_LON_MAX = 106.3, 107.1

MAX_WORKERS = 30
SEGMENT_LENGTH_KM = 0.5
POINT_DISTANCE_KM = 5

TRAFFIC_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
REVERSE_URL = "https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json"
SNAP_URL = "https://api.tomtom.com/snapToRoads/1/snapToRoads"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
INCIDENT_URL = "https://api.tomtom.com/traffic/services/5/incidentDetails"

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
                roads.append({"name": street, "lat": pos["lat"], "lon": pos["lon"]})
        if roads:
            return roads
    except:
        pass
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
                roads.append({"name": street, "lat": pos["lat"], "lon": pos["lon"]})
        return roads
    except:
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

def get_road_attributes(lat, lon):
    try:
        points_str = f"{lon},{lat};{lon + 0.0001},{lat}"
        params = {
            "key": TOMTOM_KEY,
            "points": points_str,
            "fields": "{route{properties{laneInfo{numberOfLanes},speedLimits{value,unit,type},id}}}"
        }
        r = requests.get(SNAP_URL, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            properties = data.get("route", {}).get("properties", {})
            lane_count = properties.get("laneInfo", {}).get("numberOfLanes")
            speed_limit = properties.get("speedLimits", {}).get("value") if "speedLimits" in properties else None
            segment_id = properties.get("id")
            return segment_id, lane_count, speed_limit
    except:
        pass
    return None, None, None

def get_speed_limit_osm(lat, lon):
    try:
        query = f"""
[out:json][timeout:25];
way(around:50,{lat},{lon})[highway][maxspeed];
out body;
"""
        r = requests.post(OVERPASS_URL, data={'data': query}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if 'elements' in data and data['elements']:
                for elem in data['elements']:
                    if 'tags' in elem and 'maxspeed' in elem['tags']:
                        maxspeed_str = elem['tags']['maxspeed']
                        if maxspeed_str.isdigit():
                            return int(maxspeed_str)
                        elif 'km/h' in maxspeed_str or 'kph' in maxspeed_str:
                            return int(maxspeed_str.replace(' km/h', '').replace(' kph', ''))
                        elif 'mph' in maxspeed_str:
                            return int(maxspeed_str.replace(' mph', '')) * 1.60934
        return None
    except:
        return None

def get_incidents():
    try:
        params = {
            "key": TOMTOM_KEY,
            "bbox": f"{HCM_LON_MIN},{HCM_LAT_MIN},{HCM_LON_MAX},{HCM_LAT_MAX}",
            "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code},startTime,endTime,from,to,length,delay,roadNumbers,timeValidity}}}",
            "language": "vi-VN",
            "timeValidityFilter": "present"
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

def process_road(road, incidents):
    points = generate_points_along_line(road["lat"], road["lon"],
                                        road["lat"]+0.01, road["lon"]+0.01)
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

    current_speed_avg = sum(speeds)/len(speeds)
    free_flow_speed_avg = sum(free_speeds)/len(free_speeds) if free_speeds else current_speed_avg
    valid_jams = [j for j in jams if j is not None]
    congestion_index = current_speed_avg / free_flow_speed_avg if free_flow_speed_avg else 1.0
    cross_time = SEGMENT_LENGTH_KM / current_speed_avg * 3600 if current_speed_avg else None

    name_vn = max(set(name_vn_list), key=name_vn_list.count) if name_vn_list else road["name"]
    lane_count_avg = sum(lane_counts)/len(lane_counts) if lane_counts else 1
    speed_limit_avg = sum(speed_limits)/len(speed_limits) if speed_limits else 50
    incident_flag = 1 if any(incident_flags) else 0
    segment_id = max(set(segment_ids), key=segment_ids.count) if segment_ids else hashlib.md5(road["name"].encode()).hexdigest()

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

    # Timestamp yymmddhhmm
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    day_of_week = datetime.now().weekday()

    return {
        "segmentId": segment_id,
        "name": road["name"],
        "name_vn": name_vn,
        "lat": road["lat"],
        "lon": road["lon"],
        "roadType": frc,
        "laneCount": lane_count_avg,
        "frc": frc,
        "currentSpeed": current_speed_avg,
        "freeFlowSpeed": free_flow_speed_avg,
        "jamFactor": sum(valid_jams)/len(valid_jams) if valid_jams else 0,
        "congestionIndex": congestion_index,
        "crossTime": cross_time,
        "trafficVolume": "NA",
        "occupancy": "NA",
        "speedLimit": speed_limit_avg,
        "incidentFlag": incident_flag,
        "LOS": los,
        "timeStamp": timestamp,
        "dayOfWeek": day_of_week
    }

if __name__ == "__main__":
    incidents = get_incidents()
    roads = get_main_roads_in_hcm(limit=200)
    if not roads:
        print("No roads found")
        exit()

    traffic_data = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_road, road, incidents) for road in roads]
        for future in as_completed(futures):
            result = future.result()
            if result:
                traffic_data.append(result)

    df = pd.DataFrame(traffic_data)
    if df.empty:
        print("No traffic data collected")
        exit()

    df = df.groupby("name", as_index=False).agg({
        "segmentId": "first",
        "name_vn": "first",
        "lat": "first",
        "lon": "first",
        "roadType": "first",
        "laneCount": "mean",
        "frc": "first",
        "currentSpeed": "mean",
        "freeFlowSpeed": "mean",
        "jamFactor": "mean",
        "congestionIndex": "mean",
        "crossTime": "mean",
        "trafficVolume": "first",
        "occupancy": "first",
        "speedLimit": "mean",
        "incidentFlag": "max",
        "LOS": "first",
        "timeStamp": "first",
        "dayOfWeek": "first"
    })


    output_file = "../data/traffic/traffic_hcm.csv"

    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file, encoding="utf-8-sig")
        df = pd.concat([df_existing, df], ignore_index=True)
        df.drop_duplicates(subset=["segmentId", "timeStamp"], keep="last", inplace=True)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Traffic data saved/appended to {output_file}")
