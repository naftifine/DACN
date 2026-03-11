"""
traffic_collector_split.py

Tách riêng dữ liệu tĩnh và động từ file traffic_collector_full.py gốc.
- Chạy static: thu thập các thuộc tính đường không thay đổi (lưu traffic_static.csv)
- Chạy dynamic: thu thập dữ liệu giao thông thời gian thực, kết hợp với static và ghi vào output
- Chạy full: thực hiện cả hai bước

Cách dùng:
    python traffic_collector_split.py static
    python traffic_collector_split.py dynamic
    python traffic_collector_split.py full
"""

import os
import asyncio
import math
import time
import hashlib
import random
import sys
from datetime import datetime

import aiohttp
import async_timeout
import pandas as pd
from dotenv import load_dotenv
from geopy.distance import geodesic


load_dotenv()
TOMTOM_KEY = os.getenv("TOMTOM_KEY")
HERE_KEY = os.getenv("HERE_KEY")
MAPBOX_KEY = os.getenv("MAPBOX_KEY")

print("API Keys loaded:")
print("TOMTOM_KEY:", TOMTOM_KEY[:4] + "****" if TOMTOM_KEY else "TOMTOM_KEY not set")
print("HERE_KEY:", HERE_KEY[:4] + "****" if HERE_KEY else "HERE_KEY not set")
print("MAPBOX_KEY:", MAPBOX_KEY[:4] + "****" if MAPBOX_KEY else "MAPBOX_KEY not set")

INPUT_FILE = "../data/traffic/streets_merged.csv"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

OUTPUT_FILE = f"../DATN/traffic_hcm_{timestamp}.csv"
STATIC_FILE = "../data/traffic/traffic_static.csv"

MAX_CONCURRENT = 20
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
PER_REQUEST_DELAY = 0.05
OVERPASS_DELAY = 1

SEGMENT_LENGTH_KM = 0.5


TRAFFIC_URL_TOMTOM = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
)
REVERSE_URL_TOMTOM = "https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json"
SNAP_URL_TOMTOM = "https://api.tomtom.com/snapToRoads/1/snapToRoads"
INCIDENT_URL_TOMTOM = "https://api.tomtom.com/traffic/services/5/incidentDetails"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HERE_ROUTER = "https://router.hereapi.com/v8/routes"
MAPBOX_DIRECTIONS_URL = "https://api.mapbox.com/directions/v5/mapbox/driving"
MAPBOX_MAPMATCH_URL = "https://api.mapbox.com/matching/v5/mapbox/driving"


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


def safe_int(val):
    try:
        if isinstance(val, (int, float)):
            return int(val)
        return int(val)
    except:
        return None


def estimate_traffic_volume(current_speed, free_speed, lane_count, base_capacity=1800):
    if current_speed is None or free_speed is None or free_speed <= 0:
        utilization = 0.5
    else:
        congestion_index = current_speed / free_speed
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


def curvature_index(start_lat, start_lon, end_lat, end_lon):
    return round(abs(start_lat - end_lat) + abs(start_lon - end_lon), 6)


def bearing(lat1, lon1, lat2, lon2):
    dLon = math.radians(lon2 - lon1)
    y = math.sin(dLon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.cos(dLon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360) % 360


def get_elevation(lat, lon):
    """
    PLACEHOLDER: Thay thế bằng API Elevation thực tế (VD: Google/USGS)
    Giá trị trả về giả định (dao động từ 5m đến 35m)
    """
    lat_int = int(lat * 1000)
    lon_int = int(lon * 1000)
    return 5.0 + ((lat_int + lon_int) % 30)


def get_intersection_count(lat, lon):
    """
    PLACEHOLDER: Thay thế bằng query Overpass API phức tạp hơn.
    Giá trị trả về giả định
    """
    return 1 + random.randint(0, 4)


class AsyncFetcher:
    def __init__(self, max_concurrent=MAX_CONCURRENT):
        self.sem = asyncio.Semaphore(max_concurrent)
        self.session = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    async def _request(self, method, url, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with self.sem:
                    await asyncio.sleep(PER_REQUEST_DELAY)

                    async with async_timeout.timeout(REQUEST_TIMEOUT):
                        async with self.session.request(method, url, **kwargs) as resp:
                            text = await resp.text()
                            status = resp.status
                            if status == 200:
                                try:
                                    return await resp.json()
                                except Exception:
                                    return {"_raw": text}
                            elif status in (429, 503, 502, 504):
                                wait = BACKOFF_FACTOR**attempt
                                await asyncio.sleep(wait)
                                continue
                            else:
                                return None
            except asyncio.TimeoutError:
                wait = BACKOFF_FACTOR**attempt
                await asyncio.sleep(wait)
                continue
            except Exception:
                wait = BACKOFF_FACTOR**attempt
                await asyncio.sleep(wait)
                continue
        return None

    async def get_json(self, url, params=None, headers=None, data=None, method="GET"):
        kwargs = {}
        if params:
            kwargs["params"] = params
        if headers:
            kwargs["headers"] = headers
        if data:
            kwargs["data"] = data
        return await self._request(method, url, **kwargs)


# -------------------- CÁC HÀM GỌI API (giữ nguyên) --------------------
async def get_traffic_tomtom(fetcher: AsyncFetcher, lat, lon):
    params = {"point": f"{lat},{lon}", "unit": "KMPH", "key": TOMTOM_KEY}
    res = await fetcher.get_json(TRAFFIC_URL_TOMTOM, params=params)
    if res and isinstance(res, dict):
        return res.get("flowSegmentData", {}) or {}
    return {}


async def reverse_geocode_tomtom(fetcher: AsyncFetcher, lat, lon):
    url = REVERSE_URL_TOMTOM.format(lat=lat, lon=lon)
    params = {"key": TOMTOM_KEY, "language": "vi-VN"}
    res = await fetcher.get_json(url, params=params)
    try:
        addr = res["addresses"][0]["address"]
        return addr.get("streetName")
    except Exception:
        return None


async def snap_to_road_tomtom(fetcher: AsyncFetcher, lat, lon):
    params = {"key": TOMTOM_KEY, "points": f"{lon},{lat}"}
    res = await fetcher.get_json(SNAP_URL_TOMTOM, params=params)
    if not res:
        return None, None, None
    props = {}
    if "route" in res and isinstance(res["route"], dict):
        props = res["route"].get("properties", {}) or res["route"]
    lane_count = None
    speed_limit = None
    segment_id = None
    try:
        lane_count = safe_int(props.get("laneInfo", {}).get("numberOfLanes"))
    except:
        lane_count = None
    sl = props.get("speedLimits")
    if isinstance(sl, dict):
        speed_limit = safe_int(sl.get("value"))
    elif isinstance(sl, list) and sl:
        speed_limit = safe_int(sl[0].get("value"))
    segment_id = props.get("id") or props.get("segmentId")
    return segment_id, lane_count, speed_limit


async def get_incidents_tomtom(fetcher: AsyncFetcher):
    params = {
        "key": TOMTOM_KEY,
        "bbox": f"106.3,10.3,107.1,11.2",
        "language": "vi-VN",
        "timeValidityFilter": "present",
    }
    res = await fetcher.get_json(INCIDENT_URL_TOMTOM, params=params)
    if res and isinstance(res, dict):
        return res.get("incidents", []) or []
    return []


async def get_osm_attributes(fetcher: AsyncFetcher, lat, lon, radius=50):
    query = f"""
[out:json][timeout:25];
way(around:{radius},{lat},{lon})[highway];
out tags;
"""
    res = await fetcher.get_json(OVERPASS_URL, data={"data": query}, method="POST")
    await asyncio.sleep(OVERPASS_DELAY)
    if res and isinstance(res, dict) and "elements" in res and res["elements"]:
        tags = res["elements"][0].get("tags", {}) or {}
        return {
            "roadType": tags.get("highway"),
            "lanes": safe_int(tags.get("lanes")),
            "speedLimit_osm": safe_int(tags.get("maxspeed")),
            "oneway": tags.get("oneway"),
            "surface": tags.get("surface"),
            "width": tags.get("width"),
            "ref": tags.get("ref"),
            "bus": tags.get("route") == "bus" or tags.get("public_transport"),
            "cycleway": tags.get("cycleway"),
            "sidewalk": tags.get("sidewalk"),
            "bridge": tags.get("bridge"),
            "tunnel": tags.get("tunnel"),
            "maxheight": tags.get("maxheight"),
            "maxweight": tags.get("maxweight"),
            "barrier": tags.get("barrier"),
            "lit": tags.get("lit"),
            "crossing": tags.get("crossing"),
            "amenity": tags.get("amenity"),
        }
    return {}


async def get_here_attributes(fetcher: AsyncFetcher, lat, lon):
    params = {
        "transportMode": "car",
        "origin": f"{lat},{lon}",
        "destination": f"{lat+0.0001},{lon+0.0001}",
        "return": "summary,polyline",
        "apikey": HERE_KEY,
    }
    res = await fetcher.get_json(HERE_ROUTER, params=params)
    if res and isinstance(res, dict):
        try:
            lane_count = safe_int(
                res.get("routes", [{}])[0]
                .get("sections", [{}])[0]
                .get("summary", {})
                .get("laneCount")
            )
            return {"lanes_here": lane_count}
        except:
            return {}
    return {}


async def get_mapbox_directions(
    fetcher: AsyncFetcher, start_lat, start_lon, end_lat, end_lon
):
    coords = f"{start_lon},{start_lat};{end_lon},{end_lat}"
    params = {
        "access_token": MAPBOX_KEY,
        "geometries": "geojson",
        "overview": "full",
        "steps": "true",
        "annotations": "speed,distance,duration,congestion",
    }
    url = f"{MAPBOX_DIRECTIONS_URL}/{coords}"
    res = await fetcher.get_json(url, params=params)
    if not res:
        return {}
    try:
        route = res.get("routes", [{}])[0]
        distance = route.get("distance")
        duration = route.get("duration")
        weight = route.get("weight")
        weight_name = route.get("weight_name")
        geometry = route.get("geometry", {}).get("coordinates")
        steps = route.get("legs", [{}])[0].get("steps", [])

        avg_speeds = []
        lane_counts = []
        congestions = []

        for step in steps:
            ann = step.get("annotation", {})
            if "speed" in ann:
                spd_list = [s for s in ann["speed"] if s is not None]
                if spd_list:
                    avg_speeds.extend(spd_list)

            intersections = step.get("intersections", [])
            if intersections and intersections[0].get("lanes"):
                lane_counts.append(len(intersections[0]["lanes"]))

            if "congestion" in ann:
                congestions.extend(ann["congestion"])

        avg_speed = sum(avg_speeds) / len(avg_speeds) if avg_speeds else None
        lane_count = max([lc for lc in lane_counts if lc] or [None])

        return {
            "dist_m": distance,
            "duration_sec": duration,
            "route_weight": weight,
            "mapbox_avg_speed": avg_speed,
            "lane_count_mapbox": lane_count,
            "congestion_mapbox": congestions,
        }
    except Exception as e:
        print(f"Error parsing Mapbox Directions: {e}")
        return {}


async def get_mapbox_mapmatch(fetcher: AsyncFetcher, points):
    """
    points: list [(lon, lat), ...]
    """
    if not points or len(points) < 2:
        return {}
    coords_str = ";".join([f"{lon},{lat}" for lon, lat in points])
    params = {
        "access_token": MAPBOX_KEY,
        "geometries": "geojson",
        "steps": "true",
        "annotations": "speed,distance,duration,congestion",
        "overview": "full",
    }
    url = f"{MAPBOX_MAPMATCH_URL}/{coords_str}"
    res = await fetcher.get_json(url, params=params)
    if not res:
        return {}
    try:
        match = res.get("matchings", [{}])[0]
        confidence = match.get("confidence")
        return {"match_confidence": confidence}
    except Exception as e:
        print(f"Error parsing Mapbox Map Match: {e}")
        return {}


# -------------------- THU THẬP DỮ LIỆU TĨNH --------------------
async def collect_static_segment(fetcher: AsyncFetcher, row):
    start_lat = float(row["lat_snode"])
    start_lon = float(row["long_snode"])
    end_lat = float(row["lat_enode"])
    end_lon = float(row["long_enode"])

    points = [(start_lat, start_lon), (end_lat, end_lon)]

    lane_counts = []
    speed_limits = []
    segment_ids = []
    names = []
    frc = None
    roadType = None
    surface = None
    oneway = None
    width = None
    ref = None
    osm_structural_attrs = {}

    tasks_per_point = []
    for lat, lon in points:
        snap_task = snap_to_road_tomtom(fetcher, lat, lon)
        traffic_task = get_traffic_tomtom(fetcher, lat, lon)  # chỉ lấy frc
        osm_task = get_osm_attributes(fetcher, lat, lon)
        here_task = get_here_attributes(fetcher, lat, lon)
        rev_task = reverse_geocode_tomtom(fetcher, lat, lon)

        tasks_per_point.append(
            asyncio.gather(snap_task, traffic_task, osm_task, here_task, rev_task)
        )

    results_per_point = await asyncio.gather(*tasks_per_point)

    for i, (seg_id_tt, traffic, osm_attrs, here_attrs, street_name) in enumerate(
        results_per_point
    ):
        lat, lon = points[i]

        if isinstance(seg_id_tt, tuple) and len(seg_id_tt) == 3:
            seg_id_tt, lane_tt, speed_tt = seg_id_tt
        else:
            lane_tt, speed_tt = None, None

        traffic = traffic or {}
        osm_attrs = osm_attrs or {}
        here_attrs = here_attrs or {}
        street_name = street_name if street_name else row.get("street_name")

        frc = frc or traffic.get("frc")

        if not roadType and osm_attrs:
            roadType = osm_attrs.get("roadType")
            surface = osm_attrs.get("surface")
            oneway = osm_attrs.get("oneway")
            width = osm_attrs.get("width")
            ref = osm_attrs.get("ref")
            osm_structural_attrs = {
                k: v
                for k, v in osm_attrs.items()
                if k not in ["lanes", "speedLimit_osm"]
            }

        lane_osm = osm_attrs.get("lanes")
        speed_osm = osm_attrs.get("speedLimit_osm")
        lane_here = here_attrs.get("lanes_here")

        lane_count = lane_tt or lane_here or lane_osm or 1
        speed_limit = speed_tt or speed_osm or 50

        segment_id = seg_id_tt or hashlib.md5(f"{lat},{lon}".encode()).hexdigest()

        lane_counts.append(lane_count)
        speed_limits.append(speed_limit)
        segment_ids.append(segment_id)
        names.append(street_name if street_name else row.get("street_name"))

    lane_avg = sum(lane_counts) / len(lane_counts) if lane_counts else 1
    speed_avg = sum(speed_limits) / len(speed_limits) if speed_limits else 50
    segment_id = (
        max(set(segment_ids), key=segment_ids.count)
        if segment_ids
        else row.get("segmentId_source")
    )
    name_final = max(set(names), key=names.count) if names else row.get("street_name")

    # Các chỉ số hình học (tĩnh)
    start_ele = get_elevation(start_lat, start_lon)
    end_ele = get_elevation(end_lat, end_lon)
    length_km = geodesic((start_lat, start_lon), (end_lat, end_lon)).km
    route_slope = (
        ((end_ele - start_ele) / (length_km * 1000)) * 100 if length_km > 0 else 0.0
    )
    intersection_count = get_intersection_count(start_lat, start_lon)
    curvature = curvature_index(start_lat, start_lon, end_lat, end_lon)
    bearing_deg = bearing(start_lat, start_lon, end_lat, end_lon)

    static_row = {
        "segmentId": segment_id,
        "name_vn": name_final,
        "lat_start": start_lat,
        "lon_start": start_lon,
        "lat_end": end_lat,
        "lon_end": end_lon,
        "laneCount_aggregated": lane_avg,
        "speedLimit": speed_avg,
        "frc": frc,
        "roadType": roadType,
        "surface": surface,
        "oneway": oneway,
        "width": width,
        "ref": ref,
        "busRoute": osm_structural_attrs.get("bus"),
        "bicycleLane": osm_structural_attrs.get("cycleway"),
        "sidewalk": osm_structural_attrs.get("sidewalk"),
        "bridgeTunnel": (
            osm_structural_attrs.get("bridge") or osm_structural_attrs.get("tunnel")
        ),
        "maxHeight": safe_int(osm_structural_attrs.get("maxheight")),
        "maxWeight": safe_int(osm_structural_attrs.get("maxweight")),
        "barrier": osm_structural_attrs.get("barrier"),
        "lit": osm_structural_attrs.get("lit"),
        "crossing": osm_structural_attrs.get("crossing"),
        "curvatureIndex": curvature,
        "bearing": bearing_deg,
        "lengthKm": length_km,
        "intersectionCount": intersection_count,
        "routeSlopePercent": route_slope,
        "startElevation": start_ele,
        "endElevation": end_ele,
    }
    return static_row


async def run_static_collection():
    df_roads = pd.read_csv(INPUT_FILE)
    if df_roads.empty:
        print("No roads in input file")
        return

    async with AsyncFetcher(max_concurrent=MAX_CONCURRENT) as fetcher:
        tasks = [collect_static_segment(fetcher, row) for _, row in df_roads.iterrows()]
        static_results = []
        for fut in asyncio.as_completed(tasks):
            try:
                res = await fut
                if res:
                    static_results.append(res)
            except Exception as e:
                print(f"Error in static collection: {e}")

    if not static_results:
        print("No static data collected")
        return

    df_static = pd.DataFrame(static_results)
    df_static.to_csv(STATIC_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved static data for {len(df_static)} segments to {STATIC_FILE}")
    return df_static


# -------------------- THU THẬP DỮ LIỆU ĐỘNG --------------------
async def collect_dynamic_segment(fetcher: AsyncFetcher, static_row, incidents):
    start_lat = static_row["lat_start"]
    start_lon = static_row["lon_start"]
    end_lat = static_row["lat_end"]
    end_lon = static_row["lon_end"]
    lane_avg = static_row["laneCount_aggregated"]
    speed_limit = static_row["speedLimit"]
    length_km = static_row["lengthKm"]

    points = [(start_lat, start_lon), (end_lat, end_lon)]

    # Traffic cho 2 điểm
    traffic_tasks = [get_traffic_tomtom(fetcher, lat, lon) for lat, lon in points]
    traffic_results = await asyncio.gather(*traffic_tasks)

    current_speeds = []
    free_speeds = []
    jam_factors = []
    for t in traffic_results:
        current_speeds.append(t.get("currentSpeed") or 0)
        free_speeds.append(t.get("freeFlowSpeed") or 0)
        jam_factors.append(t.get("jamFactor") or 0)

    # Incident flag
    incident_flags = [is_incident_near(lat, lon, incidents) for lat, lon in points]
    incident_flag = 1 if any(incident_flags) else 0

    # Mapbox
    mapbox_points = [(start_lon, start_lat), (end_lon, end_lat)]
    directions_task = get_mapbox_directions(
        fetcher, start_lat, start_lon, end_lat, end_lon
    )
    mapmatch_task = get_mapbox_mapmatch(fetcher, mapbox_points)
    directions, mapmatch = await asyncio.gather(directions_task, mapmatch_task)

    # Tính các chỉ số động
    current_speed_avg = (
        sum(current_speeds) / len(current_speeds) if current_speeds else 0
    )
    free_flow_avg = (
        sum(free_speeds) / len(free_speeds) if free_speeds else current_speed_avg
    )

    congestion_index = (
        current_speed_avg / free_flow_avg
        if free_flow_avg and free_flow_avg > 0
        else 1.0
    )
    traffic_volume = estimate_traffic_volume(current_speed_avg, free_flow_avg, lane_avg)
    los = get_los(congestion_index)
    occupancy = (
        100 * (1 - current_speed_avg / free_flow_avg)
        if free_flow_avg and free_flow_avg > 0
        else 0
    )
    cross_time = (
        length_km / (current_speed_avg / 3.6)
        if current_speed_avg and current_speed_avg > 0
        else None
    )
    speed_limit_ratio = (
        current_speed_avg / speed_limit if speed_limit and speed_limit > 0 else 1.0
    )
    relative_congestion_index = (
        (free_flow_avg - current_speed_avg) / free_flow_avg
        if free_flow_avg and free_flow_avg > 0
        else 0.0
    )

    now = datetime.now()
    timestamp = now.strftime("%y%m%d%H%M")
    day_of_week = now.weekday()
    hour_of_day = now.hour
    day_type = "Weekend" if day_of_week >= 5 else "Weekday"

    dynamic_fields = {
        "timeStamp": timestamp,
        "dayOfWeek": day_of_week,
        "hourOfDay": hour_of_day,
        "dayType": day_type,
        "currentSpeed": current_speed_avg,
        "freeFlowSpeed": free_flow_avg,
        "jamFactor": sum(jam_factors) / len(jam_factors) if jam_factors else 0,
        "incidentFlag": incident_flag,
        "congestionIndex": congestion_index,
        "trafficVolume": traffic_volume,
        "LOS": los,
        "occupancy": occupancy,
        "crossTime": cross_time,
        "speedLimitRatio": speed_limit_ratio,
        "relativeCongestionIndex": relative_congestion_index,
        # Mapbox fields
        "mapbox_dist_m": directions.get("dist_m"),
        "mapbox_duration_sec": directions.get("duration_sec"),
        "mapbox_route_weight": directions.get("route_weight"),
        "mapbox_avg_speed": directions.get("mapbox_avg_speed"),
        "mapbox_congestion": directions.get("congestion_mapbox"),
        "mapbox_match_confidence": mapmatch.get("match_confidence"),
        "mapbox_lane_count": directions.get("lane_count_mapbox"),
    }

    # Kết hợp với static
    full_row = {**static_row, **dynamic_fields}
    return full_row


async def run_dynamic_collection():
    if not os.path.exists(STATIC_FILE):
        print(
            f"Static file {STATIC_FILE} not found. Please run static collection first."
        )
        return

    df_static = pd.read_csv(STATIC_FILE)
    if df_static.empty:
        print("Static file is empty")
        return

    async with AsyncFetcher(max_concurrent=MAX_CONCURRENT) as fetcher:
        incidents = await get_incidents_tomtom(fetcher)

        tasks = []
        for _, row in df_static.iterrows():
            tasks.append(collect_dynamic_segment(fetcher, row, incidents))

        dynamic_results = []
        for fut in asyncio.as_completed(tasks):
            try:
                res = await fut
                if res:
                    dynamic_results.append(res)
            except Exception as e:
                print(f"Error in dynamic collection: {e}")

    if not dynamic_results:
        print("No dynamic data collected")
        return

    df_dynamic = pd.DataFrame(dynamic_results)

    # Ghi đè hoặc append vào file output
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        df_full = pd.concat([df_existing, df_dynamic], ignore_index=True)
        id_col = None
        for col in ["segmentId", "segmentId_source"]:
            if col in df_full.columns:
                id_col = col
                break
        if id_col is None:
            raise KeyError(
                "Không tìm thấy cột định danh (segmentId/segmentId_source) trong dữ liệu."
            )

        df_full.drop_duplicates(subset=[id_col, "timeStamp"], keep="last", inplace=True)
    else:
        df_full = df_dynamic

    dup_cols = ["mapbox_avg_speed", "mapbox_lane_count"]
    df_full.drop(
        columns=[c for c in dup_cols if c in df_full.columns],
        inplace=True,
        errors="ignore",
    )
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
    df_full.drop(
        columns=[c for c in cols_always_empty if c in df_full.columns],
        inplace=True,
        errors="ignore",
    )

    rename_map = {
        "mapbox_dist_m": "route_distance_m",
        "mapbox_duration_sec": "route_duration_sec",
        "mapbox_route_weight": "route_weight",
        "mapbox_congestion": "congestion_levels",
        "mapbox_match_confidence": "map_match_confidence",
    }
    df_full.rename(columns=rename_map, inplace=True)

    df_full.fillna({"surface": "asphalt", "oneway": "no"}, inplace=True)
    df_full.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(
        f"Saved {len(df_dynamic)} new records to {OUTPUT_FILE}. Total: {len(df_full)}"
    )
    return df_full


# -------------------- MAIN --------------------
async def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "dynamic"  # mặc định là dynamic (có thể đổi thành full nếu muốn)

    if mode == "static":
        print("=== Chạy thu thập dữ liệu TĨNH (một lần) ===")
        await run_static_collection()
    elif mode == "dynamic":
        print("=== Chạy thu thập dữ liệu ĐỘNG (định kỳ) ===")
        await run_dynamic_collection()
    elif mode == "full":
        print("=== Chạy FULL: thu thập tĩnh + động ===")
        if not os.path.exists(STATIC_FILE):
            await run_static_collection()
        await run_dynamic_collection()
    else:
        print("Usage: python script.py [static|dynamic|full]")


if __name__ == "__main__":
    t0 = time.time()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    print("Total time:", time.time() - t0)
