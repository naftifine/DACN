#!/usr/bin/env python3
"""
mapbox_full_async.py

Bổ sung dữ liệu Mapbox (Directions + Map Matching) vào CSV từ collector_async.py
- Lấy tối đa các attribute có thể: distance, duration, geometry, speed, lanes, surface, congestion, maneuvers, road class...
- Async + semaphore + polite delay
"""

import os
import asyncio
import aiohttp
import async_timeout
import pandas as pd
from dotenv import load_dotenv
from geopy.distance import geodesic
import math

load_dotenv()
MAPBOX_KEY = os.getenv("MAPBOX_KEY")

# CONFIG
INPUT_FILE = "../data/traffic/traffic_hcm_1107.csv"
OUTPUT_FILE = "../data/traffic/traffic_hcm_1107_full.csv"

MAX_CONCURRENT = 10
REQUEST_TIMEOUT = 15
PER_REQUEST_DELAY = 0.05
BACKOFF_FACTOR = 1.5
MAX_RETRIES = 3

MAPBOX_DIRECTIONS_URL = "https://api.mapbox.com/directions/v5/mapbox/driving"
MAPBOX_MAPMATCH_URL = "https://api.mapbox.com/matching/v5/mapbox/driving"


# ---------------- Async Fetcher ----------------
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
                            status = resp.status
                            text = await resp.text()
                            if status == 200:
                                try:
                                    return await resp.json()
                                except Exception:
                                    return {"_raw": text}
                            elif status in (429, 503, 502, 504):
                                await asyncio.sleep(BACKOFF_FACTOR**attempt)
                                continue
                            else:
                                return None
            except Exception:
                await asyncio.sleep(BACKOFF_FACTOR**attempt)
                continue
        return None

    async def get_json(self, url, params=None, data=None, method="GET"):
        kwargs = {}
        if params:
            kwargs["params"] = params
        if data:
            kwargs["data"] = data
        return await self._request(method, url, **kwargs)


# ---------------- Mapbox Directions ----------------
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
        "overview": "full",
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
        # extract per-step attributes
        avg_speeds = []
        speed_profile = []
        lane_counts = []
        surfaces = []
        maneuvers = []
        congestions = []
        start_coords = []
        end_coords = []
        durations_per_step = []
        distances_per_step = []
        road_classes = []
        for step in steps:
            # speed annotations
            ann = step.get("annotation", {})
            if "speed" in ann:
                spd_list = [s for s in ann["speed"] if s is not None]
                if spd_list:
                    avg_speeds.extend(spd_list)
                    speed_profile.append(spd_list)
            # lane info
            intersections = step.get("intersections", [])
            if intersections and intersections[0].get("lanes"):
                lane_counts.append(len(intersections[0]["lanes"]))
            else:
                lane_counts.append(None)
            # surface
            surfaces.append(step.get("driving_side", None))
            # maneuvers
            maneuvers.append(step.get("maneuver", {}).get("type"))
            # congestion
            if "congestion" in ann:
                congestions.extend(ann["congestion"])
            # step coordinates
            start_coords.append(step.get("maneuver", {}).get("location"))
            end_coords.append(
                step.get("geometry", {}).get("coordinates")[-1]
                if step.get("geometry")
                else None
            )
            durations_per_step.append(step.get("duration"))
            distances_per_step.append(step.get("distance"))
            road_classes.append(
                step.get("driving_side")
            )  # placeholder if Mapbox có `road_class`
        avg_speed = sum(avg_speeds) / len(avg_speeds) if avg_speeds else None
        lane_count = max([lc for lc in lane_counts if lc] or [None])
        return {
            "mb_distance": distance,
            "mb_duration": duration,
            "mb_weight": weight,
            "mb_weight_name": weight_name,
            "mb_geometry": geometry,
            "mb_avg_speed": avg_speed,
            "mb_speed_profile": speed_profile,
            "mb_lane_count": lane_count,
            "mb_surface": surfaces,
            "mb_maneuvers": maneuvers,
            "mb_congestion": congestions,
            "mb_start_coords": start_coords,
            "mb_end_coords": end_coords,
            "mb_durations_per_step": durations_per_step,
            "mb_distances_per_step": distances_per_step,
            "mb_road_classes": road_classes,
        }
    except Exception:
        return {}


# ---------------- Mapbox Map Matching ----------------
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
        matched_geometry = match.get("geometry", {}).get("coordinates")
        return {"mb_confidence": confidence, "mb_matched_geometry": matched_geometry}
    except Exception:
        return {}


# ---------------- Process each row ----------------
async def process_row(fetcher: AsyncFetcher, row):
    start_lat = row["lat_start"]
    start_lon = row["lon_start"]
    end_lat = row["lat_end"]
    end_lon = row["lon_end"]
    points = [(start_lon, start_lat), (end_lon, end_lat)]
    try:
        directions_attrs = await get_mapbox_directions(
            fetcher, start_lat, start_lon, end_lat, end_lon
        )
        mapmatch_attrs = await get_mapbox_mapmatch(fetcher, points)
        return {**row, **directions_attrs, **mapmatch_attrs}
    except Exception as e:
        print("Mapbox fetch error:", e)
        return row


# ---------------- Main ----------------
async def main():
    df = pd.read_csv(INPUT_FILE)
    if df.empty:
        print("No data in CSV")
        return

    async with AsyncFetcher() as fetcher:
        tasks = [process_row(fetcher, row) for _, row in df.iterrows()]
        results = []
        for fut in asyncio.as_completed(tasks):
            res = await fut
            results.append(res)

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df_out)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
