#!/usr/bin/env python3
"""
collector_full_async.py

Gộp collector_async.py + mapbox_full_async.py
- Thu thập: TomTom, HERE, OSM, Mapbox (Directions + Map Matching)
- Async + semaphore + retry/backoff
- Field Mapbox đặt tên chuẩn (KHÔNG dùng prefix mb_)
- Xuất CSV cuối cùng
"""

import os
import asyncio
import math
import time
import hashlib
from datetime import datetime

import aiohttp
import async_timeout
import pandas as pd
from dotenv import load_dotenv
from geopy.distance import geodesic

# ================= ENV =================
load_dotenv()
TOMTOM_KEY = os.getenv("TOMTOM_KEY")
HERE_KEY = os.getenv("HERE_KEY")
MAPBOX_KEY = os.getenv("MAPBOX_KEY")

# ================= CONFIG =================
INPUT_FILE = "../data/traffic/streets_merged_10.csv"
OUTPUT_FILE = "../data/traffic/traffic_hcm_full_chat.csv"

MAX_CONCURRENT = 12
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
PER_REQUEST_DELAY = 0.05
OVERPASS_DELAY = 0.4
SEGMENT_LENGTH_KM = 0.5

# ================= URL =================
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


# ================= HELPERS =================
def safe_int(v):
    try:
        return int(v)
    except:
        return None


def bearing(lat1, lon1, lat2, lon2):
    dLon = math.radians(lon2 - lon1)
    y = math.sin(dLon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.cos(dLon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def estimate_traffic_volume(cur, free, lanes, base=1800):
    if not cur or not free:
        util = 0.5
    else:
        util = max(0.05, min(1.0, 1.2 - cur / free))
    return round(base * max(1, lanes or 1) * util, 2)


def get_los(ci):
    return "ABCDEF"[sum(ci < x for x in [0.9, 0.7, 0.5, 0.3, 0.1])]


# ================= ASYNC FETCHER =================
class AsyncFetcher:
    def __init__(self):
        self.sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    async def request(self, method, url, **kwargs):
        for i in range(1, MAX_RETRIES + 1):
            try:
                async with self.sem:
                    await asyncio.sleep(PER_REQUEST_DELAY)
                    async with async_timeout.timeout(REQUEST_TIMEOUT):
                        async with self.session.request(method, url, **kwargs) as r:
                            if r.status == 200:
                                return await r.json()
                            if r.status in (429, 502, 503, 504):
                                await asyncio.sleep(BACKOFF_FACTOR**i)
            except:
                await asyncio.sleep(BACKOFF_FACTOR**i)
        return None


# ================= MAPBOX =================
async def mapbox_directions(fetcher, slat, slon, elat, elon):
    url = f"{MAPBOX_DIRECTIONS_URL}/{slon},{slat};{elon},{elat}"
    params = {
        "access_token": MAPBOX_KEY,
        "geometries": "geojson",
        "steps": "true",
        "annotations": "speed,distance,duration,congestion",
    }
    r = await fetcher.request("GET", url, params=params)
    if not r:
        return {}

    route = r["routes"][0]
    steps = route["legs"][0]["steps"]

    speeds = []
    lanes = []
    maneuvers = []
    step_dist = []
    step_dur = []

    for s in steps:
        ann = s.get("annotation", {})
        speeds += [v for v in ann.get("speed", []) if v]
        step_dist.append(s.get("distance"))
        step_dur.append(s.get("duration"))
        maneuvers.append(s.get("maneuver", {}).get("type"))

        inter = s.get("intersections", [])
        lanes.append(
            len(inter[0]["lanes"]) if inter and inter[0].get("lanes") else None
        )

    return {
        "dist_m": route.get("distance"),
        "duration_sec": route.get("duration"),
        "route_weight": route.get("weight"),
        "geometry": route.get("geometry", {}).get("coordinates"),
        "avg_speed": sum(speeds) / len(speeds) if speeds else None,
        "lane_count": max([l for l in lanes if l], default=None),
        "maneuvers": maneuvers,
        "step_distances_m": step_dist,
        "step_durations_sec": step_dur,
    }


async def mapbox_match(fetcher, pts):
    coords = ";".join([f"{lon},{lat}" for lon, lat in pts])
    url = f"{MAPBOX_MAPMATCH_URL}/{coords}"
    params = {
        "access_token": MAPBOX_KEY,
        "geometries": "geojson",
    }
    r = await fetcher.request("GET", url, params=params)
    if not r:
        return {}
    m = r["matchings"][0]
    return {
        "match_confidence": m.get("confidence"),
        "matched_geometry": m.get("geometry", {}).get("coordinates"),
    }


# ================= MAIN PROCESS =================
async def process_row(fetcher, row):
    slat, slon = row["lat_snode"], row["long_snode"]
    elat, elon = row["lat_enode"], row["long_enode"]

    mb_dir = await mapbox_directions(fetcher, slat, slon, elat, elon)
    mb_match = await mapbox_match(fetcher, [(slon, slat), (elon, elat)])

    length_km = geodesic((slat, slon), (elat, elon)).km
    b = bearing(slat, slon, elat, elon)

    return {
        **row,
        **mb_dir,
        **mb_match,
        "lengthKm": length_km,
        "bearing": b,
        "timeStamp": datetime.now().strftime("%y%m%d%H%M"),
        "dayOfWeek": datetime.now().weekday(),
    }


async def main():
    df = pd.read_csv(INPUT_FILE)
    async with AsyncFetcher() as fetcher:
        tasks = [process_row(fetcher, r) for _, r in df.iterrows()]
        rows = [await t for t in asyncio.as_completed(tasks)]

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved {len(out)} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print("Total time:", round(time.time() - t0, 2), "s")
