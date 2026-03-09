import pandas as pd
import json
import math
from collections import defaultdict

CSV_FILE = "../data/traffic/traffic_hcm.csv"

# ================== Haversine ==================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


# ================== Load ==================
df = pd.read_csv(CSV_FILE)

BIN_SIZE = 0.5   # km
MAX_BIN = 10     # max display range (0 → 10km)

bins = defaultdict(int)

valid_segments = 0
skip_null = 0


# ================== Process ==================
for _, row in df.iterrows():

    geom = row["geometry"]

    if pd.isna(geom):
        skip_null += 1
        continue

    try:
        coords = json.loads(geom)
    except:
        continue

    seg_length = 0

    for i in range(len(coords)-1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i+1]

        seg_length += haversine(lat1, lon1, lat2, lon2)

    seg_km = seg_length / 1000
    valid_segments += 1

    # ===== BINNING =====
    bin_index = int(seg_km // BIN_SIZE)

    lower = bin_index * BIN_SIZE
    upper = lower + BIN_SIZE

    label = f"{lower:.1f}-{upper:.1f} km"

    bins[label] += 1


# ================== SORT & PRINT ==================
print("\n======= SEGMENT LENGTH DISTRIBUTION (0.5 km bins) =======")
print("Valid segments:", valid_segments)
print("Skipped geometry NULL:", skip_null)
print("--------------------------------------------------------")

for k in sorted(bins.keys(), key=lambda x: float(x.split('-')[0])):
    count = bins[k]
    percent = (count / valid_segments) * 100
    print(f"{k:12s} : {count:4d} segments ({percent:.2f}%)")
