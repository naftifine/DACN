import pandas as pd
import json
import folium
import hashlib

# ================= CONFIG =================

CSV_FILE = "../data/traffic/traffic_hcm_12_16.csv"
OUTPUT_MAP = "traffic_map.html"

CENTER_LAT = 10.78
CENTER_LNG = 106.68
ZOOM_LEVEL = 12

# ================= LOAD DATA =================

df = pd.read_csv(CSV_FILE)

print("Total rows:", len(df))

# ================= INIT MAP =================

m = folium.Map(
    location=[CENTER_LAT, CENTER_LNG],
    zoom_start=ZOOM_LEVEL,
    tiles="cartodbpositron"
)

# ================= COLOR RULE =================

def get_color(ci):
    try:
        ci = float(ci)
    except:
        return "#9E9E9E"

    if ci >= 0.8:
        return "#00C853"   # green
    elif ci >= 0.6:
        return "#FFD600"   # yellow
    elif ci >= 0.4:
        return "#FF9100"   # orange
    else:
        return "#D50000"   # red


# ================= DUPLICATE CHECK =================

drawn_segments = set()
duplicate_count = 0
draw_count = 0


def geometry_hash(coords):
    """
    Hash geometry to detect duplicates
    """
    text = json.dumps(coords, sort_keys=True)
    return hashlib.md5(text.encode()).hexdigest()


# ================= DRAW =================

for idx, row in df.iterrows():

    geom = row["geometry"]

    if pd.isna(geom):
        print(f"[SKIP] Row {idx} geometry NULL")
        continue

    try:
        coords = json.loads(geom)
    except Exception as e:
        print(f"[SKIP] Row {idx} JSON error:", e)
        continue

    # ----- Duplicate detect -----

    geom_key = geometry_hash(coords)

    if geom_key in drawn_segments:
        duplicate_count += 1
        print(f"[DUPLICATE] segmentId:", row["segmentId"])
        continue

    drawn_segments.add(geom_key)

    # ----- Convert lng,lat -> lat,lng -----

    latlng = [(p[1], p[0]) for p in coords]

    # ----- Color -----

    color = get_color(row["congestionIndex"])

    # ----- Popup -----

    popup = f"""
    <b>{row['name_vn']}</b><br>
    SegmentID: {row['segmentId']}<br>
    Speed: {row['currentSpeed']} km/h<br>
    CongestionIndex: {row['congestionIndex']}<br>
    LOS: {row['LOS']}
    """

    # ----- Draw polyline -----

    folium.PolyLine(
        locations=latlng,
        color=color,
        weight=6,
        opacity=0.9,
        popup=popup
    ).add_to(m)

    draw_count += 1


# ================= SAVE =================

m.save(OUTPUT_MAP)

print("\n========== DONE ==========")
print("Drawn segments :", draw_count)
print("Duplicate found:", duplicate_count)
print("Output file    :", OUTPUT_MAP)
