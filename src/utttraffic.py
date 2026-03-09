import requests
import json

URL = "https://api.bktraffic.com/api/segment/get-status-v3"

# Bounding box khu vực TP.HCM (có thể chỉnh lại)
payload = {
    "minLat": 10.70,
    "minLon": 106.60,
    "maxLat": 10.85,
    "maxLon": 106.75
}

headers = {
    "Content-Type": "application/json"
}

def main():
    response = requests.post(URL, json=payload, headers=headers)

    if response.status_code != 200:
        print("Request failed:", response.status_code)
        print(response.text)
        return

    data = response.json()

    print("Total segments:", len(data))

    for seg in data[:10]:  # in thử 10 segment đầu
        segment_id = seg.get("segment_id")
        velocity = seg.get("velocity")
        street = seg.get("street", {}).get("name")

        print(f"{segment_id} | {street} | {velocity} km/h")


if __name__ == "__main__":
    main()
