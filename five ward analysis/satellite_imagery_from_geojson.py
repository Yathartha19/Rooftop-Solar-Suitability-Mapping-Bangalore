import os
import json
import geopandas as gpd
from shapely.geometry import Polygon, Point, shape
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# CONFIG
load_dotenv()  
API_KEY = os.getenv("GMAPS_API_KEY")
ZOOM = 20
SIZE = "640x640"
OUTPUT_DIR = "satimg"
GRID_SPACING = 0.00083  #~40–50m

# load wards
wards = gpd.read_file("closest_wards.geojson")
print(f"✅ Loaded {len(wards)} wards")

for idx, row in wards.iterrows():
    ward_name = row["KGISWardName"].strip().replace(" ", "_")
    print(f"\n🔵 === Starting ward: {ward_name} ===")

    ward_polygon = shape(row["geometry"])

    minx, miny, maxx, maxy = ward_polygon.bounds
    print(f"   Bounds: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")

    # generate grid points inside the ward polygon
    points = []
    print(f"   Generating grid points...")
    for lat in np.arange(miny, maxy, GRID_SPACING):
        for lon in np.arange(minx, maxx, GRID_SPACING):
            p = Point(lon, lat)
            if ward_polygon.contains(p):
                points.append((lat, lon))

    print(f"   ✅ Found {len(points)} grid points inside {ward_name}")

    if not points:
        print(f"   ⚠️  No grid points generated for {ward_name} — skipping.")
        continue

    # output folder
    ward_folder = os.path.join(OUTPUT_DIR, ward_name)
    os.makedirs(ward_folder, exist_ok=True)
    print(f"   📂 Output folder: {ward_folder}")

    coords_file = os.path.join(ward_folder, "coordinates.txt")
    with open(coords_file, "w") as f:
        for lat, lon in points:
            f.write(f"{lat},{lon}\n")
    print(f"   🗂️  Saved coordinates to {coords_file}")

    # download satellite imagery
    for i, (lat, lon) in enumerate(points, start=1):
        print(f"   🛰️  [{i}/{len(points)}] Downloading image for {lat:.6f}, {lon:.6f}...")

        url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}&zoom={ZOOM}&size={SIZE}&maptype=satellite&key={API_KEY}"
        )
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"   ❌ Failed to fetch image: HTTP {response.status_code} — Skipping")
                continue

            image = Image.open(BytesIO(response.content))
            filename = f"ward_{ward_name}_{lat}_{lon}.png"
            image_path = os.path.join(ward_folder, filename)
            image.save(image_path)
            print(f"   ✅ Saved: {image_path}")

        except Exception as e:
            print(f"   ❌ Error for {lat},{lon}: {e}")

    print(f"✅ Finished ward: {ward_name}")
