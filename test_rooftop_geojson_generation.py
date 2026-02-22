import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
from ultralytics import YOLO

# CONFIG
IMAGE_DIR = "images"
MODEL_PATH = "runs/segment/train/weights/best.pt"
OUTPUT_GEOJSON = "rooftops.geojson"
METERS_PER_PIXEL = 0.145  # Approximate at zoom level 20
DEG_PER_METER = 1 / 111320  # Degrees per meter at equator

def mask_to_polygons(mask):
    mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) >= 3:
            pts = cnt.squeeze()
            if pts.ndim == 2:
                poly = Polygon(pts)
                if poly.is_valid:
                    polygons.append(poly)
    return polygons

def pixel_to_latlon(x, y, lat_center, lon_center, img_width, img_height):
    # Offset from center of image
    dx = x - img_width / 2
    dy = y - img_height / 2

    delta_lat = -dy * METERS_PER_PIXEL * DEG_PER_METER
    delta_lon = dx * METERS_PER_PIXEL * DEG_PER_METER / np.cos(np.radians(lat_center))

    lat = lat_center + delta_lat
    lon = lon_center + delta_lon
    return lat, lon

def parse_coords_from_name(fname):
    parts = fname.replace(".png", "").split("_")
    return float(parts[-2]), float(parts[-1])

# Load model
model = YOLO(MODEL_PATH)
geojson = {"type": "FeatureCollection", "features": []}

for fname in os.listdir(IMAGE_DIR):
    if not fname.endswith(".png"):
        continue

    img_path = os.path.join(IMAGE_DIR, fname)
    print(f"📍 Processing {fname}")
    lat_center, lon_center = parse_coords_from_name(fname)

    # Load image to get shape
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Could not read image: {fname}")
        continue
    h, w = img.shape[:2]

    # Run model
    results = model(img_path, conf=0.3)[0]

    if results.masks is None:
        print("❌ No masks found.")
        continue

    masks = results.masks.data.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()

    for i, mask in enumerate(masks):
        polys = mask_to_polygons(mask)
        for poly in polys:
            pixel_coords = np.array(poly.exterior.coords)
            geo_coords = [pixel_to_latlon(x, y, lat_center, lon_center, w, h) for x, y in pixel_coords]
            geo_polygon = Polygon([(lon, lat) for lat, lon in geo_coords])

            geojson["features"].append({
                "type": "Feature",
                "geometry": mapping(geo_polygon),
                "properties": {
                    "image": fname,
                    "class": "rooftop",
                    "area_px": poly.area
                }
            })

# Save GeoJSON
with open(OUTPUT_GEOJSON, "w") as f:
    json.dump(geojson, f, indent=2)

print(f"✅ Done. Saved {len(geojson['features'])} polygons to {OUTPUT_GEOJSON}")
