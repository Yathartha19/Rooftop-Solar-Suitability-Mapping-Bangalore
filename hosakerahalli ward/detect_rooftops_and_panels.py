import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon, mapping, box
from shapely.ops import unary_union
from ultralytics import YOLO

# CONFIG
IMAGE_DIR = "images"
ROOF_MODEL_PATH = "runs/segment/train/weights/best.pt"
PANEL_MODEL_PATH = "model/detect/train/weights/best.pt"
OUTPUT_GEOJSON = "rooftops.geojson"
METERS_PER_PIXEL = 0.145
DEG_PER_METER = 1 / 111320

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
    dx = x - img_width / 2
    dy = y - img_height / 2
    delta_lat = -dy * METERS_PER_PIXEL * DEG_PER_METER
    delta_lon = dx * METERS_PER_PIXEL * DEG_PER_METER / np.cos(np.radians(lat_center))
    return lat_center + delta_lat, lon_center + delta_lon

def parse_coords_from_name(fname):
    parts = fname.replace(".png", "").split("_")
    return float(parts[-2]), float(parts[-1])

# Load models
roof_model = YOLO(ROOF_MODEL_PATH)
panel_model = YOLO(PANEL_MODEL_PATH)
geojson = {"type": "FeatureCollection", "features": []}

for fname in os.listdir(IMAGE_DIR):
    if not fname.endswith(".png"):
        continue

    img_path = os.path.join(IMAGE_DIR, fname)
    print(f"📍 Processing {fname}")
    lat_center, lon_center = parse_coords_from_name(fname)

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read image: {fname}")
        continue
    h, w = img.shape[:2]

    # Rooftop segmentation
    roof_results = roof_model(img_path, conf=0.3)[0]
    if roof_results.masks is None:
        continue
    masks = roof_results.masks.data.cpu().numpy()
    roof_polygons = []
    for mask in masks:
        roof_polygons.extend(mask_to_polygons(mask))

    # Solar panel object detection
    panel_results = panel_model(img_path, conf=0.3)[0]
    panel_boxes = []
    if panel_results.boxes is not None:
        xyxy = panel_results.boxes.xyxy.cpu().numpy()
        for x1, y1, x2, y2 in xyxy:
            panel_boxes.append(box(x1, y1, x2, y2))  # Shapely box

    # Convert solar boxes to geographic coordinates
    geo_solar_panels = []
    for b in panel_boxes:
        pixel_coords = list(b.exterior.coords)
        geo_coords = [pixel_to_latlon(x, y, lat_center, lon_center, w, h) for x, y in pixel_coords]
        geo_solar_panels.append(Polygon([(lon, lat) for lat, lon in geo_coords]))

    # For each rooftop polygon
    for roof_poly in roof_polygons:
        pixel_coords = np.array(roof_poly.exterior.coords)
        geo_coords = [pixel_to_latlon(x, y, lat_center, lon_center, w, h) for x, y in pixel_coords]
        geo_poly = Polygon([(lon, lat) for lat, lon in geo_coords])
        if not geo_poly.is_valid:
            continue

        # Check intersection
        has_solar = any(geo_poly.intersects(panel) for panel in geo_solar_panels)

        # Add rooftop feature
        geojson["features"].append({
            "type": "Feature",
            "geometry": mapping(geo_poly),
            "properties": {
                "image": fname,
                "class": "rooftop",
                "area_px": roof_poly.area,
                "has_solar": has_solar
            }
        })

        # Add solar box overlays inside this roof
        if has_solar:
            for panel in geo_solar_panels:
                if geo_poly.intersects(panel):
                    geojson["features"].append({
                        "type": "Feature",
                        "geometry": mapping(panel),
                        "properties": {
                            "type": "solar_box",
                            "belongs_to": fname
                        }
                    })

# --- Coverage mask with stats ---
all_roof_geos = [f for f in geojson["features"] if f["properties"].get("class") == "rooftop"]
rooftop_polys = [Polygon(f["geometry"]["coordinates"][0]) for f in all_roof_geos]

if rooftop_polys:
    coverage_polygon = unary_union(rooftop_polys)
    green = sum(1 for f in all_roof_geos if f["properties"]["has_solar"])
    red = len(all_roof_geos) - green

    geojson["features"].append({
        "type": "Feature",
        "geometry": mapping(coverage_polygon),
        "properties": {
            "type": "coverage_mask",
            "total_rooftops": len(all_roof_geos),
            "green_rooftops": green,
            "red_rooftops": red
        }
    })

# Save GeoJSON
with open(OUTPUT_GEOJSON, "w") as f:
    json.dump(geojson, f, indent=2)

print(f"✅ Done. Saved {len(geojson['features'])} features to {OUTPUT_GEOJSON}")
