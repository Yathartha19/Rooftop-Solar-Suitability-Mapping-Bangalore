import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon, mapping, box
from shapely.ops import unary_union
from ultralytics import YOLO

# --- CONFIG ---
IMAGE_DIR = "godpt/satimg"
ROOF_MODEL_PATH = "runs/segment/train/weights/best.pt"
PANEL_MODEL_PATH = "sest/panelruns/detect/train/weights/best.pt"
OUTPUT_GEOJSON = "godpt/rooftops.geojson"

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
                if poly.is_valid and poly.area > 10:
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

# --- Load models ---
roof_model = YOLO(ROOF_MODEL_PATH)
panel_model = YOLO(PANEL_MODEL_PATH)

geojson = {"type": "FeatureCollection", "features": []}

for ward_name in os.listdir(IMAGE_DIR):
    ward_path = os.path.join(IMAGE_DIR, ward_name)
    if not os.path.isdir(ward_path):
        continue

    ward_roofs = []
    for fname in os.listdir(ward_path):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(ward_path, fname)
        print(f"📍 Processing {ward_name} / {fname}")
        lat_center, lon_center = parse_coords_from_name(fname)

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read image: {fname}")
            continue
        h, w = img.shape[:2]

        # --- Rooftop Segmentation ---
        roof_results = roof_model(img_path, conf=0.3)[0]
        if roof_results.masks is None:
            continue
        masks = roof_results.masks.data.cpu().numpy()
        roof_polygons = []
        for mask in masks:
            roof_polygons.extend(mask_to_polygons(mask))

        # --- Solar Panel Detection ---
        panel_results = panel_model(img_path, conf=0.3)[0]
        panel_boxes = []
        if panel_results.boxes is not None:
            xyxy = panel_results.boxes.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in xyxy:
                panel_boxes.append(box(x1, y1, x2, y2))

        geo_solar_panels = []
        for b in panel_boxes:
            pixel_coords = list(b.exterior.coords)
            geo_coords = [pixel_to_latlon(x, y, lat_center, lon_center, w, h) for x, y in pixel_coords]
            geo_solar_panels.append(Polygon([(lon, lat) for lat, lon in geo_coords]))

        for roof_poly in roof_polygons:
            pixel_coords = np.array(roof_poly.exterior.coords)
            geo_coords = [pixel_to_latlon(x, y, lat_center, lon_center, w, h) for x, y in pixel_coords]
            geo_poly = Polygon([(lon, lat) for lat, lon in geo_coords])
            if not geo_poly.is_valid or geo_poly.area < 1e-8:
                continue

            has_solar = any(geo_poly.intersects(panel) for panel in geo_solar_panels)

            geojson["features"].append({
                "type": "Feature",
                "geometry": mapping(geo_poly),
                "properties": {
                    "ward": ward_name,
                    "image": fname,
                    "class": "rooftop",
                    "area_px": roof_poly.area,
                    "has_solar": bool(has_solar)
                }
            })

            if has_solar:
                for panel in geo_solar_panels:
                    if geo_poly.intersects(panel):
                        geojson["features"].append({
                            "type": "Feature",
                            "geometry": mapping(panel),
                            "properties": {
                                "ward": ward_name,
                                "type": "solar_box",
                                "belongs_to": fname
                            }
                        })

            ward_roofs.append(geo_poly)

    if ward_roofs:
        coverage = unary_union(ward_roofs)
        geojson["features"].append({
            "type": "Feature",
            "geometry": mapping(coverage),
            "properties": {
                "type": "ward_outline",
                "ward": ward_name,
                "rooftop_count": len(ward_roofs)
            }
        })

with open(OUTPUT_GEOJSON, "w") as f:
    json.dump(geojson, f, indent=2)

print(f"✅ Done. Saved {len(geojson['features'])} features to {OUTPUT_GEOJSON}")
