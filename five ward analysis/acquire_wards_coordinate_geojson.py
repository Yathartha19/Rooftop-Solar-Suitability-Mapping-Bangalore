import geopandas as gpd
import requests
import io
from shapely.geometry import shape
from pyproj import Transformer

# ---- CONFIG ----
CITY_CENTER_LAT = 12.9716  # MG Road
CITY_CENTER_LON = 77.5946
NUM_CLOSEST = 5 

url = "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Bangalore/BBMP.geojson"
print(f"⬇️  Downloading: {url}")
r = requests.get(url)
r.raise_for_status()
print("✅ Downloaded")
gdf = gpd.read_file(io.BytesIO(r.content))

gdf = gdf.to_crs(epsg=32643)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
cx, cy = transformer.transform(CITY_CENTER_LON, CITY_CENTER_LAT)
gdf["distance_m"] = gdf.geometry.centroid.distance(gpd.points_from_xy([cx], [cy])[0])

closest = gdf.sort_values("distance_m").head(NUM_CLOSEST)
print(closest[["KGISWardName", "distance_m"]])

closest = closest.to_crs(epsg=4326)

# ---- 6️⃣ Save valid GeoJSON ----
closest.to_file("closest_wards.geojson", driver="GeoJSON")
print("💾 Saved: closest_wards.geojson — now works in geojson.io ✅")
