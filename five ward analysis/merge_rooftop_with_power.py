import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata
import numpy as np

print("📦 Loading rooftops...")
rooftops = gpd.read_file("godpt/rooftops.geojson").to_crs(epsg=4326)
rooftops["centroid"] = rooftops.geometry.centroid
centroids = gpd.GeoDataFrame(rooftops.drop(columns="geometry"), geometry="centroid", crs="EPSG:4326")

print("📦 Loading radiation data...")
radiation = pd.read_csv("godpt/POWER_Regional_Monthly_2015_2025.csv", skiprows=9)
radiation.columns = ["PARAMETER", "YEAR", "LAT", "LON",
                     "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "ANN"]

print("🔄 Calculating multi-year average radiation...")
radiation_avg = (
    radiation.groupby(["LAT", "LON"])
    .agg({"ANN": "mean"})
    .reset_index()
)

print("🔄 Interpolating...")
points = radiation_avg[["LON", "LAT"]].values
values = radiation_avg["ANN"].values
centroid_coords = centroids.geometry.apply(lambda p: (p.x, p.y)).tolist()

# Step 1: Linear interpolation
interpolated = griddata(points, values, centroid_coords, method='linear')

# Step 2: Nearest interpolation for fallback where linear fails (NaNs)
interpolated_nearest = griddata(points, values, centroid_coords, method='nearest')

# Replace NaNs from linear with nearest
interpolated = np.where(np.isnan(interpolated), interpolated_nearest, interpolated)

# Assign back to GeoDataFrame
rooftops["ann_radiation"] = interpolated

# Save result
output_file = "godpt/rooftops_with_radiation.geojson"
rooftops.drop(columns="centroid").to_file(output_file, driver="GeoJSON")
print(f"✅ Saved rooftop polygons with interpolated radiation to {output_file}")
