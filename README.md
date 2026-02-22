# Rooftop Solar Suitability Mapping - Bangalore

This project implements a hybrid workflow to assess urban rooftop solar potential. It combines YOLO instance segmentation for rooftop extraction rule-scoring to identify the best locations for solar PV installation.

## Structure

* **acquire_wards_coordinate_geojson.py**: Fetches coordinates and grids for the 5 wards closest to the city center; outputs `closest_wards.geojson`.
* **satellite_imagery_from_geojson.py**: Downloads high-resolution satellite imagery via Google Maps API into the `satimg/` directory.
* **roof_and_panel_detection.py**: Uses YOLOv8 to detect rooftops and existing solar panels; outputs `rooftops.geojson`.
* **merge_rooftop_with_power.py**: Combines detected polygons with solar radiation data; outputs `rooftops_with_radiation.geojson`.
* **index.html**: A Mapbox/Leaflet web interface to visualize the final suitability results.

## Workflow

1. **Grid Generation**: Define study areas based on BBMP ward boundaries.
2. **Data Retrieval**: Download orthophotos for the defined grids.
3. **ML Inference**: Detect rooftop polygons and existing PV panels using yolo trained on ~500 images.
5. **Visualization**: Classifies roofs into five categories from "Highly Suitable" to "Worst".
# as
# Rooftop-Solar-Suitability-Mapping-Bangalore
