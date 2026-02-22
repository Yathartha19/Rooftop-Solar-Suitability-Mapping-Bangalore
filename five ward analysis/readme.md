Flow:
unzip geojsons.zip

1. acquire_wards_coordinate_geojson.py gets closest_wards.geojson ( basically coodinates and grids of 5 wards closest to city center ).
2. satellite_imagery_from_geojson.py creates 'satimg/' dir and gets satellite imagery from gmaps api using the closest_wards.geojson.
3. roof_and_panel_detection.py detects rooftops and solar panels from satimg/ and outputs rooftop.geojson with highlighted roofs and panels.
4. merge_rooftop_with_power.py combines rooftops.geojson with radiation data ( csv ) and outputs rooftops_with_radiation.geojson.
5. index.html uses rooftops_with_radiation.geojson to display map.