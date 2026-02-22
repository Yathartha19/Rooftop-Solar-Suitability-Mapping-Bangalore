import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

def download_google_satellite_image(lat, lon, zoom=20, size="640x640", api_key="API_KEY"):
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={size}&maptype=satellite&key={api_key}"
    )
    response = requests.get(url)
    
    if response.status_code != 200:
        print("❌ Failed to fetch image:", response.text)
        return None
    
    try:
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print("❌ Error loading image:", e)
        return None

# gmaps api key
load_dotenv()
API_KEY = os.getenv("GMAPS_API_KEY")

def load_coordinates(file_path):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            latlon = line.strip().split(",")
            if len(latlon) == 2:
                lat, lon = map(float, latlon)
                points.append((lat, lon))
    return points

for point in load_coordinates("coordinates.txt"):
    lat, lon = point
    img = download_google_satellite_image(lat, lon, api_key=API_KEY)
    if img:
        img.save(f"rooftop_example_{lat}_{lon}.png")
        print(f"✅ Saved image for {lat}, {lon} as rooftop_example_{lat}_{lon}.png")
    else:
        print(f"❌ Failed to save image for {lat}, {lon}")

