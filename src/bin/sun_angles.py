import os
import sys
from datetime import datetime, timedelta, date
import pandas as pd
from pyproj import Transformer
from pvlib import solarposition

base_path = "/your/path/input_data/patches"
output_dir = "/your/path/sun_view_angles_per_patch"
os.makedirs(output_dir, exist_ok=True)

transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

def patch_id_to_latlon(patch_id):
    x_m, y_m = [int(v)*1000 for v in patch_id.split('_')]
    lon, lat = transformer.transform(x_m + 500, y_m - 500)
    return lat, lon

def parse_sentinel_filename(filename):
    parts = filename.split('_')
    date_str = parts[1]
    return datetime.strptime(date_str, "%Y%m%d").date()

import pvlib
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo


def solar_to_local_time(date_obj, lat, lon, tz="Europe/Paris", solar_hour=10, solar_minute=30):
    """
    Convert local solar time (LST) to local civil time.
    """
    # Local solar time in minutes
    solar_minutes = solar_hour * 60 + solar_minute
    timestamp = pd.Timestamp(date_obj, tz="UTC")
    doy = timestamp.day_of_year

    # Equation of time (minutes)
    eot = pvlib.solarposition.equation_of_time_spencer71(doy)

    # Local standard meridian (degrees)
    tz_offset = pd.Timestamp(date_obj, tz=tz).utcoffset().total_seconds() / 3600
    LSTM = 15 * tz_offset

    # Time offset in minutes
    time_offset = 4 * (lon - LSTM) + eot

    # Local civil time in minutes
    local_clock_minutes = solar_minutes - time_offset
    hours = int(local_clock_minutes // 60)
    minutes = int(round(local_clock_minutes % 60))
    if minutes == 60:
        hours += 1
        minutes = 0

    # Return timezone-aware datetime
    return datetime(date_obj.year, date_obj.month, date_obj.day, hours, minutes, tzinfo=ZoneInfo(tz))


def solar_angles(date_obj, lat, lon, tz="Europe/Paris", solar_hour=10, solar_minute=30):
    """
    Compute local time and sun angles (zenith, azimuth) for a given date and location.
    """
    local_time = solar_to_local_time(date_obj, lat, lon, tz, solar_hour, solar_minute)
    # Compute solar position
    solpos = pvlib.solarposition.get_solarposition(
        time=pd.DatetimeIndex([local_time]),
        latitude=lat,
        longitude=lon,
        # altitude=0,
        # pressure=101325,
        method='nrel_numpy'
    )
    zenith = solpos['zenith'].iloc[0]
    azimuth = solpos['azimuth'].iloc[0]

    return local_time, zenith, azimuth



#
# dt = date(2025, 9, 29)
# lat, lon = 43.6, 1.44
# local_time, zenith, azimuth = solar_angles(dt, lat, lon, solar_hour=16, solar_minute=17)
#
#
# print(f"Local clock time: {local_time}")
# print(f"Solar zenith: {zenith:.2f}°")
# print(f"Solar azimuth: {azimuth:.2f}°")



def get_tile_info(tile):
    results = []
    for patch_id in os.listdir(os.path.join(base_path, tile)):
        patch_path = os.path.join(base_path, tile, patch_id)
        if not os.path.isdir(patch_path):
            continue

        lat, lon = patch_id_to_latlon(patch_id)

        img_list = [im for im in os.listdir(patch_path) if (im.endswith(".tif") and not im.startswith("MASK"))]
        for img_file in img_list:
            date = parse_sentinel_filename(img_file)
            # approximate acquisition time (9:30 local)

            local_time, zenith, azimuth = solar_angles(date, lat, lon, solar_hour=10, solar_minute=30)
            results.append({
                "patch_id": patch_id,
                "image_file": img_file,
                "date": local_time.strftime("%Y-%m-%d"),
                "lat": lat,
                "lon": lon,
                "solar_zenith": zenith,
                "solar_azimuth": azimuth,
                # "sensor_zenith": 0,   # approximate nadir
                # "sensor_azimuth": 0
            })

    patch_csv_path = os.path.join(output_dir, f"{tile}.csv")
    pd.DataFrame(results).to_csv(patch_csv_path, index=False)
    print(f"Saved {patch_csv_path}")

TILES = [
    "32TMN",
    "31TDG",
    "31TGJ",
    "31TEK",
    "30TYP",
    "32TNN",
    "31UDR",
    "30TXQ",
    "30UXV",
    "31TDN",
    "31TEJ",
    "31UCR",
    "32TLS",
    "31TCL",
    "31TEN",
    "31TDK",
    "31TFM",
    "31TFN",
    "30TXR",
    "32TML",
    "31TDH",
    "32ULU",
    "31TFH",
    "31TEM",
    "31UDQ",
    "31UFQ",
    "30TXN",
    "32TLP",
    "32UMU",
    "32ULV",
    "30TYN",
    "30TXS",
    "31UCP",
    "32TLN",
    "31TGK",
    "31UFP",
    "31UCQ",
    "31UEQ",
    "30TWP",
    "30TXT",
    "31UGQ",
    "31UDS",
    "31TGM",
    "31UGP",
    "31TFK",
    "31TFL",
    "30TXP",
    "32TLT",
    "31TEG",
    "30TWN",
    "32TLR",
    "31TCK",
    "30UXU",
    "32TNM",
    "31TDL",
    "30TYR",
    "31TGL",
    "32UMV",
    "32TLQ",
    "32TMM",
    "32TNL",
    "30UYU",
    "31UDP",
    "30TYT",
    "31UCS",
    "31TCJ",
    "30UVV",
    "31TGN",
    "31TFJ",
    "30TYQ",
    "31TCH",
    "30UYV",
    "30UVU",
    "31TEH",
    "31UEP",
    "31TDM",
    "31TGH",
    "31TEL",
    "30TYS",
    "31TDJ",
]
TILES = ["30TYQ", "31UEQ", "32UMV", "31UEP"]
# for tile in TILES:
#     get_tile_info(tile)

get_tile_info(TILES[int(sys.argv[1])])
