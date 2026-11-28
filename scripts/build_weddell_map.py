"""Align the regional satellite, ice mask and MATLAB climatology for the web map.

Inputs: a PGC Antarctic Composite Basemap export (PNG and its ArcGIS JSON extent),
the SCAR ADD v7.13 medium-resolution polygons ZIP, the original G02135 grid
(reference GeoTIFF), the local MATLAB climatology, and an NSIDC-0116 motion frame.
Requires numpy, scipy, pillow, rasterio, pyproj, pyshp and netCDF4.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import shapefile
from netCDF4 import Dataset
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import transform_geom
from scipy.io import loadmat
from scipy.ndimage import maximum_filter, minimum_filter, map_coordinates


RADIUS = 6371228
LON_MIN, LON_MAX = -75, 20
LAT_MIN, LAT_MAX = -83, -60
MERIDIAN = -30
WIDTH = 1440
DISPLAY_CRS = f"+proj=laea +lat_0=-90 +lon_0={MERIDIAN} +R={RADIUS} +units=m +no_defs"
COLORS = [
    (15, (255, 248, 208)), (31, (255, 217, 131)), (53, (250, 163, 84)),
    (74, (232, 97, 59)), (100, (193, 38, 45)),
]


def sector_radius(latitude):
    return 2 * RADIUS * np.sin(np.radians((90 + latitude) / 2))


def render(args):
    radius = sector_radius(LAT_MAX)
    left = radius * np.sin(np.radians(LON_MIN - MERIDIAN)) - 30_000
    right = radius * np.sin(np.radians(LON_MAX - MERIDIAN)) + 30_000
    top = radius + 30_000
    min_y = sector_radius(LAT_MIN) * np.cos(np.radians(LON_MAX - MERIDIAN)) - 30_000
    step = (right - left) / WIDTH
    height = int(np.ceil((top - min_y) / step))
    bottom = top - height * step
    transform = from_bounds(left, bottom, right, top, WIDTH, height)

    xx, yy = np.meshgrid(left + (np.arange(WIDTH) + .5) * step, top - (np.arange(height) + .5) * step)
    lon = np.degrees(np.arctan2(xx, yy)) + MERIDIAN
    lat = np.degrees(2 * np.arcsin(np.hypot(xx, yy) / (2 * RADIUS))) - 90
    inside = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)

    wgs_to_stereo = Transformer.from_crs(4326, 3031, always_xy=True)
    sx, sy = wgs_to_stereo.transform(lon, lat)
    src = np.asarray(Image.open(args.satellite).convert("RGBA"))
    info = json.loads(args.satellite_extent.read_text())
    extent = info["extent"]
    assert extent["spatialReference"]["wkid"] == 3031 and src.shape[:2] == (info["height"], info["width"])
    image_transform = from_bounds(extent["xmin"], extent["ymin"], extent["xmax"], extent["ymax"],
                                  info["width"], info["height"])
    sc, sr = ~image_transform * (sx, sy)
    satellite = np.stack([map_coordinates(src[..., band], [sr - .5, sc - .5], order=1) for band in range(4)], axis=-1)

    polys = []
    with shapefile.Reader(str(args.coast)) as coast:
        for feature in coast.iterShapeRecords():
            xmin, ymin, xmax, ymax = feature.shape.bbox
            if xmax < sx.min() or xmin > sx.max() or ymax < sy.min() or ymin > sy.max():
                continue
            surface = feature.record.as_dict()["surface"].lower()
            category = 1 if surface in ("land", "rumple") else 2
            geom = transform_geom("EPSG:3031", DISPLAY_CRS, feature.shape.__geo_interface__)
            polys.append((geom, category))
    # Grounded features, including ice rumples, take precedence over shelves.
    classes = rasterize(sorted(polys, key=lambda pair: -pair[1]), out_shape=inside.shape, transform=transform)
    classes[~inside] = 0

    texture = (satellite[..., 3] > 128) & (satellite[..., :3].max(axis=-1) > 28)
    rgb = np.full((*inside.shape, 3), (13, 28, 48), dtype=np.uint8)
    for category, tint in ((1, (172, 190, 203)), (2, (191, 222, 242))):
        area = classes == category
        rgb[area] = tint
        rgb[area & texture] = (
            satellite[area & texture, :3].astype(float) * .74 + np.asarray(tint) * .26
        ).astype(np.uint8)
    edge = (maximum_filter(classes, 3) != minimum_filter(classes, 3)) & (classes != 0)
    rgb[edge] = np.round(rgb[edge].astype(float) * .55).astype(np.uint8)

    mean = loadmat(args.climatology, variable_names=["concMean", "concMonths", "concDays"], squeeze_me=True)
    concentration = mean["concMean"]
    with rasterio.open(args.concentration_grid) as reference:
        if concentration.shape != reference.shape:
            raise ValueError(f"MATLAB shape {concentration.shape} != G02135 grid {reference.shape}")
        source_crs = reference.crs
        col, row = ~reference.transform * Transformer.from_crs(4326, source_crs, always_xy=True).transform(lon, lat)
    values = map_coordinates(concentration, [row - .5, col - .5], order=0, cval=np.nan)
    show = inside & (classes == 0) & (values >= 15) & (values <= 100)
    stops, shades = zip(*COLORS)
    overlay = np.zeros((*inside.shape, 4), dtype=np.uint8)
    for band in range(3):
        overlay[..., band][show] = np.interp(values[show], stops, [shade[band] for shade in shades])
    overlay[..., 3][show] = 232
    assert np.count_nonzero(show) > 20_000 and np.count_nonzero(texture & (classes != 0)) > 50_000

    with Dataset(args.motion_grid) as nc:
        x = np.asarray(nc["x"][:], dtype=float)
        y = np.asarray(nc["y"][:], dtype=float)
    assert np.allclose(np.diff(x), np.diff(x)[0]) and np.allclose(np.diff(y), np.diff(y)[0])
    x_grid, y_grid = np.meshgrid(x, y)
    angle = np.radians(MERIDIAN)
    gx = x_grid * np.cos(angle) - y_grid * np.sin(angle)
    gy = x_grid * np.sin(angle) + y_grid * np.cos(angle)
    gc, gr = np.floor((gx - left) / step).astype(int), np.floor((top - gy) / step).astype(int)
    in_image = (gc >= 0) & (gc < WIDTH) & (gr >= 0) & (gr < height)
    selectable = np.zeros(gx.shape, dtype=bool)
    selectable[in_image] = inside[gr[in_image], gc[in_image]] & (classes[gr[in_image], gc[in_image]] == 0)

    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output / "satellite.webp", quality=88, method=6)
    mask = np.where(classes > 0, 255, 0).astype(np.uint8)
    Image.fromarray(np.dstack((np.zeros_like(rgb), mask))).save(args.output / "ice-mask.png", optimize=True)
    Image.fromarray(overlay).save(args.output / "concentration.webp", lossless=True, method=6)
    metadata = {
        "width": WIDTH, "height": height, "left": left, "top": top, "step": step,
        "meridian": MERIDIAN, "sector": [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        "motion": {"x0": x[0], "y0": y[0], "dx": x[1] - x[0], "dy": y[1] - y[0],
                   "rows": y.size, "cols": x.size},
        "ocean": ["".join("1" if v else "0" for v in line) for line in selectable],
        "concentration": {"months": int(mean["concMonths"]), "weightedDays": int(mean["concDays"]),
                          "colorStops": [[value, list(color)] for value, color in COLORS]},
    }
    (args.output / "geometry.json").write_text(json.dumps(metadata, separators=(",", ":")) + "\n")
    print(f"{WIDTH}×{height}: {int(show.sum())} concentration pixels, "
          f"{int((classes == 1).sum())} land and {int((classes == 2).sum())} shelf pixels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("satellite", "satellite-extent", "coast", "climatology", "concentration-grid",
                 "motion-grid", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    render(parser.parse_args())
