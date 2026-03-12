# region imports
import datetime
import logging
import time
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import numpy as np
import spiceypy
from apexpy import Apex
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset

from .ezvislib.constants import (
    EARTH_FLATTENING,
    EARTH_RADIUS_EQUATORIAL,
)
from .ezvislib.gw_plot_methods import (
    NCDF_MISSING,
    plot_geomagnetic_references,
    save_close_figure,
)
from .ezvislib.gw_plot_params import (
    DATA_TRANSFORM,
    EZIE_DATE_FORMAT,
    GEO_LAT_LOWER_LIMIT,
    MAG_LAT_LOWER_LIMIT,
    NORTH,
    REFERENCE_ALTITUDE_KM,
    SOUTH,
)
from .ezvislib.gw_plot_utils import (
    add_product_metadata,
    map_inverted_continents,
    overlay_ezie_logo,
    parse_ezie_product_name,
)

# endregion

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def plot_b_1D_maps_with_time(
    nc_data: Dataset,
    source: Path,
    save_directory: Path,
    south_inverted: bool = False,
    figure_dpi: int = 200,
    dark_mode: bool = False,
    old_format: bool = False,
    overwrite: bool = False,
    help: bool = False,
):
    """
    - Script used to generate AEJ L3 B-up data output for B-down L2 inputs
    - Based on plot_btot_maps_with_time
    - All arrays should be 1D!

    Args:
        nc_dat (Dataset): Required. L3 file netCDF4 Dataset
        source (Path): Required. Name of L3 file used as input
        save_directory (Path) :  Required. Path to plot output root directory

    Returns:
        None
    """

    start = time.perf_counter()
    prsd = parse_ezie_product_name(source=source)
    # plot_type = "l3-l2_alt"
    plot_type = "l3-l2"

    # See if this figure already exists.
    # Do NOT remake an existing figure unless the overwrite flag is set.
    ftgt: Path = save_close_figure(
        source=source,
        save_directory=save_directory,
        obs_date=datetime.datetime.strptime(prsd.date, EZIE_DATE_FORMAT),
        spacecraft=prsd.spcv,
        tstmp=prsd.time,
        plot_type=plot_type,
        dpi=figure_dpi,
        dark_mode=dark_mode,
        old_format=old_format,
        name_only=True,
    )
    logger.debug(f"Checked file: {ftgt.as_posix()}")
    if ftgt.exists() and not overwrite:
        logger.warning(f"File exists, skipping: {ftgt.as_posix()}")
        return

    try:
        time_utc = np.array(
            [
                datetime.datetime.fromisoformat(_)
                for _ in nc_data["/l2_data/time_utc"][:]
            ]
        )
    except Exception as exc:
        logger.error(f"Encountered exception while processing L3 file: {exc}")
        return

    if not isinstance(time_utc[0], (datetime.datetime, np.datetime64)):
        raise ValueError(
            "obs_time must contain datetime.datetime or np.datetime64 objects."
        )

    # Use midpoint to select map orientation
    midpt = len(time_utc) // 2
    lat_at_midpt = nc_data["l2_data/sc_lat"][midpt]
    lon_at_midpt = nc_data["l2_data/sc_lon"][midpt]

    # Add magnetic coordinate overlay, select latitudes at which to draw magnetic
    # gridlines, play with the map projection as needed.
    if lat_at_midpt > +40:  # hemisphere = NORTH
        hemisphere = NORTH
    elif lat_at_midpt < -40:  # hemisphere = SOUTH
        hemisphere = SOUTH
    else:
        hemisphere = None
        logger.warning(
            f"L3 file appears to contain EEJ pass, skipping: {source.as_posix()}"
        )
        return

    # Grab J, B, and coordinate data arrays
    lat_mesh = nc_data["/l3_data/lat"][:]
    lon_mesh = nc_data["/l3_data/lon"][:]
    model_B = nc_data["/l3_data/Bd_geod_80"][:]
    Je = nc_data["/l3_data/Je_110"][:]
    Jn = nc_data["/l3_data/Jn_110"][:]
    lats = nc_data["/l2_data/lat"][:]
    lons = nc_data["/l2_data/lon"][:]
    obs_B = nc_data["/l2_data/Bd_geod_80"][:]
    model_B = model_B.reshape(lat_mesh.shape)

    if np.all(Jn == 0.0) and np.all(Je == 0.0) and np.all(obs_B == 0.0):
        logger.warning(
            f"L3 file Je, Jn, and observed B values are all zero, skipping: {ftgt.stem}"
        )
        return

    logger.info(f"Generating L3 plot {ftgt.stem}")

    # Get solar position, first in geodetic and then in APEX magnetic coordinates. We'll
    # display projected geodetic coordinates for now, pending addition of magnetic
    # coordinate counterparts to the L3 files.
    sun_geo_tuple_rad = []

    # Compute position of sun in both geodetic and geomagnetic coordinates for the
    # observation midpoint time.
    tdb = spiceypy.utc2et(time_utc[midpt].isoformat()[:-6])
    (subpnt, epoch, to_subpnt) = spiceypy.subslr(
        "INTERCEPT/ELLIPSOID", "EARTH", tdb, "IAU_EARTH", "LT+S", "EARTH"
    )
    # Get the subsolar point on the surface in geodetic coordinates
    sun_geo_tuple_rad = spiceypy.recgeo(
        subpnt, EARTH_RADIUS_EQUATORIAL, EARTH_FLATTENING
    )
    sun_geo_lon_mid = np.degrees(sun_geo_tuple_rad[0])
    sun_geo_lat_mid = np.degrees(sun_geo_tuple_rad[1])
    if sun_geo_lon_mid > 180.0:
        sun_geo_lon_mid -= 360.0

    # Compute the subsolar point in APEX magnetic coordinates
    apex = Apex(
        date=time_utc[midpt].year,
        refh=0,  # Leave at default (0), or set to 80 km? TBD
    )
    sun_mag_lat_mid, sun_mag_lon_mid = apex.geo2apex(
        sun_geo_lat_mid,
        sun_geo_lon_mid,
        REFERENCE_ALTITUDE_KM,
    )

    # 'lats_m' are the magnetic latitudes at which we'll overlay parallels. These need
    # to be the ACTUAL mlats for APEX, with sign appropriate for the selected hemisphere
    # regardless of any projection tricks we might apply. 'MLT_sign' is used to "trick"
    # cartopy into plotting the southern hemisphere in an inverted/transparent earth
    # format used for heliophysics, where we look down at the SOUTH pole from above the
    # NORTH pole, as if the earth were transparent. This 'south_inverted' format keeps
    # Noon, Dawn, etc. in the same relative positions and orientation for both
    # hemispheres. To get the south_inverted projection we apply the MLT_sign multiplier
    # (-1 for this case) to the "projected" quantitites. Latitudes retain their normal
    # hemisphere-appropriate sense.
    MLT_sign = +1
    if hemisphere == NORTH:
        lats_m = np.arange(MAG_LAT_LOWER_LIMIT, 90.0, 10.0)
        map_proj = ccrs.NorthPolarStereo(
            central_longitude=sun_geo_lon_mid + 180,
            true_scale_latitude=+60,
        )
    elif hemisphere == SOUTH:
        lats_m = -1 * np.arange(MAG_LAT_LOWER_LIMIT, 90.0, 10.0)
        if south_inverted:
            MLT_sign = -1
        map_proj = ccrs.SouthPolarStereo(
            central_longitude=MLT_sign * sun_geo_lon_mid,
            true_scale_latitude=-60,
        )
    else:
        lats_m = np.arange(-GEO_LAT_LOWER_LIMIT, +GEO_LAT_LOWER_LIMIT, 10.0)
        map_proj = ccrs.Stereographic(
            central_latitude=lat_at_midpt, central_longitude=lon_at_midpt
        )

    # Set up figure and projection for cartopy
    fig = plt.figure(num=prsd.time, figsize=(9.6, 8.0))
    map_axs: GeoAxes = fig.add_subplot(1, 1, 1, projection=map_proj)

    # TODO - test this on EEJ passes. It's probably missing something.
    # Set extent
    swath_wid, swath_len = 3250000, 3250000  # meters, keep it square
    lonmid, latmid = MLT_sign * lons[0, midpt], lats[0, midpt]

    # Shift center of projection slightly toward geographic pole so that LT labels do
    # not get scrunched together when the pole moves too close to (or beyond) the upper
    # boundary.
    if hemisphere == NORTH:
        ext_shift = 0.3 * (90 - latmid)
    elif hemisphere == SOUTH:
        ext_shift = 0.3 * (-90 - latmid)
    else:
        ext_shift = 0.0

    x0_m, y0_m = map_proj.transform_point(
        lonmid, latmid + ext_shift, src_crs=DATA_TRANSFORM
    )
    logger.debug(
        f"{x0_m - swath_wid} {x0_m + swath_wid} {y0_m - swath_len} {y0_m + swath_len}"
    )
    map_axs.set_extent(
        (x0_m - swath_wid, x0_m + swath_wid, y0_m - swath_len, y0_m + swath_len),
        crs=map_proj,
    )

    if (hemisphere == SOUTH) and south_inverted:
        logger.info("Mapping continents in reversed longitude coordinates")
        map_inverted_continents(ax=map_axs, linewidth=0.5)
    else:
        logger.info("Mapping continents in standard longitude coordinates")
        map_axs.coastlines(resolution="110m", linewidth=0.5)

    if hemisphere is not None:
        # Add LT grid markings - not yet "true" MLT, just LT
        _mag_lat_artist = plot_geomagnetic_references(
            map_axs,
            time_utc[midpt],
            latitudes=lats_m,
            south_inverted=(hemisphere == SOUTH) and south_inverted,
            lat_clr="black",
        )  # Might want artist for legend later?

        # Add MLT solar position notation in hours, with exceptions for Noon, etc. 'ccw'
        # is NOT the same as MLT_sign!
        if (hemisphere is None) or (hemisphere == NORTH) or south_inverted:
            ccw = +1
        else:
            ccw = -1  # SOUTH and not south_inverted case only

        mlt_desc = {}
        lon_delta = 30.0
        mlt_angs = np.arange(0, 360.0, lon_delta)
        for mm, mlt in enumerate(mlt_angs):
            mlt_desc[f"{mlt:.0f}"] = f"{mlt / 15:02.0f}H"
        mlt_desc["0"] = "Midnight"
        mlt_desc["90"] = "Dawn"
        mlt_desc["180"] = "Noon"
        mlt_desc["270"] = "Dusk"

        xp_m, yp_m = map_axs.transAxes.inverted().transform(
            map_axs.transData.transform(
                (0, +90 if hemisphere == NORTH else -90),
            )
        )  # Pole location in plot axes coordinates

        def boundary_distance(px, py, ang, stretch: int = 1.0):
            x1, y1, x2, y2 = (
                0.0 - stretch,
                0.0 - stretch,
                1.0 + stretch,
                1.0 + stretch,
            )  # Standard matpllotlib axes coordinates
            vx = np.cos(np.radians(ang))
            vy = np.sin(np.radians(ang))
            p_lrtb = np.array(
                [(x1 - px) / vx, (x2 - px) / vx, (y1 - py) / vy, (y2 - py) / vy]
            )
            wall_dist = np.min(p_lrtb[p_lrtb >= 0.0])
            return px + wall_dist * vx, py + wall_dist * vy

        for mm, mlt in enumerate(mlt_angs):
            plt_ang = (ccw * mlt - 90.0) % 360.0  # Rotate to coord sys with 0 at x axis
            x, y = boundary_distance(xp_m, yp_m, plt_ang, stretch=0.035)
            map_axs.text(
                x,
                y + 0.0,
                mlt_desc[f"{mlt:.0f}"],
                color="black",
                va="center",
                ha="center",
                transform=map_axs.transAxes,
            )

        # Overlay magnetic latitude parallels, at least until we can start plotting
        # everything in mlat, MLT.
        MLT_axes = np.arange(-180.0, 180.0, lon_delta) + MLT_sign * sun_geo_lon_mid
        while np.any(MLT_axes > 180.0):
            MLT_axes[MLT_axes > 180.0] -= 360.0
        while np.any(MLT_axes < -180.0):
            MLT_axes[MLT_axes < -180.0] += 360.0
        lat_lower_limit = GEO_LAT_LOWER_LIMIT
        lats_n = np.arange(lat_lower_limit, 90, 10)  # Lats at which to draw gridlines
        _gls = map_axs.gridlines(
            draw_labels=False,
            xlocs=np.sort(MLT_axes),
            ylocs=lats_n if hemisphere == NORTH else [-1 * lat for lat in lats_n],
            color="black",
            crs=ccrs.PlateCarree(),
        )
        _gls = map_axs.gridlines(
            draw_labels=True,
            xlocs=[],
            y_inline=True,
            ylocs=lats_n if hemisphere == NORTH else [-1 * lat for lat in lats_n],
            color="black",
            crs=ccrs.PlateCarree(),
        )

    else:  # EEJ
        map_axs.gridlines(
            draw_labels=True,
            x_inline=False,
            y_inline=False,
            rotate_labels=False,
            color="black",
            crs=ccrs.PlateCarree(),
        )
        plot_geomagnetic_references(
            map_axs,
            time_utc[midpt],
            latitudes=lats_m,
            south_inverted=(hemisphere == SOUTH) and south_inverted,
            lat_clr="black",
        )
        plot_geomagnetic_references(
            map_axs,
            time_utc[midpt],
            latitudes=[0],
            south_inverted=(hemisphere == SOUTH) and south_inverted,
            lat_clr="black",
            ls="solid",
            lw=2,
        )  # Magnetic drift equator
    # Now we can finally draw the J and B values! Define some plotting parameters.
    m_mrgn = 5  # Mesh margin to be avoided
    J = np.sqrt(Je**2 + Jn**2)
    Jmax = 10 * np.nanmax(J)  # scales the LENGTH of the quiver arrows
    qwid = 0.001  # scales the WIDTH of the quiver arrow(head?)s
    b_mrk_sz = 20  # Marker size used for observed and modeled B scatter plots

    # Shared B magnitude color scale
    # Relative
    # vmin = min(np.nanmin(obs_B), np.nanmin(model_B))
    # vmax = max(np.nanmin(obs_B), np.nanmin(model_B))
    # vmin = -1 * min(abs(vmin), abs(vmax))
    # vmax = -1 * vmin
    # Fixed
    vmin = -1200
    vmax = +1200

    # Model B
    valid = model_B != NCDF_MISSING
    sc = map_axs.scatter(
        MLT_sign * lon_mesh[valid].flatten(),
        lat_mesh[valid].flatten(),
        c=-1 * model_B[valid].flatten(),
        cmap=plt.cm.bwr,
        vmin=vmin,
        vmax=vmax,
        s=b_mrk_sz,
        transform=DATA_TRANSFORM,
    )

    # Retrieved J
    map_axs.quiver(
        MLT_sign * lon_mesh[::m_mrgn, ::m_mrgn].flatten(),
        lat_mesh[::m_mrgn, ::m_mrgn].flatten(),
        MLT_sign * Je[::m_mrgn, ::m_mrgn].flatten(),
        Jn[::m_mrgn, ::m_mrgn].flatten(),
        scale=Jmax,
        width=qwid,
        transform=DATA_TRANSFORM,
    )

    # Observed (L2) B
    valid = obs_B != NCDF_MISSING
    sc = map_axs.scatter(
        MLT_sign * lons[valid],
        lats[valid],
        c=obs_B[valid],
        cmap=plt.cm.bwr,
        vmin=vmin,
        vmax=vmax,
        s=b_mrk_sz,
        marker="o",
        transform=DATA_TRANSFORM,
    )

    # Add plot annotation and color bar, tweak margins and spacing.
    divider = make_axes_locatable(map_axs)
    cax = divider.append_axes("right", size="3%", pad=0.7, axes_class=maxes.Axes)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("B$_{{down}}$ [nT]")
    fig.suptitle(
        (
            f"L3 and L2: EZIE-{prsd.spcv.upper()} \n"
            f"{time_utc[0].strftime('%B %d, %Y %H:%M:%S')} - "
            f"{time_utc[-1].strftime('%H:%M:%S')}\n"
        ),
        y=0.99,
        va="top",
        fontsize="x-large",
        fontweight="bold",
    )
    plt.tight_layout()  # pad=0)
    plt.subplots_adjust(top=0.860, bottom=0.060)  # Modify tight_layout results slightly

    # FIXME: Metadata not yet in L3 .nc4 files
    add_product_metadata(fig=fig, nc_data=nc_data, source=source)
    # add_pipeline_metadata(fig, nc_data)
    if hemisphere is not None:
        subtitle = (
            "Plot is in Geodetic/WGS84 coordinates (solid grid). "
            "Labels indicate Local Time (LT). "
            "Magnetic latitude (dashed grid) is also shown."
        )
    else:
        subtitle = (
            "Plot is in Geodetic/WGS84 coordinates (solid grid). "
            "Magnetic latitude (dashed grid) and drift equator "
            "(thick solid line) are also shown."
        )
    fig.text(
        0.50,
        0.91,
        subtitle,
        wrap=False,
        ha="center",
        va="bottom",
        fontsize="small",
        fontweight="bold",
        # transform=map_axs.transAxes,
    )
    overlay_ezie_logo(fig)

    save_close_figure(
        source=source,
        save_directory=save_directory,
        figure=fig,
        obs_date=datetime.datetime.strptime(prsd.date, EZIE_DATE_FORMAT),
        spacecraft=prsd.spcv,
        tstmp=prsd.time,
        plot_type=plot_type,
        dpi=figure_dpi,
        dark_mode=dark_mode,
    )
    end = time.perf_counter()
    logger.info(f"Elapsed time: {end - start:.2f} seconds.")
