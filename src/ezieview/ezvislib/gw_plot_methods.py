"""
A collection of methods for generating plots from EZIE data products, primarily for use
on the science gateway. A few methods are also used for various spacecraft attitude
and/or data analyses, e.g., the star tracker loss of lock investigation.
"""

# region imports
import datetime
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.axes as maxes
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import spiceypy
from apexpy import Apex
from cartopy.feature.nightshade import Nightshade
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset, default_fillvals

from ezieview.ezvislib.gw_plot_params import (
    ALTERNATE_FIG_SIZE,
    AVERAGING_WINDOW,
    DARK_MODE_FACE_COLOR,
    DARK_MODE_FILL_COLOR,
    DARK_MODE_GRID_COLOR,
    DARK_MODE_TEXT_COLOR,
    DATA_TRANSFORM,
    DEFAULT_FIG_SIZE,
    DFLT_RES,
    EARTH_FLATTENING,
    EARTH_RADIUS_EQUATORIAL,
    EZIE_DATE_FORMAT,
    FLD_CLR,
    FLD_CMP,
    GEO_LAT_LOWER_LIMIT,
    MAG_LAT_LOWER_LIMIT,
    MEM_CLR,
    MEM_LOOK_DIRECTIONS,
    MEM_NUMBERS,
    MEM_SYMS,
    NORTH,
    NUM_FLD,
    NUM_MEM,
    O2_CTR_FREQ_MHZ,
    REFERENCE_ALTITUDE_KM,
    SAT_COL,
    SOUTH,
    SPACECRAFT,
    TERMINATOR_ALPHA,
    TERMINATOR_COLOR,
    C,
)
from ezieview.ezvislib.gw_plot_utils import (
    add_pipeline_metadata,
    add_product_metadata,
    get_datetime_from_utc_string,
    map_inverted_continents,
    map_magnetic_continents,
    moving_average,
    overlay_ezie_logo,
    parse_ezie_product_name,
    plot_geomagnetic_references,
    save_close_figure,
    set_xaxis_tick_format,
    set_yaxis_tick_format,
)

# endregion


# region globals
mpl.use("Agg")  # No interactive python window, can run headless
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
FREQ_BIN_DELTA = 205  # Number of frequency bins to extract either side of line center
FREQ_DELTA_MHZ = 3.5  # Mhz to show on either side of line center
PLT_NDX = [
    1,
    2,
    3,
    0,
]  # Positions of MEM[n] in stack of plots (top to bottom, ordered by angle WRT nadir)
SCIENCE: int = 2
NCDF_MISSING = default_fillvals["f4"]
# Offset slightly from 90 because some mapping-related package has begun throwing errors
# when 90 is used following recent updates.
POLE_LAT = 89.0
MEM_BEAM_DIAGRAM_PATH = (
    Path(__file__).parent.parent / "binary-assets" / "SV_NoSVFrame.png"
    # Path(__file__).parent.parent / "binary-assets" / "mem-beam-diagram.png"
)
# endregion


@dataclass
class L3PassData:
    """
    Data arrays extracted from a single L3 product file (and its embedded L2
    data group), sufficient to render an L3 map and related overlays.

    Attributes:
        time_utc: Observation times as timezone-aware UTC datetimes.
        midpt: Midpoint index into time_utc used to orient the map.
        lat_at_midpt: Spacecraft geodetic latitude at the midpoint.
        lon_at_midpt: Spacecraft geodetic longitude at the midpoint.
        lat_mesh: 2-D latitude mesh of the L3 grid.
        lon_mesh: 2-D longitude mesh of the L3 grid.
        model_B: Modeled B field on the L3 grid (reshaped to lat_mesh.shape).
        Je: East current density on the L3 grid.
        Jn: North current density on the L3 grid.
        lats: L2 observation geodetic latitudes (4 MEM x n_obs).
        lons: L2 observation geodetic longitudes (4 MEM x n_obs).
        obs_B: Observed (L2) B field along the MEM footprints.
        algorithm: "down" or "mag", per the L2 product's B field variant
            (None if no recognizable B variant is present).
    """

    time_utc: np.ndarray
    midpt: int
    lat_at_midpt: float
    lon_at_midpt: float
    lat_mesh: np.ndarray
    lon_mesh: np.ndarray
    model_B: np.ndarray
    Je: np.ndarray
    Jn: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    obs_B: np.ndarray
    algorithm: str | None


def _load_l3_pass(nc3: Dataset) -> L3PassData:
    """
    Extract the L3 (and embedded L2) data arrays needed for L3 mapping from an
    open L3 Dataset.

    Args:
        nc3 (Dataset): Open netCDF4 Dataset of the L3 product file.

    Returns:
        L3PassData: The extracted arrays and derived midpoint quantities.

    Raises:
        TypeError: If the L3 file's time field does not contain
            datetime.datetime or np.datetime64 objects.
    """
    time_utc = np.array(
        [datetime.datetime.fromisoformat(_) for _ in nc3["/l2_data/time_utc"][:]]
    )
    if not isinstance(time_utc[0], (datetime.datetime, np.datetime64)):
        raise TypeError(
            "obs_time must contain datetime.datetime or np.datetime64 objects."
        )

    # Use midpoint to select map orientation
    midpt = len(time_utc) // 2
    lat_at_midpt = nc3["l2_data/sc_lat"][midpt]
    lon_at_midpt = nc3["l2_data/sc_lon"][midpt]

    # Grab J, B, and coordinate data arrays
    lat_mesh = nc3["/l3_data/lat"][:]
    lon_mesh = nc3["/l3_data/lon"][:]
    model_B = nc3["/l3_data/Bd_geod_80"][:]
    Je = nc3["/l3_data/Je_110"][:]
    Jn = nc3["/l3_data/Jn_110"][:]
    lats = nc3["/l2_data/lat"][:]
    lons = nc3["/l2_data/lon"][:]
    l2grp = nc3.groups["l2_data"]
    if "Bd_geod_80" in l2grp.variables:  # NEW format
        obs_B = nc3["/l2_data/Bd_geod_80"][:]
        algorithm = "down"
    elif "Bmag_80" in l2grp.variables:  # NEW format
        obs_B = nc3["/l2_data/Bmag_80"][:]
        algorithm = "mag"
    else:  # No recognizable B field variant in this L2 group
        obs_B = np.array([])
        algorithm = None
    model_B = model_B.reshape(lat_mesh.shape)

    return L3PassData(
        time_utc=time_utc,
        midpt=midpt,
        lat_at_midpt=lat_at_midpt,
        lon_at_midpt=lon_at_midpt,
        lat_mesh=lat_mesh,
        lon_mesh=lon_mesh,
        model_B=model_B,
        Je=Je,
        Jn=Jn,
        lats=lats,
        lons=lons,
        obs_B=obs_B,
        algorithm=algorithm,
    )


def _l3_source_from_l2(l2_sorc: Path) -> Path:
    """
    Derive the corresponding L3 product path from an L2 product path by
    replacing the product-level token in the file name (not any "l2"
    substring in directory names).
    """
    # parts = l2_sorc.name.split("_")
    # if len(parts) >= 2 and parts[1].lower() == "l2":
    #     parts[1] = "l3"
    #     return l2_sorc.with_name("_".join(parts))
    # Fallback for nonstandard names: legacy full-path substitution.
    return Path(l2_sorc.as_posix().replace("l2", "l3"))


def _apply_dark_mode(dark_mode: bool) -> None:
    """
    Switch to the 'dark_background' style and matching face colors.
    """
    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
        plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR


def _plot_needs_update(
    source: Path,
    save_directory: Path,
    obs_date: datetime.datetime,
    plot_type: str,
    *,
    type_desc: str | None = None,
    spacecraft: str | None = None,
    orbit: int | None = None,
    tstmp: int | str | None = None,
    version: str | None = None,
    dpi: int = DFLT_RES,
    dark_mode: bool = False,
    old_format: bool = False,
    overwrite: bool = False,
) -> Path | None:
    """
    Check whether the plot for the given source and plot type needs to be
    (re)generated.

    Args:
        source (Path): Path of the source data file.
        save_directory (Path): Root directory in which to save the plot file.
        obs_date (datetime.datetime): UTC date of the observations.
        plot_type (str): Short identifier of the plot type, used in the
            file name.
        type_desc (str | None, optional): Human-readable plot description
            for logging; defaults to plot_type.
        spacecraft (str | None, optional): Spacecraft identifier.
        orbit (int | None, optional): Orbit number.
        tstmp (int | str | None, optional): Pass start timestamp.
        version (str | None, optional): Product version/revision string.
        dpi (int, optional): Resolution (DPI) of the plot.
        dark_mode (bool, optional): Dark-mode plot.
        old_format (bool, optional): Legacy file naming scheme.
        overwrite (bool, optional): Overwrite existing plot files.

    Returns:
        Path | None: Target plot path if plotting should proceed, or None
            if an up-to-date plot already exists and overwrite is not set.
    """
    ftgt, old_hash, new_hash = save_close_figure(
        source=source,
        save_directory=save_directory,
        obs_date=obs_date,
        spacecraft=spacecraft,
        orbit=orbit,
        tstmp=tstmp,
        version=version,
        plot_type=plot_type,
        dpi=dpi,
        dark_mode=dark_mode,
        old_format=old_format,
        name_only=True,
    )
    logger.debug(f"Checked file: {ftgt.as_posix()}")
    if not plot_update_required(
        source=source,
        ftgt=ftgt,
        plot_type=type_desc if type_desc is not None else plot_type,
        old_hash=old_hash,
        new_hash=new_hash,
        overwrite=overwrite,
    ):
        return None
    return ftgt


def _finalize_figure(
    fig: plt.Figure,
    nc_data: Dataset,
    source: Path,
    title: str,
    *,
    git_branch: str | None = None,
    git_commit: str | None = None,
    size: str = "small",
    dark_mode: bool = False,
) -> None:
    """
    Add the suptitle, product and pipeline metadata, and the EZIE logo to a
    figure.
    """
    fig.suptitle(title, weight="bold", fontsize="x-large", y=0.99, va="top")
    add_product_metadata(fig=fig, nc_data=nc_data, source=source, size=size)
    add_pipeline_metadata(
        fig=fig,
        nc_data=nc_data,
        git_branch=git_branch,
        git_commit=git_commit,
        size=size,
    )
    overlay_ezie_logo(fig=fig, dark_mode=dark_mode)


def _subsolar_geodetic(time_utc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the subsolar point in geodetic coordinates at each UTC time using
    SPICE.

    Args:
        time_utc (np.ndarray): Array of timezone-aware UTC datetimes.

    Returns:
        tuple[np.ndarray, np.ndarray]: (longitude_deg, latitude_deg) arrays
            of the subsolar point at each time.
    """
    subpnt_rad = []
    for tutc in time_utc:
        tdb = spiceypy.utc2et(tutc.isoformat()[:-6])
        (subpnt, _epoch, _to_subpnt) = spiceypy.subslr(
            "INTERCEPT/ELLIPSOID", "EARTH", tdb, "IAU_EARTH", "LT+S", "EARTH"
        )
        # Fix the subsolar point on the surface in geodetic coordinates
        subpnt_rad.append(
            spiceypy.recgeo(subpnt, EARTH_RADIUS_EQUATORIAL, EARTH_FLATTENING)
        )
    sun_geo_lon_deg = np.array([np.degrees(x[0]) for x in subpnt_rad])
    sun_geo_lat_deg = np.array([np.degrees(x[1]) for x in subpnt_rad])
    return sun_geo_lon_deg, sun_geo_lat_deg


def _subsolar_geodetic_at(tutc: datetime.datetime) -> tuple[float, float]:
    """
    Compute the subsolar point in geodetic coordinates at a single UTC time.

    Returns:
        tuple[float, float]: (longitude_deg, latitude_deg), with longitude
            folded into the [-180, +180] range.
    """
    tdb = spiceypy.utc2et(tutc.isoformat()[:-6])
    (subpnt, _epoch, _to_subpnt) = spiceypy.subslr(
        "INTERCEPT/ELLIPSOID", "EARTH", tdb, "IAU_EARTH", "LT+S", "EARTH"
    )
    sun_rad = spiceypy.recgeo(subpnt, EARTH_RADIUS_EQUATORIAL, EARTH_FLATTENING)
    sun_lon_mid = np.degrees(sun_rad[0])
    if sun_lon_mid > 180.0:
        sun_lon_mid -= 360.0
    return float(sun_lon_mid), float(np.degrees(sun_rad[1]))


def _select_hemisphere(lat: float) -> str | None:
    """
    Select the plotting hemisphere from a geographic latitude.
    """
    if lat > +40.0:
        return NORTH
    if lat < -40.0:
        return SOUTH
    return None


def _inset_map_projection(
    hemisphere: str | None,
    lat: float,
    lon: float,
) -> ccrs.Projection:
    """
    Projection for the per-orbit footprint map insets: polar stereographic
    for the polar hemispheres, orthographic centered on the pass midpoint
    otherwise.
    """
    if hemisphere == NORTH:
        return ccrs.NorthPolarStereo(central_longitude=0)  # 0 is default
    if hemisphere == SOUTH:
        return ccrs.SouthPolarStereo(central_longitude=180)  # 0 is default
    return ccrs.Orthographic(central_latitude=lat, central_longitude=lon)


def _make_l3_projection(
    hemisphere: str | None,
    sun_geo_lon_mid: float,
    lat_at_midpt: float,
    lon_at_midpt: float,
    south_inverted: bool,
) -> tuple[np.ndarray, int, ccrs.Projection]:
    """
    Select the magnetic-latitude parallels, MLT sense, and cartopy projection
    for an L3 map of the given hemisphere.

    'lats_m' are the magnetic latitudes at which to overlay parallels: the
    ACTUAL APEX mlats, with sign appropriate for the selected hemisphere
    regardless of any projection tricks. 'MLT_sign' (-1 for a south-inverted
    map) is used to "trick" cartopy into plotting the southern hemisphere in
    the heliophysics "transparent earth" format, where we look down at the
    SOUTH pole from above the NORTH pole. That 'south_inverted' format keeps
    Noon, Dawn, etc. in the same relative positions and orientation for both
    hemispheres; the multiplier is applied to the "projected" quantities while
    latitudes retain their normal hemisphere-appropriate sense.

    Returns:
        tuple[np.ndarray, int, ccrs.Projection]: (lats_m, MLT_sign, map_proj)
    """
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
    return lats_m, MLT_sign, map_proj


def _mlt_label(mlt_deg: float) -> str:
    """
    MLT axis label for a magnetic local time angle (degrees), with cardinal
    names at midnight, dawn, noon, and dusk.
    """
    if mlt_deg == 0.0:
        return "Midnight"
    if mlt_deg == 90.0:
        return "Dawn"
    if mlt_deg == 180.0:
        return "Noon"
    if mlt_deg == 270.0:
        return "Dusk"
    return f"{mlt_deg / 15:02.0f}H"


def _stack_mem_fields(nc_data: Dataset, field_tmpl: str) -> np.ndarray:
    """
    Stack the MEM-numbered netCDF fields `field_tmpl{1..4}` into a single
    (NUM_MEM, n_obs) array.
    """
    return np.vstack([nc_data[f"{field_tmpl}{mm}"] for mm in MEM_NUMBERS])


def _update_range(
    pair: np.ndarray, low_sample: np.ndarray, high_sample: np.ndarray | None = None
) -> None:
    """
    Widen a length-2 [min, max] pair (in place) to include all finite values
    of the given samples. The minimum is widened by low_sample and the maximum
    by high_sample (low_sample when not given). NaN entries are ignored.
    """
    if high_sample is None:
        high_sample = low_sample
    pair[0] = np.nanmin(
        np.concatenate(([pair[0]], np.asarray(low_sample, dtype=float)))
    )
    pair[1] = np.nanmax(
        np.concatenate(([pair[1]], np.asarray(high_sample, dtype=float)))
    )


def _add_mem_beam_diagram(
    fig: plt.Figure,
    hi: float,
    *,
    left: float | None = None,
    bottom: float | None = None,
    center_x: float | None = None,
    bottom_frac: float = 0.0,
    zorder: int | None = None,
) -> plt.Axes:
    """
    Overlay the MEM beam numbering/pointing diagram on a figure and return
    its axes.

    Placement is either an explicit (left, bottom) figure-fraction rect
    origin, or a center_x (horizontally centered on that figure fraction)
    with a bottom measured as a fraction of the figure aspect ratio.
    """
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    fig_aspect_ratio = fw / fh
    mem_beam_img = plt.imread(MEM_BEAM_DIAGRAM_PATH)
    wi = hi * (mem_beam_img.shape[1] / mem_beam_img.shape[0]) / fig_aspect_ratio
    if left is None:
        left = (center_x - wi / 2) if center_x is not None else 0.0
    if bottom is None:
        bottom = bottom_frac * fig_aspect_ratio
    img_axs: plt.Axes = fig.add_axes(rect=(left, bottom, wi, hi))
    if zorder is None:
        img_axs.imshow(mem_beam_img, aspect="auto")
    else:
        img_axs.imshow(mem_beam_img, aspect="auto", zorder=zorder)
    img_axs.axis("off")
    return img_axs


def _setup_mlat_mlt_axes(lat_axs: plt.Axes) -> plt.Axes:
    """
    Configure a magnetic-latitude time axis with a twin magnetic-local-time
    axis; return the twin axis.
    """
    lat_axs.grid(axis="both")
    lat_axs.set_ylabel("Magnetic Latitude\n(APEX, degrees)", weight="bold")
    lat_axs.set_xlabel("Time (UTC)", weight="bold")
    mlt_axs = lat_axs.twinx()
    mlt_axs.set_ylim(-0.5, 24.5)
    mlt_axs.set_yticks(range(0, 25, 6))
    mlt_axs.set_yticklabels(
        ["Midnight", "Dawn", "Noon", "Dusk", "Midnight"], size="x-small"
    )
    mlt_axs.set_ylabel("Magnetic Local Time\n(APEX, hours)", weight="bold")
    return mlt_axs


def plot_update_required(
    source: Path,
    ftgt: Path,
    plot_type: str,
    old_hash: str,
    new_hash: str,
    overwrite: bool = False,
):
    if ftgt.exists():
        if old_hash == new_hash:
            if not overwrite:
                logger.info(
                    f"File exists, source hash unchanged, skipping: {ftgt.name}"
                )
                return False
            else:
                logger.info(
                    f"File exists but overwrite flag is set, replacing: {ftgt.name}"
                )
        else:
            logger.info(
                f"Source file hash has changed, updating plot for: {source.name}"
            )
    else:
        logger.info(f"File does not exist, creating {plot_type} plot: {ftgt.name}")
    return True


def clip_to_circle(
    map_axs: GeoAxes,
    proj_method: ccrs.Projection,
    hemisphere: str,
):
    """_summary_

    Args:
        map_axs (GeoAxes): cartopy axes that we wish to modify
        proj_method ():
        hemisphere (str): NORTH or SOUTH for this polar projection tweak
    """
    x_0, y_0 = proj_method.transform_point(
        0,
        +POLE_LAT if hemisphere == NORTH else -POLE_LAT,
        src_crs=DATA_TRANSFORM,
    )
    x_1, y_1 = proj_method.transform_point(
        45,
        MAG_LAT_LOWER_LIMIT if hemisphere == NORTH else -MAG_LAT_LOWER_LIMIT,
        src_crs=DATA_TRANSFORM,
    )
    map_meters: float = np.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2)
    map_axs.set_extent(
        (
            -map_meters,
            +map_meters,
            -map_meters,
            +map_meters,
        ),
        crs=proj_method,
    )
    # Compute a circle in axes coordinates that will be used as a clipping boundary for
    # the map.
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    map_axs.set_boundary(circle, transform=map_axs.transAxes)


def coverage_plot_polar_layout(
    observation_date: datetime.datetime,
    hemisphere: str,
    show_terminator_at: datetime.datetime | None = None,
    show_mag_lat: bool = False,
    south_inverted: bool = False,
    dark_mode: bool = False,
):
    """
    Instantiate the polar magnetic-coordinate (MLT vs magnetic latitude) layout
    used for per-hemisphere coverage maps.

    Creates one figure with a single row of polar stereographic axes, one per
    spacecraft, oriented so that magnetic local time (MLT) runs around the
    pole. Adds MLT and magnetic-latitude gridlines, an optional day/night
    terminator, and magnetic continent overlays.

    Args:
        observation_date (datetime.datetime): UTC date, used for the plot
            title.
        hemisphere (str): Hemisphere to plot (e.g., NORTH or SOUTH).
        show_terminator_at (datetime.datetime | None, optional): Time at which
            to draw the day/night terminator. Defaults to None.
        show_mag_lat (bool, optional): Show magnetic latitude gridlines.
            Defaults to False.
        south_inverted (bool, optional): Invert the southern hemisphere MLT
            axis (heliospheric community mapping style). Defaults to False.
        dark_mode (bool, optional): Use 'dark mode' styling. Defaults to False.

    Returns:
        tuple(plt.Figure, dict[int, GeoAxes]): The figure and a dict of the
            per-spacecraft axes (keyed by spacecraft index).
    """
    # Instantiate figure and axes for plotting - tweak some rcparams as needed
    _apply_dark_mode(dark_mode)

    fig = plt.figure(figsize=ALTERNATE_FIG_SIZE)
    hndx = 0
    rows, cols = 1, len(SPACECRAFT)
    maprowspan = 1
    lat_lower_limit = MAG_LAT_LOWER_LIMIT
    lats_n = np.arange(lat_lower_limit, 90.0, 10.0)  # Lats at which to draw gridlines

    ccw = +1
    if hemisphere == NORTH or hemisphere is None:
        proj_method = ccrs.NorthPolarStereo(central_longitude=0)  # 0 is default
    elif hemisphere == SOUTH:
        proj_method = ccrs.SouthPolarStereo(central_longitude=180)  # 0 is default
        if not south_inverted:
            ccw = -1

    all_axs: dict = {}
    for sndx, spcrft in enumerate(SPACECRAFT):
        map_axs: GeoAxes = plt.subplot2grid(
            (rows, cols),
            (hndx, sndx),
            rowspan=maprowspan,
            colspan=1,
            projection=proj_method,
        )
        # ty:ignore[invalid-assignment]

        if hemisphere is not None:
            clip_to_circle(
                map_axs=map_axs, proj_method=proj_method, hemisphere=hemisphere
            )

        lon_delta = 30.0
        gl = map_axs.gridlines(
            draw_labels=True,
            x_inline=False,
            xlocs=np.arange(-180.0, 180.0, lon_delta),
            y_inline=True,
            ylocs=lats_n if hemisphere == NORTH else [-1 * lat for lat in lats_n],
            color="black" if not dark_mode else DARK_MODE_TEXT_COLOR,
        )
        gl.xlabel_style = {
            "size": 1,
            "color": "white" if not dark_mode else DARK_MODE_FACE_COLOR,
        }  # Suppress longitudes, we'll add MLT ourselves
        gl.ylabel_style = {"size": "small", "zorder": 5}

        # Add labels in MLT to the polar stereographic plot, with 0 at bottom (-y axis)
        mlt_locs = np.arange(0, 360.0, lon_delta)
        mlt_desc = {f"{mlt:.0f}": _mlt_label(mlt) for mlt in mlt_locs}
        for mm, mlt in enumerate(mlt_locs):
            mlt_pos = (ccw * mlt - 90.0) % 360.0  # Rotate to coord sys with 0 at x axis
            x = 0.5 + 0.55 * np.cos(np.radians(mlt_pos))
            y = 0.5 + 0.55 * np.sin(np.radians(mlt_pos))
            map_axs.text(
                x,
                y,
                mlt_desc[f"{mlt:.0f}"],
                transform=map_axs.transAxes,
                color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
                va="center",
                horizontalalignment="center",
            )

        if show_terminator_at is not None:
            # QUERY: Could we take difference from noon or midnight & flip sign to make
            # this work when plotting with south_inverted set, i.e., "trick" cartopy?
            map_axs.add_feature(
                Nightshade(
                    show_terminator_at,
                    alpha=TERMINATOR_ALPHA,
                    color=TERMINATOR_COLOR,
                    zorder=2,
                )
            )

        clip_to_circle(map_axs=map_axs, proj_method=proj_method, hemisphere=hemisphere)

        # Set axes title and save axes object in dictionary for use below
        map_axs.set_title(f"{spcrft} - {hemisphere}", loc="left", pad=12)
        all_axs[sndx] = map_axs

    return fig, all_axs


def coverage_plot_stereographic_layout(
    observation_date: datetime.datetime,
    hemisphere: str,
    show_terminator_at: datetime.datetime | None = None,
    show_mag_lat: bool = False,
    south_inverted: bool = False,  # for compatibility w/other plot method calls
    dark_mode: bool = False,
):
    """
    Instantiate the geographic (geodetic lat/lon) polar stereographic layout
    used for per-hemisphere coverage maps.

    Creates one figure with a single row of polar stereographic axes, one per
    spacecraft, in geographic coordinates. Adds geographic lat/lon gridlines,
    an optional day/night terminator, and coastlines; optionally overlays
    magnetic latitude gridlines.

    Args:
        observation_date (datetime.datetime): UTC date, used for the plot
            title.
        hemisphere (str): Hemisphere to plot (e.g., NORTH or SOUTH).
        show_terminator_at (datetime.datetime | None, optional): Time at which
            to draw the day/night terminator. Defaults to None.
        show_mag_lat (bool, optional): Overlay magnetic latitude gridlines in
            addition to the geographic grid. Defaults to False.
        south_inverted (bool, optional): Invert the southern hemisphere
            (heliospheric community mapping style). Defaults to False.
        dark_mode (bool, optional): Use 'dark mode' styling. Defaults to False.

    Returns:
        tuple(plt.Figure, dict[int, GeoAxes]): The figure and a dict of the
            per-spacecraft axes (keyed by spacecraft index).
    """
    # Instantiate figure and axes for plotting - tweak some rcparams as needed
    _apply_dark_mode(dark_mode)
    # cartopy does NOT use these, adjust with gridlines artist below
    # plt.rcParams["xtick.labelsize"] = "small"
    # plt.rcParams["ytick.labelsize"] = "small"

    fig = plt.figure(figsize=ALTERNATE_FIG_SIZE)
    hndx = 0
    rows, cols = 1, len(SPACECRAFT)
    maprowspan = 1
    lat_lower_limit = GEO_LAT_LOWER_LIMIT if show_mag_lat else MAG_LAT_LOWER_LIMIT
    lats_n = np.arange(lat_lower_limit, 90, 10)  # Latitudes at which to draw gridlines

    if hemisphere is None:
        hemisphere = NORTH

    if hemisphere == NORTH:
        proj_method = ccrs.NorthPolarStereo()
    else:
        proj_method = ccrs.SouthPolarStereo()

    all_axs = {}
    for sndx, spcrft in enumerate(SPACECRAFT):
        map_axs: GeoAxes = plt.subplot2grid(
            (rows, cols),
            (hndx, sndx),
            rowspan=maprowspan,
            colspan=1,
            projection=proj_method,
        )  # ty:ignore[invalid-assignment]

        gl = map_axs.gridlines(
            draw_labels=True,
            x_inline=False,
            xlocs=range(-180, 180, 30),
            y_inline=True,
            ylocs=lats_n if hemisphere == NORTH else [-1 * lat for lat in lats_n],
            color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
        )
        gl.xlabel_style = {"size": "small"}
        gl.ylabel_style = {"size": "small"}

        if show_terminator_at is not None:
            map_axs.add_feature(
                Nightshade(
                    show_terminator_at,
                    alpha=TERMINATOR_ALPHA,
                    color=TERMINATOR_COLOR,
                    zorder=2,
                )
            )

        clip_to_circle(map_axs=map_axs, proj_method=proj_method, hemisphere=hemisphere)

        if show_mag_lat:
            # Add _first_ legend, indicating latitude coordinate types
            geo_lat_artist = plt.Line2D(
                (0, 1),
                (0, 0),
                color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
                linestyle="solid",
                lw=1,
            )
            mag_lat_artist = plot_geomagnetic_references(
                map_axs,
                observation_date,
                latitudes=(
                    lats_n if hemisphere == NORTH else [-1 * lat for lat in lats_n]
                ),
                lat_clr="black" if not dark_mode else DARK_MODE_GRID_COLOR,
            )
            _lat_lgd = map_axs.legend(
                [geo_lat_artist, mag_lat_artist],
                ["Geodetic", "Geomagnetic"],
                bbox_to_anchor=(1.10, 1.00),
                loc="lower right",
                # Use raw string to avoid invalid escape sequence warning from \circ
                title=r"Latitude: $10^\circ$grid",
                title_fontsize="small",
                fontsize="x-small",
            )
            map_axs.add_artist(_lat_lgd)

        # Set axes title and save axes object in dictionary for use below
        map_axs.set_title(f"{spcrft} - {hemisphere}", loc="left", pad=12)
        all_axs[sndx] = map_axs

    return fig, all_axs


def mollweide_layout(
    observation_date: datetime.datetime,
    show_terminator_at: datetime.datetime | None = None,
    num_sc: int = 1,
    show_mag_lat: bool = False,
    dark_mode: bool = False,
):
    """
    Instantiate the equatorial Mollweide layout used for equatorial coverage
    maps.

    Creates one figure with a single column of Mollweide-projection axes, one
    per spacecraft (num_sc of them), in geographic coordinates. Adds lat/lon
    gridlines, an optional day/night terminator, and coastlines; optionally
    overlays magnetic latitude gridlines.

    Args:
        observation_date (datetime.datetime): UTC date, used for the plot
            title.
        show_terminator_at (datetime.datetime | None, optional): Time at which
            to draw the day/night terminator. Defaults to None.
        num_sc (int, optional): Number of spacecraft (axes) to create.
            Defaults to 1.
        show_mag_lat (bool, optional): Overlay magnetic latitude gridlines in
            addition to the geographic grid. Defaults to False.
        dark_mode (bool, optional): Use 'dark mode' styling. Defaults to False.

    Returns:
        tuple(plt.Figure, dict[int, GeoAxes]): The figure and a dict of the
            per-spacecraft axes (keyed by spacecraft index).
    """
    # Instantiate figure and axes for plotting - tweak some rcparams as needed
    _apply_dark_mode(dark_mode)
    # cartopy does NOT use these, adjust with gridlines artist below
    # plt.rcParams["xtick.labelsize"] = "small"
    # plt.rcParams["ytick.labelsize"] = "small"

    fig = plt.figure(figsize=(ALTERNATE_FIG_SIZE[0], num_sc * ALTERNATE_FIG_SIZE[1]))
    rows, cols = 2 * num_sc, 3
    maprowspan = 2
    mapcolspan = 3
    lats_n = np.arange(-30, 31, 10)  # Latitudes at which to draw gridlines
    proj_method = ccrs.Mollweide()

    all_axs = {}
    for sndx in range(num_sc):
        map_axs: GeoAxes = plt.subplot2grid(
            (rows, cols),
            (2 * sndx, 0),
            rowspan=maprowspan,
            colspan=mapcolspan,
            projection=proj_method,
        )  # ty:ignore[invalid-assignment]

        map_axs.set_global()
        gl = map_axs.gridlines(
            draw_labels=True,
            # x_inline=True,
            # y_inline=True,
            xlocs=range(-180, 180, 30),
            ylocs=range(-90, 91, 15),
            color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
        )
        gl.xlabel_style = {"size": "small"}
        gl.ylabel_style = {"size": "small"}

        if show_terminator_at is not None:
            map_axs.add_feature(
                Nightshade(
                    show_terminator_at,
                    alpha=TERMINATOR_ALPHA,
                    color=TERMINATOR_COLOR,
                    zorder=2,
                )
            )

        if show_mag_lat:
            # Add _first_ legend, indicating latitude coordinate types
            geo_lat_artist = plt.Line2D(
                (0, 1),
                (0, 0),
                color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
                linestyle="solid",
                lw=1,
            )
            _mag_lat_artist = plot_geomagnetic_references(
                map_axs,
                observation_date,
                latitudes=[0],
                ls="solid",
                lw=2,
                lat_clr="black" if not dark_mode else DARK_MODE_GRID_COLOR,
            )
            mag_lat_artist = plot_geomagnetic_references(
                map_axs,
                observation_date,
                latitudes=lats_n,
                lat_clr="black" if not dark_mode else DARK_MODE_GRID_COLOR,
            )
            _lat_lgd = map_axs.legend(
                [geo_lat_artist, mag_lat_artist],
                [
                    r"Geodetic, $15^\circ$",
                    r"Geomagnetic, $10^\circ$",
                ],
                bbox_to_anchor=(1.00, 0.90),
                loc="lower right",
                # Use raw string to avoid invalid escape sequence warning from \circ
                title="Latitude Grid",
                title_fontsize="small",
                fontsize="x-small",
            )
            map_axs.add_artist(_lat_lgd)
            all_axs[sndx] = map_axs

        # Set axes title and save axes object in dictionary for use below
        map_axs.set_title(f"{SPACECRAFT[sndx]} ", loc="left", weight="bold")  # ,pad=12)

    return fig, all_axs


def plot_geolocation(
    nc_data: Dataset,
    source: Path,
    indices: tuple,
    save_directory: Path,
    git_branch: str | None = None,
    git_commit: str | None = None,
    figure_dpi: int = DFLT_RES,
    dark_mode: bool = False,
    overwrite: bool = False,
):
    """
    Plot geolocation parameters using L2 files (or any other product level containing
    the required geolocation fields).

    Args:
        nc_data (Dataset): Open netCDF4 Dataset of the product file to plot.
        source (Path): Path of the source product file, used for plot naming
            and metadata.
        indices (tuple): (Start, stop) observation index pair delimiting the
            science pass to plot.
        save_directory (Path): Root directory in which to save the generated
            plot files.
        git_branch (str | None, optional): Pipeline branch to display when the
            product does not carry one. Defaults to None.
        git_commit (str | None, optional): Pipeline commit hash to display
            when the product does not carry one. Defaults to None.
        figure_dpi (int, optional): Resolution (DPI) at which to generate the
            plots. Defaults to DFLT_RES.
        dark_mode (bool, optional): Generate plots using 'dark mode' format.
            Defaults to False.
        overwrite (bool, optional): Overwrite existing plot files; otherwise
            skip. Defaults to False.
    """
    _apply_dark_mode(dark_mode)

    geo_group = nc_data.groups["Geolocation"]
    sc_id = nc_data.getncattr("SpaceVehicle")

    # Get time data and convert to datetime objects
    time_utc, obs_date = get_datetime_from_utc_string(nc_data.groups["Time"])
    i0, i1 = indices
    use_obs = np.s_[i0:i1]
    t_stamp = time_utc[i0].strftime("%H%M%S")
    orb_num = nc_data["Science/orbit_number"][i0]
    time_utc = time_utc[use_obs]
    plot_type = "geolocation"

    if (
        _plot_needs_update(
            source=source,
            save_directory=save_directory,
            obs_date=obs_date,
            plot_type=plot_type,
            spacecraft=sc_id,
            tstmp=t_stamp,
            overwrite=overwrite,
        )
        is None
    ):
        return

    # Prepare subplots for grouped geolocation parameters
    rows = 5
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=DEFAULT_FIG_SIZE, sharex=True)

    # Define groups of variables to plot in each window
    groups = [
        [("sat_pos_eci", "Cart"), ("sat_pos_ecef", "Cart")],
        [("sat_vel_eci", "Cart"), ("sat_vel_ecef", "Cart")],
        [
            ("sat_att_quaternion_eci", "Quaternion"),
            ("sat_att_quaternion_ecef", "Quaternion"),
            ("sat_att_quaternion_lvlh", "Quaternion"),
        ],
        [("sat_roll",), ("sat_pitch",), ("sat_yaw",)],
        [("sat_roll_rate",), ("sat_pitch_rate",), ("sat_yaw_rate",)],
        [("sat_solar_zen",), ("sat_solar_az",)],
        [("reference_altitude",)],
        [
            ("look_dir1", "Cart"),
            ("look_dir2", "Cart"),
            ("look_dir3", "Cart"),
            ("look_dir4", "Cart"),
        ],
        [
            ("earth_inc_ang1",),
            ("earth_inc_ang2",),
            ("earth_inc_ang3",),
            ("earth_inc_ang4",),
        ],
        [("data_flag",)],
    ]

    # Plot each group of variables
    for ax, group in zip(axs.flat, groups, strict=False):
        long_name = "Unknown"
        units = "None"
        for var_name, *dims in group:
            if var_name in geo_group.variables:
                var_data = geo_group.variables[var_name][use_obs]
                # Get long name and units for plot axes labels, if available.
                if hasattr(geo_group.variables[var_name], "long_name"):
                    long_name = geo_group.variables[var_name].long_name
                if hasattr(geo_group.variables[var_name], "units"):
                    units = geo_group.variables[var_name].units
                if dims:
                    for i in range(var_data.shape[1]):
                        ax.plot(time_utc, var_data[:, i], label=f"{var_name}_{i}")
                else:
                    ax.plot(time_utc, var_data, label=var_name)
        try:
            trim_pos = long_name.index("Unit")
            long_name = long_name[0:trim_pos]
        except ValueError:
            pass
        ax.set_ylabel(f"{long_name}\n({units})", fontsize="x-small")
        _h, _l = ax.get_legend_handles_labels()
        ax.legend(loc="upper left", ncols=2 if len(_l) > 4 else 1, fontsize="x-small")
        ax.grid(True)
        set_xaxis_tick_format(
            ax,
            use_seconds=(time_utc[-1] - time_utc[0]).total_seconds() < 120,
        )

    for col in range(cols):
        axs[rows - 1, col].set_xlabel(
            f"Time (UTC) - {time_utc[0].strftime('%Y-%m-%d')}", weight="bold"
        )

    # Tweak position and add any figure-level annotation
    fig_title = (
        f"Geolocation Parameters: {sc_id} - Orbit {orb_num}\n"
        f"{time_utc[0].strftime('%Y-%m-%d (%j) %H:%M:%S')} - "
        f"{time_utc[-1].strftime('%Y-%m-%d (%j) %H:%M:%S')}"
    )
    _finalize_figure(
        fig=fig,
        nc_data=nc_data,
        source=source,
        title=fig_title,
        git_branch=git_branch,
        git_commit=git_commit,
    )
    plt.subplots_adjust(
        left=0.06, right=0.98, bottom=0.07, top=0.93, wspace=0.12, hspace=0.01
    )
    save_close_figure(
        source=source,
        save_directory=save_directory,
        figure=fig,
        obs_date=obs_date,
        spacecraft=sc_id,
        tstmp=t_stamp,
        plot_type=plot_type,
        dpi=figure_dpi,
    )


def plot_ancillary(
    nc_data: Dataset,
    source: Path,
    indices: tuple,
    save_directory: Path,
    git_branch: str | None = None,
    git_commit: str | None = None,
    figure_dpi: int = DFLT_RES,
    dark_mode: bool = False,
    overwrite: bool = False,
):
    """
    Plot the reference IGRF-14 B field values stored in the Ancillary group.

    Args:
        nc_data (Dataset): Open netCDF4 Dataset of the product file to plot.
        source (Path): Path of the source product file, used for plot naming
            and metadata.
        indices (tuple): (Start, stop) observation index pair delimiting the
            science pass to plot.
        save_directory (Path): Root directory in which to save the generated
            plot files.
        git_branch (str | None, optional): Pipeline branch to display when the
            product does not carry one. Defaults to None.
        git_commit (str | None, optional): Pipeline commit hash to display
            when the product does not carry one. Defaults to None.
        figure_dpi (int, optional): Resolution (DPI) at which to generate the
            plots. Defaults to DFLT_RES.
        dark_mode (bool, optional): Generate plots using 'dark mode' format.
            Defaults to False.
        overwrite (bool, optional): Overwrite existing plot files; otherwise
            skip. Defaults to False.
    """
    _apply_dark_mode(dark_mode)

    # Get time data and convert to datetime objects
    time_utc, obs_date = get_datetime_from_utc_string(nc_data.groups["Time"])
    i0, i1 = indices
    use_obs = np.s_[i0:i1]
    t_stamp = time_utc[i0].strftime("%H%M%S")
    orb_num = nc_data["Science/orbit_number"][i0]
    time_utc = time_utc[use_obs]
    sc_id = nc_data.getncattr("SpaceVehicle")
    plot_type = "ancillary"

    if (
        _plot_needs_update(
            source=source,
            save_directory=save_directory,
            obs_date=obs_date,
            plot_type=plot_type,
            type_desc="ancillary (IGRF)",
            spacecraft=sc_id,
            tstmp=t_stamp,
            overwrite=overwrite,
        )
        is None
    ):
        return

    # Prepare figure for plotting with sharex and sharey
    fig, axs = plt.subplots(
        1, NUM_FLD, figsize=DEFAULT_FIG_SIZE, sharex=False, sharey=True
    )

    # Plot each anc_model_bgeo variable in separate subplots
    for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
        ax = axs[mem_ndx]
        var_bgeon = nc_data[f"Ancillary/anc_model_bgeon{mem_num}"][use_obs]
        var_bgeoe = nc_data[f"Ancillary/anc_model_bgeoe{mem_num}"][use_obs]
        var_bgeod = nc_data[f"Ancillary/anc_model_bgeod{mem_num}"][use_obs]
        var_bgeo = nc_data[f"Ancillary/anc_model_bgeo{mem_num}"][use_obs]

        b_vars = [var_bgeon, var_bgeoe, var_bgeod, var_bgeo]
        for vv, b_var in enumerate(b_vars):
            ax.plot(time_utc, b_var, label=FLD_CMP[vv], color=FLD_CLR[vv])

        # Format the x-axis to show only hours and minutes
        set_xaxis_tick_format(ax, max_ticks=16 // NUM_FLD + 1, rotation=45)
        ax.set_title(
            f"MEM {mem_num} - "
            f"Look Direction {MEM_LOOK_DIRECTIONS[mem_ndx]}"
            r"º",
            # r"$\bf{^\circ}$",
            weight="bold",
            size="medium",
        )
        ax.grid(True)
        ax.set_xlabel("Time (UTC)", weight="bold")
        if mem_ndx == 0:
            ax.set_ylabel("IGRF-14 B-Fields\n(nT)", weight="bold")
            ax.legend(loc="upper left")

    # Tweak position and add any figure-level annotation
    fig_title = "IGRF-14 B Field Values: "
    fig_title = fig_title + (
        f"{sc_id} - Orbit {orb_num}\n"
        f"{time_utc[0].strftime('%Y-%m-%d (%j) %H:%M:%S UT')} - "
        f"{time_utc[-1].strftime('%Y-%m-%d (%j) %H:%M:%S UT')}"
    )
    _finalize_figure(
        fig=fig,
        nc_data=nc_data,
        source=source,
        title=fig_title,
        git_branch=git_branch,
        git_commit=git_commit,
    )
    plt.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.10,
        top=0.91,
        hspace=0.00,
        wspace=0.00,
    )
    save_close_figure(
        source=source,
        save_directory=save_directory,
        figure=fig,
        obs_date=obs_date,
        spacecraft=sc_id,
        tstmp=t_stamp,
        plot_type=plot_type,
        dpi=figure_dpi,
    )


def plot_calibration(
    nc_data: Dataset,
    source: Path,
    indices: tuple,
    save_directory: Path,
    git_branch: str | None = None,
    git_commit: str | None = None,
    t_diff: bool = False,
    figure_dpi: int = DFLT_RES,
    dark_mode: bool = False,
    overwrite: bool = False,
):
    """
    Plot the calibrated Ta and Tb scene temperatures as images with frequency on the x
    axis and time on the y axis.

    Args:
        nc_data (Dataset): Open netCDF4 Dataset of the L1 product file to plot.
        source (Path): Path of the source product file, used for plot naming
            and metadata.
        indices (tuple): (Start, stop) observation index pair delimiting the
            science pass to plot.
        save_directory (Path): Root directory in which to save the generated
            plot files.
        git_branch (str | None, optional): Pipeline branch to display when the
            product does not carry one. Defaults to None.
        git_commit (str | None, optional): Pipeline commit hash to display
            when the product does not carry one. Defaults to None.
        t_diff (bool, optional): Plot the temperature difference from one time
            step to the next, rather than the value at each timestep.
            Defaults to False.
        figure_dpi (int, optional): Resolution (DPI) at which to generate the
            plots. Defaults to DFLT_RES.
        dark_mode (bool, optional): Generate plots using 'dark mode' format.
            Defaults to False.
        overwrite (bool, optional): Overwrite existing plot files; otherwise
            skip. Defaults to False.
    """
    _apply_dark_mode(dark_mode)

    # Get time data and convert to datetime objects
    time_utc, obs_date = get_datetime_from_utc_string(
        nc_data.groups["Time"], indices=indices
    )
    t_stamp = time_utc[0].strftime("%H%M%S")
    # i0, i1 = indices  # Whole file with JPL "padding"

    # Plot only times when the acquisition mode is SCIENCE
    obs_flg = nc_data["ChannelOrderedCounts/acquisition_mode"][:] == SCIENCE
    i0, i1 = np.argmax(obs_flg), obs_flg.size - np.argmax(obs_flg[::-1])
    # Truncated per-orbit L1 file may end while SCIENCE mode flag is still turned on.
    if i1 == obs_flg.size:
        i1 -= 1  # Adjust end of range when this occurs

    time_utc = time_utc[i0:i1]
    sc_id = nc_data.getncattr("SpaceVehicle")
    product = nc_data.getncattr("product")
    orb_num = nc_data["Science/orbit_number"][i0]  # Always use starting orbit number?

    t_kinds = ["TA", "TB"]
    # t_kinds = ["TB"]
    stokes = ["V", "H", "S3", "S4"]
    num_cols = len(stokes)
    num_rows = len(MEM_NUMBERS)

    if product == "L1":
        # Working from L1
        tkind_vars = [
            "CalibratedSceneTemperatures/ta",
            "CalibratedSceneTemperatures/tb",
        ]
        # 1e3 => GHZ to MHz
        frequency_MHZ = nc_data["FrequencyGrid/frequency"][:] * 1.0e3
        ckind = "calibration"

    else:
        # Working from L0B
        ckind = "calibration_l0b"
        tkind_vars = [
            "Calibration/Anc_ta",
            "Calibration/Anc_tb",
        ]
        wavenumbers = nc_data["FrequencyGrid/wavenumbers"][:]  # frequency data
        # Constants: wavenumbers in cm^-1, C in m/s: 100 => m to cm, 1e-6 => Hz to MHz
        frequency_MHZ = C * 100.0 * wavenumbers * 1e-6

    freq_mhz_delta = frequency_MHZ - O2_CTR_FREQ_MHZ
    fc = np.argmin(np.abs(freq_mhz_delta))
    f0 = max(0, fc - FREQ_BIN_DELTA)
    f1 = min(len(freq_mhz_delta), fc + FREQ_BIN_DELTA)
    freq_mhz_delta = freq_mhz_delta[f0:f1]
    tgrd = np.tile(time_utc, (freq_mhz_delta.shape[0], 1)).T
    logger.debug(
        f"{freq_mhz_delta.shape} {tgrd.shape} "
        f"{nc_data['CalibratedSceneTemperatures/tb1'][i0:i1, f0:f1, 0].shape}"
    )

    if not t_diff:
        cmap = mpl.colormaps["viridis"]
        cmap.set_bad("white", 0.0)
    else:
        cmap = mpl.colormaps["bwr"]
        cmap.set_bad("black", 0.0)
    for kk, t_kind in enumerate(t_kinds):
        t_kind_str = t_kind if not t_diff else f"{t_kind}_difference"

        if (
            _plot_needs_update(
                source=source,
                save_directory=save_directory,
                obs_date=obs_date,
                plot_type=f"{ckind}-{t_kind_str}",
                type_desc=f"calibrated scene temperature ({t_kind})",
                spacecraft=sc_id,
                tstmp=t_stamp,
                overwrite=overwrite,
            )
            is None
        ):
            continue

        fig, axs = plt.subplots(
            num_rows + 1,
            num_cols,
            figsize=DEFAULT_FIG_SIZE,
            sharey=True,
            height_ratios=[1, 1, 1, 1, 0.25],
        )
        for col in range(num_cols):  # Iterate over columns (Stokes parameters)
            # Set fixed scale for Ta and Tb
            if not t_diff:
                match stokes[col].upper():
                    case "H" | "V":
                        vmin, vmax = +150, +300
                    case "S3" | "S4":
                        vmin, vmax = -40, +40
            else:
                match stokes[col].upper():
                    case "H" | "V":
                        vmin, vmax = -25, +25
                    case "S3" | "S4":
                        vmin, vmax = -50, +50

            for row in range(num_rows):
                ax = axs[row, col]
                var_name = f"{tkind_vars[kk]}{row + 1}"  # {TA or TB}{MEM number}
                var_title = f"{t_kind.upper()}{row + 1}"
                data = nc_data[var_name][i0:i1, :, col]  # Shape: (ObsRate, Freq_Array)
                if t_diff:
                    data -= data[0, :]

                # TODO - Mask off time steps where we were not in EARTHLOOK mode?
                tb_mesh = ax.pcolormesh(
                    freq_mhz_delta,
                    tgrd,
                    data[:, f0:f1],
                    shading="auto",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                ax.set_title(f"{var_title} Stokes {stokes[col]}")
                ax.set_xlim([-FREQ_DELTA_MHZ, +FREQ_DELTA_MHZ])
                if row == num_rows - 1:
                    ax.set_xlabel("Offset from Center Frequency (MHz)\n   ")
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel("Time (UTC)")
                    set_yaxis_tick_format(ax)
                if col == num_cols - 1:
                    ax.text(
                        1.01,
                        0.50,
                        f"MEM {row + 1}",
                        ha="left",
                        va="center",
                        transform=ax.transAxes,
                        rotation=90,
                    )

            # Add colorbar
            ax = axs[num_rows, col]  # axis row just for colorbar
            ax.set_axis_off()  # turn off visible axes components
            _cbar = plt.colorbar(
                tb_mesh,
                ax=ax,
                orientation="horizontal",
                location="bottom",
                fraction=0.45,
                extend="both",
            )
            if t_diff:
                _cbar.set_label("Brightness Temperature Difference (K)")
            else:
                _cbar.set_label("Brightness Temperature (K)")

        # Tweak position and add any figure-level annotation
        plt.subplots_adjust(
            left=0.05,
            right=0.98,
            bottom=0.08,
            top=0.915,
            wspace=0.02,
            hspace=0.22,
        )
        fig_title = (
            f"{product} Brightness Temperature - {t_kind_str} : "
            f"{sc_id} - Orbit {orb_num}\n"
            f"{time_utc[0].strftime('%Y-%m-%d (%j) %H:%M:%S')} - "
            f"{time_utc[-1].strftime('%Y-%m-%d (%j) %H:%M:%S')}"
        )
        _finalize_figure(
            fig=fig,
            nc_data=nc_data,
            source=source,
            title=fig_title,
            git_branch=git_branch,
            git_commit=git_commit,
        )
        save_close_figure(
            source=source,
            save_directory=save_directory,
            figure=fig,
            obs_date=obs_date,
            spacecraft=sc_id,
            tstmp=t_stamp,
            plot_type=f"{ckind}-{t_kind_str}",
            dpi=figure_dpi,
        )


def plot_retrieved_bd_only(
    nc_data: Dataset,
    source: Path,
    save_directory: Path,
    version: str | None = None,
    figure_dpi: int = DFLT_RES,
    dark_mode: bool = False,
    south_inverted: bool = False,
    overwrite: bool = False,
):
    """
    Plot the retrieved dB fields for each MEM in two formats, along with their estimated
    errors. Errors for individual dBs are calculated from the derived covariance values.

    Args:
        nc_data (Dataset): Open netCDF4 Dataset of the L2 product file to plot.
        source (Path): Path of the source product file, used for plot naming
            and metadata.
        save_directory (Path): Root directory in which to save the generated
            plot files.
        version (str | None, optional): Product version/revision string
            included in the plot file name. Defaults to None.
        figure_dpi (int, optional): Resolution (DPI) at which to generate the
            plots. Defaults to DFLT_RES.
        dark_mode (bool, optional): Generate plots using 'dark mode' format.
            Defaults to False.
        south_inverted (bool, optional): Invert the southern hemisphere MLT
            axis (heliospheric community mapping style). Defaults to False.
        overwrite (bool, optional): Overwrite existing plot files; otherwise
            skip. Defaults to False.

    Returns:
        None
    """
    # Deprecated option, will be removed
    # mode (str, optional): Plot mode label, "Corrected" or "Uncorrected".
    #     Defaults to "Uncorrected".
    mode = "Corrected"  # FIXME - no longer using this with new CDL+retrieval algorithm
    _apply_dark_mode(dark_mode)

    # Collate netcdf data for all spacecraft and orbits on the selected day and orbit.

    # Get time data (strings) and convert to datetime objects.
    # L2 files are already broken up into discrete science passes.
    # if len(nc_data["Time/time_utc"]) == 0:
    try:
        time_utc, obs_date = get_datetime_from_utc_string(nc_data.groups["Time"])
    except Exception as exc:
        logger.error(f"Exception encountered: {exc}")
        logger.error("Processing skipped--truncated or corrupted data file?")
        logger.error(f"Problem file (datetime values): {source.as_posix()}")
        return
    t_stamp = time_utc[0].strftime("%H%M%S")
    orb_num = nc_data["Science/orbit_number"][0]
    sc_id = nc_data.getncattr("SpaceVehicle")

    if orb_num == "" or sc_id == "":
        logger.warning(
            f"File missing basic metadata values, skipping: {source.as_posix()}"
        )
        return

    # QUERY: Attempt to further trim SV slewing from observations at start and finish?
    # if len(time_utc) > 12:
    #     use_obs = np.s_[5:-5]
    # else:
    #     use_obs = np.s_[:]
    use_obs = np.s_[:]  # See how things look without the haircut now

    # Define smoothing parameters for curve to be overlain on (noisy) dB plots.
    max_win = len(time_utc[use_obs])
    ave_win = min(AVERAGING_WINDOW, max_win)

    # Extract MEM retrieved dBs, geolocation and magnetic coordinates, covariance fields
    if mode.lower() == "corrected":
        # mem_dbd_val = [
        #     f"RetrievedParameters/retrieved_dbgeod{mm}" for mm in MEM_NUMBERS
        # ]
        plot_type = "retrieved_b_fields_dBs"
    else:
        # mem_dbd_val = [
        #     f"RetrievedParameters/{mode.lower()}_retrieved_dbgeod{mm}"
        #     for mm in MEM_NUMBERS
        # ]
        plot_type = f"retrieved_{mode.lower()}_b_fields_dBs"

    logger.debug(f"spacecraft ID is {sc_id}")
    if (
        _plot_needs_update(
            source=source,
            save_directory=save_directory,
            obs_date=obs_date,
            plot_type=plot_type,
            type_desc="retrieved B_d",
            spacecraft=sc_id,
            tstmp=t_stamp,
            overwrite=overwrite,
        )
        is None
    ):
        return

    rtrvgrp = nc_data.groups["RetrievedParameters"]
    if "observation_valid" in rtrvgrp.variables:  # presence implies this is NEW format
        # TODO: Need to support dbmag or dbdown?
        algorithm = "mag"
        dbd_tmpl = "debiased_dbmag"
        cov_tmpl = "retrieved_dbmag_error"
    else:
        algorithm = "down"
        if mode.lower() == "corrected":
            dbd_tmpl = "retrieved_dbgeod"
        else:
            dbd_tmpl = f"{mode.lower()}_retrieved_dbgeod"
        cov_tmpl = "cov_dd"

    # Stack MEM arrays so we can index and loop through them by number rather than
    # using 4 separate variable names.
    mem_dbd = _stack_mem_fields(nc_data, f"RetrievedParameters/{dbd_tmpl}")
    mem_cdd = _stack_mem_fields(nc_data, f"RetrievedParameters/{cov_tmpl}")
    if algorithm == "down":
        mem_cdd = np.sqrt(mem_cdd)  # Covariance is in nc4 files, not error
    mag_lat = _stack_mem_fields(nc_data, "MagneticCoords/obs_maglat")
    mag_ltm = _stack_mem_fields(nc_data, "MagneticCoords/obs_magLT")
    obs_lat = _stack_mem_fields(nc_data, "Geolocation/obs_lat")
    obs_lon = _stack_mem_fields(nc_data, "Geolocation/obs_lon")

    missing_val = mem_dbd == NCDF_MISSING
    mem_dbd = np.where(missing_val, np.nan, mem_dbd)
    mag_lat = np.where(missing_val, np.nan, mag_lat)
    mag_ltm = np.where(missing_val, np.nan, mag_ltm)
    obs_lat = np.where(missing_val, np.nan, obs_lat)
    obs_lon = np.where(missing_val, np.nan, obs_lon)

    valid_pnts = np.sum(~np.isnan(mem_dbd[0, :]))
    if valid_pnts < 3:
        logger.debug(~np.isnan(mem_dbd[0, :]))
        logger.error(
            f"Insufficient non-NaN samples to plot, skipping: {source.as_posix()}"
        )
        return

    # Plot magnetic field reference values and deltas from EZIE OSSE retrieval along
    # EZIE MEM lines of sight.
    logger.info("Generating plot of retrieved B field dBs")
    nrows, ncols = NUM_MEM + 1, 2

    # use = np.full_like(time_utc[:], fill_value=True, dtype=bool)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=DEFAULT_FIG_SIZE,
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    # Hide all axes on RHS - we'll add "special" axes manually there.
    for row in range(nrows):
        axs[row, 1].set_visible(False)

    # We'll save the total variable range for each field component across all MEMs here,
    # adjusting plot limits afterwards when we know what the correct range is for the
    # whole ensemble.
    b_rng = np.full((2), fill_value=np.nan)  # 4 B x [min,max]

    col = 0
    axs[-1, 1].set_visible(False)
    lat_axs = axs[-1, col]
    mlt_axs = _setup_mlat_mlt_axes(lat_axs)
    for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
        row = PLT_NDX[mem_ndx]
        mem_axs = axs[row, col]
        lat_axs.plot(
            time_utc[use_obs],
            mag_lat[mem_ndx, use_obs],
            color=MEM_CLR[mem_ndx],
            linestyle="solid",
        )
        mlt_axs.plot(
            time_utc[use_obs],
            mag_ltm[mem_ndx, use_obs],
            color=MEM_CLR[mem_ndx],
            linestyle="dotted",
        )
        mem_axs.errorbar(
            time_utc[use_obs],
            mem_dbd[mem_ndx][use_obs],
            yerr=mem_cdd[mem_ndx][use_obs],
            color=MEM_CLR[mem_ndx],
            ecolor="purple",
            linestyle="solid",
            errorevery=3,
            capsize=2.0,
            barsabove=True,
        )
        mem_axs.plot(
            time_utc[use_obs],
            moving_average(mem_dbd[mem_ndx][use_obs], ave_win),
            color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
            linestyle="solid",
            lw=1.0,
            zorder=3,
        )
        try:
            _update_range(
                b_rng,
                mem_dbd[mem_ndx][use_obs] - mem_cdd[mem_ndx][use_obs],
                mem_dbd[mem_ndx][use_obs] + mem_cdd[mem_ndx][use_obs],
            )
        except Exception as exc:
            mem_axs.set_ylim((-1.0, +1.0))
            logger.error(f"Exception while attempting to save axis y range: {exc}")
            logger.error(f"Problem file (axis range): {source.as_posix()}")
            # plt.close(fig)
            return

        mem_axs.grid(axis="both")
        mem_axs.axhline(0, ls="dotted", color="black")

        # Place label with receiver number at top of each column
        if algorithm == "down":
            mem_axs.set_ylabel(f"MEM {mem_num} dB$_\\mathbf{{D}}$ (nT)", weight="bold")
        else:
            mem_axs.set_ylabel(f"MEM {mem_num} d|B| (nT)", weight="bold")

        offnad_offset = 0.01
        fw, fh = fig.get_figwidth(), fig.get_figheight()
        mem_axs.text(
            offnad_offset,
            1.0 - offnad_offset * fw / fh,
            f"Off-Nadir Angle: {MEM_LOOK_DIRECTIONS[mem_ndx]}" r"º",
            weight="bold",
            size="medium",
            va="top",
            ha="left",
            transform=mem_axs.transAxes,
        )

        # Label only leftmost column y axis
        if col == 0 and row == 0:
            mem_axs.set_title(
                (
                    "Retrieved Current-Induced "
                    f"B$_\\mathbf{{{REFERENCE_ALTITUDE_KM}\\ km}}$"
                    " Field Vectors (nT)"
                ),
                weight="bold",
                size="medium",
            )

    # Adjust axis y limits to be uniform for a given field component across all MEMs
    for mem_ndx in range(NUM_MEM):
        mem_axs = axs[PLT_NDX[mem_ndx], col]
        tot_rng = b_rng[1] - b_rng[0]
        logger.debug(f"{b_rng[0]} : {b_rng[1]}")
        # FIXME - kludge to use old L2 file - comment out line below
        try:
            mem_axs.set_ylim((b_rng[0] - 0.05 * tot_rng, b_rng[1] + 0.05 * tot_rng))
        except Exception as exc:
            mem_axs.set_ylim((-1.0, +1.0))
            logger.error(f"Exception while attempting to set axes y limits: {exc}")
            logger.error(f"Problem file (data range): {source.as_posix()}")
            # plt.close(fig)
            return

    # Debugging bazillion-tick failures (when only a single non-NaN value is plotted?)
    # logger.info("Adding time (x axis) tick labels...")
    # logger.info(f"Start and stop times (UTC) are {time_utc[0]} - {time_utc[-1]}")
    # logger.info(f"Start+1 and stop-1 times (UTC) are {time_utc[1]} - {time_utc[-2]}")
    # logger.info(f"Length of time array is {len(time_utc)}")
    # if (time_utc[-1] - time_utc[0]).total_seconds() < 120:
    #     logger.info(f"{mem_dbd[0, :]}")
    #     logger.info(f"{np.sum(~np.isnan(mem_dbd[0, :]))}")

    # Format x-axis to show only hours and minutes unless sample is shorter than 120
    # seconds. Label only bottom row left time/x axis.
    set_xaxis_tick_format(
        axs[-1, 0],
        max_ticks=20 // ncols + 1,
        use_seconds=(time_utc[-1] - time_utc[0]).total_seconds() < 120,
    )

    # NEW geomagnetic coordinate map inset
    midpt = len(time_utc) // 2
    lat_at_midpt = nc_data["Geolocation/sat_lat"][midpt]
    lon_at_midpt = nc_data["Geolocation/sat_lon"][midpt]

    hemisphere = _select_hemisphere(lat_at_midpt)
    proj_method = _inset_map_projection(hemisphere, lat_at_midpt, lon_at_midpt)

    maprowspan, mapcolspan = 3, 1  # 5 rows for geographic coordinate inset
    map_axs: GeoAxes = plt.subplot2grid(
        (nrows, ncols),
        (0, ncols - mapcolspan),
        fig=fig,
        projection=proj_method,
        rowspan=maprowspan,
        colspan=mapcolspan,
    )  # ty:ignore[invalid-assignment]

    if hemisphere is not None:
        clip_to_circle(map_axs=map_axs, proj_method=proj_method, hemisphere=hemisphere)

    # Map S/C and MEM footprint lat/lon
    dtlim = [time_utc[0], time_utc[1]]
    duration = (dtlim[1] - dtlim[0]).total_seconds()
    midpoint = dtlim[0] + datetime.timedelta(seconds=int(0.5 * duration))

    # Get solar position, first in geodetic and then in APEX magnetic coordinates.
    sun_geo_lon_deg, sun_geo_lat_deg = _subsolar_geodetic(time_utc)

    # Compute the subsolar point in APEX magnetic coordinates
    apex = Apex(date=midpoint.year, refh=0)
    _sun_mlat, sun_mlon = apex.geo2apex(
        sun_geo_lat_deg,
        sun_geo_lon_deg,
        REFERENCE_ALTITUDE_KM,
    )

    # Compute satellite footprint in APEX magnetic coordinates
    sat_maglat, sat_maglon = apex.geo2apex(
        nc_data["Geolocation/sat_lat"][:],
        nc_data["Geolocation/sat_lon"][:],
        nc_data["Geolocation/sat_alt"][:],
    )

    # Convert magnetic longitude to MLT
    sat_magLT_deg = 180.0 + sat_maglon - sun_mlon

    # Debugging "mirrored" southern hemisphere plots
    # logger.debug(f"{hemisphere} {lat_at_midpt}")
    # logger.debug(f"{np.min(sun_geo_lat_deg)}, {np.max(sun_geo_lat_deg)}")
    # logger.debug(f"{np.min(sun_geo_lon_deg)}, {np.max(sun_geo_lon_deg)}")
    # logger.debug(
    #     f"Sun magnetic latitude, longitude = "
    #     f"{sun_mlat[tndx // 2]:7.2f}, "
    #     f"{sun_mlon[tndx // 2]:7.2f}"
    # )
    # logger.debug(f"{np.min(sat_magLT_deg)}, {np.max(sat_magLT_deg)}")

    # Transform continent outlines from geographic to APEX magnetic coordinates
    MLT_sign = -1 if (south_inverted and hemisphere == SOUTH) else +1
    if hemisphere is not None:
        map_magnetic_continents(
            ax=map_axs,
            time4mag=midpoint,
            sun_mlon=sun_mlon[len(sun_mlon) // 2],
            alt4mag=REFERENCE_ALTITUDE_KM,
            south_inverted=(south_inverted and (hemisphere == SOUTH)),
            color="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
        )
        map_sc_mem_footprints_magnetic(
            map_axs=map_axs,
            sat_lat=sat_maglat,
            sat_lon=MLT_sign * sat_magLT_deg,
            obs_lat=mag_lat[:, :].T,
            obs_lon=MLT_sign * mag_ltm[:, :].T * 15,  # Convert hours to degrees
            at_time=midpoint,
            dark_mode=dark_mode,
            zorder=4,
            small_text=False,
            plain=False,
            legend_top=False,
            terminator=False,
            hemisphere=hemisphere,
            south_inverted=(south_inverted and (hemisphere == SOUTH)),
        )

    else:
        map_sc_mem_footprints(
            map_axs=map_axs,
            sat_lat=nc_data["Geolocation/sat_lat"][use_obs],
            sat_lon=nc_data["Geolocation/sat_lon"][use_obs],
            obs_lat=obs_lat[:, use_obs].T,
            obs_lon=obs_lon[:, use_obs].T,
            at_time=midpoint,
            dark_mode=dark_mode,
            terminator=True,
            zorder=4,
            small_text=False,
            plain=False,
            legend_top=False,
            mid_lat_mag=True,
        )

    # Add MEM beam numbering/pointing diagram at bottom right corner of figure
    _add_mem_beam_diagram(fig, hi=0.29, center_x=0.76, bottom_frac=0.025)

    # Tweak position and add any figure-level annotation
    fig_title = f"{mode} Retrieved Magnetic Field Deltas: "
    fig_title = fig_title + (
        f"{sc_id} - Orbit {orb_num}\n"
        f"{time_utc[use_obs][0].strftime('%Y-%m-%d (%j) %H:%M:%S')} - "
        f"{time_utc[use_obs][-1].strftime('%Y-%m-%d (%j) %H:%M:%S')}"
    )
    _finalize_figure(
        fig=fig,
        nc_data=nc_data,
        source=source,
        title=fig_title,
        dark_mode=dark_mode,
    )
    plt.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.07,
        top=0.91,
        hspace=0.00,
        wspace=0.05,
    )
    save_close_figure(
        source=source,
        save_directory=save_directory,
        figure=fig,
        obs_date=obs_date,
        spacecraft=sc_id,
        tstmp=t_stamp,
        plot_type=plot_type,
        dpi=figure_dpi,
        dark_mode=dark_mode,
    )


def map_sc_mem_footprints(
    map_axs,
    sat_lat: np.ndarray,
    sat_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_lon: np.ndarray,
    at_time: datetime.datetime,
    polar: bool = False,
    terminator: bool = False,
    dark_mode: bool = False,
    zorder: int = 4,
    small_text: bool = False,
    add_title: bool = True,
    legend_top: bool = False,
    mid_lat_mag: bool = False,
    plain: bool = False,
    south_inverted: bool = False,
):
    """
    Map geolocated S/C and MEM footprints in geographic coordinates.

    Overlays the spacecraft track and each MEM's observation footprint on the
    supplied cartopy axes, with gridlines, coastlines, optional day/night
    terminator, and a legend.

    Args:
        map_axs (plt.Axes): Cartopy axes on which to draw the footprints.
        sat_lat (np.ndarray): Spacecraft geodetic latitude per observation.
        sat_lon (np.ndarray): Spacecraft geodetic longitude per observation.
        obs_lat (np.ndarray): MEM footprint geodetic latitude, 4 MEMs (rows) x
            observations (cols).
        obs_lon (np.ndarray): MEM footprint geodetic longitude, 4 MEMs (rows)
            x observations (cols).
        at_time (datetime.datetime): Time used for the terminator and any
            time-dependent overlays.
        polar (bool, optional): Use a polar-style single-axis layout.
            Defaults to False.
        terminator (bool, optional): Show day/night with Nightshade.
            Defaults to False.
        dark_mode (bool, optional): Use 'dark_background' style.
            Defaults to False.
        zorder (int, optional): Base draw order for the footprint artists.
            Defaults to 4.
        small_text (bool, optional): Use smaller axis text. Defaults to False.
        add_title (bool, optional): Add an axes title. Defaults to True.
        legend_top (bool, optional): Place the legend at the top.
            Defaults to False.
        mid_lat_mag (bool, optional): Label latitude ticks in magnetic
            latitude. Defaults to False.
        plain (bool, optional): Draw footprints only, without grids, labels,
            or legend. Defaults to False.
        south_inverted (bool, optional): Invert the southern hemisphere
            (heliospheric community mapping style). Defaults to False.

    Returns:
        None
    """
    map_axs.set_global()
    map_axs.add_feature(
        cfeature.OCEAN,
        alpha=1.0 if dark_mode else 0.3,
        facecolor="#305080" if dark_mode else "#60A0F0",
    )
    map_axs.add_feature(
        cfeature.LAND,
        alpha=0.7 if dark_mode else 0.3,
        facecolor="#d2c2a2" if dark_mode else "#d0c0a0",
    )

    gl = map_axs.gridlines(
        draw_labels=True,
        x_inline=True,  # FIXME
        xlocs=range(-180, 180, 30),
        ylocs=range(-90, 91, 15),
        color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
    )
    gl.xlabel_style = {
        "size": "xx-small" if small_text else "x-small",
        "color": "black" if not dark_mode else DARK_MODE_TEXT_COLOR,
    }
    gl.ylabel_style = {
        "size": "xx-small" if small_text else "x-small",
        "color": "black" if not dark_mode else DARK_MODE_TEXT_COLOR,
    }
    if mid_lat_mag:
        plot_geomagnetic_references(
            ax=map_axs,
            at_time=at_time,
            latitudes=[
                -30,
                -20,
                -10,
                +10,
                +20,
                +30,
            ],
            lat_clr="#a03030",
        )
        plot_geomagnetic_references(
            ax=map_axs,
            at_time=at_time,
            latitudes=[0],
            ls="-",
            lat_clr="#a03030",
        )
    else:
        plot_geomagnetic_references(
            ax=map_axs,
            at_time=at_time,
            latitudes=[
                -80,
                -70,
                -60,
                -50,
                0,
                +50,
                +60,
                +70,
                +80,
            ],
            lat_clr="#a03030",
        )

    if terminator:
        # Take difference from noon or midnight and flip sign to make this work when
        # plotting with south_inverted set
        noon = at_time.replace(hour=12, minute=0, second=0)
        noon_delt = noon - at_time
        at_time_inverted = noon + noon_delt
        map_axs.add_feature(
            Nightshade(
                at_time if not south_inverted else at_time_inverted,
                alpha=TERMINATOR_ALPHA,
                color=TERMINATOR_COLOR,
                zorder=2,
            )
        )

    # FIXME: We might need to adjust these algorithmicallly based on the time spacing of
    # the L0A points
    # mew, mkevry = 0.6, 2  # for "sparse" files
    mew, mkevry = 0.9, 10  # for "dense" files
    map_kws = {
        "transform": ccrs.PlateCarree(),
        "ls": "none",
        "ms": 4,
        "mfc": "none",
        "mew": mew,
        "markevery": mkevry,
    }  # Same for S/C and all MEMs

    # MEM lat/lon from L0A file
    if not plain:
        for mm, mem in enumerate(MEM_NUMBERS):
            mem_kws = {
                "marker": MEM_SYMS[mm],
                "color": MEM_CLR[mm],  # MEM_CB_CLR[mm],
                "label": f"MEM {mem}",
                "zorder": zorder + 1,  # Always place ABOVE spacecraft footprint
            }  # MEM-specific
            map_axs.plot(obs_lon[:, mm], obs_lat[:, mm], **(map_kws | mem_kws))

    # S/C lat/lon from L0A file
    sat_kws = {
        "marker": "d",
        "ms": 2,
        "color": SAT_COL if dark_mode else "black",
        "label": "SV Ground Track",
        # "label": "L0A - EARTHLOOK or SKYLOOK",
        "zorder": zorder,
    }  # S/C-specific
    map_axs.plot(sat_lon, sat_lat, **(map_kws | sat_kws))

    # Add map annotation
    ncols = 3  # if not polar else 1
    fontsize = "xx-small" if small_text else "x-small"
    if not plain:
        if legend_top:
            # loc = "upper right"  # if not polar else "lower left"
            # bbta = (1.04, -0.05)  # if not polar else (1.05, 0.05)
            loc = "lower right"
            bbta = (0.50, +1.02)
        else:
            loc = "upper right"
            bbta = (0.50, -0.05)
        map_axs.legend(
            loc=loc, ncols=ncols, fontsize=fontsize, markerscale=2, bbox_to_anchor=bbta
        )
    if add_title:
        map_axs.set_title(
            "Geolocated Spacecraft and MEM Footprints",
            size="small" if small_text else "medium",
            pad=12,
        )


def map_sc_mem_footprints_magnetic(
    map_axs,
    sat_lat: np.ndarray,
    sat_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_lon: np.ndarray,
    at_time: datetime.datetime,
    hemisphere: str | None = None,
    dark_mode: bool = False,
    zorder: int = 4,
    small_text: bool = False,
    legend_top: bool = False,
    geo_labels: bool = False,
    geo_offset: float = 0.0,
    terminator: bool = False,
    plain: bool = False,
    south_inverted: bool = False,
):
    """
    Map geolocated S/C and MEM footprints in geomagnetic (APEX) coordinates.

    Overlays the spacecraft track and each MEM's observation footprint on the
    supplied cartopy axes in magnetic lat/MLT coordinates, with MLT and
    magnetic-latitude gridlines, magnetic continent overlays, optional
    day/night terminator, and a legend.

    Args:
        map_axs (plt.Axes): Cartopy axes on which to draw the footprints.
        sat_lat (np.ndarray): Spacecraft magnetic latitude per observation.
        sat_lon (np.ndarray): Spacecraft MLT (in degrees) per observation.
        obs_lat (np.ndarray): MEM footprint magnetic latitude, 4 MEMs (rows)
            x observations (cols).
        obs_lon (np.ndarray): MEM footprint MLT (in degrees), 4 MEMs (rows)
            x observations (cols).
        at_time (datetime.datetime): Time used for the magnetic model and
            terminator.
        hemisphere (str | None, optional): Hemisphere to plot (e.g., NORTH,
            SOUTH), or None. Defaults to None.
        dark_mode (bool, optional): Use 'dark_background' style.
            Defaults to False.
        zorder (int, optional): Base draw order for the footprint artists.
            Defaults to 4.
        small_text (bool, optional): Use smaller axis text. Defaults to False.
        legend_top (bool, optional): Place the legend at the top.
            Defaults to False.
        geo_labels (bool, optional): Show geographic coordinate labels in
            addition to the magnetic ones. Defaults to False.
        geo_offset (float, optional): Angular offset applied to the
            geographic label positions. Defaults to 0.0.
        terminator (bool, optional): Show day/night with Nightshade.
            Defaults to False.
        plain (bool, optional): Draw footprints only, without grids, labels,
            or legend. Defaults to False.
        south_inverted (bool, optional): Invert the southern hemisphere MLT
            axis (heliospheric community mapping style). Defaults to False.

    Returns:
        None
    """
    if (hemisphere is None) or (hemisphere == NORTH) or south_inverted:
        ccw = +1
    else:
        ccw = -1
    lats_n = np.arange(
        MAG_LAT_LOWER_LIMIT, 90.0, 10.0
    )  # Lats at which to draw gridlines

    if south_inverted:
        cardinal_labels = {"south": "N", "north": "S"}
    else:
        cardinal_labels = {"south": "S", "north": "N"}

    lon_delta = 30.0
    gl = map_axs.gridlines(
        draw_labels=True,
        x_inline=False,
        xlocs=np.arange(-180.0, 180.0, lon_delta),
        y_inline=True,
        ylocs=lats_n if hemisphere == NORTH else [-1 * lat for lat in lats_n],
        color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
    )
    if False:  # Not supported for non-rectangular projections =(
        lat_formatter = LatitudeFormatter(cardinal_labels=cardinal_labels)
        map_axs.yaxis.set_major_formatter(lat_formatter)

    # gl.ylabel_style = {
    #     "size": 1,
    #     "zorder": 0,
    #     "color": "white" if not dark_mode else DARK_MODE_FACE_COLOR,
    # }  # Suppress latitude labels

    if not geo_labels:
        # Add labels in MLT to the polar stereographic plot, with noon at top
        gl.xlabel_style = {
            "size": 1,
            "zorder": 0,
            "color": "white" if not dark_mode else DARK_MODE_FACE_COLOR,
        }  # Suppress longitudes, we'll add MLT ourselves
        mlt_locs = np.arange(0, 360.0, lon_delta)
        mlt_desc = {f"{mlt:.0f}": _mlt_label(mlt) for mlt in mlt_locs}

        for mm, mlt in enumerate(mlt_locs):
            mlt_pos = (ccw * mlt - 90.0) % 360.0  # Rotate to coord sys with 0 at x axis
            x = 0.5 + 0.57 * np.cos(np.radians(mlt_pos))
            y = 0.5 + 0.54 * np.sin(np.radians(mlt_pos))
            map_axs.text(
                x,
                y,
                mlt_desc[f"{mlt:.0f}"],
                transform=map_axs.transAxes,
                color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
                va="center",
                horizontalalignment="center",
            )

    elif geo_labels and not (south_inverted and (hemisphere == SOUTH)):
        gl.xlabel_style = {"size": "small", "zorder": 5}
    else:
        gl.xlabel_style = {
            "size": 1,
            "zorder": 0,
            "color": "white" if not dark_mode else DARK_MODE_FACE_COLOR,
        }  # Suppress longitudes, we'll add MLT ourselves
        mlt_locs = np.arange(-180, 180.0, lon_delta)
        mlt_desc = {}
        for mm, mlt in enumerate(mlt_locs):
            mlt_desc[f"{mlt:.0f}"] = f"{mlt:.0f}" + r"$^\circ$"
        for mm, mlt in enumerate(mlt_locs):
            # Rotate positions to align with coord sys central longitude definition
            mlt_pos = (ccw * mlt + 90.0 + geo_offset) % 360.0
            x = 0.5 + 0.55 * np.cos(np.radians(mlt_pos))
            y = 0.5 + 0.55 * np.sin(np.radians(mlt_pos))
            map_axs.text(
                x,
                y,
                mlt_desc[f"{mlt:.0f}"],
                transform=map_axs.transAxes,
                color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
                va="center",
                horizontalalignment="center",
            )

    # FIXME: We might need to adjust these algorithmicallly based on the time spacing of
    # the L0A points
    # mew, mkevry = 0.6, 2  # for "sparse" files
    mew, mkevry = 0.9, 10  # for "dense" files
    map_kws = {
        "transform": ccrs.PlateCarree(),
        "ls": "none",
        "ms": 4,
        "mfc": "none",
        "mew": mew,
        "markevery": mkevry,
    }  # Same for S/C and all MEMs

    # MEM lat/lon from L0A file
    if not plain:
        for mm, mem in enumerate(MEM_NUMBERS):
            mem_kws = {
                "marker": MEM_SYMS[mm],
                "color": MEM_CLR[mm],  # MEM_CB_CLR[mm],
                "label": f"MEM {mem}",
                "zorder": zorder + 1,  # Always place ABOVE spacecraft footprint
            }  # MEM-specific
            map_axs.plot(obs_lon[:, mm], obs_lat[:, mm], **(map_kws | mem_kws))

    # S/C lat/lon from L0A file
    sat_kws = {
        "marker": "d",
        "ms": 2,
        "color": SAT_COL if dark_mode else "black",
        "label": "SV Ground Track",
        # "label": "L0A - EARTHLOOK or SKYLOOK",
        "zorder": zorder,
    }  # S/C-specific
    map_axs.plot(sat_lon, sat_lat, **(map_kws | sat_kws))

    if terminator:
        map_axs.add_feature(
            Nightshade(
                at_time,
                alpha=TERMINATOR_ALPHA,
                color=TERMINATOR_COLOR,
                zorder=2,
            )
        )

    # Add map annotation
    ncols = 3  # if not polar else 1
    fontsize = "xx-small" if small_text else "x-small"
    if not plain:
        if legend_top:
            # loc = "upper right"  # if not polar else "lower left"
            # bbta = (1.04, -0.05)  # if not polar else (1.05, 0.05)
            loc = "lower right"
            bbta = (0.50, +1.02)
        else:
            loc = "upper right"
            bbta = (0.50, -0.06)
        map_axs.legend(
            loc=loc, ncols=ncols, fontsize=fontsize, markerscale=2, bbox_to_anchor=bbta
        )


def l3_map(
    l3_map_axs: GeoAxes,
    l3_data: L3PassData,
    lats_m: np.ndarray,
    fig: plt.figure,
    hemisphere: str,
    MLT_sign: int,
    sun_geo_lon_mid: float,
    map_proj: ccrs.Projection,
    south_inverted: bool = False,
    dark_mode: bool = False,
    show_b_value: bool = False,
    stretch: float = 0.06,
):
    """Plot a map of modeled and observed dB and current values on the supplied axis.

    Args:
        l3_map_axs (GeoAxes): Cartopy axes on which to draw the L3 map.
        l3_data (L3PassData): Arrays extracted from the L3 product (and its
            embedded L2 group) for this pass.
        lats_m (np.ndarray): Magnetic latitudes at which to overlay parallels.
        fig (plt.figure): Parent figure (used for the color bar).
        hemisphere (str): NORTH or SOUTH (or None for an EEJ pass).
        MLT_sign (int): MLT sign multiplier (+1, or -1 for south-inverted).
        sun_geo_lon_mid (float): Subsolar geodetic longitude (degrees) at the
            pass midpoint.
        map_proj (ccrs.Projection): Cartopy projection of the map axes.
        south_inverted (bool, optional): Invert the southern hemisphere MLT
            axis (heliospheric community mapping style). Defaults to False.
        dark_mode (bool, optional): Generate plots using 'dark mode' format.
            Defaults to False.
        show_b_value (bool, optional): Color MEM footprints on the L3 map subplot
            according to the observed value of B; otherwise just overlay black markers.
            Defaults to False.
        stretch (float, optional): Adjusts location of MLT labels. Defaults to 0.06.

    Returns:
        None
    """
    # TODO: Test this on EEJ passes. It's probably missing something.

    # Instantiate figure and axes for plotting - tweak some rcparams as needed
    _apply_dark_mode(dark_mode)

    # Unpack the pass arrays for use throughout this method.
    time_utc = l3_data.time_utc
    lons = l3_data.lons
    lon_mesh = l3_data.lon_mesh
    lats = l3_data.lats
    lat_mesh = l3_data.lat_mesh
    Jn = l3_data.Jn
    Je = l3_data.Je
    model_B = l3_data.model_B
    obs_B = l3_data.obs_B
    midpt = l3_data.midpt

    # Set extent
    swath_wid, swath_len = 3250000, 3250000  # meters, keep it square
    lonmid, latmid = MLT_sign * lons[0, midpt], lats[0, midpt]

    # Shift center of projection slightly toward geographic pole so that LT labels do
    # not get scrunched together when the pole moves too close to (or beyond) the upper
    # boundary.
    if hemisphere == NORTH:
        ext_shift = 0.3 * (90 - latmid)
        b_sign_mod = +1
    elif hemisphere == SOUTH:
        ext_shift = 0.3 * (-90 - latmid)
        b_sign_mod = -1
    else:  # EEJ pass
        ext_shift = 0.0
        b_sign_mod = +1

    x0_m, y0_m = map_proj.transform_point(
        lonmid, latmid + ext_shift, src_crs=DATA_TRANSFORM
    )
    logger.debug(
        f"{x0_m - swath_wid} {x0_m + swath_wid} {y0_m - swath_len} {y0_m + swath_len}"
    )
    l3_map_axs.set_extent(
        (x0_m - swath_wid, x0_m + swath_wid, y0_m - swath_len, y0_m + swath_len),
        crs=map_proj,
    )

    if (hemisphere == SOUTH) and south_inverted:
        logger.info("Mapping continents in reversed longitude coordinates")
        map_inverted_continents(
            ax=l3_map_axs,
            linewidth=0.5,
            color="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
        )
    else:
        logger.info("Mapping continents in standard longitude coordinates")
        l3_map_axs.coastlines(
            resolution="110m",
            linewidth=0.5,
            color="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
        )

    if hemisphere is not None:
        # Add LT grid markings - not yet "true" MLT, just LT
        _mag_lat_artist = plot_geomagnetic_references(
            l3_map_axs,
            time_utc[midpt],
            latitudes=lats_m,
            south_inverted=(hemisphere == SOUTH) and south_inverted,
            lat_clr="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
        )  # Might want artist for legend later?

        # Add MLT solar position notation in hours, with exceptions for Noon, etc. 'ccw'
        # is NOT the same as MLT_sign!
        if (hemisphere is None) or (hemisphere == NORTH) or south_inverted:
            ccw = +1
        else:
            ccw = -1  # SOUTH and not south_inverted case only

        lon_delta = 30.0
        mlt_angs = np.arange(0, 360.0, lon_delta)
        mlt_desc = {f"{mlt:.0f}": _mlt_label(mlt) for mlt in mlt_angs}

        xp_m, yp_m = l3_map_axs.transAxes.inverted().transform(
            l3_map_axs.transData.transform(
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
            x, y = boundary_distance(xp_m, yp_m, plt_ang, stretch=stretch)
            l3_map_axs.text(
                x,
                y + 0.0,
                mlt_desc[f"{mlt:.0f}"],
                color="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
                va="center",
                ha="center",
                transform=l3_map_axs.transAxes,
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
        _gls = l3_map_axs.gridlines(
            draw_labels=False,
            xlocs=np.sort(MLT_axes),
            ylocs=lats_n if hemisphere == NORTH else [-1 * lat for lat in lats_n],
            color="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
            crs=ccrs.PlateCarree(),
        )
        _gls = l3_map_axs.gridlines(
            draw_labels=True,
            xlocs=[],
            y_inline=True,
            ylocs=lats_n if hemisphere == NORTH else [-1 * lat for lat in lats_n],
            color="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
            crs=ccrs.PlateCarree(),
        )

    else:  # EEJ
        l3_map_axs.gridlines(
            draw_labels=True,
            x_inline=False,
            y_inline=False,
            rotate_labels=False,
            color="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
            crs=ccrs.PlateCarree(),
        )
        plot_geomagnetic_references(
            l3_map_axs,
            time_utc[midpt],
            latitudes=lats_m,
            south_inverted=(hemisphere == SOUTH) and south_inverted,
            lat_clr="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
        )
        plot_geomagnetic_references(
            l3_map_axs,
            time_utc[midpt],
            latitudes=[0],
            south_inverted=(hemisphere == SOUTH) and south_inverted,
            lat_clr="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
            ls="solid",
            lw=2,
        )  # Magnetic drift equator

    # Now we can finally draw the J and B values! Define some plotting parameters.
    m_stride = 5  # Overlay current arrows (quiver plot) every m_stride positions
    J = np.sqrt(Je**2 + Jn**2)
    Jmax = 10 * np.nanmax(J)  # scales the LENGTH of the quiver arrows
    qwid = 0.001  # scales the WIDTH of the quiver arrow(head?)s
    b_mrk_sz = 20  # Marker size used for observed and modeled B scatter plots

    # Shared B magnitude color scales, choose one.
    # Relative:
    # vmin = min(np.nanmin(obs_B), np.nanmin(model_B))
    # vmax = max(np.nanmin(obs_B), np.nanmin(model_B))
    # vmin = -1 * min(abs(vmin), abs(vmax)) # make it symmetric about zero
    # vmax = -1 * vmin
    # Fixed:
    vmin = -1200
    vmax = +1200

    if dark_mode:
        color_min = "#4203ff"
        color_center = DARK_MODE_FILL_COLOR
        color_max = "#ff0342"
        cmap = LinearSegmentedColormap.from_list(
            "cmap_name", [color_min, color_center, color_max]
        )
    else:
        # cmap = plt.cm.bwr
        cmap = plt.cm.coolwarm

    # Model B
    valid = model_B != NCDF_MISSING

    vmin = -1000
    vmax = +1000

    # sc = l3_map_axs.scatter(
    #     MLT_sign * lon_mesh[valid].flatten(),
    #     lat_mesh[valid].flatten(),
    #     c=b_sign_mod * model_B[valid].flatten(),
    #     cmap=cmap,
    #     vmin=vmin,
    #     vmax=vmax,
    #     s=b_mrk_sz,
    #     transform=DATA_TRANSFORM,
    # )

    def smooth_along_track(values, window=5):
        kernel = np.ones(window, dtype=float)
        smoothed = np.empty_like(values, dtype=float)

        if values.shape[1] <= kernel.shape[0]:
            return values

        for i in range(values.shape[0]):
            row = np.asarray(values[i], dtype=float)
            valid = np.isfinite(row).astype(float)
            filled = np.where(np.isfinite(row), row, 0.0)

            numer = np.convolve(filled, kernel, mode="same")
            denom = np.convolve(valid, kernel, mode="same")

            smoothed[i] = np.where(denom > 0, numer / denom, np.nan)

        return smoothed

    obs_bmag_smooth = smooth_along_track(model_B, window=5)

    sc = l3_map_axs.pcolormesh(
        lon_mesh,
        lat_mesh,
        b_sign_mod * obs_bmag_smooth,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        transform=DATA_TRANSFORM,
    )

    # Retrieved J
    l3_map_axs.quiver(
        MLT_sign * lon_mesh[::m_stride, ::m_stride].flatten(),
        lat_mesh[::m_stride, ::m_stride].flatten(),
        MLT_sign * Je[::m_stride, ::m_stride].flatten(),
        Jn[::m_stride, ::m_stride].flatten(),
        scale=Jmax,
        width=qwid,
        color="xkcd:black" if not dark_mode else "orange",
        transform=DATA_TRANSFORM,
    )

    # Observed (L2) B
    valid = obs_B != NCDF_MISSING

    if show_b_value:
        _sc = l3_map_axs.scatter(
            x=MLT_sign * lons[valid],
            y=lats[valid],
            s=b_mrk_sz / 2,
            c=obs_B[valid],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker="o",
            edgecolors="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
            linewidths=0.1,
            transform=DATA_TRANSFORM,
        )
    else:
        _sc = l3_map_axs.scatter(
            x=MLT_sign * lons[valid],
            y=lats[valid],
            s=b_mrk_sz / 4,
            c="none",
            edgecolors="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
            linewidths=0.5,
            marker=".",
            transform=DATA_TRANSFORM,
        )
    # Add plot annotation and color bar, tweak margins and spacing.
    divider = make_axes_locatable(l3_map_axs)
    cax = divider.append_axes("right", size="3%", pad=0.7, axes_class=maxes.Axes)
    cbar = fig.colorbar(sc, cax=cax)
    # cbar.set_label(f"B$_{{{algorithm}}}$ [nT]")
    cbar.set_label("dB$_{{down}}$ [nT]")


def plot_b_1D_maps_with_time(
    nc_data: Dataset,
    source: Path,
    save_directory: Path,
    south_inverted: bool = False,
    git_branch: str = "Unavailable",
    git_commit: str = "Unavailable",
    figure_dpi: int = 200,
    dark_mode: bool = False,
    old_format: bool = False,
    overwrite: bool = False,
    show_b_value: bool = True,
):
    """
    Plot the retrieved 1-D B field maps versus time for L3 products.

    Generates the AEJ L3 B-up data output plots from B-down L2 inputs (based
    on the plot_btot_maps_with_time approach). All input arrays are expected to
    be 1-D.

    Args:
        nc_data (Dataset): Open netCDF4 Dataset of the L3 file used as input.
        source (Path): Path of the L3 source file, used for plot naming and
            metadata.
        save_directory (Path): Root directory in which to save the generated
            plot files.
        south_inverted (bool, optional): Invert the southern hemisphere MLT
            axis (heliospheric community mapping style). Defaults to False.
        git_branch (str, optional): Pipeline branch to display (used because
            L3 products do not yet carry pipeline metadata).
            Defaults to "Unavailable".
        git_commit (str, optional): Pipeline commit hash to display (used
            because L3 products do not yet carry pipeline metadata).
            Defaults to "Unavailable".
        figure_dpi (int, optional): Resolution (DPI) at which to generate the
            plots. Defaults to 200.
        dark_mode (bool, optional): Generate plots using 'dark mode' format.
            Defaults to False.
        old_format (bool, optional): Use the legacy plot file naming scheme.
            Defaults to False.
        overwrite (bool, optional): Overwrite existing plot files; otherwise
            skip. Defaults to False.

    Returns:
        None
    """
    start = time.perf_counter()
    _apply_dark_mode(dark_mode)
    prsd = parse_ezie_product_name(source=source)
    plot_type: str = "l3-l2"
    obs_date = datetime.datetime.strptime(prsd.date, EZIE_DATE_FORMAT).replace(
        tzinfo=datetime.UTC
    )

    # See if this figure already exists.
    # Do NOT remake an existing figure unless the overwrite flag is set.
    ftgt = _plot_needs_update(
        source=source,
        save_directory=save_directory,
        obs_date=obs_date,
        plot_type=plot_type,
        type_desc="L3 retrieved B_d and current",
        spacecraft=prsd.spcv,
        tstmp=prsd.time,
        dpi=figure_dpi,
        dark_mode=dark_mode,
        old_format=old_format,
        overwrite=overwrite,
    )
    if ftgt is None:
        return

    try:
        l3_data = _load_l3_pass(nc_data)
    except TypeError:
        raise
    except Exception as exc:
        logger.error(f"Encountered exception while processing L3 file: {exc}")
        return

    time_utc = l3_data.time_utc
    midpt = l3_data.midpt
    lat_at_midpt = l3_data.lat_at_midpt
    lon_at_midpt = l3_data.lon_at_midpt

    # Add magnetic coordinate overlay, select latitudes at which to draw magnetic
    # gridlines, play with the map projection as needed.
    hemisphere = _select_hemisphere(lat_at_midpt)
    if hemisphere is None:
        logger.warning(f"L3 file appears to contain EEJ pass, skipping: {source.name}")
        return

    if (
        np.all(l3_data.Jn == 0.0)
        and np.all(l3_data.Je == 0.0)
        and np.all(l3_data.obs_B == 0.0)
    ):
        logger.warning(
            f"L3 file Je, Jn, and observed B values are all zero, skipping: {ftgt.stem}"
        )
        return

    # Get solar position, first in geodetic and then in APEX magnetic coordinates. We'll
    # display projected geodetic coordinates for now, pending addition of magnetic
    # coordinate counterparts to the L3 files.

    # Compute position of sun in both geodetic and geomagnetic coordinates for the
    # observation midpoint time.
    sun_geo_lon_mid, sun_geo_lat_mid = _subsolar_geodetic_at(time_utc[midpt])

    # Compute the subsolar point in APEX magnetic coordinates
    apex = Apex(
        date=time_utc[midpt].year,
        refh=0,  # Leave at default (0), or set to 80 km? TBD
    )
    _sun_mag_lat_mid, _sun_mag_lon_mid = apex.geo2apex(
        sun_geo_lat_mid,
        sun_geo_lon_mid,
        REFERENCE_ALTITUDE_KM,
    )

    lats_m, MLT_sign, map_proj = _make_l3_projection(
        hemisphere,
        sun_geo_lon_mid,
        lat_at_midpt,
        lon_at_midpt,
        south_inverted,
    )

    # Set up figure and projection for cartopy
    fig = plt.figure(num=prsd.time, figsize=(9.6, 8.0))
    l3_map_axs: GeoAxes = fig.add_subplot(1, 1, 1, projection=map_proj)

    l3_map(
        l3_map_axs=l3_map_axs,
        l3_data=l3_data,
        lats_m=lats_m,
        fig=fig,
        hemisphere=hemisphere,
        MLT_sign=MLT_sign,
        sun_geo_lon_mid=sun_geo_lon_mid,
        map_proj=map_proj,
        south_inverted=False,
        dark_mode=dark_mode,
        stretch=0.035,
        show_b_value=show_b_value,
    )

    fig_title = (
        f"L3 and L2: EZIE-{prsd.spcv.upper()} \n"
        f"{time_utc[0].strftime('%B %d, %Y %H:%M:%S')} - "
        f"{time_utc[-1].strftime('%H:%M:%S')}\n"
    )
    _finalize_figure(
        fig=fig,
        nc_data=nc_data,
        source=source,
        title=fig_title,
        git_branch=git_branch,
        git_commit=git_commit,
        dark_mode=dark_mode,
        size="x-small",
    )
    plt.tight_layout()  # pad=0)
    plt.subplots_adjust(top=0.860, bottom=0.060)  # Modify tight_layout results slightly

    if hemisphere is not None:
        qvclr = "orange" if dark_mode else "black"
        subtitle = (
            f"Currents ({qvclr} arrows) are plotted in Geodetic/WGS84 coordinates ("
            "solid grid).\nLabels indicate Local Time (LT). Magnetic latitude (dashed "
            "grid) is also shown."
        )
    else:
        subtitle = f"""
           Currents ({qvclr} arrows) are plotted in Geodetic/WGS84 coordinates (solid
           grid). Magnetic latitude (dashed grid) and drift equator (thick solid line)
           are also shown.
           """
    fig.text(
        0.50,
        0.90,
        subtitle,
        wrap=False,
        ha="center",
        va="bottom",
        fontsize="small",
        fontweight="bold",
    )

    save_close_figure(
        source=source,
        save_directory=save_directory,
        figure=fig,
        obs_date=obs_date,
        spacecraft=prsd.spcv,
        tstmp=prsd.time,
        plot_type=plot_type,
        dpi=figure_dpi,
        dark_mode=dark_mode,
    )
    end = time.perf_counter()
    logger.info(f"Elapsed time: {end - start:.2f} seconds.")


def plot_retrieved_B_and_J(
    nc2_data: Dataset,
    nc2_sorc: Path,
    save_directory: Path,
    version: str | None = None,
    figure_dpi: int = DFLT_RES,
    dark_mode: bool = False,
    south_inverted: bool = False,
    overwrite: bool = False,
    show_b_value: bool = False,
):
    """
    Plot the retrieved B field (L2) together with the derived current (J)
    vectors (L3) on a combined map.

    The L3 source file is derived from the L2 source by substituting the "l2"
    product token for "l3" in the file name. Shows the model and observed B
    fields and the retrieved currents, with MEM footprints, magnetic reference
    grids, and MLT labels.

    Args:
        nc2_data (Dataset): Open netCDF4 Dataset of the L2 product file to
            plot.
        nc2_sorc (Path): Path of the L2 source product file, used for plot
            naming, metadata, and deriving the L3 source path.
        save_directory (Path): Root directory in which to save the generated
            plot files.
        version (str | None, optional): Product version/revision string
            included in the plot file name. Defaults to None.
        figure_dpi (int, optional): Resolution (DPI) at which to generate the
            plots. Defaults to DFLT_RES.
        dark_mode (bool, optional): Generate plots using 'dark mode' format.
            Defaults to False.
        south_inverted (bool, optional): Invert the southern hemisphere MLT
            axis (heliospheric community mapping style). Defaults to False.
        overwrite (bool, optional): Overwrite existing plot files; otherwise
            skip. Defaults to False.
        show_b_value (bool, optional): Color MEM footprints on the L3 map subplot
            according to the observed value of B; otherwise just overlay black markers.
            Defaults to False.

    Raises:
        TypeError: If the L3 file's time field does not contain
            datetime.datetime or np.datetime64 objects.

    Returns:
        None
    """
    # Deprecated option, will be removed
    # mode (str, optional): Plot mode label, "Corrected" or "Uncorrected".
    #     Defaults to "Uncorrected".

    nc3_sorc: Path = _l3_source_from_l2(nc2_sorc)
    if not nc3_sorc.exists():
        logger.error(
            f"L3 corresponding to supplied L2 absent, skipping: {nc3_sorc.as_posix()}"
        )
        return

    _apply_dark_mode(dark_mode)

    # Collate netcdf data for all spacecraft and orbits on the selected day and orbit.

    # Get time data (strings) and convert to datetime objects.
    # L2 files are already broken up into discrete science passes.
    # if len(nc_data["Time/time_utc"]) == 0:
    try:
        time_utc, obs_date = get_datetime_from_utc_string(nc2_data.groups["Time"])
    except Exception as exc:
        logger.error(f"Exception encountered: {exc}")
        logger.error("Processing skipped--truncated or corrupted data file?")
        logger.error(f"Problem file (datetime values): {nc2_sorc.as_posix()}")
        return
    t_stamp = time_utc[0].strftime("%H%M%S")
    orb_num = nc2_data["Science/orbit_number"][0]
    sc_id = nc2_data.getncattr("SpaceVehicle")

    # This field _should_ be constant but do the following for robustness?
    reference_altitude_km = np.nanmedian(nc2_data["Geolocation/reference_altitude"])

    if orb_num == "" or sc_id == "":
        logger.warning(
            f"File missing basic metadata values, skipping: {nc2_sorc.as_posix()}"
        )
        return

    # FIXME: Attempt to trim slewing observations at start and finish
    # if len(time_utc) > 12:
    #     use_obs = np.s_[5:-5]
    # else:
    #     use_obs = np.s_[:]
    use_obs = np.s_[:]  # See how things look without the haircut now

    # Define smoothing parameters for curve to be overlain on (noisy) dB plots.
    num_tim = len(time_utc[use_obs])
    ave_win = min(AVERAGING_WINDOW, num_tim)

    # Extract MEM retrieved dBs, geolocation and magnetic coordinates, covariance fields
    plot_type = "retrieved_b_and_j"

    ftgt = _plot_needs_update(
        source=nc2_sorc,
        save_directory=save_directory,
        obs_date=obs_date,
        plot_type=plot_type,
        type_desc="retrieved B field dBs",
        spacecraft=sc_id,
        tstmp=t_stamp,
        overwrite=overwrite,
    )
    if ftgt is None:
        return

    rtrvgrp = nc2_data.groups["RetrievedParameters"]
    if "observation_valid" in rtrvgrp.variables:  # NEW format
        # TODO: Need to support dbmag or dbdown?
        dbd_tmpl = "debiased_dbmag"
        cov_tmpl = "retrieved_dbmag_error"
        algorithm = "mag"
    else:
        dbd_tmpl = "retrieved_dbgeod"
        cov_tmpl = "cov_dd"
        algorithm = "down"
    mem_db_str = f"RetrievedParameters/{dbd_tmpl}"

    # Stack MEM arrays so we can index and loop through them by number rather than
    # using 4 separate variable names.
    mem_dbd = _stack_mem_fields(nc2_data, mem_db_str)
    mem_err = _stack_mem_fields(nc2_data, f"RetrievedParameters/{cov_tmpl}")
    if algorithm == "down":
        mem_err = np.sqrt(mem_err)  # Covariance is in nc4 files, not error
    mag_lat = _stack_mem_fields(nc2_data, "MagneticCoords/obs_maglat")
    mag_ltm = _stack_mem_fields(nc2_data, "MagneticCoords/obs_magLT")
    obs_lat = _stack_mem_fields(nc2_data, "Geolocation/obs_lat")
    obs_lon = _stack_mem_fields(nc2_data, "Geolocation/obs_lon")

    missing_val = mem_dbd == NCDF_MISSING
    mem_dbd = np.where(missing_val, np.nan, mem_dbd)
    mag_lat = np.where(missing_val, np.nan, mag_lat)
    mag_ltm = np.where(missing_val, np.nan, mag_ltm)
    obs_lat = np.where(missing_val, np.nan, obs_lat)
    obs_lon = np.where(missing_val, np.nan, obs_lon)

    valid_pnts = np.sum(~np.isnan(mem_dbd[0, :]))
    if valid_pnts < 3:
        logger.debug(~np.isnan(mem_dbd[0, :]))
        logger.error(
            f"Insufficient non-NaN samples to plot, skipping: {nc2_sorc.as_posix()}"
        )
        return

    # Plot magnetic field reference values and deltas from EZIE OSSE retrieval along
    # EZIE MEM lines of sight.
    nrows, ncols = NUM_MEM + 1, 5
    fig = plt.figure(figsize=DEFAULT_FIG_SIZE)
    dbcols = 3
    ags = GridSpec(nrows, ncols, figure=fig)
    ax0 = fig.add_subplot(ags[0, 0:dbcols])
    ax1 = fig.add_subplot(ags[1, 0:dbcols], sharex=ax0)
    ax2 = fig.add_subplot(ags[2, 0:dbcols], sharex=ax0)
    ax3 = fig.add_subplot(ags[3, 0:dbcols], sharex=ax0)
    ax4 = fig.add_subplot(ags[4, 0:dbcols], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    with Dataset(nc3_sorc, mode="r") as nc3_data:
        try:
            l3_data = _load_l3_pass(nc3_data)
        except TypeError:
            raise
        except Exception as exc:
            logger.error(f"Encountered exception while processing L3 file: {exc}")
            return

        time_utc = l3_data.time_utc
        midpt = l3_data.midpt
        lat_at_midpt = l3_data.lat_at_midpt
        lon_at_midpt = l3_data.lon_at_midpt
        lons = l3_data.lons
        lats = l3_data.lats
        lat_mesh = l3_data.lat_mesh
        lon_mesh = l3_data.lon_mesh
        model_B = l3_data.model_B

        # Add magnetic coordinate overlay, select latitudes at which to draw magnetic
        # gridlines, play with the map projection as needed.
        hemisphere = _select_hemisphere(lat_at_midpt)
        if hemisphere is None:
            logger.warning(
                f"L3 file appears to contain EEJ pass, skipping: {nc3_sorc.as_posix()}"
            )
            return

    if l3_data.algorithm is not None:
        algorithm = l3_data.algorithm

    if (
        np.all(l3_data.Jn == 0.0)
        and np.all(l3_data.Je == 0.0)
        and np.all(l3_data.obs_B == 0.0)
    ):
        logger.warning(
            f"L3 file Je, Jn, and observed B values are all zero, skipping: {ftgt.stem}"
        )
        return

    logger.info(f"Generating L3 plot {ftgt.stem}")

    # Get solar position, first in geodetic and then in APEX magnetic coordinates. We'll
    # display projected geodetic coordinates for now, pending addition of magnetic
    # coordinate counterparts to the L3 files.

    # Compute position of sun in both geodetic and geomagnetic coordinates for the
    # observation midpoint time.
    sun_geo_lon_mid, _sun_geo_lat_mid = _subsolar_geodetic_at(time_utc[midpt])

    # Save for potential later use?
    # Compute the subsolar point in APEX magnetic coordinates
    # apex = Apex(
    #     date=time_utc[midpt].year,
    #     refh=0,  # Leave at default (0), or set to 80 km? TBD
    # )
    # _sun_mag_lat_mid, _sun_mag_lon_mid = apex.geo2apex(
    #     sun_geo_lat_mid,
    #     sun_geo_lon_mid,
    #     reference_altitude_km,
    # )

    lats_m, MLT_sign, map_proj = _make_l3_projection(
        hemisphere,
        sun_geo_lon_mid,
        lat_at_midpt,
        lon_at_midpt,
        south_inverted,
    )

    # Set up axis and projection for cartopy
    maprowspan, mapcolspan = 2, 2  # 5 rows for geographic coordinate inset
    l3_map_axs: GeoAxes = plt.subplot2grid(
        (nrows, ncols),
        (0, ncols - mapcolspan),
        fig=fig,
        projection=map_proj,
        rowspan=maprowspan,
        colspan=mapcolspan,
    )  # ty:ignore[invalid-assignment]

    l3_map(
        l3_map_axs=l3_map_axs,
        l3_data=l3_data,
        lats_m=lats_m,
        fig=fig,
        hemisphere=hemisphere,
        MLT_sign=MLT_sign,
        sun_geo_lon_mid=sun_geo_lon_mid,
        map_proj=map_proj,
        south_inverted=False,
        dark_mode=dark_mode,
        stretch=0.06,
        show_b_value=show_b_value,
    )

    fpclr = "gray" if dark_mode else "black"
    qvclr = "orange" if dark_mode else "black"
    if algorithm == "down":
        mag_label = "dB$_\\mathbf{{D}}$"
    else:
        mag_label = "d|B|"
    if hemisphere is not None:
        if show_b_value:
            mem_fp_desc = (
                f"circles, colored according to the observed value of {mag_label}"
            )
        else:
            mem_fp_desc = f"{fpclr} dots"

        subtitle1 = f"""Currents ({qvclr} arrows) are plotted above in Geodetic/WGS84
        coordinates (solid grid). Labels indicate Local Time (LT). Magnetic latitude
        (dashed grid) is also shown. MEM footprint locations, shown above as
        {mem_fp_desc}, are also plotted below in Apex Geomagnetic coordinates."""

    else:
        subtitle1 = f"""Currents ({qvclr} arrows) are plotted in Geodetic/WGS84
        coordinates (solid grid). Magnetic latitude (dashed grid) and drift equator
        (thick solid line) are also shown. MEM footprints are plotted below in Apex
        Geomagnetic coordinates."""

    subtitle2 = f"""The MEM d|B| data plotted at left is sampled at a 2 second cadence.
    The {fpclr} lines are the running average of {AVERAGING_WINDOW} samples. The magenta
    lines show the modeled dB$_{{down}}$ values sampled along each MEM
    footprint."""

    fig.text(
        0.705,
        0.51,
        " ".join(subtitle1.split()),
        wrap=True,
        ha="left",
        va="center",
        fontsize="x-small",
        fontweight="bold",
    )

    fig.text(
        0.705,
        0.46,
        " ".join(subtitle2.split()),
        wrap=True,
        ha="left",
        va="center",
        fontsize="x-small",
        fontweight="bold",
    )

    # We'll save the total variable range for each field component across
    # all MEMs here, adjusting plot limits afterwards when we know what the
    # correct range is for the whole ensemble.
    b_rng = np.full((2), fill_value=np.nan)  # 4 B x [min,max]

    lat_axs = axs[-1]
    mlt_axs = _setup_mlat_mlt_axes(lat_axs)
    lat_axs.text(
        x=-0.05,
        y=-0.19,
        s=f"B perturbation source: {mem_db_str}",
        size="x-small",
        transform=lat_axs.transAxes,
        va="top",
        weight="bold",
    )
    for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
        row = PLT_NDX[mem_ndx]
        mem_axs = axs[row]
        lat_axs.plot(
            time_utc[use_obs],
            mag_lat[mem_ndx, use_obs],
            color=MEM_CLR[mem_ndx],
            linestyle="solid",
        )
        mlt_axs.plot(
            time_utc[use_obs],
            mag_ltm[mem_ndx, use_obs],
            color=MEM_CLR[mem_ndx],
            linestyle="dotted",
        )
        mem_axs.errorbar(
            time_utc[use_obs],
            mem_dbd[mem_ndx][use_obs],
            yerr=mem_err[mem_ndx][use_obs],
            color=MEM_CLR[mem_ndx],
            ecolor="purple",
            linestyle="solid",
            errorevery=3,
            capsize=2.0,
            barsabove=True,
        )
        mem_axs.plot(
            time_utc[use_obs],
            moving_average(mem_dbd[mem_ndx][use_obs], ave_win),
            color="black" if not dark_mode else DARK_MODE_TEXT_COLOR,
            linestyle="solid",
            lw=1.0,
            zorder=3,
        )

        def mem_nearest(mem_lon, mem_lat):
            b_nearest = np.zeros((num_tim, 2), dtype=int)
            new_shape = lon_mesh.shape
            for xx in range(num_tim):
                b_nearest[xx, :] = np.unravel_index(
                    np.argmin(
                        (mem_lon[xx] - lon_mesh) ** 2 + (mem_lat[xx] - lat_mesh) ** 2
                    ),
                    new_shape,
                )
            return b_nearest

        sample_pts = mem_nearest(lons[mem_ndx, use_obs], lats[mem_ndx, use_obs])
        mem_axs.plot(
            time_utc[use_obs],
            [model_B[x, y] for x, y in sample_pts],
            color="magenta",
            linestyle="solid",
            lw=1.0,
            zorder=2,
        )

        try:
            _update_range(
                b_rng,
                mem_dbd[mem_ndx][use_obs] - mem_err[mem_ndx][use_obs],
                mem_dbd[mem_ndx][use_obs] + mem_err[mem_ndx][use_obs],
            )
        except Exception as exc:
            mem_axs.set_ylim((-1.0, +1.0))
            logger.error(f"Exception while attempting to save axis y range: {exc}")
            logger.error(f"Problem file (axis range): {nc2_sorc.as_posix()}")
            # plt.close(fig)
            return

        mem_axs.grid(axis="both")
        mem_axs.axhline(0, ls="dotted", color="black")

        # Place label with receiver number at top of each column
        if algorithm == "down":
            mem_axs.set_ylabel(f"MEM {mem_num} dB$_\\mathbf{{D}}$ (nT)", weight="bold")
        else:
            mem_axs.set_ylabel(f"MEM {mem_num} d|B| (nT)", weight="bold")
        offnad_offset = 0.01
        fw, fh = fig.get_figwidth(), fig.get_figheight()
        axs_asp_rat = nrows * dbcols / ncols * fw / fh  # ~  the _axis_ aspect ratio?
        mem_axs.text(
            x=offnad_offset,
            y=1.0 - offnad_offset * axs_asp_rat,
            s=f"Off-Nadir Angle: {MEM_LOOK_DIRECTIONS[mem_ndx]}" r"º",
            weight="bold",
            size="medium",
            va="top",
            ha="left",
            transform=mem_axs.transAxes,
        )

        # Label only leftmost column y axis
        if row == 0:
            mem_axs.set_title(
                f"Retrieved Magnetic Field Magnitude Perturbation (nT) at Reference "
                f"Altitude ({reference_altitude_km} km)",
                weight="bold",
                size="medium",
            )

    # Adjust axis y limits to be uniform for a given field component across all MEMs
    for mem_ndx in range(NUM_MEM):
        mem_axs = axs[PLT_NDX[mem_ndx]]
        tot_rng = b_rng[1] - b_rng[0]
        logger.debug(f"{b_rng[0]} : {b_rng[1]}")
        # FIXME - kludge to use old L2 file - comment out line below
        try:
            mem_axs.set_ylim((b_rng[0] - 0.05 * tot_rng, b_rng[1] + 0.05 * tot_rng))
        except Exception as exc:
            mem_axs.set_ylim((-1.0, +1.0))
            logger.error(f"Exception while attempting to set axes y limits: {exc}")
            logger.error(f"Problem file (data range): {nc2_sorc.as_posix()}")
            # plt.close(fig)
            return

    # Debugging bazillion-tick failures (when only a single non-NaN value is plotted?)
    # Should no longer be an issue with updated L2 retrieval algorithm.
    # logger.info("Adding time (x axis) tick labels...")
    # logger.info(f"Start and stop times (UTC) are {time_utc[0]} - {time_utc[-1]}")
    # logger.info(f"Start+1 and stop-1 times (UTC) are {time_utc[1]} - {time_utc[-2]}")
    # logger.info(f"Length of time array is {len(time_utc)}")
    # if (time_utc[-1] - time_utc[0]).total_seconds() < 120:
    #     logger.info(f"{mem_dbd[0, :]}")
    #     logger.info(f"{np.sum(~np.isnan(mem_dbd[0, :]))}")

    # Format x-axis to show only hours and minutes unless sample is shorter than 120
    # seconds. Label only bottom row left time/x axis.
    set_xaxis_tick_format(
        axs[-1],
        max_ticks=20 // ncols + 1,
        use_seconds=(time_utc[-1] - time_utc[0]).total_seconds() < 120,
    )

    # NEW geomagnetic coordinate map inset
    midpt = len(time_utc) // 2
    lat_at_midpt = nc2_data["Geolocation/sat_lat"][midpt]
    lon_at_midpt = nc2_data["Geolocation/sat_lon"][midpt]

    hemisphere = _select_hemisphere(lat_at_midpt)
    proj_method = _inset_map_projection(hemisphere, lat_at_midpt, lon_at_midpt)

    maprowspan, mapcolspan = 2, 2  # 5 rows for geographic coordinate inset
    fp_map_axs: GeoAxes = plt.subplot2grid(
        (nrows, ncols),
        (nrows - maprowspan, ncols - mapcolspan),
        fig=fig,
        projection=proj_method,
        rowspan=maprowspan,
        colspan=mapcolspan,
    )  # ty:ignore[invalid-assignment]

    if hemisphere is not None:
        clip_to_circle(
            map_axs=fp_map_axs, proj_method=proj_method, hemisphere=hemisphere
        )

    # Map S/C and MEM footprint lat/lon
    dtlim = [time_utc[0], time_utc[1]]
    duration = (dtlim[1] - dtlim[0]).total_seconds()
    midpoint = dtlim[0] + datetime.timedelta(seconds=int(0.5 * duration))

    # Get solar position, first in geodetic and then in APEX magnetic coordinates.
    sun_geo_lon_deg, sun_geo_lat_deg = _subsolar_geodetic(time_utc)

    # Compute the subsolar point in APEX magnetic coordinates
    apex = Apex(date=midpoint.year, refh=0)
    _sun_mlat, sun_mlon = apex.geo2apex(
        sun_geo_lat_deg,
        sun_geo_lon_deg,
        reference_altitude_km,
    )

    # Compute satellite footprint in APEX magnetic coordinates
    sat_maglat, sat_maglon = apex.geo2apex(
        nc2_data["Geolocation/sat_lat"][:],
        nc2_data["Geolocation/sat_lon"][:],
        nc2_data["Geolocation/sat_alt"][:],
    )

    # Convert magnetic longitude to MLT
    sat_magLT_deg = 180.0 + sat_maglon - sun_mlon

    # Debugging "mirrored" southern hemisphere plots
    # logger.debug(f"{hemisphere} {lat_at_midpt}")
    # logger.debug(f"{np.min(sun_geo_lat_deg)}, {np.max(sun_geo_lat_deg)}")
    # logger.debug(f"{np.min(sun_geo_lon_deg)}, {np.max(sun_geo_lon_deg)}")
    # logger.debug(
    #     f"Sun magnetic latitude, longitude = "
    #     f"{sun_mlat[tndx // 2]:7.2f}, "
    #     f"{sun_mlon[tndx // 2]:7.2f}"
    # )
    # logger.debug(f"{np.min(sat_magLT_deg)}, {np.max(sat_magLT_deg)}")

    # Transform continent outlines from geographic to APEX magnetic coordinates
    MLT_sign = -1 if (south_inverted and hemisphere == SOUTH) else +1
    if hemisphere is not None:
        map_magnetic_continents(
            ax=fp_map_axs,
            time4mag=midpoint,
            sun_mlon=sun_mlon[len(sun_mlon) // 2],
            alt4mag=reference_altitude_km,
            south_inverted=(south_inverted and (hemisphere == SOUTH)),
            color="xkcd:black" if not dark_mode else DARK_MODE_GRID_COLOR,
        )
        map_sc_mem_footprints_magnetic(
            map_axs=fp_map_axs,
            sat_lat=sat_maglat,
            sat_lon=MLT_sign * sat_magLT_deg,
            obs_lat=mag_lat[:, :].T,
            obs_lon=MLT_sign * mag_ltm[:, :].T * 15,  # Convert hours to degrees
            at_time=midpoint,
            dark_mode=dark_mode,
            zorder=4,
            small_text=False,
            plain=False,
            legend_top=False,
            terminator=False,
            hemisphere=hemisphere,
            south_inverted=(south_inverted and (hemisphere == SOUTH)),
        )

    else:
        map_sc_mem_footprints(
            map_axs=fp_map_axs,
            sat_lat=nc2_data["Geolocation/sat_lat"][use_obs],
            sat_lon=nc2_data["Geolocation/sat_lon"][use_obs],
            obs_lat=obs_lat[:, use_obs].T,
            obs_lon=obs_lon[:, use_obs].T,
            at_time=midpoint,
            dark_mode=dark_mode,
            terminator=True,
            zorder=4,
            small_text=False,
            plain=False,
            legend_top=False,
            mid_lat_mag=True,
        )

    # Add MEM beam numbering/pointing diagram at bottom right corner of figure
    _add_mem_beam_diagram(fig, hi=0.22, left=0.62, bottom=0.44 - 0.22 / 2, zorder=3)

    # Tweak position and add any figure-level annotation
    fig_title = "Magnetic Field Perturbations and Currents: "
    fig_title = fig_title + (
        f"{sc_id} - Orbit {orb_num}\n"
        f"{time_utc[use_obs][0].strftime('%Y-%m-%d (%j) %H:%M:%S')} - "
        f"{time_utc[use_obs][-1].strftime('%Y-%m-%d (%j) %H:%M:%S')}"
    )
    _finalize_figure(
        fig=fig,
        nc_data=nc2_data,
        source=nc2_sorc,
        title=fig_title,
        size="x-small",
        dark_mode=dark_mode,
    )
    with Dataset(nc3_sorc, mode="r") as nc3_data:
        add_product_metadata(
            fig=fig, nc_data=nc3_data, source=nc3_sorc, second=True, size="x-small"
        )
    plt.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.08,
        top=0.91,
        hspace=0.00,
        wspace=0.05,
    )
    save_close_figure(
        source=nc2_sorc,
        save_directory=save_directory,
        figure=fig,
        obs_date=obs_date,
        spacecraft=sc_id,
        tstmp=t_stamp,
        plot_type=plot_type,
        dpi=figure_dpi,
        dark_mode=dark_mode,
    )


def ingest_full_day_all_sv(
    run_date: str,  # YYYYMMDD
    used_files: list,  # list of L1 paths
):
    """
    Ingest and aggregate geolocation and metadata data from all EZIE Level-1 files
    for a specified date across all spacecraft (~15 orbits each).

    This function processes a list of L1 NetCDF files, extracts relevant variables
    (such as satellite latitude/longitude, MEM observation coordinates, magnetic
    latitude, MLT, and solar zenith angles), and combines them into a unified
    dictionary containing data for all orbits of the day.

    Args:
        run_date (str): The date to process, formatted as 'YYYYMMDD'.
        used_files (list): A list of file paths (Path objects or strings) pointing to
            the L1 NetCDF files to be ingested.

    Returns:
        (tuple):

            full_day (dict): A dictionary where keys are variable names (e.g.,
            'sat_lat', 'obs_maglat1') and values are numpy arrays of the aggregated data
            for all spacecraft and orbits on the given date.

            uniq_svid (numpy.ndarray): An array of unique spacecraft identifiers found
            in the processed files.

            uniq_orbs (numpy.ndarray): An array of unique orbit numbers found in the
            processed files.
    """

    # Generate dictionary of all locations sampled by the EZIE spacecraft in the course
    # of one day. The dictionary key hierarchy is:
    # 1) SV
    # 2) L1 var names (from coverage_db_list below)
    coverage_db_list = [
        # "SpaceVehicle",  # Now extracted from global attributes instead
        "/Science/orbit_number",
        "/Geolocation/sat_lat",
        "/Geolocation/sat_lon",
        "/Geolocation/obs_lat1",
        "/Geolocation/obs_lat2",
        "/Geolocation/obs_lat3",
        "/Geolocation/obs_lat4",
        "/Geolocation/obs_lon1",
        "/Geolocation/obs_lon2",
        "/Geolocation/obs_lon3",
        "/Geolocation/obs_lon4",
        "/Geolocation/obs_solar_zen1",
        "/Geolocation/obs_solar_zen2",
        "/Geolocation/obs_solar_zen3",
        "/Geolocation/obs_solar_zen4",
        "/MagneticCoords/obs_maglat1",
        "/MagneticCoords/obs_maglat2",
        "/MagneticCoords/obs_maglat3",
        "/MagneticCoords/obs_maglat4",
        "/MagneticCoords/obs_magLT1",
        "/MagneticCoords/obs_magLT2",
        "/MagneticCoords/obs_magLT3",
        "/MagneticCoords/obs_magLT4",
    ]

    date_dict = {}
    full_day = {}
    for each_file in used_files:
        # Ingest latest version/revision L1 file for each SV
        prsd = parse_ezie_product_name(each_file)
        if prsd.date != run_date:
            continue

        # Gather entries for ALL orbits by this SV on this date
        if prsd.spcv not in date_dict:
            date_dict[prsd.spcv] = {}
        if "l1" in each_file.stem.lower():
            try:
                with Dataset(each_file, mode="r") as nc_data:
                    product = nc_data.getncattr("product")
                    if product == "L1":
                        logger.info(f"Processing {product} file {Path(each_file).name}")

                        if prsd.orbt not in date_dict[prsd.spcv]:
                            date_dict[prsd.spcv][prsd.orbt] = {}
                        # TODO: Check to make sure global attribute matches filename!
                        # Then eliminate this semi-redundant quantity from date_dict.
                        spcv = f"EZIE-{prsd.spcv.upper()}"
                        logger.debug(f"Setting SpaceVehicle to {spcv}")
                        date_dict[prsd.spcv][prsd.orbt]["SpaceVehicle"] = np.repeat(
                            str(spcv),
                            nc_data.dimensions["ObsRate"].size,
                        )
                        for dbvar in coverage_db_list:
                            try:
                                date_dict[prsd.spcv][prsd.orbt][dbvar] = nc_data[dbvar][
                                    :
                                ].data
                            except AttributeError:
                                date_dict[prsd.spcv][prsd.orbt][dbvar] = np.repeat(
                                    str(nc_data[dbvar][:]),
                                    nc_data.dimensions["ObsRate"].size,
                                )
                        logger.debug(
                            f"{date_dict[prsd.spcv][prsd.orbt]['SpaceVehicle']}"
                        )
                    else:
                        logger.error(f"{product} misidentified as L1-skipping")
                        date_dict[prsd.spcv][prsd.orbt] = None
            except Exception as exc:
                logger.error(f"Unrecoverable problem, skipping file: {exc}")
                date_dict[prsd.spcv][prsd.orbt] = None

    # Combine entries from each SV/L1 file into single unified dictionary
    for sv_dict in date_dict.values():
        for orb_dict in sv_dict.values():
            if orb_dict is None:  # This orbit failed to ingest; skip it
                continue
            if "SpaceVehicle" not in full_day:
                full_day["SpaceVehicle"] = []
            if orb_dict["SpaceVehicle"] is not None:
                full_day["SpaceVehicle"].extend(orb_dict["SpaceVehicle"])
            for dbvar in coverage_db_list:
                full_key = dbvar.split("/")[-1]  # Use only variable name, not group
                if full_key not in full_day:
                    full_day[full_key] = []
                try:
                    if orb_dict[dbvar] is not None:
                        full_day[full_key].extend(orb_dict[dbvar])
                except Exception as exc:
                    logger.error(
                        f"Selected field could not be extracted from file: {exc}"
                    )

    # Recast each dictionary value (list) as a numpy array
    for dbvar in coverage_db_list:
        full_key = dbvar.split("/")[-1]  # Use only variable name, not group
        full_day[full_key] = np.array(full_day[full_key])
    full_day["SpaceVehicle"] = np.array(full_day["SpaceVehicle"])

    # FIXME: Not in .nc4 files! Borrowing MEM 1.
    # FIXME: Have these added to netcdf products? Otherwise must compute w/SPICE _here_.
    full_day["sat_mlat"] = full_day["obs_maglat1"]
    full_day["sat_MLT"] = full_day["obs_magLT1"]

    # Get unique values of orbits and SVs
    uniq_svid = np.unique(full_day["SpaceVehicle"])
    uniq_orbs = np.unique(full_day["orbit_number"])

    logger.info(f"Distinct orbit numbers: {[int(_) for _ in uniq_orbs]!r}")
    logger.info(f"Distinct space vehicles: {[str(_) for _ in uniq_svid]!r}")

    return full_day, uniq_svid, uniq_orbs
