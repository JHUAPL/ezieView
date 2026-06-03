"""
A collection of methods for generating plots from EZIE data products, primarily for use
on the science gateway. A few methods are also used for various spacecraft attitude
and/or data analyses, e.g., the star tracker loss of lock investigation.

TODO: Refactor the various mapping methods used by multiple types of plot generation
routines, both geodetic and magnetic, into a single set of modular methods (rather than
the organic mess that grew out of shifting requirements over the last two years). Once
that is done, the methods specific to a single type of plot should probably be moved to
their respective update_*_plots.py files in their presumably slimmed down new forms.
This file should contain only the methods common to all, and the gw_*.py routines are
already used by both the gateway plotting codes and various analysis coes, e.g., for
sci-ops investigations (star tracker loss of lock, etc.).
"""

# region imports
import datetime
import logging
import time
import warnings
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
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset, default_fillvals

from ezieview.ezvislib.gw_plot_params import (
    ALTERNATE_FIG_SIZE,
    AVERAGING_WINDOW,
    BT,
    DARK_MODE_FACE_COLOR,
    DARK_MODE_FILL_COLOR,
    DARK_MODE_GRID_COLOR,
    DARK_MODE_TEXT_COLOR,
    DATA_TRANSFORM,
    DBS_MOD,
    DEFAULT_FIG_SIZE,
    DFLT_RES,
    EARTH_FLATTENING,
    EARTH_RADIUS_EQUATORIAL,
    EZIE_DATE_FORMAT,
    FLD_CLR,
    FLD_CMP,
    FLD_MODES,
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
    TOT_MOD,
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
# endregion

# rc_fonts = {
#     "text.usetex": True,
#     # "text.latex.preview": True,
#     "font.size": 20,
#     "axes.titlesize": 22,
#     "axes.labelsize": 22,
#     "legend.fontsize": 20,
#     "xtick.labelsize": 20,
#     "ytick.labelsize": 20,
#     "figure.titlesize": 22,
#     "mathtext.default": "regular",
#     # "text.latex.preamble": [r"""\usepackage{bm}"""],
# }
# mpl.rcParams.update(rc_fonts)


def coverage_plot_polar_layout(
    observation_date: datetime.datetime,
    hemisphere: str,
    show_terminator_at: datetime.datetime | None = None,
    show_mag_lat: bool = False,
    south_inverted: bool = False,
    dark_mode: bool = False,
):
    # Instantiate figure and axes for plotting - tweak some rcparams as needed
    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
        plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR

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
            x_0, y_0 = proj_method.transform_point(
                0,
                90 if hemisphere == NORTH else -90,
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
            # Compute a circle in axes coordinates that will be used as a clipping
            # boundary for the map.
            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            map_axs.set_boundary(circle, transform=map_axs.transAxes)

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
        mlt_desc = {}
        for mm, mlt in enumerate(mlt_locs):
            mlt_desc[f"{mlt:.0f}"] = f"{mlt / 15:02.0f}H"
        mlt_desc["0"] = "Midnight"
        mlt_desc["90"] = "Dawn"
        mlt_desc["180"] = "Noon"
        mlt_desc["270"] = "Dusk"
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
            # FIXME: Take difference from noon or midnight and flip sign to make this
            # work when plotting with south_inverted set
            map_axs.add_feature(
                Nightshade(
                    show_terminator_at,
                    alpha=TERMINATOR_ALPHA,
                    color=TERMINATOR_COLOR,
                    zorder=2,
                )
            )

        x_0, y_0 = proj_method.transform_point(
            0,
            90 if hemisphere == NORTH else -90,
            src_crs=DATA_TRANSFORM,
        )
        x_1, y_1 = proj_method.transform_point(
            45,
            MAG_LAT_LOWER_LIMIT if hemisphere == NORTH else -MAG_LAT_LOWER_LIMIT,
            src_crs=DATA_TRANSFORM,
        )
        map_meters = np.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2)
        map_axs.set_extent(
            (
                -map_meters,
                +map_meters,
                -map_meters,
                +map_meters,
            ),
            crs=proj_method,
        )

        # Compute a circle in axes coordinates that will be used as a clipping boundary
        # for the map.
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        map_axs.set_boundary(circle, transform=map_axs.transAxes)

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
    # Instantiate figure and axes for plotting - tweak some rcparams as needed
    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
        plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR
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
            # FIXME: Take difference from noon or midnight and flip sign to make this
            # work when plotting with south_inverted set
            map_axs.add_feature(
                Nightshade(
                    show_terminator_at,
                    alpha=TERMINATOR_ALPHA,
                    color=TERMINATOR_COLOR,
                    zorder=2,
                )
            )
        x_0, y_0 = proj_method.transform_point(
            0,
            90 if hemisphere == NORTH else -90,
            src_crs=DATA_TRANSFORM,
        )
        x_1, y_1 = proj_method.transform_point(
            45,
            lat_lower_limit if hemisphere == NORTH else -lat_lower_limit,
            src_crs=DATA_TRANSFORM,
        )
        map_meters = np.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2)
        map_axs.set_extent(
            (
                -map_meters,
                +map_meters,
                -map_meters,
                +map_meters,
            ),
            crs=proj_method,
        )

        # Compute a circle in axes coordinates that will be used as a clipping boundary
        # for the map.
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        map_axs.set_boundary(circle, transform=map_axs.transAxes)

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
    # Instantiate figure and axes for plotting - tweak some rcparams as needed
    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
        plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR
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
            # FIXME: Take difference from noon or midnight and flip sign to make this
            # work when plotting with south_inverted set
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
    """
    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
        plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR

    geo_group = nc_data.groups["Geolocation"]
    sc_id = nc_data["Metadata/SpaceVehicle"][0]

    # Get time data and convert to datetime objects
    time_utc, obs_date = get_datetime_from_utc_string(nc_data.groups["Time"])
    i0, i1 = indices
    t_stamp = time_utc[i0].strftime("%H%M%S")
    orb_num = nc_data["Science/orbit_number"][i0]
    # FIXME: Attempt to trim slewing observations at start and finish
    # if i1 - i0 > 12:
    #     use_obs = np.s_[i0 + 5 : i1 - 5]
    # else:
    use_obs = np.s_[i0:i1]
    time_utc = time_utc[use_obs]
    plot_type = "geolocation"

    ftgt, old_hash, new_hash = save_close_figure(
        source=source,
        save_directory=save_directory,
        obs_date=obs_date,
        spacecraft=sc_id,
        tstmp=t_stamp,
        plot_type=plot_type,
        name_only=True,
    )
    logger.debug(f"Checked file: {ftgt.as_posix()}")

    if ftgt.exists() and (old_hash == new_hash) and not overwrite:
        logger.info(f"File exists, source hash unchanged, skipping: {ftgt.as_posix()}")
        return
    if ftgt.exists() and (old_hash != new_hash):
        logger.info("Source file hash has changed, updating plot")

    logger.info("Generating geolocation plot")

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
    fig.suptitle(fig_title, weight="bold", fontsize="x-large", y=0.99, va="top")
    add_product_metadata(fig=fig, nc_data=nc_data, source=source)
    add_pipeline_metadata(
        fig=fig,
        nc_data=nc_data,
        git_branch=git_branch,
        git_commit=git_commit,
    )
    overlay_ezie_logo(fig)
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
    Plot the reference IGRF B field values stored in the Ancillary group
    """
    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
        plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR

    # Get time data and convert to datetime objects
    time_utc, obs_date = get_datetime_from_utc_string(nc_data.groups["Time"])
    i0, i1 = indices
    t_stamp = time_utc[i0].strftime("%H%M%S")
    orb_num = nc_data["Science/orbit_number"][i0]
    # FIXME: Attempt to trim slewing observations at start and finish
    # if i1 - i0 > 12:  # Slew from SkyCal at start? Trim a few steps?
    #     use_obs = np.s_[i0 + 5 : i1 - 5]
    # else:
    use_obs = np.s_[i0:i1]
    time_utc = time_utc[use_obs]
    sc_id = nc_data["Metadata/SpaceVehicle"][0]
    plot_type = "ancillary"

    ftgt, old_hash, new_hash = save_close_figure(
        source=source,
        save_directory=save_directory,
        obs_date=obs_date,
        spacecraft=sc_id,
        tstmp=t_stamp,
        plot_type=plot_type,
        name_only=True,
    )
    logger.debug(f"Checked file: {ftgt.as_posix()}")
    if ftgt.exists() and (old_hash == new_hash) and not overwrite:
        logger.info(f"File exists, source hash unchanged, skipping: {ftgt.as_posix()}")
        return
    if ftgt.exists() and (old_hash != new_hash):
        logger.info("Source file hash has changed, updating plot")
    logger.info("Generating plot of ancillary (IGRF) data")

    # Prepare figure for plotting with sharex and sharey
    fig, axs = plt.subplots(
        1, NUM_FLD, figsize=DEFAULT_FIG_SIZE, sharex=False, sharey=True
    )

    # Plot each anc_model_bgeo variable in separate subplots
    for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
        # ax = axs[(mem_ndx - 1) // 2, (mem_ndx - 1) % 2]
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
    # title_date_time = time_utc[len(time_utc) // 2]  # get midpoint of observation
    fig_title = "IGRF-14 B Field Values: "
    fig_title = fig_title + (
        f"{sc_id} - Orbit {orb_num}\n"
        f"{time_utc[0].strftime('%Y-%m-%d (%j) %H:%M:%S UT')} - "
        f"{time_utc[-1].strftime('%Y-%m-%d (%j) %H:%M:%S UT')}"
    )
    fig.suptitle(fig_title, weight="bold", fontsize="x-large", y=0.99, va="top")
    add_product_metadata(fig=fig, nc_data=nc_data, source=source)
    add_pipeline_metadata(
        fig=fig,
        nc_data=nc_data,
        git_branch=git_branch,
        git_commit=git_commit,
    )
    overlay_ezie_logo(fig)
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
    """
    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
        plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR

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

    # logger.debug(f"{obs_flg[0:10]}")
    # logger.debug(f"{obs_flg[i0]}")
    # logger.debug(f"{obs_flg[i1 - 1]}")
    # logger.debug(f"{obs_flg[-10:]}")
    # logger.debug(f"{obs_flg.size} - {i0} : {i1}")

    time_utc = time_utc[i0:i1]
    sc_id = nc_data["Metadata/SpaceVehicle"][0]
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

        ftgt, old_hash, new_hash = save_close_figure(
            source=source,
            save_directory=save_directory,
            obs_date=obs_date,
            spacecraft=sc_id,
            tstmp=t_stamp,
            plot_type=f"{ckind}-{t_kind_str}",
            name_only=True,
        )
        logger.debug(f"Checked file: {ftgt.as_posix()}")

        if ftgt.exists() and (old_hash == new_hash) and not overwrite:
            logger.info(
                f"File exists, source hash unchanged, skipping: {ftgt.as_posix()}"
            )
            continue
        if ftgt.exists() and (old_hash != new_hash):
            logger.info("Source file hash has changed, updating plot")

        logger.info(f"Generating calibrated scene temperature plot ({t_kind})")

        fig, axs = plt.subplots(
            num_rows + 1,
            num_cols,
            figsize=DEFAULT_FIG_SIZE,
            sharey=True,
            height_ratios=[1, 1, 1, 1, 0.25],
        )
        for col in range(num_cols):  # Iterate over columns (Stokes parameters)
            # Auto-scaling of Ta and Tb range (not a good idea, but save for later?)
            # vmins, vmaxs = [], []
            # for row in range(1, 5):  # Iterate over TA1...TA4 or TB1...TB4
            #     var_name = f"{tkind_vars[kk]}{row}"
            #     data = nc_data[var_name][i0:i1, :, col]  # Shape:(ObsRate, Freq_Array)
            #     vmins.append(data.min())
            #     vmaxs.append(data.max())
            # vmin, vmax = min(vmins), max(vmaxs)

            # Set fixed scale for Ta and Tb
            if not t_diff:
                match stokes[col].upper():
                    case "H" | "V":
                        vmin, vmax = +150, +300
                    case "S3" | "S4":
                        vmin, vmax = -40, +40
            else:
                match stokes[col].upper():
                    case "H" | "V" | "S3":
                        vmin, vmax = -25, +25
                    case "S3" | "S4":
                        vmin, vmax = -50, +50

            for row in range(0, num_rows):
                ax = axs[row, col]
                var_name = f"{tkind_vars[kk]}{row + 1}"
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
                # ax.xaxis.set_major_locator(plt.MaxNLocator(7))
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
        fig.suptitle(
            fig_title,
            weight="bold",
            fontsize="x-large",
            y=0.99,
            va="top",
        )
        add_product_metadata(fig=fig, nc_data=nc_data, source=source)
        add_pipeline_metadata(
            fig=fig,
            nc_data=nc_data,
            git_branch=git_branch,
            git_commit=git_commit,
        )
        overlay_ezie_logo(fig)
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


def plot_retrieved_b_fields(
    nc_data: Dataset,
    source: Path,
    save_directory: Path,
    version: str | None = None,
    dark_mode: bool = False,
    figure_dpi: int = DFLT_RES,
    overwrite: bool = False,
):
    """
    Plot the retrieved B fields for each MEM in two formats, along with their estimated
    errors. Errors for individual dBs are calculated from the derived covariance values.
    Errors for the total B field are calculated as a weighted average of the errors from
    each inidiviual N-E-D component. The two formats are:
    1) Just the dBs for the N-E-D components ( 4 MEM x 3 B )
    2) Both the N-E-D components and the B totals for each MEM ( 4 MEM x 4 B )
    """
    logger.info("Generating retrieved and reference B field plots")
    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
        plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR

    # Collate netcdf data for all spacecraft and orbits on the selected day

    # Get time data and convert to datetime objects
    # L2 files are already broken up into discrete science passes
    time_utc, obs_date = get_datetime_from_utc_string(nc_data.groups["Time"])
    t_stamp = time_utc[0].strftime("%H%M%S")
    orb_num = nc_data["Science/orbit_number"][0]
    sc_id = nc_data["Metadata/SpaceVehicle"][0]
    # FIXME: Attempt to trim slewing observations at start and finish
    # if len(time_utc) > 12:
    #     use_obs = np.s_[5:-5]
    # else:
    use_obs = np.s_[:]

    max_win = len(time_utc[use_obs])
    ave_win = min(AVERAGING_WINDOW, max_win)

    # MEM retrieved dBs
    mem_dbn_val = [f"RetrievedParameters/retrieved_dbgeon{mm}" for mm in MEM_NUMBERS]
    mem_dbe_val = [f"RetrievedParameters/retrieved_dbgeoe{mm}" for mm in MEM_NUMBERS]
    mem_dbd_val = [f"RetrievedParameters/retrieved_dbgeod{mm}" for mm in MEM_NUMBERS]
    mem_dbt_val = [f"RetrievedParameters/retrieved_dbgeo{mm}" for mm in MEM_NUMBERS]

    # MEM footprint reference model (IGRF) B fields
    igrf_bn_val = [f"Ancillary/anc_model_bgeon{mm}" for mm in MEM_NUMBERS]
    igrf_be_val = [f"Ancillary/anc_model_bgeoe{mm}" for mm in MEM_NUMBERS]
    igrf_bd_val = [f"Ancillary/anc_model_bgeod{mm}" for mm in MEM_NUMBERS]
    igrf_bt_val = [f"Ancillary/anc_model_bgeo{mm}" for mm in MEM_NUMBERS]

    # Covariance fields
    mem_bnn_cov = [f"RetrievedParameters/cov_nn{mm}" for mm in MEM_NUMBERS]
    mem_bee_cov = [f"RetrievedParameters/cov_ee{mm}" for mm in MEM_NUMBERS]
    mem_bdd_cov = [f"RetrievedParameters/cov_dd{mm}" for mm in MEM_NUMBERS]
    mem_bne_cov = [f"RetrievedParameters/cov_ne{mm}" for mm in MEM_NUMBERS]
    mem_bnd_cov = [f"RetrievedParameters/cov_nd{mm}" for mm in MEM_NUMBERS]
    mem_bed_cov = [f"RetrievedParameters/cov_ed{mm}" for mm in MEM_NUMBERS]

    # Retrieved Btot value and Btot error
    mem_btv_val = [f"RetrievedParameters/retrieved_b_tot{mm}" for mm in MEM_NUMBERS]
    mem_bte_val = [f"RetrievedParameters/retrieved_b_tot_err{mm}" for mm in MEM_NUMBERS]

    # Stack MEM arrays so we can index and loop through them by number rather than using
    # 4 separate variable names.
    mem_dbn = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_dbn_val],))  # 4xn_obs
    mem_dbe = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_dbe_val],))
    mem_dbd = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_dbd_val],))
    mem_dbt = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_dbt_val],))
    igrf_bn = np.vstack((*[nc_data[nc_fld] for nc_fld in igrf_bn_val],))  # 4xn_obs
    igrf_be = np.vstack((*[nc_data[nc_fld] for nc_fld in igrf_be_val],))
    igrf_bd = np.vstack((*[nc_data[nc_fld] for nc_fld in igrf_bd_val],))
    igrf_bt = np.vstack((*[nc_data[nc_fld] for nc_fld in igrf_bt_val],))
    rtrv_bt = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_btv_val],))
    rtrv_be = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_bte_val],))
    mem_cnn = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_bnn_cov],))
    mem_cee = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_bee_cov],))
    mem_cdd = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_bdd_cov],))
    mem_cne = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_bne_cov],))
    mem_ced = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_bed_cov],))
    mem_cnd = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_bnd_cov],))

    # Stack field components in addition to MEMs to create 3-D array, nB x 4 MEM x n_obs
    igrf_bf = np.stack((*[igrf_bn, igrf_be, igrf_bd, igrf_bt],))
    mem_dbs = np.stack((*[mem_dbn, mem_dbe, mem_dbd, mem_dbt],))
    mem_cov = np.stack((*[mem_cnn, mem_cee, mem_cdd, mem_cne, mem_ced, mem_cnd],))

    # Plot magnetic field reference values and deltas from EZIE OSSE retrieval along
    # EZIE MEM lines of sight.
    for fld_mod in FLD_MODES:
        # FLD_MODES ==> Plot reference (IGRF) B field vectors or dBs?
        logger.info(f"Generating plot of B field {fld_mod}")
        if fld_mod == TOT_MOD:
            rows, cols = NUM_FLD, NUM_MEM  # Include B_Total
        else:
            rows, cols = NUM_FLD - 1, NUM_MEM  # Do NOT include [meaningless] dB_Total

        # use = np.full_like(time_utc[:], fill_value=True, dtype=bool)
        fig, axs = plt.subplots(
            rows,
            cols,
            figsize=DEFAULT_FIG_SIZE,
            sharex=True,
            sharey=False,
        )

        # We'll save the total variable range for each field component across
        # all MEMs here, adjusting plot limits afterwards when we know what the
        # correct range is for the whole ensemble.
        b_rng = np.full((NUM_FLD, 2), fill_value=np.nan)  # 4 B x [min,max]

        for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
            col = mem_ndx % cols
            for fld_ndx, fld_cmp in enumerate(FLD_CMP):
                # Tweak axis row position depending on whether or not we're plotting
                # just the N-E-D components of B or adding B total as well.
                if fld_mod == DBS_MOD:
                    if fld_cmp == BT:
                        # No B_TOT plots for dBs, only for full fields
                        continue
                    row = (mem_ndx // cols) * NUM_FLD + fld_ndx
                else:
                    row = (mem_ndx // cols) * NUM_FLD + (fld_ndx + 1) % NUM_FLD
                mem_axs = axs[row, col]

                if fld_mod == TOT_MOD:
                    mem_axs.plot(
                        time_utc[use_obs],
                        igrf_bf[fld_ndx][mem_ndx][use_obs],
                        color=FLD_CLR[fld_ndx],
                        linewidth=2.0,
                        linestyle="solid",
                        zorder=3,
                    )
                    if fld_cmp == BT:  # B total
                        mbt = np.sqrt(
                            (
                                igrf_bf[0][mem_ndx][use_obs]
                                + mem_dbs[0][mem_ndx][use_obs]
                            )
                            ** 2
                            + (
                                igrf_bf[1][mem_ndx][use_obs]
                                + mem_dbs[1][mem_ndx][use_obs]
                            )
                            ** 2
                            + (
                                igrf_bf[2][mem_ndx][use_obs]
                                + mem_dbs[2][mem_ndx][use_obs]
                            )
                            ** 2
                        )
                        # The calculated 'mbt' values above should be identical to the
                        # corresponding retrieved_b_tot field values, plotted just
                        # below. If they are not, we have a problem!
                        mem_axs.errorbar(
                            time_utc[use_obs],
                            rtrv_bt[mem_ndx][use_obs],
                            yerr=rtrv_be[mem_ndx][use_obs],
                            color=FLD_CLR[NUM_FLD - 1],
                            linestyle="dotted",
                            errorevery=5,
                        )
                    else:
                        mbt = (
                            igrf_bf[fld_ndx][mem_ndx][use_obs]
                            + mem_dbs[fld_ndx][mem_ndx][use_obs]
                        )

                    mem_axs.plot(
                        time_utc[use_obs],
                        mbt,
                        # color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
                        color=FLD_CLR[fld_ndx],
                        linewidth=3.0,
                        linestyle="dotted",
                    )

                    new_sample = list(mbt) + list(igrf_bf[fld_ndx][mem_ndx][use_obs])
                    b_rng[fld_ndx, 0] = np.nanmin(
                        list(b_rng[fld_ndx, 0:1]) + new_sample
                    )
                    b_rng[fld_ndx, 1] = np.nanmax(
                        list(b_rng[fld_ndx, 1:2]) + new_sample
                    )
                    del new_sample

                else:
                    mem_axs.errorbar(
                        time_utc[use_obs],
                        mem_dbs[fld_ndx][mem_ndx][use_obs],
                        yerr=np.sqrt(mem_cov[fld_ndx][mem_ndx][use_obs]),
                        color=FLD_CLR[fld_ndx],
                        linestyle="solid",
                        errorevery=5,
                        label=f"{fld_cmp}",
                    )
                    mem_axs.plot(
                        time_utc[use_obs],
                        moving_average(
                            mem_dbs[fld_ndx][mem_ndx][use_obs],
                            ave_win,
                        ),
                        color="black" if not dark_mode else DARK_MODE_GRID_COLOR,
                        linestyle="solid",
                        lw=1.0,
                        label=f"{fld_cmp} - Running Average",
                        zorder=3,
                    )
                    b_rng[fld_ndx, 0] = np.nanmin(
                        list(b_rng[fld_ndx, 0:1])
                        + list(
                            mem_dbs[fld_ndx][mem_ndx][use_obs]
                            - np.sqrt(mem_cov[fld_ndx][mem_ndx][use_obs])
                        )
                    )
                    b_rng[fld_ndx, 1] = np.nanmax(
                        list(b_rng[fld_ndx, 1:2])
                        + list(
                            mem_dbs[fld_ndx][mem_ndx][use_obs]
                            + np.sqrt(mem_cov[fld_ndx][mem_ndx][use_obs])
                        )
                    )

                # Do _these_ things for both dBs and total Bs

                mem_axs.grid(axis="both")
                mem_axs.axhline(0, ls="dotted", color="black")
                mem_axs.text(
                    0.06,
                    0.99,
                    f"{fld_cmp}",
                    transform=mem_axs.transAxes,
                    ha="left",
                    va="top",
                    weight="bold",
                    size="large",
                    color=FLD_CLR[fld_ndx],
                )
                # Draw horizontal separators between stacked B field component subplots?
                # if row != 0:
                #     mem_axs.spines["top"].set_visible(False)
                # if row != rows - 1:
                #     mem_axs.spines["bottom"].set_visible(False)

                # Place title with receiver number at top of each column
                if row == 0:
                    mem_axs.set_title(
                        f"MEM {mem_num} - "
                        f"Look Direction {MEM_LOOK_DIRECTIONS[mem_ndx]}"
                        r"º",
                        # r"$\bf{^\circ}$",
                        weight="bold",
                        size="medium",
                    )
                # Label only leftmost column y axis
                if col != 0:
                    mem_axs.set_yticklabels([])
                # Label only bottom row x axis
                if row == rows - 1:
                    mem_axs.set_xlabel("Time (UTC)", weight="bold")

        # Adjust axis y limits to be uniform for a given field component across all MEMs
        for mem_ndx in [x - 1 for x in MEM_NUMBERS]:
            col = mem_ndx
            for fld_ndx, fld_cmp in enumerate(FLD_CMP):
                if fld_mod == DBS_MOD:
                    if fld_cmp == BT:
                        # No B_TOT plots for dBs, only for full field
                        continue
                    row = (mem_ndx // cols) * NUM_FLD + fld_ndx
                else:
                    # Stick B_total on top, even though it's last in list of components
                    row = (mem_ndx // cols) * NUM_FLD + (fld_ndx + 1) % NUM_FLD
                mem_axs = axs[row, col]
                # mem_axs.set_ylim([-2.0e4, 4.0e4])  # FIXME
                tot_rng = b_rng[fld_ndx][1] - b_rng[fld_ndx][0]
                mem_axs.set_ylim(
                    b_rng[fld_ndx][i] + tot_rng * x
                    for i, x in enumerate([-0.05, +0.05])
                )
                # Format the x-axis to show only hours and minutes
                if row == rows - 1:
                    set_xaxis_tick_format(
                        mem_axs,
                        max_ticks=16 // cols + 1,
                        rotation=45,
                        use_seconds=(time_utc[-1] - time_utc[0]).total_seconds() < 120,
                    )

        # Tweak position and add any figure-level annotation
        if fld_mod == TOT_MOD:
            axs[-1, -1].plot(
                time_utc[use_obs][0:2],
                [0, 0],
                color=("black" if not dark_mode else DARK_MODE_GRID_COLOR),
                linestyle="solid",
                linewidth=2,
                label="Background B vectors",
            )
            axs[-1, -1].plot(
                time_utc[use_obs][0:2],
                [0, 0],
                color=("black" if not dark_mode else DARK_MODE_GRID_COLOR),
                linestyle="dotted",
                linewidth=3,
                label="Retrieved B vectors",
            )
            axs[-1, -1].legend(loc="lower right")
            fig_title = "Background and Retrieved Magnetic Fields: "
            y_label = (
                "Reference (IGRF-14) and Retrieved "
                f"B$_\\mathbf{{{REFERENCE_ALTITUDE_KM}\\ km}}$"
                " Field Vectors (nT)"
            )
        else:
            fig_title = "Retrieved Magnetic Field Deltas: "
            y_label = (
                "Retrieved Current-Induced "
                f"B$_\\mathbf{{{REFERENCE_ALTITUDE_KM}\\ km}}$"
                " Field Vectors (nT)"
            )
        fig_title = fig_title + (
            f"{sc_id} - Orbit {orb_num}\n"
            f"{time_utc[use_obs][0].strftime('%Y-%m-%d (%j) %H:%M:%S')} - "
            f"{time_utc[use_obs][-1].strftime('%Y-%m-%d (%j) %H:%M:%S')}"
        )
        fig.suptitle(fig_title, weight="bold", fontsize="x-large", y=0.99, va="top")
        add_product_metadata(fig=fig, nc_data=nc_data, source=source)
        add_pipeline_metadata(fig=fig, nc_data=nc_data)
        overlay_ezie_logo(fig)
        fig.text(
            0.01,
            0.5,
            y_label,
            ha="left",
            va="center",
            rotation=90,
            weight="bold",
            size="large",
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
            plot_type=f"retrieved_b_fields_{fld_mod}",  # _{i0:04d}-{i1:04d}",
            dpi=figure_dpi,
        )


def plot_retrieved_bd_only(
    nc_data: Dataset,
    source: Path,
    save_directory: Path,
    version: str | None = None,
    mode: str = "Uncorrected",
    figure_dpi: int = DFLT_RES,
    dark_mode: bool = False,
    south_inverted: bool = False,
    overwrite: bool = False,
):
    """
    Plot the retrieved dB fields for each MEM in two formats, along with their estimated
    errors. Errors for individual dBs are calculated from the derived covariance values.

    Args:

    Returns:
        None
    """
    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
        plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR

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
    sc_id = nc_data["Metadata/SpaceVehicle"][0]

    # FIXME: Attempt to trim slewing observations at start and finish
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
        mem_dbd_val = [
            f"RetrievedParameters/retrieved_dbgeod{mm}" for mm in MEM_NUMBERS
        ]
        plot_type = "retrieved_b_fields_dBs"
    else:
        mem_dbd_val = [
            f"RetrievedParameters/{mode.lower()}_retrieved_dbgeod{mm}"
            for mm in MEM_NUMBERS
        ]
        plot_type = f"retrieved_{mode.lower()}_b_fields_dBs"

    ftgt, old_hash, new_hash = save_close_figure(
        source=source,
        save_directory=save_directory,
        obs_date=obs_date,
        spacecraft=sc_id,
        tstmp=t_stamp,
        plot_type=plot_type,
        name_only=True,
    )
    logger.debug(f"Checked file: {ftgt.as_posix()}")

    if ftgt.exists() and (old_hash == new_hash) and not overwrite:
        logger.info(f"File exists, source hash unchanged, skipping: {ftgt.as_posix()}")
        return
    if ftgt.exists() and (old_hash != new_hash):
        logger.info("Source file hash has changed, updating plot")

    mem_bdd_cov = [f"RetrievedParameters/cov_dd{mm}" for mm in MEM_NUMBERS]
    mem_obs_lat = [f"Geolocation/obs_lat{mm}" for mm in MEM_NUMBERS]
    mem_obs_lon = [f"Geolocation/obs_lon{mm}" for mm in MEM_NUMBERS]
    mem_mag_lat = [f"MagneticCoords/obs_maglat{mm}" for mm in MEM_NUMBERS]
    mem_mag_LTm = [f"MagneticCoords/obs_magLT{mm}" for mm in MEM_NUMBERS]

    # Stack MEM arrays so we can index and loop through them by number rather than using
    # 4 separate variable names.
    # FIXME - kludge to use old L2 file - comment out line below
    mem_dbd = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_dbd_val],))
    mem_cdd = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_bdd_cov],))
    mag_lat = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_mag_lat],))
    mag_ltm = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_mag_LTm],))
    obs_lat = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_obs_lat],))
    obs_lon = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_obs_lon],))

    for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
        missing_val = mem_dbd[mem_ndx, :] == NCDF_MISSING
        mem_dbd[mem_ndx, missing_val] = np.nan
        mag_lat[mem_ndx, missing_val] = np.nan
        mag_ltm[mem_ndx, missing_val] = np.nan
        obs_lat[mem_ndx, missing_val] = np.nan
        obs_lon[mem_ndx, missing_val] = np.nan

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

    # We'll save the total variable range for each field component across
    # all MEMs here, adjusting plot limits afterwards when we know what the
    # correct range is for the whole ensemble.
    b_rng = np.full((2), fill_value=np.nan)  # 4 B x [min,max]

    col = 0
    axs[-1, 1].set_visible(False)
    lat_axs = axs[-1, col]
    lat_axs.grid(axis="both")
    lat_axs.set_ylabel("Magnetic Latitude\n(APEX, degrees)", weight="bold")
    lat_axs.set_xlabel("Time (UTC)", weight="bold")
    mlt_axs = lat_axs.twinx()
    mlt_axs.set_ylim(-0.5, 24.5)
    mlt_axs.set_yticks(range(0, 25, 6))
    mlt_lbl = [
        "Midnight",
        "Dawn",
        "Noon",
        "Dusk",
        "Midnight",
    ]
    mlt_axs.set_yticklabels(mlt_lbl, size="x-small")
    mlt_axs.set_ylabel("Magnetic Local Time\n(APEX, hours)", weight="bold")
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
            yerr=np.sqrt(mem_cdd[mem_ndx][use_obs]),
            color=MEM_CLR[mem_ndx],
            linestyle="solid",
            errorevery=5,
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
            b_rng[0] = np.nanmin(
                list(b_rng[0:1])
                + list(mem_dbd[mem_ndx][use_obs] - np.sqrt(mem_cdd[mem_ndx][use_obs]))
            )
            b_rng[1] = np.nanmax(
                list(b_rng[1:2])
                + list(mem_dbd[mem_ndx][use_obs] + np.sqrt(mem_cdd[mem_ndx][use_obs]))
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
        mem_axs.set_ylabel(f"MEM {mem_num} dB$_\\mathbf{{D}}$ (nT)", weight="bold")
        mem_axs.text(
            0.01,
            0.99,
            f"Off-Nadir Angle: {MEM_LOOK_DIRECTIONS[mem_ndx]}"
            r"º",
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
    for mem_ndx in [x - 1 for x in MEM_NUMBERS]:
        mem_axs = axs[PLT_NDX[mem_ndx], col]
        tot_rng = b_rng[1] - b_rng[0]
        logger.debug(f"{b_rng[0]} : {b_rng[1]}")
        # FIXME - kludge to use old L2 file - comment out line below
        try:
            mem_axs.set_ylim(
                b_rng[i] + tot_rng * x for i, x in enumerate([-0.05, +0.05])
            )
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

    if lat_at_midpt > +40:  # hemisphere = NORTH
        hemisphere = NORTH
    elif lat_at_midpt < -40:  # hemisphere = SOUTH
        hemisphere = SOUTH
    else:
        hemisphere = None

    if hemisphere == NORTH:
        proj_method = ccrs.NorthPolarStereo(central_longitude=0)  # 0 is default
    elif hemisphere == SOUTH:
        proj_method = ccrs.SouthPolarStereo(central_longitude=180)
        if south_inverted:
            MLT_sign = -1
    else:
        proj_method = ccrs.Orthographic(
            central_latitude=lat_at_midpt, central_longitude=lon_at_midpt
        )

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
        x_0, y_0 = proj_method.transform_point(
            0,
            90 if hemisphere == NORTH else -90,
            src_crs=DATA_TRANSFORM,
        )
        x_1, y_1 = proj_method.transform_point(
            45,
            MAG_LAT_LOWER_LIMIT if hemisphere == NORTH else -MAG_LAT_LOWER_LIMIT,
            src_crs=DATA_TRANSFORM,
        )
        map_meters = np.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2)
        map_axs.set_extent(
            (
                -map_meters,
                +map_meters,
                -map_meters,
                +map_meters,
            ),
            crs=proj_method,
        )
        # Compute a circle in axes coordinates that will be used as a clipping boundary
        # for the map.
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        map_axs.set_boundary(circle, transform=map_axs.transAxes)

    # Map S/C and MEM footprint lat/lon
    dtlim = [time_utc[0], time_utc[1]]
    duration = (dtlim[1] - dtlim[0]).total_seconds()
    midpoint = dtlim[0] + datetime.timedelta(seconds=int(0.5 * duration))

    # Get solar position, first in geodetic and then in APEX magnetic coordinates.
    sun_geo_tuple_rad = []

    # Geodetic
    for tndx, tutc in enumerate(time_utc):
        tdb = spiceypy.utc2et(tutc.isoformat()[:-6])
        (subpnt, epoch, to_subpnt) = spiceypy.subslr(
            "INTERCEPT/ELLIPSOID", "EARTH", tdb, "IAU_EARTH", "LT+S", "EARTH"
        )
        sun_geo_tuple_rad.append(
            spiceypy.recgeo(subpnt, EARTH_RADIUS_EQUATORIAL, EARTH_FLATTENING)
        )  # Fix the subsolar point on the surface in geodetic coordinates
    sun_geo_lon_deg = np.array([np.degrees(x[0]) for x in sun_geo_tuple_rad])
    sun_geo_lat_deg = np.array([np.degrees(x[1]) for x in sun_geo_tuple_rad])

    # Compute the subsolar point in APEX magnetic coordinates
    apex = Apex(date=midpoint.year, refh=0)
    sun_mlat, sun_mlon = apex.geo2apex(
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
        )
        map_sc_mem_footprints_magnetic(
            map_axs=map_axs,
            sat_lat=sat_maglat,
            sat_lon=MLT_sign * sat_magLT_deg,
            obs_lat=mag_lat[:, :].T,
            obs_lon=MLT_sign * mag_ltm[:, :].T * 15,  # Convert hours to degrees
            at_time=midpoint,
            dark_mode=False,
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
            dark_mode=False,
            terminator=True,
            zorder=4,
            small_text=False,
            plain=False,
            legend_top=False,
            mid_lat_mag=True,
        )

    # Add MEM beam numbering/pointing diagram at bottom right corner of figure
    fw = fig.get_figwidth()
    fh = fig.get_figheight()
    fig_aspect_ratio = fw / fh
    # ll, bb, ww, hh = map_axs.get_position().bounds
    mem_beam_img = plt.imread(
        Path(__file__).parent.parent / "binary-assets" / "mem-beam-diagram.png"
    )
    img_aspect_ratio = mem_beam_img.shape[1] / mem_beam_img.shape[0]
    hi = 0.29
    wi = hi * img_aspect_ratio / fig_aspect_ratio
    # img_axs = fig.add_axes([0.975 - wi, 0.025 * fig_aspect_ratio, wi, hi])  # LR
    img_axs: plt.Axes = fig.add_axes(
        rect=(0.76 - wi / 2, 0.025 * fig_aspect_ratio, wi, hi)
    )
    img_axs.imshow(mem_beam_img, aspect="auto")
    # Just turn ticks off so we get a border around image.
    img_axs.set_xticks([])
    img_axs.set_yticks([])
    # img_axs.axis("off")

    # Tweak position and add any figure-level annotation
    fig_title = f"{mode} Retrieved Magnetic Field Deltas: "
    fig_title = fig_title + (
        f"{sc_id} - Orbit {orb_num}\n"
        f"{time_utc[use_obs][0].strftime('%Y-%m-%d (%j) %H:%M:%S')} - "
        f"{time_utc[use_obs][-1].strftime('%Y-%m-%d (%j) %H:%M:%S')}"
    )
    fig.suptitle(fig_title, weight="bold", fontsize="x-large", y=0.99, va="top")
    add_product_metadata(fig=fig, nc_data=nc_data, source=source)
    add_pipeline_metadata(fig=fig, nc_data=nc_data)
    overlay_ezie_logo(fig)
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
    """Map geolocated S/C and MEM footprints

    Args:
        map_axs (_type_, optional): _description_. Defaults to None.
        sat_lat (np.ndarray, optional): _description_. Defaults to None.
        sat_lon (np.ndarray, optional): _description_. Defaults to None.
        obs_lat (np.ndarray, optional): _description_. Defaults to None.
        obs_lon (np.ndarray, optional): _description_. Defaults to None.
        at_time (datetime.datetime, optional): _description_. Defaults to None.
        terminator (bool, optional): Show day/night with Nightshade. Defaults to False.
        dark_mode (bool, optional): Use 'dark_background' style. Defaults to False.
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
        facecolor="#d0c0a0" if dark_mode else "#d0c0a0",
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
    """Map geolocated S/C and MEM footprints

    Args:
        map_axs (_type_, optional): _description_. Defaults to None.
        sat_lat (np.ndarray, optional): _description_. Defaults to None.
        sat_lon (np.ndarray, optional): _description_. Defaults to None.
        obs_lat (np.ndarray, optional): _description_. Defaults to None.
        obs_lon (np.ndarray, optional): _description_. Defaults to None.
        at_time (datetime.datetime, optional): _description_. Defaults to None.
        terminator (bool, optional): Show day/night with Nightshade. Defaults to False.
        dark_mode (bool, optional): Use 'dark_background' style. Defaults to False.
    """
    if (hemisphere is None) or (hemisphere == NORTH) or south_inverted:
        ccw = +1
    else:
        ccw = -1
    lats_n = np.arange(
        MAG_LAT_LOWER_LIMIT, 90.0, 10.0
    )  # Lats at which to draw gridlines

    if south_inverted:
        cardinal_labels = dict(south="N", north="S")
    else:
        cardinal_labels = dict(south="S", north="N")

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
        mlt_desc = {}
        mlt_ha = {}
        mlt_va = {}
        for mm, mlt in enumerate(mlt_locs):
            mlt_desc[f"{mlt:.0f}"] = f"{mlt / 15:02.0f}H"
            mlt_ha[f"{mlt:.0f}"] = "center"
            mlt_va[f"{mlt:.0f}"] = "center"
        mlt_desc["0"] = "Midnight"
        mlt_desc["90"] = "Dawn"
        mlt_desc["180"] = "Noon"
        mlt_desc["270"] = "Dusk"
        mlt_ha["0"] = "center"
        mlt_ha["90"] = "center"
        mlt_ha["180"] = "center"
        mlt_ha["270"] = "center"
        mlt_va["0"] = "center"
        mlt_va["90"] = "center"
        mlt_va["180"] = "center"
        mlt_va["270"] = "center"

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
                va=mlt_va[f"{mlt:.0f}"],
                horizontalalignment=mlt_ha[f"{mlt:.0f}"],
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


def plot_mag_and_geo_maps(
    nc_data: Dataset,
    source: Path,
    save_directory: Path,
    figure_dpi: int = 300,
    dark_mode: bool = False,
    south_inverted: bool = False,
    overwrite: bool = False,
):
    """
    Purpose:

    Args:

    Returns:
        None
    """
    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
        plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR

    try:
        time_utc, obs_date = get_datetime_from_utc_string(nc_data.groups["Time"])
    except Exception as exc:
        logger.error(f"Exception encountered: {exc}")
        logger.error("Processing skipped--truncated or corrupted data file?")
        logger.error(f"Problem file (datetime values): {source.as_posix()}")
        return

    t_stamp = time_utc[0].strftime("%H%M%S")
    orb_num = nc_data["Science/orbit_number"][0]
    sc_id = nc_data["Metadata/SpaceVehicle"][0]

    # FIXME: Attempt to trim slewing observations at start and finish
    # if len(time_utc) > 12:
    #     use_obs = np.s_[5:-5]
    # else:
    use_obs = np.s_[:]

    plot_type = "mag_and_geo_maps"
    ftgt, old_hash, new_hash = save_close_figure(
        source=source,
        save_directory=save_directory,
        obs_date=obs_date,
        spacecraft=sc_id,
        tstmp=t_stamp,
        plot_type=plot_type,
        name_only=True,
    )
    logger.debug(f"Checked file: {ftgt.as_posix()}")

    if ftgt.exists() and (old_hash == new_hash) and not overwrite:
        logger.info(f"File exists, source hash unchanged, skipping: {ftgt.as_posix()}")
        return
    if ftgt.exists() and (old_hash != new_hash):
        logger.info("Source file hash has changed, updating plot")

    logger.info("Generating plot of MEM and spacecraft footprints, geo and mag coords")

    mem_obs_lat = [f"Geolocation/obs_lat{mm}" for mm in MEM_NUMBERS]
    mem_obs_lon = [f"Geolocation/obs_lon{mm}" for mm in MEM_NUMBERS]
    mem_mag_lat = [f"MagneticCoords/obs_maglat{mm}" for mm in MEM_NUMBERS]
    mem_mag_LTm = [f"MagneticCoords/obs_magLT{mm}" for mm in MEM_NUMBERS]

    # Stack MEM arrays so we can index and loop through them by number rather than using
    # 4 separate variable names.
    mag_lat = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_mag_lat],))
    mag_ltm = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_mag_LTm],))
    obs_lat = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_obs_lat],))
    obs_lon = np.vstack((*[nc_data[nc_fld] for nc_fld in mem_obs_lon],))

    for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
        missing_val = obs_lat[mem_ndx, :] == NCDF_MISSING
        mag_lat[mem_ndx, missing_val] = np.nan
        mag_ltm[mem_ndx, missing_val] = np.nan
        obs_lat[mem_ndx, missing_val] = np.nan
        obs_lon[mem_ndx, missing_val] = np.nan

    # Plot magnetic field reference values and deltas from EZIE OSSE retrieval along
    # EZIE MEM lines of sight.
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
    for col in range(ncols):
        for row in range(nrows):
            axs[row, col].set_visible(False)

    col = 0
    axs[-1, col].set_visible(True)
    lat_axs = axs[-1, col]
    lat_axs.grid(axis="both")
    lat_axs.set_ylabel("Magnetic Latitude\n(APEX, degrees)", weight="bold")
    lat_axs.set_xlabel("Time (UTC)", weight="bold")
    mlt_axs = lat_axs.twinx()
    mlt_axs.set_ylim(-0.5, 24.5)
    mlt_axs.set_yticks(range(0, 25, 6))
    mlt_lbl = [
        "Midnight",
        "Dawn",
        "Noon",
        "Dusk",
        "Midnight",
    ]
    mlt_axs.set_yticklabels(mlt_lbl, size="x-small")
    mlt_axs.set_ylabel("Magnetic Local Time\n(APEX, hours)", weight="bold")
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

        mem_axs.grid(axis="both")
        mem_axs.axhline(0, ls="dotted", color="black")

        # Place label with receiver number at top of each column
        mem_axs.set_ylabel(f"MEM {mem_num} dB$_\\mathbf{{D}}$ (nT)", weight="bold")
        mem_axs.text(
            0.01,
            0.99,
            f"Off-Nadir Angle: {MEM_LOOK_DIRECTIONS[mem_ndx]}"
            r"º",
            weight="bold",
            size="medium",
            va="top",
            ha="left",
            transform=mem_axs.transAxes,
        )

        # Label only leftmost column y axis
        if col == 0 and row == 0:
            mem_axs.set_title(
                "MEM and SV footprint locations",
                weight="bold",
                size="medium",
            )

    # Adjust axis y limits to be uniform for a given field component across all MEMs
    for mem_ndx in [x - 1 for x in MEM_NUMBERS]:
        mem_axs = axs[PLT_NDX[mem_ndx], col]

    # Format the x-axis to show only hours and minutes
    # Label only bottom row left time/x axis
    set_xaxis_tick_format(
        axs[-1, 0],
        max_ticks=20 // ncols + 1,
        rotation=45,
        use_seconds=(time_utc[-1] - time_utc[0]).total_seconds() < 120,
    )

    # NEW geomagnetic coordinate map inset
    midpt = len(time_utc) // 2
    lat_at_midpt = nc_data["Geolocation/sat_lat"][midpt]
    lon_at_midpt = nc_data["Geolocation/sat_lon"][midpt]

    # Map S/C and MEM footprint lat/lon
    dtlim = [time_utc[0], time_utc[1]]
    duration = (dtlim[1] - dtlim[0]).total_seconds()
    midpoint = dtlim[0] + datetime.timedelta(seconds=int(0.5 * duration))

    # Add magnetic coordinate overlay, select latitudes at which to draw gridlines
    if lat_at_midpt > +40:  # hemisphere = NORTH
        hemisphere = NORTH
        lats_p = np.arange(+MAG_LAT_LOWER_LIMIT, +81.0, +10.0)
    elif lat_at_midpt < -40:  # hemisphere = SOUTH
        hemisphere = SOUTH
        lats_p = np.arange(-MAG_LAT_LOWER_LIMIT, -81.0, -10.0)
    else:
        hemisphere = None
        lats_p = np.array([-20.0, -10.0, +10.0, +20.0])

    # Get solar position, first in geodetic and then in APEX magnetic coordinates.
    sun_geo_tuple_rad = []

    # Geodetic
    for tndx, tutc in enumerate(time_utc):
        tdb = spiceypy.utc2et(tutc.isoformat()[:-6])
        (subpnt, epoch, to_subpnt) = spiceypy.subslr(
            "INTERCEPT/ELLIPSOID", "EARTH", tdb, "IAU_EARTH", "LT+S", "EARTH"
        )
        sun_geo_tuple_rad.append(
            spiceypy.recgeo(subpnt, EARTH_RADIUS_EQUATORIAL, EARTH_FLATTENING)
        )  # Fix the subsolar point on the surface in geodetic coordinates
    sun_geo_lat_deg = np.array([np.degrees(x[1]) for x in sun_geo_tuple_rad])
    sun_geo_lon_deg = np.array([np.degrees(x[0]) for x in sun_geo_tuple_rad])
    sun_geo_lat_mid = sun_geo_lat_deg[sun_geo_lat_deg.shape[0] // 2]
    sun_geo_lon_mid = sun_geo_lon_deg[sun_geo_lon_deg.shape[0] // 2]

    apex = Apex(date=midpoint.year, refh=0)
    sun_mlat, sun_mlon = apex.geo2apex(
        sun_geo_lat_deg,
        sun_geo_lon_deg,
        REFERENCE_ALTITUDE_KM,
    )  # Compute the subsolar point in APEX magnetic coordinates
    sun_mag_lat_mid = sun_mlat[sun_mlat.shape[0] // 2]
    sun_mag_lon_mid = sun_mlon[sun_mlon.shape[0] // 2]

    # Compute satellite footprint in APEX magnetic coordinates
    sat_maglat, sat_maglon = apex.geo2apex(
        nc_data["Geolocation/sat_lat"][:],
        nc_data["Geolocation/sat_lon"][:],
        nc_data["Geolocation/sat_alt"][:],
    )

    # Convert magnetic longitude to MLT
    sat_magLT_deg = 180.0 + sat_maglon - sun_mlon

    # Debugging "mirrored" southern hemisphere plots
    logger.debug(f"{hemisphere} {lat_at_midpt}")
    logger.debug(f"{np.min(sun_geo_lat_deg)}, {np.max(sun_geo_lat_deg)}")
    logger.debug(f"{np.min(sun_geo_lon_deg)}, {np.max(sun_geo_lon_deg)}")
    logger.info(f"Hemisphere is {hemisphere}")
    logger.info(
        f"Sun geodetic latitude, longitude = "
        f"{sun_geo_lat_mid:7.2f}, "
        f"{sun_geo_lon_mid:7.2f}"
    )
    logger.info(
        f"Sun magnetic latitude, longitude = "
        f"{sun_mag_lat_mid:7.2f}, "
        f"{sun_mag_lon_mid:7.2f}"
    )
    logger.debug(f"{np.min(sat_magLT_deg)}, {np.max(sat_magLT_deg)}")

    # NEW geomagnetic coordinate map inset
    MLT_sign = +1
    if hemisphere == NORTH:
        proj_method_mag = ccrs.NorthPolarStereo(central_longitude=0)  # 0 is default
        proj_method_geo = ccrs.NorthPolarStereo(central_longitude=sun_geo_lon_mid + 180)
    elif hemisphere == SOUTH:
        if south_inverted:
            MLT_sign = -1
        proj_method_mag = ccrs.SouthPolarStereo(central_longitude=180)
        proj_method_geo = ccrs.SouthPolarStereo(
            central_longitude=MLT_sign * sun_geo_lon_mid
        )
    else:
        proj_method_mag = ccrs.Orthographic(
            central_latitude=lat_at_midpt, central_longitude=lon_at_midpt
        )
        proj_method_geo = ccrs.Orthographic(
            central_latitude=lat_at_midpt, central_longitude=lon_at_midpt
        )

    maprowspan, mapcolspan = 3, 1  # 5 rows for geographic coordinate inset
    geo_axs: GeoAxes = plt.subplot2grid(
        (nrows, ncols),
        (0, 0),
        fig=fig,
        projection=proj_method_geo,
        rowspan=maprowspan,
        colspan=mapcolspan,
    )  # ty:ignore[invalid-assignment]

    if hemisphere is not None:
        x_0, y_0 = proj_method_geo.transform_point(
            0,
            90 if hemisphere == NORTH else -90,
            src_crs=DATA_TRANSFORM,
        )
        x_1, y_1 = proj_method_geo.transform_point(
            45,
            MAG_LAT_LOWER_LIMIT if hemisphere == NORTH else -MAG_LAT_LOWER_LIMIT,
            src_crs=DATA_TRANSFORM,
        )
        map_meters = np.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2)
        geo_axs.set_extent(
            (
                -map_meters,
                +map_meters,
                -map_meters,
                +map_meters,
            ),
            crs=proj_method_geo,
        )

        # Compute a circle in axes coordinates that will be used as a clipping boundary
        # for the map.
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        geo_axs.set_boundary(circle, transform=geo_axs.transAxes)

        logger.info("Mapping continents in reversed longitude coordinates")
        if south_inverted:
            map_inverted_continents(
                ax=geo_axs,
            )
        map_sc_mem_footprints_magnetic(
            map_axs=geo_axs,
            sat_lat=nc_data["Geolocation/sat_lat"][use_obs],
            sat_lon=MLT_sign * nc_data["Geolocation/sat_lon"][use_obs],
            obs_lat=obs_lat[:, use_obs].T,
            obs_lon=MLT_sign * obs_lon[:, use_obs].T,
            at_time=midpoint,
            dark_mode=False,
            zorder=4,
            small_text=False,
            plain=False,
            geo_labels=True,
            geo_offset=MLT_sign * sun_geo_lon_mid,
            legend_top=False,
            hemisphere=hemisphere,
            south_inverted=(south_inverted and (hemisphere == SOUTH)),
        )
        plot_geomagnetic_references(
            geo_axs,
            midpoint,
            latitudes=lats_p,
            south_inverted=south_inverted,
            lat_clr="red" if not dark_mode else DARK_MODE_GRID_COLOR,
        )

    # =================================================================================
    # fig.autofmt_xdate()
    # plt.subplots_adjust(
    #     left=0.06,
    #     right=0.99,
    #     bottom=0.10,
    #     top=0.91,
    #     hspace=0.00,
    #     wspace=0.05,
    # )
    # save_close_figure(
    #     source=source,
    #     save_directory=save_directory,
    #     figure=fig,
    #     obs_date=obs_date,
    #     spacecraft=sc_id,
    #     # orbit=orb_num,
    #     tstmp=t_stamp,
    #     plot_type=plot_type,
    #     dpi=300,
    # )
    # return
    # =================================================================================

    maprowspan, mapcolspan = 3, 1  # 5 rows for geographic coordinate inset
    mag_axs: GeoAxes = plt.subplot2grid(
        shape=(nrows, ncols),
        loc=(0, ncols - mapcolspan),
        fig=fig,
        projection=proj_method_mag,
        rowspan=maprowspan,
        colspan=mapcolspan,
    )  # ty:ignore[invalid-assignment]

    # logger.info("Adding geographic coordinate SZA overlay")
    # sza_overlay(
    #     geo_axs,
    #     MLT_sign,
    #     np.radians(sun_geo_lat_mid),
    #     np.radians(sun_geo_lon_mid),
    # )
    # logger.info("Adding APEX magnetic coordinate SZA overlay")
    # sza_overlay(
    #     mag_axs,
    #     MLT_sign,
    #     np.radians(sun_geo_lat_mid),
    #     np.radians(sun_geo_lon_mid),
    #     sun_mlon_deg=sun_mag_lon_mid,
    #     time4mag=midpoint,
    #     magnetic=True,
    # )

    if hemisphere is not None:
        # if (hemisphere == NORTH) or south_inverted:
        #     mag_axs.set_extent(
        #         extents=[-180, 180, MAG_LAT_LOWER_LIMIT, +90], crs=DATA_TRANSFORM
        #     )
        # else:
        #     mag_axs.set_extent(
        #         extents=[-180, 180, -90, -MAG_LAT_LOWER_LIMIT], crs=DATA_TRANSFORM
        #     )

        x_0, y_0 = proj_method_mag.transform_point(
            0,
            90 if hemisphere == NORTH else -90,
            src_crs=DATA_TRANSFORM,
        )
        x_1, y_1 = proj_method_mag.transform_point(
            45,
            MAG_LAT_LOWER_LIMIT if hemisphere == NORTH else -MAG_LAT_LOWER_LIMIT,
            src_crs=DATA_TRANSFORM,
        )
        map_meters = np.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2)
        mag_axs.set_extent(
            (
                -map_meters,
                +map_meters,
                -map_meters,
                +map_meters,
            ),
            crs=proj_method_mag,
        )

        # Compute a circle in axes coordinates that will be used as a clipping boundary
        # for the map.
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        mag_axs.set_boundary(circle, transform=mag_axs.transAxes)

    # Transform continent outlines from geographic to APEX magnetic coordinates
    if hemisphere is not None:
        logger.info("Mapping continents in APEX magnetic coordinates")
        map_magnetic_continents(
            ax=mag_axs,
            time4mag=midpoint,
            sun_mlon=sun_mlon[len(sun_mlon) // 2],
            alt4mag=REFERENCE_ALTITUDE_KM,
            south_inverted=(south_inverted and (hemisphere == SOUTH)),
        )
        map_sc_mem_footprints_magnetic(
            map_axs=mag_axs,
            sat_lat=sat_maglat,
            sat_lon=MLT_sign * sat_magLT_deg,
            obs_lat=mag_lat[:, :].T,
            obs_lon=MLT_sign * mag_ltm[:, :].T * 15,  # Convert hours to degrees
            at_time=midpoint,
            dark_mode=False,
            zorder=4,
            small_text=False,
            plain=False,
            legend_top=False,
            hemisphere=hemisphere,
            south_inverted=(south_inverted and (hemisphere == SOUTH)),
        )

    # Add MEM beam numbering/pointing diagram at bottom right corner of figure
    fw = fig.get_figwidth()
    fh = fig.get_figheight()
    fig_aspect_ratio = fw / fh
    mem_beam_img = plt.imread(
        Path(__file__).parent.parent / "binary-assets" / "mem-beam-diagram.png"
    )
    img_aspect_ratio = mem_beam_img.shape[1] / mem_beam_img.shape[0]
    hi = 0.31
    wi = hi * img_aspect_ratio / fig_aspect_ratio
    # img_axs = fig.add_axes([0.975 - wi, 0.025 * fig_aspect_ratio, wi, hi])  # LR
    img_axs = fig.add_axes(rect=(0.76 - wi / 2, 0.025 * fig_aspect_ratio, wi, hi))
    img_axs.imshow(mem_beam_img, aspect="auto")
    # Just turn ticks off so we get a border around image, not entire axis.
    # img_axs.axis("off")
    img_axs.set_xticks([])
    img_axs.set_yticks([])

    # Tweak position and add any figure-level annotation
    fig_title = (
        "Spacecraft and MEM Footprints, Geographic (L) and Magnetic (R) Coordinates - "
    )
    fig_title = fig_title + (
        f"{sc_id} - Orbit {orb_num}\n"
        f"{time_utc[use_obs][0].strftime('%Y-%m-%d (%j) %H:%M:%S')} - "
        f"{time_utc[use_obs][-1].strftime('%Y-%m-%d (%j) %H:%M:%S')}"
    )
    fig.suptitle(fig_title, weight="bold", fontsize="x-large", y=0.99, va="top")

    add_product_metadata(fig=fig, nc_data=nc_data, source=source)
    add_pipeline_metadata(fig=fig, nc_data=nc_data)
    overlay_ezie_logo(fig)

    fig.autofmt_xdate()
    plt.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.10,
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
        # orbit=orb_num,
        tstmp=t_stamp,
        plot_type=plot_type,
        overwrite=overwrite,
        dpi=figure_dpi,
        dark_mode=dark_mode,
    )


def sza_overlay(
    map_axs,
    MLT_sign,
    sun_lat_rad,
    sun_lon_rad,
    sun_mlon_deg=None,
    magnetic=False,
    time4mag=None,
):
    # Compute dot product of normal vector at each location on earth with the vector to
    # the sun at the appropriate time to establish SZA grid.
    lonvec = np.radians(np.linspace(-180, 180, 91))
    latvec = np.radians(np.linspace(-90, 90, 46))
    # lonvec = np.radians(np.linspace(-180, 180, 121))
    # latvec = np.radians(np.linspace(-90, 90, 61))
    # lonvec = np.radians(np.linspace(-180, 180, 181))
    # latvec = np.radians(np.linspace(-90, 90, 91))
    longrd, latgrd = np.meshgrid(lonvec, latvec)
    pgrd = np.array(
        [
            np.cos(MLT_sign * longrd) * np.cos((latgrd)),
            np.sin(MLT_sign * longrd) * np.cos((latgrd)),
            np.sin((latgrd)),
        ]
    ).T
    psun = np.array(
        [
            np.cos(MLT_sign * sun_lon_rad) * np.cos((sun_lat_rad)),
            np.sin(MLT_sign * sun_lon_rad) * np.cos((sun_lat_rad)),
            np.sin((sun_lat_rad)),
        ]
    )
    szagrd = np.degrees(np.acos(np.dot(pgrd, psun))).T

    # Transform coordintes to magnetic lat/lon if required. note that this must be done
    # AFTER computation of the dot product. The magnetic coordinate system is not
    # orthogonal and regular (TODO: describe this better).
    if magnetic:
        logger.info("Performing magnetic coordinate transformation")
        bgn = time.perf_counter()

        apex = Apex(date=time4mag.year, refh=0)
        lat_map, lon_map = apex.geo2apex(
            np.degrees(latgrd).ravel(),
            np.degrees(longrd).ravel(),
            REFERENCE_ALTITUDE_KM,
        )
        # lat_map, lon_map = apex.gg_gm_apex(
        #     int(time4mag.year),
        #     np.degrees(latgrd).ravel(),
        #     np.degrees(longrd).ravel(),
        #     REFERENCE_ALTITUDE_KM,
        #     apex.GEO_TO_MAG,
        # )

        lat_map = np.reshape(lat_map, latgrd.shape)
        lon_map = np.reshape(lon_map, longrd.shape) + 180 - sun_mlon_deg
        end = time.perf_counter()
        logger.info(f"Elapsed time: {end - bgn:.2f} seconds")
    else:
        lat_map = np.degrees(latgrd)
        lon_map = np.degrees(longrd)

    if magnetic:
        map_axs.pcolormesh(
            MLT_sign * lon_map,
            lat_map,
            180.0 - szagrd,
            shading="auto",
            cmap="afmhot",
            alpha=0.9,
            transform=DATA_TRANSFORM,
        )

    else:
        map_axs.contourf(
            MLT_sign * lon_map,
            lat_map,
            180.0 - szagrd,
            levels=90,
            cmap="afmhot",
            alpha=0.90,
            transform=DATA_TRANSFORM,
        )
    conval = map_axs.contour(
        MLT_sign * lon_map,
        lat_map,
        szagrd,
        levels=np.linspace(0, 180, 19),
        colors="k",
        linestyles=":",
        transform=DATA_TRANSFORM,
        alpha=0.5,
    )
    map_axs.clabel(
        conval,
        colors="#808080",
        fontsize="small",
    )


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
    plot_type: str = "l3-l2"

    # See if this figure already exists.
    # Do NOT remake an existing figure unless the overwrite flag is set.
    ftgt, old_hash, new_hash = save_close_figure(
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

    if ftgt.exists() and (old_hash == new_hash) and not overwrite:
        logger.info(f"File exists, source hash unchanged, skipping: {ftgt.as_posix()}")
        return
    if ftgt.exists() and (old_hash != new_hash):
        logger.info("Source file hash has changed, updating plot")

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

    add_product_metadata(fig=fig, nc_data=nc_data, source=source, size="x-small")
    add_pipeline_metadata(
        fig=fig,
        nc_data=nc_data,
        git_branch=git_branch,
        git_commit=git_commit,
        size="x-small",
    )

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

    Parameters
    ----------
    run_date : str
        The date to process, formatted as 'YYYYMMDD'.
    used_files : list
        A list of file paths (Path objects or strings) pointing to the L1 NetCDF
        files to be ingested.

    Returns
    -------
    tuple
        A tuple containing:
        - full_day (dict): A dictionary where keys are variable names (e.g., 'sat_lat',
          'obs_maglat1') and values are numpy arrays of the aggregated data for all
          spacecraft and orbits on the given date.
        - uniq_svid (numpy.ndarray): An array of unique spacecraft identifiers found
          in the processed files.
        - uniq_orbs (numpy.ndarray): An array of unique orbit numbers found in the
          processed files.
    """

    # Generate dictionary of all locations sampled by the EZIE spacecraft in the course
    # of one day. The dictionary key hierarchy is:
    # 1) SV
    # 2) L1 var names (from coverage_db_list below)
    coverage_db_list = [
        "/Metadata/SpaceVehicle",
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
        if prsd.spcv not in date_dict.keys():
            date_dict[prsd.spcv] = {}
        if "l1" in each_file.stem.lower():
            try:
                with Dataset(each_file, mode="r") as nc_data:
                    product = nc_data.getncattr("product")
                    if product == "L1":
                        logger.info(f"Processing {product} file {Path(each_file).name}")

                        if prsd.orbt not in date_dict[prsd.spcv].keys():
                            date_dict[prsd.spcv][prsd.orbt] = {}
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
                    else:
                        logger.error(f"{product} misidentified as L1-skipping")
                        date_dict[prsd.spcv][prsd.orbt][dbvar] = None
            except Exception as exc:
                logger.error(f"Unrecoverable problem, skipping file: {exc}")
                date_dict[prsd.spcv][prsd.orbt][dbvar] = None

    # Combine entries from each SV/L1 file into single unified dictionary
    for sv_id, sv_dict in date_dict.items():
        for orb_id, orb_dict in sv_dict.items():
            for dbvar in coverage_db_list:
                full_key = dbvar.split("/")[-1]  # Use only variable name, not group
                if full_key not in full_day.keys():
                    full_day[full_key] = []
                try:
                    if orb_dict[dbvar] is not None:
                        full_day[full_key].extend(orb_dict[dbvar])
                except Exception as exc:
                    logger.error(
                        f"Selected field could not be extracted from file: {exc}"
                    )

    # Recast each dictionary value (list) as a numpy array
    for sv_id, sv_dict in date_dict.items():
        for orb_id, orb_dict in sv_dict.items():
            for dbvar in coverage_db_list:
                full_key = dbvar.split("/")[-1]  # Use only variable name, not group
                full_day[full_key] = np.array(full_day[full_key])

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


def plot_retrieved_B_and_J(
    nc2_data: Dataset,
    nc2_sorc: Path,
    save_directory: Path,
    version: str | None = None,
    mode: str = "Uncorrected",
    figure_dpi: int = DFLT_RES,
    dark_mode: bool = False,
    south_inverted: bool = False,
    overwrite: bool = False,
):
    """
    Plot the retrieved dB fields for each MEM in two formats, along with their estimated
    errors. Errors for individual dBs are calculated from the derived covariance values.

    Args:

    Returns:
        None
    """
    nc3_sorc: Path = Path(nc2_sorc.as_posix().replace("l2", "l3"))

    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
        plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR

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
    sc_id = nc2_data["Metadata/SpaceVehicle"][0]

    # FIXME: Attempt to trim slewing observations at start and finish
    # if len(time_utc) > 12:
    #     use_obs = np.s_[5:-5]
    # else:
    #     use_obs = np.s_[:]
    use_obs = np.s_[:]  # See how things look without the haircut now

    # Define smoothing parameters for curve to be overlain on (noisy) dB plots.
    max_win = len(time_utc[use_obs])
    ave_win = min(AVERAGING_WINDOW, max_win)

    # Extract MEM retrieved dBs, geolocation and magnetic coordinates, covariance fields
    mem_dbd_val = [f"RetrievedParameters/retrieved_dbgeod{mm}" for mm in MEM_NUMBERS]
    plot_type = "retrieved_b_and_j"

    ftgt, old_hash, new_hash = save_close_figure(
        source=nc2_sorc,
        save_directory=save_directory,
        obs_date=obs_date,
        spacecraft=sc_id,
        tstmp=t_stamp,
        plot_type=plot_type,
        name_only=True,
    )
    logger.debug(f"Checked file: {ftgt.as_posix()}")

    if ftgt.exists() and (old_hash == new_hash) and not overwrite:
        logger.info(f"File exists, source hash unchanged, skipping: {ftgt.as_posix()}")
        return
    if ftgt.exists() and (old_hash != new_hash):
        logger.info("Source file hash has changed, updating plot")

    mem_bdd_cov = [f"RetrievedParameters/cov_dd{mm}" for mm in MEM_NUMBERS]
    mem_obs_lat = [f"Geolocation/obs_lat{mm}" for mm in MEM_NUMBERS]
    mem_obs_lon = [f"Geolocation/obs_lon{mm}" for mm in MEM_NUMBERS]
    mem_mag_lat = [f"MagneticCoords/obs_maglat{mm}" for mm in MEM_NUMBERS]
    mem_mag_LTm = [f"MagneticCoords/obs_magLT{mm}" for mm in MEM_NUMBERS]

    # Stack MEM arrays so we can index and loop through them by number rather than using
    # 4 separate variable names.
    # FIXME - kludge to use old L2 file - comment out line below
    mem_dbd = np.vstack((*[nc2_data[nc_fld] for nc_fld in mem_dbd_val],))
    mem_cdd = np.vstack((*[nc2_data[nc_fld] for nc_fld in mem_bdd_cov],))
    mag_lat = np.vstack((*[nc2_data[nc_fld] for nc_fld in mem_mag_lat],))
    mag_ltm = np.vstack((*[nc2_data[nc_fld] for nc_fld in mem_mag_LTm],))
    obs_lat = np.vstack((*[nc2_data[nc_fld] for nc_fld in mem_obs_lat],))
    obs_lon = np.vstack((*[nc2_data[nc_fld] for nc_fld in mem_obs_lon],))

    for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
        missing_val = mem_dbd[mem_ndx, :] == NCDF_MISSING
        mem_dbd[mem_ndx, missing_val] = np.nan
        mag_lat[mem_ndx, missing_val] = np.nan
        mag_ltm[mem_ndx, missing_val] = np.nan
        obs_lat[mem_ndx, missing_val] = np.nan
        obs_lon[mem_ndx, missing_val] = np.nan

    valid_pnts = np.sum(~np.isnan(mem_dbd[0, :]))
    if valid_pnts < 3:
        logger.debug(~np.isnan(mem_dbd[0, :]))
        logger.error(
            f"Insufficient non-NaN samples to plot, skipping: {nc2_sorc.as_posix()}"
        )
        return

    # Plot magnetic field reference values and deltas from EZIE OSSE retrieval along
    # EZIE MEM lines of sight.
    logger.info("Generating plot of retrieved B field dBs")
    nrows, ncols = NUM_MEM + 1, 5

    # use = np.full_like(time_utc[:], fill_value=True, dtype=bool)
    fig = plt.figure(figsize=DEFAULT_FIG_SIZE)
    ags = GridSpec(nrows, ncols, figure=fig)
    ax0 = fig.add_subplot(ags[0, 0:3])
    ax1 = fig.add_subplot(ags[1, 0:3], sharex=ax0)
    ax2 = fig.add_subplot(ags[2, 0:3], sharex=ax0)
    ax3 = fig.add_subplot(ags[3, 0:3], sharex=ax0)
    ax4 = fig.add_subplot(ags[4, 0:3], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    # # Hide all axes on RHS - we'll add "special" axes manually there.
    # RHS0, RHS1 = ncols - 2, ncols - 1
    # for row in range(nrows):
    #     for col in range(RHS0, RHS1):
    #         axs[row, col].set_visible(False)

    # We'll save the total variable range for each field component across
    # all MEMs here, adjusting plot limits afterwards when we know what the
    # correct range is for the whole ensemble.
    b_rng = np.full((2), fill_value=np.nan)  # 4 B x [min,max]

    # col = np.s_[0:2]
    # axs[-1, 1].set_visible(False)
    lat_axs = axs[-1]
    lat_axs.grid(axis="both")
    lat_axs.set_ylabel("Magnetic Latitude\n(APEX, degrees)", weight="bold")
    lat_axs.set_xlabel("Time (UTC)", weight="bold")
    mlt_axs = lat_axs.twinx()
    mlt_axs.set_ylim(-0.5, 24.5)
    mlt_axs.set_yticks(range(0, 25, 6))
    mlt_lbl = [
        "Midnight",
        "Dawn",
        "Noon",
        "Dusk",
        "Midnight",
    ]
    mlt_axs.set_yticklabels(mlt_lbl, size="x-small")
    mlt_axs.set_ylabel("Magnetic Local Time\n(APEX, hours)", weight="bold")
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
            yerr=np.sqrt(mem_cdd[mem_ndx][use_obs]),
            color=MEM_CLR[mem_ndx],
            linestyle="solid",
            errorevery=5,
        )
        mem_axs.plot(
            time_utc[use_obs],
            moving_average(mem_dbd[mem_ndx][use_obs], ave_win),
            color="black" if not dark_mode else DARK_MODE_TEXT_COLOR,
            linestyle="solid",
            lw=1.0,
            zorder=3,
        )
        try:
            b_rng[0] = np.nanmin(
                list(b_rng[0:1])
                + list(mem_dbd[mem_ndx][use_obs] - np.sqrt(mem_cdd[mem_ndx][use_obs]))
            )
            b_rng[1] = np.nanmax(
                list(b_rng[1:2])
                + list(mem_dbd[mem_ndx][use_obs] + np.sqrt(mem_cdd[mem_ndx][use_obs]))
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
        mem_axs.set_ylabel(f"MEM {mem_num} dB$_\\mathbf{{D}}$ (nT)", weight="bold")
        mem_axs.text(
            0.01,
            0.99,
            f"Off-Nadir Angle: {MEM_LOOK_DIRECTIONS[mem_ndx]}"
            r"º",
            weight="bold",
            size="medium",
            va="top",
            ha="left",
            transform=mem_axs.transAxes,
        )

        # Label only leftmost column y axis
        if row == 0:
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
    for mem_ndx in [x - 1 for x in MEM_NUMBERS]:
        mem_axs = axs[PLT_NDX[mem_ndx]]
        tot_rng = b_rng[1] - b_rng[0]
        logger.debug(f"{b_rng[0]} : {b_rng[1]}")
        # FIXME - kludge to use old L2 file - comment out line below
        try:
            mem_axs.set_ylim(
                b_rng[i] + tot_rng * x for i, x in enumerate([-0.05, +0.05])
            )
        except Exception as exc:
            mem_axs.set_ylim((-1.0, +1.0))
            logger.error(f"Exception while attempting to set axes y limits: {exc}")
            logger.error(f"Problem file (data range): {nc2_sorc.as_posix()}")
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
        axs[-1],
        max_ticks=20 // ncols + 1,
        use_seconds=(time_utc[-1] - time_utc[0]).total_seconds() < 120,
    )

    # NEW geomagnetic coordinate map inset
    midpt = len(time_utc) // 2
    lat_at_midpt = nc2_data["Geolocation/sat_lat"][midpt]
    lon_at_midpt = nc2_data["Geolocation/sat_lon"][midpt]

    if lat_at_midpt > +40:  # hemisphere = NORTH
        hemisphere = NORTH
    elif lat_at_midpt < -40:  # hemisphere = SOUTH
        hemisphere = SOUTH
    else:
        hemisphere = None

    if hemisphere == NORTH:
        proj_method = ccrs.NorthPolarStereo(central_longitude=0)  # 0 is default
    elif hemisphere == SOUTH:
        proj_method = ccrs.SouthPolarStereo(central_longitude=180)
        if south_inverted:
            MLT_sign = -1
    else:
        proj_method = ccrs.Orthographic(
            central_latitude=lat_at_midpt, central_longitude=lon_at_midpt
        )

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
        x_0, y_0 = proj_method.transform_point(
            0,
            90 if hemisphere == NORTH else -90,
            src_crs=DATA_TRANSFORM,
        )
        x_1, y_1 = proj_method.transform_point(
            45,
            MAG_LAT_LOWER_LIMIT if hemisphere == NORTH else -MAG_LAT_LOWER_LIMIT,
            src_crs=DATA_TRANSFORM,
        )
        map_meters = np.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2)
        fp_map_axs.set_extent(
            (
                -map_meters,
                +map_meters,
                -map_meters,
                +map_meters,
            ),
            crs=proj_method,
        )
        # Compute a circle in axes coordinates that will be used as a clipping boundary
        # for the map.
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        fp_map_axs.set_boundary(circle, transform=fp_map_axs.transAxes)

    # Map S/C and MEM footprint lat/lon
    dtlim = [time_utc[0], time_utc[1]]
    duration = (dtlim[1] - dtlim[0]).total_seconds()
    midpoint = dtlim[0] + datetime.timedelta(seconds=int(0.5 * duration))

    # Get solar position, first in geodetic and then in APEX magnetic coordinates.
    sun_geo_tuple_rad = []

    # Geodetic
    for tndx, tutc in enumerate(time_utc):
        tdb = spiceypy.utc2et(tutc.isoformat()[:-6])
        (subpnt, epoch, to_subpnt) = spiceypy.subslr(
            "INTERCEPT/ELLIPSOID", "EARTH", tdb, "IAU_EARTH", "LT+S", "EARTH"
        )
        sun_geo_tuple_rad.append(
            spiceypy.recgeo(subpnt, EARTH_RADIUS_EQUATORIAL, EARTH_FLATTENING)
        )  # Fix the subsolar point on the surface in geodetic coordinates
    sun_geo_lon_deg = np.array([np.degrees(x[0]) for x in sun_geo_tuple_rad])
    sun_geo_lat_deg = np.array([np.degrees(x[1]) for x in sun_geo_tuple_rad])

    # Compute the subsolar point in APEX magnetic coordinates
    apex = Apex(date=midpoint.year, refh=0)
    sun_mlat, sun_mlon = apex.geo2apex(
        sun_geo_lat_deg,
        sun_geo_lon_deg,
        REFERENCE_ALTITUDE_KM,
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
            alt4mag=REFERENCE_ALTITUDE_KM,
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
    fw = fig.get_figwidth()
    fh = fig.get_figheight()
    fig_aspect_ratio = fw / fh
    mem_beam_img = plt.imread(
        Path(__file__).parent.parent / "binary-assets" / "mem-beam-diagram.png"
    )
    img_aspect_ratio = mem_beam_img.shape[1] / mem_beam_img.shape[0]
    hi = 0.18
    wi = hi * img_aspect_ratio / fig_aspect_ratio
    beamsx, beamsy = 0.62, 0.46 - hi / 2
    # img_axs = fig.add_axes([0.975 - wi, 0.025 * fig_aspect_ratio, wi, hi])  # LR
    img_axs: plt.Axes = fig.add_axes(rect=(beamsx, beamsy, wi, hi))
    img_axs.imshow(mem_beam_img, aspect="auto", zorder=3)
    # Just turn ticks off so we get a border around image.
    img_axs.set_xticks([])
    img_axs.set_yticks([])
    # img_axs.axis("off")

    with Dataset(nc3_sorc, mode="r") as nc3_data:
        try:
            time_utc = np.array(
                [
                    datetime.datetime.fromisoformat(_)
                    for _ in nc3_data["/l2_data/time_utc"][:]
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
        lat_at_midpt = nc3_data["l2_data/sc_lat"][midpt]
        lon_at_midpt = nc3_data["l2_data/sc_lon"][midpt]

        # Add magnetic coordinate overlay, select latitudes at which to draw magnetic
        # gridlines, play with the map projection as needed.
        if lat_at_midpt > +40:  # hemisphere = NORTH
            hemisphere = NORTH
        elif lat_at_midpt < -40:  # hemisphere = SOUTH
            hemisphere = SOUTH
        else:
            hemisphere = None
            logger.warning(
                f"L3 file appears to contain EEJ pass, skipping: {nc3_sorc.as_posix()}"
            )
            return

        # Grab J, B, and coordinate data arrays
        lat_mesh = nc3_data["/l3_data/lat"][:]
        lon_mesh = nc3_data["/l3_data/lon"][:]
        model_B = nc3_data["/l3_data/Bd_geod_80"][:]
        Je = nc3_data["/l3_data/Je_110"][:]
        Jn = nc3_data["/l3_data/Jn_110"][:]
        lats = nc3_data["/l2_data/lat"][:]
        lons = nc3_data["/l2_data/lon"][:]
        obs_B = nc3_data["/l2_data/Bd_geod_80"][:]
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

        mlt_desc = {}
        lon_delta = 30.0
        mlt_angs = np.arange(0, 360.0, lon_delta)
        for mm, mlt in enumerate(mlt_angs):
            mlt_desc[f"{mlt:.0f}"] = f"{mlt / 15:02.0f}H"
        mlt_desc["0"] = "Midnight"
        mlt_desc["90"] = "Dawn"
        mlt_desc["180"] = "Noon"
        mlt_desc["270"] = "Dusk"

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
            # x, y = boundary_distance(xp_m, yp_m, plt_ang, stretch=0.035)
            x, y = boundary_distance(xp_m, yp_m, plt_ang, stretch=0.06)
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

    from matplotlib.colors import LinearSegmentedColormap

    if dark_mode:
        color_min = "#4203ff"
        color_center = DARK_MODE_FILL_COLOR
        color_max = "#ff0342"
        cmap = LinearSegmentedColormap.from_list(
            "cmap_name", [color_min, color_center, color_max]
        )
    else:
        cmap = plt.cm.bwr

    # Model B
    valid = model_B != NCDF_MISSING
    sc = l3_map_axs.scatter(
        MLT_sign * lon_mesh[valid].flatten(),
        lat_mesh[valid].flatten(),
        c=-1 * model_B[valid].flatten(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=b_mrk_sz,
        transform=DATA_TRANSFORM,
    )

    # Retrieved J
    l3_map_axs.quiver(
        MLT_sign * lon_mesh[::m_mrgn, ::m_mrgn].flatten(),
        lat_mesh[::m_mrgn, ::m_mrgn].flatten(),
        MLT_sign * Je[::m_mrgn, ::m_mrgn].flatten(),
        Jn[::m_mrgn, ::m_mrgn].flatten(),
        scale=Jmax,
        width=qwid,
        color="xkcd:black" if not dark_mode else "orange",
        transform=DATA_TRANSFORM,
    )

    # Observed (L2) B
    valid = obs_B != NCDF_MISSING
    sc = l3_map_axs.scatter(
        MLT_sign * lons[valid],
        lats[valid],
        c=obs_B[valid],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        # s=b_mrk_sz,
        # marker="o",
        s=b_mrk_sz / 4,
        marker="s",
        transform=DATA_TRANSFORM,
    )

    # Add plot annotation and color bar, tweak margins and spacing.
    divider = make_axes_locatable(l3_map_axs)
    cax = divider.append_axes("right", size="3%", pad=0.7, axes_class=maxes.Axes)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("B$_{{down}}$ [nT]")
    # fig.suptitle(
    #     (
    #         f"L3 and L2: EZIE-{prsd.spcv.upper()} \n"
    #         f"{time_utc[0].strftime('%B %d, %Y %H:%M:%S')} - "
    #         f"{time_utc[-1].strftime('%H:%M:%S')}\n"
    #     ),
    #     y=0.99,
    #     va="top",
    #     fontsize="x-large",
    #     fontweight="bold",
    # )

    if hemisphere is not None:
        subtitle = """
            Currents are plotted above in Geodetic/WGS84 coordinates (solid grid).
            Labels indicate Local Time (LT). Magnetic latitude (dashed grid) is also
            shown. MEM footprints are plotted below in Apex Geomagnetic coordinates.
            """
    else:
        subtitle = """
            Current plot is in Geodetic/WGS84 coordinates (solid grid). Magnetic
            latitude (dashed grid) and drift equator (thick solid line) are also
            shown. MEM footprints are plotted below in Apex Geomagnetic coordinates.
            """
    fig.text(
        0.71,
        0.50,
        subtitle,
        wrap=False,
        ha="left",
        va="center",
        fontsize="x-small",
        fontweight="bold",
    )

    # Tweak position and add any figure-level annotation
    fig_title = f"{mode} Retrieved Magnetic Field Deltas and Currents: "
    fig_title = fig_title + (
        f"{sc_id} - Orbit {orb_num}\n"
        f"{time_utc[use_obs][0].strftime('%Y-%m-%d (%j) %H:%M:%S')} - "
        f"{time_utc[use_obs][-1].strftime('%Y-%m-%d (%j) %H:%M:%S')}"
    )
    fig.suptitle(fig_title, weight="bold", fontsize="x-large", y=0.99, va="top")
    add_product_metadata(fig=fig, nc_data=nc2_data, source=nc2_sorc, size="x-small")
    with Dataset(nc3_sorc, mode="r") as nc3_data:
        add_product_metadata(
            fig=fig, nc_data=nc3_data, source=nc3_sorc, second=True, size="x-small"
        )
    add_pipeline_metadata(fig=fig, nc_data=nc2_data, size="x-small")
    overlay_ezie_logo(fig=fig, dark_mode=dark_mode)
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
