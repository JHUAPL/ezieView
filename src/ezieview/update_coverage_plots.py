"""
Generate EZIE full-day coverage summary plots.

For each date in the requested range, produces:
- MEM coverage histograms (magnetic latitude, MLT, and solar zenith angle),
  broken down by orbit and MEM, and
- polar stereographic and Mollweide maps of spacecraft passes and MEM
  observation locations, broken down by orbit.

    Example command line--run from repo root, in an activated python venv:

    view_coverage \\
        -d0 2025-06-20 \\
        -d1 2025-06-22 \\
        -fd '/project/ezie/data/l1' \\
        -fp 'ezie_l1_*.nc4' \\
        -pd "/project/ezie/plots/daily-summary"
"""

# region imports
import datetime
import logging
from contextlib import contextmanager
from pathlib import Path

import cartopy.feature as cfeature
import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ezieview.ezvislib import gw_logger
from ezieview.ezvislib.gw_plot_methods import (
    coverage_plot_polar_layout,
    coverage_plot_stereographic_layout,
    ingest_full_day_all_sv,
    mollweide_layout,
    save_close_figure,
)
from ezieview.ezvislib.gw_plot_params import (
    ALTERNATE_FIG_SIZE,
    BGN_COLOR,
    BOUNDS,
    CLI_DATE_FORMAT,
    COVERAGE_TYPES,
    COVERAGE_UNITS,
    DARK_MODE_FACE_COLOR,
    DARK_MODE_FILL_COLOR,
    DARK_MODE_GRID_COLOR,
    DATA_TRANSFORM,
    DFLT_RES,
    END_COLOR,
    EQUATORIAL,
    EZIE_COLOR,
    EZIE_DATE_FORMAT,
    EZIE_LABEL,
    MAGLAT,
    MAX_ORBITS,
    MEM_CLR,
    MEM_NUMBERS,
    MLT,
    NORTH,
    NUM_MEM,
    NUM_SPC,
    REGIONS,
    SCIENCE_COLOR,
    SCIENCE_LABEL,
    SOUTH,
    SPACECRAFT,
    SPCLOOK_COLOR,
    SPCLOOK_LABEL,
    SZA,
    UTC_TZ,
)
from ezieview.ezvislib.gw_plot_utils import (
    filter_files_by_version,
    parse_ezie_product_name,
)

# endregion imports

mpl.use("Agg")

# Root logger, configured by gw_logger.initialize_logging() inside main().
logger = logging.getLogger()

# ===== Daily map plot parameters ======================================================
# Plot ID suffixes for the daily map figures
DATA_COLLECTION_GEO = "mem-data-collection-events-geo"
MEM_OBS_LOCN_GEO = "mem-observation-locations-geo"
DATA_COLLECTION_MAG = "mem-data-collection-events-mag"
MEM_OBS_LOCN_MAG = "mem-observation-locations-mag"
POLAR_PLOT_TYPES = [
    DATA_COLLECTION_GEO,
    MEM_OBS_LOCN_GEO,
    DATA_COLLECTION_MAG,
    MEM_OBS_LOCN_MAG,
]
MOLLWEIDE_PLOT_TYPES = [DATA_COLLECTION_GEO, MEM_OBS_LOCN_GEO]
# Corresponding figure title for each map plot type
MAP_TITLES = {
    DATA_COLLECTION_GEO: "EZIE Data Collection Events: Geographic Lat/Lon",
    DATA_COLLECTION_MAG: "EZIE Data Collection Events: Magnetic Lat/MLT",
    MEM_OBS_LOCN_GEO: "EZIE MEM Observation Locations: Geographic Lat/Lon",
    MEM_OBS_LOCN_MAG: "EZIE MEM Observation Locations: Magnetic Lat/MLT",
}
MAP_MARKER = "o"
SC_MS = 2
MEM_MS = 2
END_MS = 4
END_MEW = 0.5
LEGEND_MARKER_SCALE = 2  # Magnify marker size for legend display

# ===== MEM coverage histogram parameters ==============================================
# Per-region magnetic latitude histogram bin limits (bin_max is padded by 2 x bin_res)
MAGLAT_BIN_LIMITS = {
    NORTH: (+45.0, +90.0),
    EQUATORIAL: (-35.0, +35.0),
    SOUTH: (-90.0, -45.0),
}
# Display (y-axis) limits for magnetic latitude panels. Note the equatorial display
# range (+/-30) is narrower than its bin range (+/-35): samples with
# 30 < |mlat| < 35 count toward the colorbar scaling but are not displayed.
MAGLAT_DISPLAY_LIMITS = {
    NORTH: (+45.0, +90.0),
    EQUATORIAL: (-30.0, +30.0),
    SOUTH: (-90.0, -45.0),
}
# Figure-fraction margins for the MEM coverage histogram figures
COV_FIG_MARGINS = {
    "left": 0.045,
    "right": 1.05,
    "bottom": 0.070,
    "top": 0.91,
    "wspace": 0.0,
}
CBAR_WIDTH = 0.012
CBAR_OFFSET = 0.008


@contextmanager
def _dark_style(dark_mode: bool):
    """
    Apply 'dark mode' styling for the duration of the with-block, restoring all
    rcParams (including those set by the cartopy layout methods) on exit.
    """
    if not dark_mode:
        yield
        return
    saved_rc = dict(mpl.rcParams)
    plt.style.use("dark_background")
    plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
    plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR
    try:
        yield
    finally:
        mpl.rcParams.clear()
        mpl.rcParams.update(saved_rc)


def in_region(lat: np.ndarray, regn: str) -> np.ndarray:
    """Boolean mask of samples whose (magnetic) latitude lies within region regn."""
    lo, hi = BOUNDS[regn]
    return (lat >= lo) & (lat <= hi)


def _skip_existing(
    obs_date: datetime.datetime,
    plot_type: str,
    output_dir: Path,
    dark_mode: bool,
    figure_dpi: int,
    overwrite: bool,
) -> bool:
    """
    Return True, logging the skip, if the plot file for this plot type already
    exists and overwrite is disabled.
    """
    fig_path, _old_hash, _new_hash = save_close_figure(
        obs_date=obs_date,
        plot_type=plot_type,
        save_directory=output_dir,
        dark_mode=dark_mode,
        dpi=figure_dpi,
        name_only=True,
    )
    if fig_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {fig_path.name}")
        return True
    return False


# ===== MEM coverage histogram plotting ================================================


def _orbit_edges(orbit: np.ndarray) -> np.ndarray:
    """Orbit histogram bin edges: MAX_ORBITS unit bins starting at the first orbit."""
    orb_min = np.min(orbit)
    return np.arange(orb_min, orb_min + MAX_ORBITS + 1, 1) - 0.5


def _coverage_bin_edges(cvrg_type: str, regn: str) -> np.ndarray:
    """Bin edges for the 1-D (non-orbit) coverage histogram axis."""
    if cvrg_type == MAGLAT:
        bin_res = 2.0
        bin_min, bin_max = MAGLAT_BIN_LIMITS[regn]
        bin_max += 2.0 * bin_res
    elif cvrg_type == SZA:
        bin_res, bin_min, bin_max = 5.0, 0.0, 180.0 + 2.0 * 5.0
    else:  # MLT
        bin_res, bin_min, bin_max = 0.5, 0.0, 24.0 + 0.5
    return np.arange(bin_min, bin_max, bin_res) - bin_res / 2


def _coverage_histograms(
    cvrg_type: str,
    regn: str,
    sc_id: np.ndarray,
    orbit: np.ndarray,
    mem_mlat: np.ndarray,
    mem_qty: np.ndarray,
    mem_mode: np.ndarray,
    bin_edg: np.ndarray,
    orb_edg: np.ndarray,
) -> tuple[dict[int, dict[int, np.ndarray]], float, np.ndarray]:
    """
    Make 2-D histograms of coverage versus orbit, per spacecraft and MEM, for the
    samples of region regn in earth-look (science collection) mode. The region
    filter always applies to magnetic latitude, regardless of the coverage type
    being binned.

    Returns:
        (cvrg, cmax, sampled) where cvrg[sndx][mem_ndx] is the histogram with
        empty bins set to NaN, cmax is the maximum bin count (at least 2), and
        sampled is the (spacecraft, MEM) count matrix.
    """
    cvrg: dict[int, dict[int, np.ndarray]] = {}
    sampled = np.zeros((len(SPACECRAFT), NUM_MEM))
    cmax = 2
    for sndx, spcrft in enumerate(SPACECRAFT):
        cvrg[sndx] = {}
        for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
            # Determine MEM-specific science-collection/space-look status,
            # creating a data collection event (dce) mask tied to the time step.
            scuse = sc_id == spcrft
            rguse = in_region(mem_mlat[mem_ndx], regn)
            dce = mem_mode[mem_ndx] & scuse & rguse
            sampled[sndx, mem_ndx] += np.sum(dce)
            # Diagnostics - sample sizes for each plot
            logger.debug(
                f"{regn} "
                f"{cvrg_type:20s} "
                f"{spcrft} "
                f"{mem_num} "
                f"{np.sum(scuse):5d} "
                f"{np.sum(rguse):5d} "
                f"{np.sum(mem_mode[mem_ndx]):5d} "
                f"{np.sum(scuse & rguse):5d} "
                f"{np.sum(dce):5d}"
            )

            if np.any(dce):
                # Boolean indexing copies, so the MLT wraparound below is local
                vals = mem_qty[mem_ndx][dce]
                if cvrg_type == MLT:
                    # Wrap MLT values near 24 h around to the start of the axis
                    vals = np.where(vals > bin_edg[-1], vals - 24.0, vals)
                hst, _y_edg, _x_edg = np.histogram2d(
                    vals,
                    orbit[dce],
                    bins=(bin_edg, orb_edg),
                )
            else:
                # No valid observations, create blank template.
                hst = np.zeros((len(bin_edg) - 1, len(orb_edg) - 1))
            hst[hst == 0] = np.nan
            cvrg[sndx][mem_ndx] = hst
            if np.any(dce):
                cmax = max(cmax, np.nanmax(hst))
        logger.info(
            f"Number of samples for SV {spcrft} in region {regn} "
            f"was {np.sum(sampled[sndx, :])}"
        )
    return cvrg, cmax, sampled


def _style_coverage_axes(haxs, cvrg_type: str, regn: str, bin_edg: np.ndarray) -> None:
    """Set axis limits and ticks for a MEM coverage histogram panel."""
    haxs.set_ylim(bin_edg[0], bin_edg[-1])
    if cvrg_type == MAGLAT:
        haxs.set_ylim(*MAGLAT_DISPLAY_LIMITS[regn])
    elif cvrg_type == MLT:
        haxs.set_yticks(range(0, 24, 2))
    elif cvrg_type == SZA:
        haxs.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))


def _add_coverage_colorbar(cb_img, sndx: int, cpsc: int) -> None:
    """Add the colorbar following the last MEM panel of spacecraft column sndx."""
    mrgn = COV_FIG_MARGINS
    left = (
        mrgn["left"]
        + CBAR_OFFSET
        + (mrgn["right"] - mrgn["left"]) / (NUM_SPC * cpsc) * (sndx * cpsc + NUM_MEM)
    )
    cbar_axs = plt.axes(
        (left, mrgn["bottom"], CBAR_WIDTH, mrgn["top"] - mrgn["bottom"])
    )
    cbar = plt.colorbar(cb_img, cax=cbar_axs, extend="min", fraction=0.20)
    cbar.set_label("Number of samples")
    # Massage colorbar tick placement and labeling
    cbar.locator = plt.MaxNLocator(integer=True, nbins="auto")


def _render_coverage_figure(
    obs_date: datetime.datetime,
    regn: str,
    cvrg_type: str,
    cvrg_indx: int,
    cvrg: dict[int, dict[int, np.ndarray]],
    cmax: float,
    bin_edg: np.ndarray,
    orb_edg: np.ndarray,
    plot_type: str,
    output_dir: Path,
    dark_mode: bool,
    figure_dpi: int,
) -> None:
    """Build, annotate, and save one MEM coverage histogram figure."""
    rows = 1
    cpsc = NUM_MEM + 2
    cols = len(SPACECRAFT) * cpsc
    fig = plt.figure(figsize=ALTERNATE_FIG_SIZE)

    # "plasma" gives an abrupt 0 to >0 transition; under-range (empty) bins in white
    cmap = plt.get_cmap("plasma", int(cmax))
    # cmap.set_under("white")  # deprecated
    cmap.with_extremes(under="white")

    for sndx, spcrft in enumerate(SPACECRAFT):
        for mem_ndx in range(NUM_MEM):
            # Create and customize axes for each spacecraft and MEM
            haxs = plt.subplot2grid(
                (rows, cols),
                (0, sndx * cpsc + mem_ndx),
                rowspan=1,
                colspan=1,
            )
            if mem_ndx == 0:
                # Label axes of first ("exterior") MEM axis
                haxs.set_title(f"{spcrft} - {regn}", loc="left")
                haxs.set_ylabel(f"{cvrg_type} ({COVERAGE_UNITS[cvrg_indx]})")
            else:
                # Turn off ticks on "interior" MEM axes
                haxs.set_yticklabels([])
                haxs.tick_params(axis="y", length=0, width=0)

            _style_coverage_axes(haxs, cvrg_type, regn, bin_edg)

            cb_img = haxs.imshow(
                cvrg[sndx][mem_ndx],
                cmap=cmap,
                vmin=0.5,
                vmax=cmax + 0.5,
                origin="lower",
                aspect="auto",
                extent=(
                    orb_edg[0],
                    orb_edg[-1],
                    bin_edg[0],
                    bin_edg[-1],
                ),
            )
            haxs.grid(axis="y")
            haxs.set_xticks([])
            haxs.set_xlabel(f"MEM {mem_ndx + 1}")
            if mem_ndx == NUM_MEM - 1:
                _add_coverage_colorbar(cb_img, sndx, cpsc)

    # Tweak position and add any figure-level annotation
    fig.suptitle(
        f"MEM {cvrg_type} Coverage, All Orbits: {obs_date.strftime('%Y-%m-%d (%j)')}",
        weight="bold",
        fontsize="x-large",
    )
    plt.subplots_adjust(**COV_FIG_MARGINS)

    save_close_figure(
        figure=fig,
        obs_date=obs_date,
        plot_type=plot_type,
        save_directory=output_dir,
        dark_mode=dark_mode,
        dpi=figure_dpi,
    )


def _plot_region_coverage(
    obs_date: datetime.datetime,
    regn: str,
    sc_id: np.ndarray,
    orbit: np.ndarray,
    mem_mlat: np.ndarray,
    mem_mlt: np.ndarray,
    mem_sza: np.ndarray,
    mem_mode: np.ndarray,
    output_dir: Path,
    overwrite: bool,
    dark_mode: bool,
    figure_dpi: int,
) -> None:
    """Generate the per-coverage-type histogram figures for one region."""
    orb_edg = _orbit_edges(orbit)
    mem_qty = {MAGLAT: mem_mlat, MLT: mem_mlt, SZA: mem_sza}
    for cvrg_indx, cvrg_type in enumerate(COVERAGE_TYPES):
        plot_type = f"{regn.lower()}-mem-coverage-{cvrg_type.lower().replace(' ', '-')}"
        if _skip_existing(
            obs_date, plot_type, output_dir, dark_mode, figure_dpi, overwrite
        ):
            continue
        logger.info(f"Generating MEM coverage plot of type ({cvrg_type})")

        bin_edg = _coverage_bin_edges(cvrg_type, regn)
        cvrg, cmax, sampled = _coverage_histograms(
            cvrg_type=cvrg_type,
            regn=regn,
            sc_id=sc_id,
            orbit=orbit,
            mem_mlat=mem_mlat,
            mem_qty=mem_qty[cvrg_type],
            mem_mode=mem_mode,
            bin_edg=bin_edg,
            orb_edg=orb_edg,
        )
        if np.sum(sampled) == 0:
            # Don't bother to generate figures if region had 0 observations
            logger.info(f"Number of samples in region {regn} was zero, skipping.")
            continue

        _render_coverage_figure(
            obs_date=obs_date,
            regn=regn,
            cvrg_type=cvrg_type,
            cvrg_indx=cvrg_indx,
            cvrg=cvrg,
            cmax=cmax,
            bin_edg=bin_edg,
            orb_edg=orb_edg,
            plot_type=plot_type,
            output_dir=output_dir,
            dark_mode=dark_mode,
            figure_dpi=figure_dpi,
        )


def plot_mlt_sza_coverage(
    obs_date: datetime.datetime,
    sc_id: np.ndarray,
    orbit: np.ndarray,
    mem_mlat: np.ndarray,
    mem_mlt: np.ndarray,
    mem_sza: np.ndarray,
    mem_mode: np.ndarray,
    output_dir: Path,
    regions: list,
    dark_mode: bool = False,
    overwrite: bool = False,
    figure_dpi: int = DFLT_RES,
) -> None:
    """
    Generate 2-D histograms showing daily coverage, broken down by orbit and MEM,
    for:
    1. Magnetic latitude
    2. Magnetic Local Time
    3. Solar Zenith Angle

    Breaks L1 data for each SV into orbits so we can make a 2-D histogram of
    coverage versus orbit for this day, rather than just determining coverage for
    the day as a whole. One figure is generated per region and coverage type, with
    one panel per spacecraft and MEM.

    Args:
        obs_date: UTC date of the observations being plotted.
        sc_id: Spacecraft ID for each observation sample.
        orbit: Orbit number for each observation sample.
        mem_mlat: Magnetic latitude of each MEM footprint, stacked
            over the 4 MEMs (rows) x samples (cols).
        mem_mlt: Magnetic local time of each MEM footprint,
            stacked as above.
        mem_sza: Solar zenith angle of each MEM footprint,
            stacked as above.
        mem_mode: Boolean flag (True = EARTHLOOK, False = SKYLOOK)
            for each MEM, stacked as above.
        output_dir: Root directory in which to save the generated plot files.
        regions: Regions (e.g., NORTH, SOUTH, EQUATORIAL) that had
            observations and therefore need plots.
        dark_mode: Generate plots using 'dark mode' format.
            Defaults to False.
        overwrite: Overwrite existing plot files. If not set,
            any existing files will be skipped. Defaults to False.
        figure_dpi: Resolution (DPI) at which to generate the
            plots. Defaults to DFLT_RES.
    """
    with _dark_style(dark_mode):
        for regn in regions:
            _plot_region_coverage(
                obs_date=obs_date,
                regn=regn,
                sc_id=sc_id,
                orbit=orbit,
                mem_mlat=mem_mlat,
                mem_mlt=mem_mlt,
                mem_sza=mem_sza,
                mem_mode=mem_mode,
                output_dir=output_dir,
                overwrite=overwrite,
                dark_mode=dark_mode,
                figure_dpi=figure_dpi,
            )


# ===== Daily map plotting =============================================================


def _scatter(
    map_axs,
    x,
    y,
    *,
    label: str,
    zorder: int,
    color: str | None = None,
    ms: float = SC_MS,
    mfc: str | None = None,
    mec: str | None = None,
    mew: float | None = None,
) -> None:
    """Plot rasterized point markers at data-coordinate locations on a map axis."""
    style = {}
    if color is not None:
        style["color"] = color
    if mfc is not None:
        style["mfc"] = mfc
    if mec is not None:
        style["mec"] = mec
    if mew is not None:
        style["mew"] = mew
    map_axs.plot(
        x,
        y,
        transform=DATA_TRANSFORM,
        label=label,
        zorder=zorder,
        marker=MAP_MARKER,
        ms=ms,
        lw=0,
        ls="none",
        rasterized=True,
        **style,
    )


def _orbit_endpoints(map_axs, x, y, end_mec: str) -> None:
    """Mark the first and last in-region samples of a spacecraft track."""
    _scatter(
        map_axs,
        x[0],
        y[0],
        label="Start",
        zorder=5,
        ms=END_MS,
        mfc=BGN_COLOR,
        mec=end_mec,
        mew=END_MEW,
    )
    _scatter(
        map_axs,
        x[-1],
        y[-1],
        label="End",
        zorder=5,
        ms=END_MS,
        mfc=END_COLOR,
        mec=end_mec,
        mew=END_MEW,
    )


def _dedup_legend(map_axs, bbox_to_anchor: tuple, loc: str) -> None:
    """Rebuild the axes legend, keeping one entry per unique label."""
    handles, labels = map_axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    map_axs.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=bbox_to_anchor,
        loc=loc,
        markerscale=LEGEND_MARKER_SCALE,
        fontsize="x-small",
    )


def _plot_sv_orbits(
    all_axs: dict,
    ptype: str,
    sc_id: np.ndarray,
    orbit: np.ndarray,
    sc_mlat: np.ndarray,
    mem_mode: np.ndarray,
    xcrd: np.ndarray,
    ycrd: np.ndarray,
    xmem: np.ndarray,
    ymem: np.ndarray,
    mlt_sign: float,
    regn: str,
    end_mec: str,
    legend_anchor: tuple,
    legend_loc: str,
) -> np.ndarray:
    """
    Plot each spacecraft's daily orbit activity on the given per-spacecraft axes.

    For 'mem-observation' plot types, each orbit's spacecraft track is drawn in
    EZIE_COLOR and each MEM's earth-look footprints in its own color; otherwise
    the track samples are split into science (any MEM earth-look) and space-look
    (all MEMs sky-look) categories. Orbit start/end markers and a per-spacecraft
    legend are added.

    Returns:
        np.ndarray: (spacecraft, MEM) matrix of earth-look sample counts.
    """
    mem_obs = "mem-observation" in ptype
    sci_mode = np.any(mem_mode, axis=0)
    sampled = np.zeros((len(SPACECRAFT), NUM_MEM))
    orbnums = np.unique(orbit)
    rguse = in_region(sc_mlat, regn)  # filter on samples in proper location

    for sndx, spcrft in enumerate(SPACECRAFT):
        map_axs = all_axs[sndx]
        scuse = sc_id == spcrft
        plotted = False
        for orbnum in orbnums:
            in_rgn = scuse & (orbit == orbnum) & rguse
            if not np.any(in_rgn):
                continue  # no observations in region for this spacecraft/orbit
            plotted = True

            for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
                # Determine MEM science-collection/space-look status, creating a
                # data collection event (dce) mask tied to the time step.
                dce = mem_mode[mem_ndx] & in_rgn
                slk = ~mem_mode[mem_ndx] & in_rgn
                sampled[sndx, mem_ndx] += np.sum(dce)
                # Diagnostics - sample sizes for each orbit
                logger.debug(
                    f"{regn} "
                    f"{orbnum:5d} "
                    f"{spcrft} "
                    f"{mem_num} "
                    f"{np.sum(scuse):5d} "
                    f"{np.sum(rguse):5d} "
                    f"{np.sum(in_rgn):5d} "
                    f"{np.sum(slk):5d} "
                    f"{np.sum(dce):5d}"
                )

            if mem_obs:  # plot MEM footprint[s]
                _scatter(
                    map_axs,
                    mlt_sign * xcrd[in_rgn],
                    ycrd[in_rgn],
                    color=EZIE_COLOR,
                    label=EZIE_LABEL,
                    zorder=2,
                )
                for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
                    dce = mem_mode[mem_ndx] & in_rgn
                    if np.any(dce):
                        _scatter(
                            map_axs,
                            mlt_sign * xmem[mem_ndx][dce],
                            ymem[mem_ndx][dce],
                            color=MEM_CLR[mem_ndx],
                            label=f"MEM {mem_num}",
                            zorder=3,
                        )
            else:  # plot spacecraft science/space-look activity
                # SV-level status: science if ANY MEM is earth-look,
                # space-look if ALL MEMs are sky-look.
                sv_dce = in_rgn & sci_mode
                sv_slk = in_rgn & ~sci_mode
                if np.any(sv_dce):
                    _scatter(
                        map_axs,
                        mlt_sign * xcrd[sv_dce],
                        ycrd[sv_dce],
                        color=SCIENCE_COLOR,
                        label=SCIENCE_LABEL,
                        zorder=3,
                    )
                if np.any(sv_slk):
                    _scatter(
                        map_axs,
                        mlt_sign * xcrd[sv_slk],
                        ycrd[sv_slk],
                        color=SPCLOOK_COLOR,
                        label=SPCLOOK_LABEL,
                        zorder=3,
                    )

            _orbit_endpoints(map_axs, mlt_sign * xcrd[in_rgn], ycrd[in_rgn], end_mec)

        # Add legend denoting MEM footprint or spacecraft activity versus position
        if plotted:
            _dedup_legend(map_axs, legend_anchor, legend_loc)
        logger.info(
            f"Number of samples for SV {spcrft} in region {regn} "
            f"was {np.sum(sampled[sndx, :])}"
        )
    return sampled


def _render_polar_maps(
    obs_date: datetime.datetime,
    sc_id: np.ndarray,
    orbit: np.ndarray,
    sclat: np.ndarray,
    sclon: np.ndarray,
    sc_mlat: np.ndarray,
    sc_mlt: np.ndarray,
    mem_lat: np.ndarray,
    mem_lon: np.ndarray,
    mem_mlat: np.ndarray,
    mem_mlt: np.ndarray,
    mem_mode: np.ndarray,
    output_dir: Path,
    regions: list,
    south_inverted: bool,
    overwrite: bool,
    dark_mode: bool,
    figure_dpi: int,
) -> None:
    """Polar stereographic coverage maps for the north and south regions."""
    pol_regn = [regn for regn in regions if regn in (NORTH, SOUTH)]
    end_mec = DARK_MODE_GRID_COLOR if dark_mode else "black"
    for ptype in POLAR_PLOT_TYPES:
        for regn in pol_regn:
            plot_type = f"{regn.lower()}-{ptype}"
            if _skip_existing(
                obs_date, plot_type, output_dir, dark_mode, figure_dpi, overwrite
            ):
                continue
            logger.info(f"Generating MEM coverage plot of type {plot_type}")

            if "geo" in ptype:
                xcrd, ycrd, xmem, ymem = sclon, sclat, mem_lon, mem_lat
                show_mag_lat = True
                plot_method = coverage_plot_stereographic_layout
                mlt_sign = 1.0
            else:
                # NOTE: MEM mlat used as proxy for SV mlat right now
                xcrd, ycrd = sc_mlt * 15.0, sc_mlat
                xmem, ymem = mem_mlt * 15.0, mem_mlat
                show_mag_lat = False
                plot_method = coverage_plot_polar_layout
                mlt_sign = -1.0 if (south_inverted and regn == SOUTH) else 1.0

            fig, all_axs = plot_method(
                observation_date=obs_date,
                hemisphere=regn,
                dark_mode=dark_mode,
                show_mag_lat=show_mag_lat,
                south_inverted=south_inverted,
            )

            sampled = _plot_sv_orbits(
                all_axs=all_axs,
                ptype=ptype,
                sc_id=sc_id,
                orbit=orbit,
                sc_mlat=sc_mlat,
                mem_mode=mem_mode,
                xcrd=xcrd,
                ycrd=ycrd,
                xmem=xmem,
                ymem=ymem,
                mlt_sign=mlt_sign,
                regn=regn,
                end_mec=end_mec,
                legend_anchor=(1.10, 0.05),
                legend_loc="upper right",
            )
            if np.sum(sampled) == 0:
                # Don't bother to make figure if hemisphere had 0 observations
                logger.info(f"Number of samples in region {regn} was zero, skipping.")
                plt.close(fig)
                continue

            # Tweak position and add any figure-level annotation
            fig.suptitle(
                f"{MAP_TITLES[ptype]}, All Orbits: "
                f"{obs_date.strftime('%Y-%m-%d (%j)')}",
                weight="bold",
                fontsize="x-large",
            )
            plt.subplots_adjust(
                left=0.04,
                right=0.94,
                bottom=0.05,
                top=0.90,
                hspace=0.15,  # irrelevant at the moment, only one row
                wspace=0.23 if plot_method == coverage_plot_polar_layout else 0.20,
            )
            save_close_figure(
                figure=fig,
                obs_date=obs_date,
                plot_type=plot_type,
                save_directory=output_dir,
                dark_mode=dark_mode,
                dpi=figure_dpi,
            )


def _render_mollweide_maps(
    obs_date: datetime.datetime,
    sc_id: np.ndarray,
    orbit: np.ndarray,
    sclat: np.ndarray,
    sclon: np.ndarray,
    sc_mlat: np.ndarray,
    mem_lat: np.ndarray,
    mem_lon: np.ndarray,
    mem_mode: np.ndarray,
    output_dir: Path,
    regions: list,
    overwrite: bool,
    dark_mode: bool,
    figure_dpi: int,
) -> None:
    """Mollweide coverage maps for the equatorial region."""
    if EQUATORIAL not in regions:  # Otherwise do not bother with this plot type
        return
    regn = EQUATORIAL
    end_mec = DARK_MODE_GRID_COLOR if dark_mode else "black"
    for ptype in MOLLWEIDE_PLOT_TYPES:
        plot_type = f"equatorial-{ptype}"
        if _skip_existing(
            obs_date, plot_type, output_dir, dark_mode, figure_dpi, overwrite
        ):
            continue
        logger.info(f"Generating MEM coverage plot of type {plot_type}")

        fig, all_axs = mollweide_layout(
            observation_date=obs_date,
            dark_mode=dark_mode,
            show_mag_lat=True,
            num_sc=NUM_SPC,
        )
        for map_axs in all_axs.values():
            map_axs.add_feature(cfeature.OCEAN)

        sampled = _plot_sv_orbits(
            all_axs=all_axs,
            ptype=ptype,
            sc_id=sc_id,
            orbit=orbit,
            sc_mlat=sc_mlat,
            mem_mode=mem_mode,
            xcrd=sclon,
            ycrd=sclat,
            xmem=mem_lon,
            ymem=mem_lat,
            mlt_sign=1.0,
            regn=regn,
            end_mec=end_mec,
            legend_anchor=(1.00, 0.00),
            legend_loc="lower right",
        )
        if np.sum(sampled) == 0:
            # Don't bother to make figure if region had 0 observations
            logger.info(f"Number of samples in region {regn} was zero, skipping.")
            plt.close(fig)
            continue

        # Tweak position and add any figure-level annotation
        fig.suptitle(
            f"{MAP_TITLES[ptype]}, All Orbits: {obs_date.strftime('%Y-%m-%d (%j)')}",
            x=0.5,
            y=0.995,
            weight="bold",
            fontsize="x-large",
        )
        plt.subplots_adjust(
            left=0.04,
            right=0.94,
            bottom=0.01,
            top=0.97,
            hspace=0.05,  # irrelevant at the moment, only one row
            wspace=0.20,
        )  # Triple map projection
        save_close_figure(
            figure=fig,
            obs_date=obs_date,
            plot_type=plot_type,
            save_directory=output_dir,
            dark_mode=dark_mode,
            dpi=figure_dpi,
        )


def plot_daily_maps(
    obs_date: datetime.datetime,
    sc_id: np.ndarray,
    orbit: np.ndarray,
    sclat: np.ndarray,
    sclon: np.ndarray,
    sc_mlat: np.ndarray,
    sc_mlt: np.ndarray,
    mem_lat: np.ndarray,
    mem_lon: np.ndarray,
    mem_mlat: np.ndarray,
    mem_mlt: np.ndarray,
    mem_mode: np.ndarray,
    output_dir: Path,
    regions: list,
    south_inverted: bool = False,
    overwrite: bool = False,
    dark_mode: bool = False,
    figure_dpi: int = DFLT_RES,
) -> None:
    """
    Generate polar stereographic plots showing spacecraft passes, broken down by
    orbit, in two formats:
    1) Distinguishing between data collection events and space-look operations, and
    2) Mapping the locations observed during each orbit by each individual MEM.

    Mollweide maps are also generated when EQUATORIAL is included in regions.

    Args:
        obs_date: UTC date of the observations being plotted.
        sc_id: Spacecraft ID for each observation sample.
        orbit: Orbit number for each observation sample.
        sclat: Geographic latitude of the spacecraft for each sample.
        sclon: Geographic longitude of the spacecraft for each sample.
        sc_mlat: Magnetic latitude of the spacecraft for each sample.
        sc_mlt: Magnetic local time of the spacecraft for each sample.
        mem_lat: Geographic latitude of each MEM footprint, stacked
            over the 4 MEMs (rows) x samples (cols).
        mem_lon: Geographic longitude of each MEM footprint,
            stacked as above.
        mem_mlat: Magnetic latitude of each MEM footprint,
            stacked as above.
        mem_mlt: Magnetic local time of each MEM footprint,
            stacked as above.
        mem_mode: Boolean flag (True = EARTHLOOK, False = SKYLOOK)
            for each MEM, stacked as above.
        output_dir: Root directory in which to save the generated plot files.
        regions: Regions (e.g., NORTH, SOUTH, EQUATORIAL) that had
            observations and therefore need plots.
        south_inverted: Invert the MLT axis on the southern
            hemisphere plot (heliospheric community mapping style).
            Defaults to False.
        overwrite: Overwrite existing plot files. If not set,
            any existing files will be skipped. Defaults to False.
        dark_mode: Generate plots using 'dark mode' format.
            Defaults to False.
        figure_dpi: Resolution (DPI) at which to generate the
            plots. Defaults to DFLT_RES.
    """
    with _dark_style(dark_mode):
        _render_polar_maps(
            obs_date=obs_date,
            sc_id=sc_id,
            orbit=orbit,
            sclat=sclat,
            sclon=sclon,
            sc_mlat=sc_mlat,
            sc_mlt=sc_mlt,
            mem_lat=mem_lat,
            mem_lon=mem_lon,
            mem_mlat=mem_mlat,
            mem_mlt=mem_mlt,
            mem_mode=mem_mode,
            output_dir=output_dir,
            regions=regions,
            south_inverted=south_inverted,
            overwrite=overwrite,
            dark_mode=dark_mode,
            figure_dpi=figure_dpi,
        )
        _render_mollweide_maps(
            obs_date=obs_date,
            sc_id=sc_id,
            orbit=orbit,
            sclat=sclat,
            sclon=sclon,
            sc_mlat=sc_mlat,
            mem_lat=mem_lat,
            mem_lon=mem_lon,
            mem_mode=mem_mode,
            output_dir=output_dir,
            regions=regions,
            overwrite=overwrite,
            dark_mode=dark_mode,
            figure_dpi=figure_dpi,
        )


def _valid_files_and_dates(
    keep_files: list[Path],
) -> tuple[list[Path], list[str]]:
    """
    Split the candidate files into (valid_files, dates), excluding any file whose
    name fails to parse a version (presumed corrupt).
    """
    valid_files: list[Path] = []
    dates: list[str] = []
    for each_file in keep_files:
        prsd = parse_ezie_product_name(each_file)
        logger.debug(f"{prsd!s}")
        if str(prsd.vrsn) == "nan":
            logger.error(f"Corrupted file name? {each_file}; excluding from run")
            continue
        valid_files.append(each_file)
        dates.append(prsd.date)
    return valid_files, dates


def _stack_mem_arrays(fdd: dict) -> tuple:
    """
    Stack the per-MEM fdd arrays into (NUM_MEM, n_samples) arrays.

    Returns:
        (mem_lat, mem_lon, mem_mlat, mem_mlt, mem_sza, mem_mode) where mem_mode
        is a boolean mask of earth-look (True) versus sky-look (False) samples.
    """
    mem_lat = np.vstack([fdd[f"obs_lat{m}"] for m in MEM_NUMBERS])
    mem_lon = np.vstack([fdd[f"obs_lon{m}"] for m in MEM_NUMBERS])
    mem_mlat = np.vstack([fdd[f"obs_maglat{m}"] for m in MEM_NUMBERS])
    mem_mlt = np.vstack([fdd[f"obs_magLT{m}"] for m in MEM_NUMBERS])
    mem_sza = np.vstack([fdd[f"obs_solar_zen{m}"] for m in MEM_NUMBERS])
    # NaN MEM footprints are sky-look; otherwise earth-look.
    mem_mode = ~np.isnan(mem_mlat)
    return mem_lat, mem_lon, mem_mlat, mem_mlt, mem_sza, mem_mode


def _regions_with_observations(fdd: dict, uniq_orbs: np.ndarray) -> list:
    """
    Return the regions in which at least one spacecraft made observations, based
    on the median magnetic latitude of each SV/orbit slice.
    """
    sat_mlat = fdd["sat_mlat"]
    sv_regns = np.zeros((len(REGIONS), len(SPACECRAFT)), dtype=bool)
    for sndx, spcrft in enumerate(SPACECRAFT):
        for orbnum in uniq_orbs:
            posns = (fdd["SpaceVehicle"] == spcrft) & (fdd["orbit_number"] == orbnum)
            if not np.any(posns):
                continue
            mlat_slice = sat_mlat[posns]
            if np.all(np.isnan(mlat_slice)):
                logger.error(
                    f"Unexpected result - all-NaN sat_mlat slice "
                    f"({spcrft}, orbit {orbnum})"
                )
                continue
            loctn = np.nanmedian(mlat_slice)
            for rndx, regn in enumerate(REGIONS):
                if loctn >= BOUNDS[regn][0] and loctn <= BOUNDS[regn][1]:
                    sv_regns[rndx, sndx] = True

    for rndx, regn in enumerate(REGIONS):
        logger.info(
            f"{regn:10s} {bool(sv_regns[rndx].any())!s:5s} {sv_regns[rndx, :]!r}"
        )
    return list(np.array(REGIONS)[sv_regns.any(axis=1)])


# region cli
# Define command line options using 'click' package decorators.
@click.command()
@click.option(
    "-d0",
    "--start_date",
    type=click.DateTime(formats=[CLI_DATE_FORMAT]),
    required=True,
    help="Start date of entries to be processed (YYYY-MM-DD)",
)
@click.option(
    "-d1",
    "--stop_date",
    type=click.DateTime(formats=[CLI_DATE_FORMAT]),
    required=True,
    help="Stop date (inclusive) of entries to be processed (YYYY-MM-DD)",
)
@click.option(
    "-fd",
    "--file_directory",
    required=True,
    help="Root directory path to be searched for EZIE input files",
)
@click.option(
    "-fp",
    "--file_pattern",
    default="ezie_l1*.nc4",
    help="""
Pattern of file path to be searched for EZIE input files, e.g., "ezie_l2\\*.nc4"
Note that you MUST escape all wildcard characters!
""",
)
@click.option(
    "-pd",
    "--plots_directory",
    required=True,
    help="""
Root directory in which to save generated plot files, to which 'single-orbit' will
be appended. Files will be further organized by subdirectories for each day of the
year.
""",
)
@click.option(
    "-over",
    "--overwrite",
    type=bool,
    help="Overwrite existing files. If not set, any existing files will be skipped ",
)
@click.option(
    "-dm",
    "--dark_mode",
    is_flag=True,
    help="Generate plots using 'dark mode' format",
)
@click.option(
    "-dpi",
    "--figure_dpi",
    type=int,
    default=DFLT_RES,
    help=f"Generate plots at specified DPI rather than default value of {DFLT_RES} ",
)
# endregion cli
def main(
    start_date: datetime.datetime,
    stop_date: datetime.datetime,
    file_directory: str,
    file_pattern: str,
    plots_directory: str,
    overwrite: bool,
    dark_mode: bool,
    figure_dpi: int,
) -> int:
    """
    Generate a set of EZIE full-day summary plots based on the specified date and
    supplied file types, as determined by the supplied file pattern input. Options
    to make plots at higher resolution (-dpi/--figure_dpi, default is 100 DPI) and
    in 'dark mode' are also available.

    Args:
        start_date: Start date of entries to be processed.
        stop_date: Stop date (inclusive) of entries to be processed.
        file_directory: Root directory path to be searched for EZIE input files.
        file_pattern: Pattern of file path to be searched for EZIE input files.
        plots_directory: Root directory in which to save generated plot files.
        overwrite: Overwrite existing files. If not set, existing files will be
            skipped.
        dark_mode: Generate plots using 'dark mode' format.
        figure_dpi: Resolution (DPI) at which to generate the plots.

    Returns:
        int: 0 on success, 1 if no (valid) files were found matching the date
        specifications.
    """
    t_bgn = datetime.datetime.now(UTC_TZ)
    start_date = start_date.replace(tzinfo=UTC_TZ)
    stop_date = stop_date.replace(tzinfo=UTC_TZ).replace(hour=23, minute=59, second=59)

    # Configure logging (root logger) using default log output folder and level (INFO)
    gw_logger.initialize_logging(log_to_file=True, rotating=True)
    logger.info("======= NEW RUN =======")
    context = click.get_current_context()
    for nam, val in context.params.items():
        logger.info(f"{nam:18s} : {val}")

    # Create list of highest version/revision for a given spacecraft and date,
    # for the per-orbit L1 files
    keep_files = filter_files_by_version(
        file_directory=file_directory,
        file_pattern=file_pattern,
        start_date=start_date,
        stop_date=stop_date,
        remove_near_dupes=True,
    )
    if keep_files is None:
        logger.error("No files found matching date specifications. Exiting")
        return 1
    logger.info(
        f"Found {len(keep_files)} files matching specified pattern and date range"
    )

    run_files, dates = _valid_files_and_dates(keep_files)
    if not run_files:
        logger.error("No valid files remain after file name parsing. Exiting")
        return 1

    # Create output directory for storage of plot files if it does not already exist
    out_dir_path = Path(plots_directory)
    if not out_dir_path.exists():
        out_dir_path.mkdir(parents=True, exist_ok=True)

    for run_date in sorted(set(dates)):
        logger.info(f"Generating coverage plots and maps for {run_date!s}")
        fdd, _uniq_svid, uniq_orbs = ingest_full_day_all_sv(
            run_date=run_date,
            used_files=run_files,
        )  # Full Day Dictionary

        # Stack MEM coordinate arrays so we can loop through them in plotting routines
        (mem_lat, mem_lon, mem_mlat, mem_mlt, mem_sza, mem_mode) = _stack_mem_arrays(
            fdd
        )

        # Determine earth/sky look status at each time step for each MEM. The SV
        # science collection status as a whole is valid if any one of the four MEM
        # is looking at the earth.
        sci_mode = np.sum(mem_mode, axis=0) > 0
        logger.info(
            f"MEM Obs/Earth/Sky {mem_mode.shape} {np.sum(mem_mode)} {np.sum(~mem_mode)}"
        )
        logger.info(
            f"SCI Obs/Earth/Sky {sci_mode.shape} {np.sum(sci_mode)} {np.sum(~sci_mode)}"
        )

        # Determine which regions need to be plotted, having had one or more SVs
        # make observations there.
        regions_to_plot = _regions_with_observations(fdd, uniq_orbs)
        logger.info(f"Plotting regions {regions_to_plot!s}")

        # Plot coverage for this date using combined daily entries for all SVs & orbits
        obs_date = datetime.datetime.strptime(run_date, EZIE_DATE_FORMAT).replace(
            tzinfo=UTC_TZ
        )
        sc_id = np.array(fdd["SpaceVehicle"])
        orbit = np.array(fdd["orbit_number"])

        plot_mlt_sza_coverage(
            obs_date=obs_date,
            sc_id=sc_id,
            orbit=orbit,
            mem_mlat=mem_mlat,
            mem_mlt=mem_mlt,
            mem_sza=mem_sza,
            mem_mode=mem_mode,
            output_dir=out_dir_path,
            regions=regions_to_plot,
            overwrite=overwrite,
            figure_dpi=figure_dpi,
            dark_mode=dark_mode,
        )
        plot_daily_maps(
            obs_date=obs_date,
            sc_id=sc_id,
            orbit=orbit,
            sclat=np.array(fdd["sat_lat"]),
            sclon=np.array(fdd["sat_lon"]),
            sc_mlat=np.array(fdd["sat_mlat"]),
            sc_mlt=np.array(fdd["sat_MLT"]),
            mem_lat=mem_lat,
            mem_lon=mem_lon,
            mem_mlat=mem_mlat,
            mem_mlt=mem_mlt,
            mem_mode=mem_mode,
            output_dir=out_dir_path,
            regions=regions_to_plot,
            south_inverted=True,
            overwrite=overwrite,
            figure_dpi=figure_dpi,
            dark_mode=dark_mode,
        )

    # Finished - log elapsed execution time
    t_end = datetime.datetime.now(UTC_TZ)
    logger.info(f"Elapsed time (seconds): {(t_end - t_bgn).total_seconds():7.3f}")
    return 0


if __name__ == "__main__":
    main(max_content_width=120)
