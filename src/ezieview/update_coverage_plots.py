# region imports
import datetime
from pathlib import Path

import cartopy.feature as cfeature
import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from ezieview.ezvislib import gw_logger
from ezieview.ezvislib.gw_plot_methods import (
    coverage_plot_polar_layout,
    coverage_plot_stereographic_layout,
    mollweide_layout,
    save_close_figure,
)
from ezieview.ezvislib.gw_plot_params import (
    ALTERNATE_FIG_SIZE,
    BGN_COLOR,
    # BOUND_EQUATORIAL,
    BOUNDS,
    CLI_DATE_FORMAT,
    COVERAGE_TYPES,
    COVERAGE_UNITS,
    DARK_MODE_FACE_COLOR,
    DARK_MODE_FILL_COLOR,
    DARK_MODE_GRID_COLOR,
    DATA_TRANSFORM,
    DFLT_RES,
    EARTHLOOK,
    END_COLOR,
    EQUATORIAL,
    EZIE_COLOR,
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
    SKYLOOK,
    SOUTH,
    SPACECRAFT,
    SPCLOOK_COLOR,
    SPCLOOK_LABEL,
    SZA,
    UTC_TZ,
)
from ezieview.ezvislib.gw_plot_utils import (
    filter_daily_files_by_version,
    parse_ezie_product_name,
)

# endregion imports

mpl.use("Agg")
logger = gw_logger.initialize_logging(log_to_file=True, rotating=True)


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
        date_dict[prsd.spcv] = {}
        if "l1" in each_file.stem.lower():
            try:
                with Dataset(each_file, mode="r") as nc_data:
                    product = nc_data.getncattr("product")
                    if product == "L1":
                        logger.info(f"Processing {product} file {Path(each_file).name}")

                        for dbvar in coverage_db_list:
                            try:
                                date_dict[prsd.spcv][dbvar] = nc_data[dbvar][:].data
                            except AttributeError:
                                date_dict[prsd.spcv][dbvar] = np.repeat(
                                    str(nc_data[dbvar][:]),
                                    nc_data.dimensions["ObsRate"].size,
                                )
                    else:
                        logger.error(f"{product} misidentified as L1-skipping")
                        date_dict[prsd.spcv][dbvar] = None
            except Exception as exc:
                logger.error(f"Unrecoverable problem, skipping file: {exc}")
                date_dict[prsd.spcv][dbvar] = None

    # Combine entries from each SV/L1 file into single unified dictionary
    for sv_id, sv_dict in date_dict.items():
        for dbvar in coverage_db_list:
            full_key = dbvar.split("/")[-1]  # Use only variable name, not group
            if full_key not in full_day.keys():
                full_day[full_key] = []
            try:
                if sv_dict[dbvar] is not None:
                    full_day[full_key].extend(sv_dict[dbvar])
            except Exception as exc:
                logger.error(f"Selected field could not be extracted from file: {exc}")

    # Recast each dictionary value (list) as a numpy array
    for sv_id, sv_dict in date_dict.items():
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


def plot_mlt_sza_coverage(
    obs_date: datetime.datetime,
    sc_id: np.ndarray,
    orbit: np.ndarray,
    mem_mlat: np.ndarray,
    mem_MLT: np.ndarray,
    mem_sza: np.ndarray,
    mem_mode: np.ndarray,  # EARTHLOOK or SKYLOOK or ???
    output_dir: Path,
    regions: list,
    dark_mode: bool = False,
    overwrite: bool = False,
    figure_dpi: int = DFLT_RES,
):
    """
    Generate bar plots showing daily coverage, broken down by orbit and MEM, for:
        1. Magnetic latitude
        2. Magnetic Local Time
        3. Solar Zenith Angle
        Break L1 for each SV into orbits so we can make 2-D histogram of coverage versus
        orbit for this day, rather than just determining coverage for day as a whole.

    Args:
        obs_date (datetime.datetime): _description_
        sc_id (np.ndarray): _description_
        orbit (np.ndarray): _description_
        mem_mlat (np.ndarray): _description_
        mem_MLT (np.ndarray): _description_
        mem_sza (np.ndarray): _description_
        mem_mode (np.ndarray): _description_
        regions (list): _description_
        dark_mode (bool, optional): _description_. Defaults to False.
        overwrite (bool, optional): _description_. Defaults to False.
        figure_dpi (int, optional): _description_. Defaults to DFLT_RES.
    """

    orb_min = np.min(orbit)
    orb_max = MAX_ORBITS
    orb_res = 1
    orb_edg = np.arange(orb_min, orb_min + orb_max + orb_res, orb_res) - orb_res / 2

    cvrg: dict = {}
    for regn in regions:
        cvrg[regn] = {}
        # Generate plot with appropriate bounds based on the observed location.
        for cvrg_indx, cvrg_type in enumerate(COVERAGE_TYPES):
            cvrg_type_str = cvrg_type.lower().replace(" ", "-")
            plot_type = f"{regn.lower()}-mem-coverage-{cvrg_type_str}"
            fig_path = save_close_figure(
                obs_date=obs_date,
                plot_type=plot_type,
                save_directory=output_dir,
                dark_mode=dark_mode,
                dpi=figure_dpi,
                name_only=True,
            )

            if fig_path.exists() and not overwrite:
                logger.info(f"Skipping existing file: {fig_path.name}")
                continue
            logger.info(f"Generating MEM coverage plot of type ({cvrg_type})")

            if cvrg_type == MAGLAT:
                bin_res = 2.0
                bin_min = -90.0 if regn == SOUTH else +45.0 if regn == NORTH else -35.0
                bin_max = -45.0 if regn == SOUTH else +90.0 if regn == NORTH else +35.0
                bin_max += 2.0 * bin_res
                bin_qty = mem_mlat

            elif cvrg_type == SZA:
                bin_res = 5.0
                bin_min = 0.0
                bin_max = 180.0 + 2.0 * bin_res
                bin_qty = mem_sza

            elif cvrg_type == MLT:
                bin_res = 0.5
                bin_min = 0.0
                bin_max = 24.0 + bin_res
                bin_qty = mem_MLT

            bin_edg = np.arange(bin_min, bin_max, bin_res) - bin_res / 2

            sampled = np.zeros((len(SPACECRAFT), len(MEM_NUMBERS)))
            cmax = 2
            # Make 2-D histograms of coverage, filtering on hemisphere, spacecraft,
            # and science data collection status. Gather data for ALL spacecraft and
            # MEMs first, so we can standardize the scaling for all to a uniform
            # range.
            for sndx, spcrft in enumerate(SPACECRAFT):
                cvrg[regn][sndx] = {}
                for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
                    # Determine spacecraft science-collection/space-look status
                    scuse = sc_id == spcrft
                    rguse = (mem_mlat[mem_ndx] >= BOUNDS[regn][0]) & (
                        mem_mlat[mem_ndx] <= BOUNDS[regn][1]
                    )
                    lvuse = mem_mode[mem_ndx]
                    # Determine spacecraft's science-collection/space-look status
                    # TOD: Do we really need to track the difference between 'use' and
                    # dce' anymore, can we not just generate a single boolean mask for
                    # the points meeting all criteria?
                    dce = lvuse & scuse & rguse
                    sampled[sndx, mem_ndx] += np.sum(dce)
                    # Diagnostics - sample sizes for each plot
                    logger.debug(
                        f"{regn} "
                        f"{cvrg_type:20s} "
                        f"{spcrft} "
                        f"{mem_num} "
                        f"{np.sum(scuse):5d} "
                        f"{np.sum(rguse):5d} "
                        f"{np.sum(lvuse):5d} "
                        f"{np.sum(scuse & rguse):5d} "
                        f"{np.sum(dce):5d}"
                    )

                    # Get global statistics for each coverage type.
                    if np.sum(dce) > 0:
                        if cvrg_type == MLT:
                            bin_qty[mem_ndx][dce][
                                bin_qty[mem_ndx][dce] > bin_edg[-1]
                            ] -= 24.0
                        cvrg[regn][sndx][mem_ndx], _y_edg, _x_edg = np.histogram2d(
                            bin_qty[mem_ndx][dce],
                            orbit[dce],
                            bins=(bin_edg, orb_edg),
                        )
                        cmax = max(cmax, np.nanmax(cvrg[regn][sndx][mem_ndx]))
                    else:
                        # No valid observations, create blank template.
                        cvrg[regn][sndx][mem_ndx] = np.zeros(
                            (len(bin_edg) - 1, len(orb_edg) - 1)
                        )
                    cvrg[regn][sndx][mem_ndx][cvrg[regn][sndx][mem_ndx] == 0] = np.nan

                logger.info(
                    f"Number of samples for SV {spcrft} in region {regn} "
                    f"was {np.sum(sampled[sndx, :])}"
                )
                # end spacecraft loop

            if np.sum(sampled) == 0:
                # Don't bother to generate figures if region had 0 observations
                logger.info(f"Number of samples in region {regn} was zero, skipping.")
                continue

            if dark_mode:
                # Make dark mode modifications
                plt.style.use("dark_background")
                plt.rcParams["savefig.facecolor"] = DARK_MODE_FACE_COLOR
                plt.rcParams["axes.facecolor"] = DARK_MODE_FILL_COLOR

            # Instantiate figure and axes for plotting
            rows = 1
            cpsc = len(MEM_NUMBERS) + 2
            cols = len(SPACECRAFT) * cpsc
            fig = plt.figure(figsize=ALTERNATE_FIG_SIZE)

            lmrgn = +0.045
            rmrgn = -0.050
            bmrgn = +0.070
            tmrgn = +0.090
            wmrgn = +0.000
            cwdth = +0.012
            cmrgn = +0.008

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

                    # cmap = plt.get_cmap("OrRd", int(cmax))  # gradual 0 to > 0
                    cmap = plt.get_cmap("plasma", int(cmax))  # abrupt  0 to > 0
                    cmap.set_under("white")  # For bins that had NO observations

                    haxs.set_ylim(bin_edg[0], bin_edg[-1])
                    if cvrg_type == MAGLAT:
                        haxs.set_ylim(
                            (-90.0, -45.0)
                            if regn == SOUTH
                            else (+45.0, +90.0)
                            if regn == NORTH
                            else (-30.0, +30.0)
                        )
                    elif cvrg_type == MLT:
                        haxs.set_yticks(range(0, 24, 2))
                    elif cvrg_type == SZA:
                        haxs.yaxis.set_major_locator(
                            plt.MaxNLocator(integer=True, nbins=10)
                        )

                    cb_img = haxs.imshow(
                        cvrg[regn][sndx][mem_ndx],
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
                    haxs.set_xlabel(
                        f"MEM {mem_ndx + 1}"
                        # f" - Look Direction {MEM_LOOK_DIRECTIONS[mem_ndx]}"
                        # r"º"
                        # r"$\bf{^\circ}$"
                    )

                    if mem_ndx == NUM_MEM - 1:
                        cbar_axs = plt.axes(
                            (
                                lmrgn
                                + cmrgn
                                + (1.0 - lmrgn - rmrgn)
                                / (NUM_SPC * cpsc)
                                * (sndx * cpsc + NUM_MEM),
                                bmrgn,
                                cwdth,
                                1.0 - bmrgn - tmrgn,
                            )
                        )
                        cbar = plt.colorbar(
                            cb_img,
                            cax=cbar_axs,
                            extend="min",
                            fraction=0.20,
                        )
                        cbar.set_label("Number of samples")
                        # Massage colorbar tick placement and labeling
                        cbar.locator = plt.MaxNLocator(integer=True, nbins="auto")

            # Tweak position and add any figure-level annotation
            fig.suptitle(
                f"MEM {cvrg_type} Coverage, All Orbits: "
                f"{obs_date.strftime('%Y-%m-%d (%j)')}",
                weight="bold",
                fontsize="x-large",
            )
            plt.subplots_adjust(
                left=lmrgn,
                right=1.0 - rmrgn,
                bottom=bmrgn,
                top=1.0 - tmrgn,
                wspace=wmrgn,
                # wspace=0.18,
            )

            save_close_figure(
                figure=fig,
                obs_date=obs_date,
                plot_type=plot_type,
                save_directory=output_dir,
                dark_mode=dark_mode,
                dpi=figure_dpi,
            )
            # Restore default rcParams values
            mpl.rcParams.update(mpl.rcParamsDefault)


def plot_daily_maps(
    obs_date: datetime.datetime,
    sc_id: np.ndarray,
    orbit: np.ndarray,
    sclat: np.ndarray,
    sclon: np.ndarray,
    sc_mlat: np.ndarray,
    sc_MLT: np.ndarray,
    mem_lat: np.ndarray,
    mem_lon: np.ndarray,
    mem_mlat: np.ndarray,
    mem_MLT: np.ndarray,
    mem_mode: np.ndarray,  # EARTHLOOK or SKYLOOK or ???
    output_dir: Path,
    regions: list,
    south_inverted: bool = False,
    overwrite: bool = False,
    dark_mode: bool = False,
    figure_dpi: int = DFLT_RES,
):
    """
    Generate polar stereographic plots showing spacecraft passes, broken down by
    orbit, in two formats:
    1) Distinguishing between data collection events and space-look operations, and
    2) Mapping the locations observed during each orbit by each individual MEM.
    """

    # Plot marker type and size specs
    spv_mrk = "o"
    mem_mrk = "o"
    end_mrk = "o"
    spv_siz = 2
    mem_siz = 2
    end_siz = 4
    mrkr_scal = 2  # Magnify marker size for legend display
    locn_stride = 1  # don't need to plot every point? TBD
    rasterized = True  # Speed rendering of PNG? -- It does not need to be zoomable!

    # Plot ID suffixes...
    data_collect_geo = "mem-data-collection-events-geo"
    mem_obs_locn_geo = "mem-observation-locations-geo"
    data_collect_mag = "mem-data-collection-events-mag"
    mem_obs_locn_mag = "mem-observation-locations-mag"
    # Corresponding title formats for each plot format
    sup_ttl = {}
    sup_ttl[data_collect_geo] = "EZIE Data Collection Events: Geographic Lat/Lon"
    sup_ttl[data_collect_mag] = "EZIE Data Collection Events: Magnetic Lat/MLT"
    sup_ttl[mem_obs_locn_geo] = "EZIE MEM Observation Locations: Geographic Lat/Lon"
    sup_ttl[mem_obs_locn_mag] = "EZIE MEM Observation Locations: Magnetic Lat/MLT"

    # Select region[s] and plot ID suffixes to be generated, collected in lists. Loop
    # through all plot types and regions, skipping formats that are inappropriate for a
    # given type or region.
    # Polar plots, north and south AEJ
    for ptype in [
        data_collect_geo,
        mem_obs_locn_geo,
        data_collect_mag,
        mem_obs_locn_mag,
    ]:
        for regn in regions:
            sampled = np.zeros((len(SPACECRAFT), len(MEM_NUMBERS)))
            MLT_sign = +1
            if regn in [NORTH, SOUTH]:
                # Instantiate figure and axes for plotting, with desired projection
                if "geo" in ptype:
                    xcrd = sclon
                    ycrd = sclat
                    xmem = mem_lon
                    ymem = mem_lat
                    show_mag_lat = True
                    plot_method = coverage_plot_stereographic_layout
                else:
                    MLT_sign = -1 if (south_inverted and regn == SOUTH) else +1
                    # FIXME: SC mlat is faked with MEM mlat right now
                    xcrd = sc_MLT * 15.0
                    ycrd = sc_mlat
                    xmem = mem_MLT * 15.0
                    ymem = mem_mlat
                    show_mag_lat = False
                    plot_method = coverage_plot_polar_layout
                    # Uses MLT kludge instead of longitude

                plot_type = f"{regn.lower()}-{ptype}"
                fig_path = save_close_figure(
                    obs_date=obs_date,
                    plot_type=plot_type,
                    save_directory=output_dir,
                    dark_mode=dark_mode,
                    dpi=figure_dpi,
                    name_only=True,
                )

                if fig_path.exists() and not overwrite:
                    logger.info(f"Skipping existing file: {fig_path.name}")
                    continue
                logger.info(f"Generating MEM coverage plot of type {plot_type}")

                fig, all_axs = plot_method(
                    observation_date=obs_date,
                    hemisphere=regn,
                    dark_mode=dark_mode,
                    show_mag_lat=show_mag_lat,
                    south_inverted=True,
                )

                for sndx, spcrft in enumerate(SPACECRAFT):
                    unique_orbits = np.unique(orbit)
                    for ondx, orbnum in enumerate(unique_orbits):
                        # Filter on spacecraft ID and orbit number
                        scuse = (sc_id == spcrft) & (orbit == orbnum)
                        rguse = (sc_mlat >= BOUNDS[regn][0]) & (
                            sc_mlat <= BOUNDS[regn][1]
                        )  # filter on samples in proper location

                        if not np.any(scuse & rguse):
                            continue  # no observations in region for this spacecraft

                        for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
                            # Determine overall SV science-collection/space-look status
                            # rguse = (mem_mlat[mem_ndx] >= BOUNDS[regn][0]) & (
                            #     mem_mlat[mem_ndx] <= BOUNDS[regn][1]
                            # )
                            dce = mem_mode[mem_ndx] & scuse & rguse
                            slk = (~mem_mode[mem_ndx]) & scuse & rguse
                            sampled[sndx, mem_ndx] += np.sum(dce)

                            logger.debug(
                                f"{regn} "
                                f"{orbnum:5d} "
                                f"{spcrft} "
                                f"{mem_num} "
                                f"{np.sum(scuse):5d} "
                                f"{np.sum(rguse):5d} "
                                f"{np.sum(scuse & rguse):5d} "
                                f"{np.sum(slk):5d} "
                                f"{np.sum(dce):5d}"
                            )

                        # TODO: Speed up plotting of maps. Why are they so slow? Is it
                        # the calculating or the rendering that is so slow? TBD
                        if "mem-observation" in ptype:  # plot MEM footprint[s]
                            all_axs[sndx].plot(
                                MLT_sign * xcrd[scuse & rguse],
                                ycrd[scuse & rguse],
                                transform=DATA_TRANSFORM,
                                color=EZIE_COLOR,
                                label=EZIE_LABEL,  # if ondx == 0 else None,
                                zorder=2,
                                marker=spv_mrk,
                                ms=spv_siz,
                                lw=0,
                                ls="none",
                                markevery=locn_stride,
                                rasterized=rasterized,
                            )

                            for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
                                # Determine MEM science-collection/space-look status
                                dce = mem_mode[mem_ndx] & scuse & rguse
                                if np.sum(dce) > 0:
                                    all_axs[sndx].plot(
                                        MLT_sign * xmem[mem_ndx][dce],
                                        ymem[mem_ndx, dce],
                                        label=f"MEM {mem_num}",
                                        transform=DATA_TRANSFORM,
                                        color=MEM_CLR[mem_ndx],
                                        zorder=3,
                                        marker=mem_mrk,
                                        ms=mem_siz,
                                        lw=0,
                                        ls="none",
                                        markevery=locn_stride,
                                        rasterized=rasterized,
                                    )

                        else:  # plot spacecraft footprint
                            if np.sum(sampled[sndx, :]) > 0:
                                # Had at least one MEM with data collection event
                                all_axs[sndx].plot(
                                    MLT_sign * xcrd[dce],
                                    ycrd[dce],
                                    ms=spv_siz,
                                    marker=spv_mrk,
                                    zorder=3,
                                    transform=DATA_TRANSFORM,
                                    color=SCIENCE_COLOR,
                                    label=SCIENCE_LABEL,  # if ondx == 0 else None,
                                    lw=0,
                                    ls="none",
                                    markevery=locn_stride,
                                    rasterized=rasterized,
                                )

                            if np.sum(slk) > 0:  # Had at least one space look event
                                all_axs[sndx].plot(
                                    MLT_sign * xcrd[slk],
                                    ycrd[slk],
                                    ms=spv_siz,
                                    marker=spv_mrk,
                                    zorder=3,
                                    transform=DATA_TRANSFORM,
                                    color=SPCLOOK_COLOR,
                                    label=SPCLOOK_LABEL,  # if ondx == 0 else None,
                                    lw=0,
                                    ls="none",
                                    markevery=locn_stride,
                                    rasterized=rasterized,
                                )

                        all_axs[sndx].plot(
                            MLT_sign * xcrd[scuse & rguse][0],
                            ycrd[scuse & rguse][0],
                            marker=end_mrk,
                            ms=end_siz,
                            mfc=BGN_COLOR,
                            mec="black" if not dark_mode else DARK_MODE_GRID_COLOR,
                            label="Start",  # if not labeled else None,
                            transform=DATA_TRANSFORM,
                            zorder=5,
                            lw=0,
                            ls="none",
                            markevery=locn_stride,
                            rasterized=rasterized,
                        )
                        all_axs[sndx].plot(
                            MLT_sign * xcrd[scuse & rguse][-1],
                            ycrd[scuse & rguse][-1],
                            marker=end_mrk,
                            ms=end_siz,
                            mfc=END_COLOR,
                            mec="black" if not dark_mode else DARK_MODE_GRID_COLOR,
                            label="End",  # if not labeled else None,
                            transform=DATA_TRANSFORM,
                            zorder=5,
                            lw=0,
                            ls="none",
                            markevery=locn_stride,
                            rasterized=rasterized,
                        )
                        # labeled = True

                        # Add _second_ legend denoting MEM footprint or spacecraft
                        # activity versus position
                        handles, labels = all_axs[sndx].get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        all_axs[sndx].legend(
                            by_label.values(),
                            by_label.keys(),
                            bbox_to_anchor=(1.10, +0.05),
                            loc="upper right",
                            markerscale=mrkr_scal,
                            fontsize="x-small"
                            if ptype == data_collect_geo
                            else "x-small",
                        )

                    logger.info(
                        f"Number of samples for SV {spcrft} in region {regn} "
                        f"was {np.sum(sampled[sndx, :])}"
                    )
                    # end spacecraft loop

                if np.sum(sampled) == 0:
                    # Don't bother to make figure if hemisphere had 0 observations
                    logger.info(
                        f"Number of samples in region {regn} was zero, skipping."
                    )
                    plt.close(fig)
                    continue

                # Tweak position and add any figure-level annotation
                fig.suptitle(
                    f"{sup_ttl[ptype]}, All Orbits: "
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
            # end AEJ mode

    # Mollweide plots, EEJ
    if EQUATORIAL in regions:  # Otherwise do not bother with this plot type
        regn = EQUATORIAL
        for ptype in [
            data_collect_geo,
            mem_obs_locn_geo,
        ]:
            xcrd = sclon
            ycrd = sclat
            xmem = mem_lon
            ymem = mem_lat

            plot_type = f"equatorial-{ptype}"
            fig_path = save_close_figure(
                obs_date=obs_date,
                plot_type=plot_type,
                save_directory=output_dir,
                dark_mode=dark_mode,
                dpi=figure_dpi,
                name_only=True,
            )

            if fig_path.exists() and not overwrite:
                logger.info(f"Skipping existing file: {fig_path.name}")
                continue

            fig, all_axs = mollweide_layout(
                observation_date=obs_date,
                dark_mode=dark_mode,
                show_mag_lat=True,
                num_sc=NUM_SPC,
            )
            sampled = np.zeros((len(SPACECRAFT), len(MEM_NUMBERS)))
            for sndx, spcrft in enumerate(SPACECRAFT):
                map_axs = all_axs[sndx]
                map_axs.add_feature(cfeature.OCEAN)
                unique_orbits = np.unique(orbit)
                for ondx, orbnum in enumerate(unique_orbits):
                    # Filter on spacecraft ID and orbit number
                    scuse = (sc_id == spcrft) & (orbit == orbnum)
                    rguse = (sc_mlat >= BOUNDS[regn][0]) & (sc_mlat <= BOUNDS[regn][1])
                    if not np.any(scuse & rguse):
                        continue  # no observations for this spacecraft

                    # Determine overall SV science-collection/space-look status
                    for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
                        dce = mem_mode[mem_ndx] & scuse & rguse
                        slk = (~mem_mode[mem_ndx]) & scuse & rguse
                        sampled[sndx, mem_ndx] += np.sum(dce)
                        logger.debug(
                            f"{regn} "
                            f"{orbnum:5d} "
                            f"{spcrft} "
                            f"{mem_num} "
                            f"{np.sum(scuse):5d} "
                            f"{np.sum(rguse):5d} "
                            f"{np.sum(scuse & rguse):5d} "
                            f"{np.sum(slk):5d} "
                            f"{np.sum(dce):5d}"
                        )

                    if "mem-observation" in ptype:
                        map_axs.plot(
                            xcrd[scuse & rguse],
                            ycrd[scuse & rguse],
                            transform=DATA_TRANSFORM,
                            color=EZIE_COLOR,
                            label=EZIE_LABEL,  # if ondx == 0 else None,
                            zorder=2,
                            marker=spv_mrk,
                            ms=spv_siz,
                            lw=0,
                            ls="none",
                            markevery=locn_stride,
                            rasterized=rasterized,
                        )

                        for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
                            # Determine s/c's science-collection/space-look status
                            dce = mem_mode[mem_ndx] & scuse & rguse
                            slk = ~mem_mode[mem_ndx] & scuse & rguse
                            if np.sum(dce) > 0:
                                map_axs.plot(
                                    xmem[mem_ndx][dce],
                                    ymem[mem_ndx][dce],
                                    label=f"MEM {mem_num}",  # if ondx == 0 else None,
                                    transform=DATA_TRANSFORM,
                                    color=MEM_CLR[mem_ndx],
                                    zorder=3,
                                    marker=mem_mrk,
                                    ms=mem_siz,
                                    lw=0,
                                    ls="none",
                                    markevery=locn_stride,
                                    rasterized=rasterized,
                                )

                    else:
                        if np.sum(sampled[sndx, :]) > 0:
                            # Had at least one data collection event
                            map_axs.plot(
                                xcrd[dce],
                                ycrd[dce],
                                marker=spv_mrk,
                                ms=spv_siz,
                                zorder=3,
                                transform=DATA_TRANSFORM,
                                color=SCIENCE_COLOR,
                                label=SCIENCE_LABEL,  # if ondx == 0 else None,
                                lw=0,
                                ls="none",
                                markevery=locn_stride,
                                rasterized=rasterized,
                            )

                        if np.sum(slk) > 0:  # Had at least one space look event
                            map_axs.plot(
                                xcrd[slk],
                                ycrd[slk],
                                marker=spv_mrk,
                                ms=spv_siz,
                                zorder=3,
                                transform=DATA_TRANSFORM,
                                color=SPCLOOK_COLOR,
                                label=SPCLOOK_LABEL,  # if ondx == 0 else None,
                                lw=0,
                                ls="none",
                                rasterized=rasterized,
                            )

                    # Mark orbit start and endpoints
                    map_axs.plot(
                        xcrd[scuse & rguse][0],
                        ycrd[scuse & rguse][0],
                        marker=end_mrk,
                        ms=end_siz,
                        mfc=BGN_COLOR,
                        mec="black" if not dark_mode else DARK_MODE_GRID_COLOR,
                        mew=0.25,
                        label="Start",  # if ondx == 0 else None,
                        transform=DATA_TRANSFORM,
                        zorder=5,
                        lw=0,
                        ls="none",
                    )
                    map_axs.plot(
                        xcrd[scuse & rguse][-1],
                        ycrd[scuse & rguse][-1],
                        marker=end_mrk,
                        ms=end_siz,
                        mfc=END_COLOR,
                        mec="black" if not dark_mode else DARK_MODE_GRID_COLOR,
                        mew=0.25,
                        label="End",  # if ondx == 0 else None,
                        transform=DATA_TRANSFORM,
                        zorder=5,
                        lw=0,
                        ls="none",
                    )

                    # Add _second_ legend denoting MEM footprint or spacecraft
                    # activity versus position
                    handles, labels = map_axs.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    map_axs.legend(
                        by_label.values(),
                        by_label.keys(),
                        bbox_to_anchor=(1.00, +0.00),
                        loc="lower right",
                        markerscale=mrkr_scal,
                        fontsize="x-small" if ptype == data_collect_geo else "x-small",
                    )

                logger.info(
                    f"Number of samples for SV {spcrft} in region {regn} "
                    f"was {np.sum(sampled)}"
                )
            if np.sum(sampled) == 0:
                # Don't bother to make figure if hemisphere had 0 observations
                logger.info(f"Number of samples in region {regn} was zero, skipping.")
                plt.close(fig)
                continue

            # Tweak position and add any figure-level annotation
            fig.suptitle(
                f"{sup_ttl[ptype]}, All Orbits: {obs_date.strftime('%Y-%m-%d (%j)')}",
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
        # end EEJ mode

    # Restore default rcParams values before returning
    mpl.rcParams.update(mpl.rcParamsDefault)


# region cli
# Define command line options using 'click' package decorators.
# @click.command(context_settings={"show_default": True}, no_args_is_help=True)
# @click.option("-h", "--help", is_flag=True, help="Display this help message")
@click.command()
@click.option(
    "-d0",
    "--start_date",
    type=click.DateTime(formats=[CLI_DATE_FORMAT]),
    default=None,
    required=True,
    help="Start date of entries to be processed (YYYY-MM-DD)",
)
@click.option(
    "-d1",
    "--stop_date",
    type=click.DateTime(formats=[CLI_DATE_FORMAT]),
    default=None,
    required=True,
    help="Stop date (inclusive) of entries to be processed (YYYY-MM-DD)",
)
@click.option(
    "-fd",
    "--file_directory",
    default=None,
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
    default=None,
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
    start_date,
    stop_date,
    file_directory,
    file_pattern,
    plots_directory,
    overwrite,
    dark_mode,
    figure_dpi,
) -> int:
    """
    Generate a set of EZIE full-day summary plots based on the specified date and
    supplied file types, as determined by the supplied file pattern input. Options to
    make plots at higher resolution (-dpi/--figure_dpi, default is 100 DPI)
    and in 'dark mode' are also available.

    Example command line--run from repo root, in an activated python venv:

    \b
    glow_daily \\
        -d0 2025-06-20 \\
        -d1 2025-06-22 \\
        -fd '/project/ezie/data/l1' \\
        -fp 'ezie_l1_*.nc4'  \\
        -pd "/project/ezie/plots/daily-summary"

    """
    # python3 update_coverage_plots.py \\
    # Begin - configure logging, log all supplied arguments
    t_bgn = datetime.datetime.now(UTC_TZ)
    start_date = start_date.replace(tzinfo=UTC_TZ)
    stop_date = stop_date.replace(tzinfo=UTC_TZ).replace(hour=23, minute=59, second=59)
    # Using method's default log output folder and log level (INFO)
    logger.info("======= NEW RUN =======")
    context = click.get_current_context()
    for nam, val in context.params.items():
        logger.info(f"{nam:18s} : {val}")

    # Create list of highest version/revision for a given spacecraft and date. For
    # products that require separation of full-day files into individual science passes,
    # that segmentation will be performed in the individual product method.
    keep_files = filter_daily_files_by_version(
        file_directory=file_directory,
        file_pattern=file_pattern,
        start_date=start_date,
        stop_date=stop_date,
    )

    if keep_files is None:
        logger.error("No files found matching date specifications. Exiting")
        return 1
    else:
        logger.info(
            f"Found {len(keep_files)} files matching specified pattern and date range"
        )

    # Create output directory for storage of plot files if it does not already exist
    out_dir_path = Path(plots_directory)
    if not out_dir_path.exists():
        out_dir_path.mkdir(parents=True, exist_ok=True)

    dates = []
    for each_file in keep_files:
        prsd = parse_ezie_product_name(each_file)
        dates.append(prsd.date)
        logger.info(f"{prsd!s}")
        if str(prsd.vrsn) == "nan":
            logger.error(f"Corrupted file name? {each_file}")
            continue

    run_dates = sorted(set(dates))
    run_dates = run_dates[::-1]  # Newest first -- optional

    for run_date in run_dates:
        logger.info(f"Generating coverage plots and maps for {run_date!s}")
        fdd, uniq_svid, uniq_orbs = ingest_full_day_all_sv(
            run_date=run_date,
            used_files=keep_files,
        )  # Full Day Dictionary

        for key, val in fdd.items():
            logger.debug(f"{key:22s} {len(val)}")

        # Stack MEM coordinate arrays so we can loop through them in plotting routines
        mem_lat = np.vstack(
            (fdd["obs_lat1"], fdd["obs_lat2"], fdd["obs_lat3"], fdd["obs_lat4"])
        )
        mem_lon = np.vstack(
            (fdd["obs_lon1"], fdd["obs_lon2"], fdd["obs_lon3"], fdd["obs_lon4"])
        )
        mem_mlat = np.vstack(
            (
                fdd["obs_maglat1"],
                fdd["obs_maglat2"],
                fdd["obs_maglat3"],
                fdd["obs_maglat4"],
            )
        )
        mem_MLT = np.vstack(
            (
                fdd["obs_magLT1"],
                fdd["obs_magLT2"],
                fdd["obs_magLT3"],
                fdd["obs_magLT4"],
            )
        )
        mem_sza = np.vstack(
            (
                fdd["obs_solar_zen1"],
                fdd["obs_solar_zen2"],
                fdd["obs_solar_zen3"],
                fdd["obs_solar_zen4"],
            )
        )

        # Determine earth/sky look status at each time step for each MEM. The SV science
        # collection status as a whole is assumed to be valid if any one of the four MEM
        # is looking at the earth. Might change if JPL "Science" flag used instead? TBD
        mem_look = np.full_like(mem_mlat, dtype=object, fill_value=EARTHLOOK)
        mem_look[np.where(np.isnan(mem_mlat))] = SKYLOOK
        mem_mode = mem_look == EARTHLOOK
        sci_mode = np.sum(mem_mode, axis=0) > 0

        logger.info(
            f"MEM Obs/Earth/Sky {mem_mode.shape} {np.sum(mem_mode)} {np.sum(~mem_mode)}"
        )
        logger.info(
            f"SCI Obs/Earth/Sky {sci_mode.shape} {np.sum(sci_mode)} {np.sum(~sci_mode)}"
        )

        # Add region observation flag for each SV
        sv_regns = np.full(
            (len(REGIONS), len(SPACECRAFT)), fill_value=False, dtype=bool
        )
        for sndx, spcrft in enumerate(SPACECRAFT):
            for ondx, orbnum in enumerate(uniq_orbs):
                this_sv = fdd["SpaceVehicle"] == spcrft
                this_or = fdd["orbit_number"] == orbnum
                posns = this_sv & this_or
                loctn = np.nan
                if np.sum(posns) > 0:
                    if np.all(np.isnan(fdd["sat_mlat"][posns])):
                        logger.error("Unexpected result - all-NaN sat_mlat slice")
                    else:
                        loctn = np.nanmedian(fdd["sat_mlat"][posns])
                        for rndx, regn in enumerate(REGIONS):
                            if loctn >= BOUNDS[regn][0] and loctn <= BOUNDS[regn][1]:
                                sv_regns[rndx, sndx] = True

        # Determine which regions need to be plotted, having had one or more SVs make
        # observations there.
        plot_regn = np.full((len(REGIONS),), fill_value=False, dtype=bool)
        for rndx, regn in enumerate(REGIONS):
            if np.sum(sv_regns[rndx, :]) > 0:
                plot_regn[rndx] = True
            logger.info(f"{regn:10s} {plot_regn[rndx]!s:5s} {sv_regns[rndx, :]!r}")
        regions_to_plot = np.array(REGIONS)[plot_regn]
        logger.info(f"Plotting regions {regions_to_plot!s}")

        # Plot coverage for this date using combined daily entries for all SVs & orbits
        plot_mlt_sza_coverage(
            obs_date=datetime.datetime.strptime(run_date, "%Y%m%d"),
            sc_id=np.array(fdd["SpaceVehicle"]),
            orbit=np.array(fdd["orbit_number"]),
            mem_mlat=mem_mlat,
            mem_MLT=mem_MLT,
            mem_sza=mem_sza,
            mem_mode=mem_mode,
            output_dir=out_dir_path,
            regions=regions_to_plot,
            overwrite=overwrite,
            figure_dpi=figure_dpi,
            dark_mode=dark_mode,
        )

        plot_daily_maps(
            obs_date=datetime.datetime.strptime(run_date, "%Y%m%d"),
            sc_id=np.array(fdd["SpaceVehicle"]),
            orbit=np.array(fdd["orbit_number"]),
            sclat=np.array(fdd["sat_lat"]),
            sclon=np.array(fdd["sat_lon"]),
            sc_mlat=np.array(fdd["sat_mlat"]),
            sc_MLT=np.array(fdd["sat_MLT"]),
            mem_lat=mem_lat,
            mem_lon=mem_lon,
            mem_mlat=mem_mlat,
            mem_MLT=mem_MLT,
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
