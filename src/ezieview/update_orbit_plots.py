# region imports
import datetime
from pathlib import Path

import click
import spiceypy
from netCDF4 import Dataset

from . import plot_L3_Bdown
from .ezvislib import gw_logger, gw_plot_methods
from .ezvislib.gw_plot_params import (
    CLI_DATE_FORMAT,
    DFLT_RES,
    UTC_TZ,
)
from .ezvislib.gw_plot_utils import (
    filter_daily_files_by_version,
    filter_files_by_version,
    parse_ezie_product_name,
    split_on_science_segments,
)
from .ezvislib.kernel_mgr import spice_kernel_mgr

# endregion imports


# region cli
# Define command line options using 'click' package decorators.
@click.command(context_settings={"show_default": True})  #  , no_args_is_help=True)
# @click.option("-h", "--help", is_flag=True, help="Display this help message")
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
    default="ezie_l*.nc4",
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
    Generate a set of EZIE single-orbit summary plots based on the specified date and
    supplied file types, as determined by the supplied file pattern input. Options to
    make plots at higher resolution (-hr/--high_res_dpi = 300 DPI, default is 100 DPI)
    and in 'dark mode' are also available.

    Example command line--run from repo root, in an activated python venv:

    \b
    python3 update_orbit_plots.py \\
        -d0 2025-06-20 \\
        -d1 2025-06-22 \\
        -fd '/project/ezie/data/' \\
        -fp 'ezie_l1_*.nc4'  \\
        -pd "/project/ezie/gateway/plots"

    """
    # Begin - configure logging, log all supplied arguments
    t_bgn = datetime.datetime.now(UTC_TZ)
    # Using method's default log output folder and log level (INFO)
    start_date = start_date.replace(tzinfo=UTC_TZ)
    stop_date = stop_date.replace(tzinfo=UTC_TZ).replace(hour=23, minute=59, second=59)
    logger = gw_logger.initialize_logging(log_to_file=True, rotating=True)
    logger.info("======= NEW RUN =======")
    context = click.get_current_context()
    for nam, val in context.params.items():
        logger.info(f"{nam:18s} : {val}")

    # Create list of highest version/revision for a given spacecraft and date. For
    # products that require separation of full-day files into individual science passes,
    # that segmentation will be performed in the individual product method.
    if "_l1_" in file_pattern:
        keep_files = filter_daily_files_by_version(
            file_directory=file_directory,
            file_pattern=file_pattern,
            start_date=start_date,
            stop_date=stop_date,
        )
    else:
        keep_files = filter_files_by_version(
            file_directory=file_directory,
            file_pattern=file_pattern,
            start_date=start_date,
            stop_date=stop_date,
            remove_near_dupes=True,  # FIXME: Set to False for test files lacking revisions
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

    # FIXME: Need to load these until such time as all relevant data is available
    # directly from .nc4 files. We are currently lacking:
    # 1) SV magnetic coordinates (lat, lon, MLT)
    # 2) Solar subpoint magnetic (and geodetic?) coordinates to correctly orient MLT
    #    plots. Geodetic probably not needed? TBD
    spk, lsk, pck = spice_kernel_mgr()
    spiceypy.furnsh([spk.as_posix(), lsk.as_posix(), pck.as_posix()])

    # Diagnostics
    # count = spiceypy.ktotal("ALL")
    # logger.info(f"Kernel count after load: {count}")
    # for i in range(0, count):
    #     [file, ktype, _source, _handle] = spiceypy.kdata(i, "ALL")
    #     logger.info(f"File   {file}")
    #     logger.info(f"Type   {ktype}")

    keep_files = keep_files[::-1]  # FIXME: Process newest data first? OPTIONAL
    for each_file in keep_files:
        prsd = parse_ezie_product_name(each_file)
        if str(prsd.vrsn) == "nan":
            logger.error(f"Corrupted file name? {each_file}")
            continue
        version = f"{prsd.vrsn}_{prsd.rvsn}"

        if "l1" in each_file.stem.lower():
            with Dataset(each_file, mode="r") as nc_data:
                try:
                    nobs = nc_data.dimensions["ObsRate"].size
                except Exception as exc:
                    logger.error(
                        f"Exception: {exc} obtaining ObsRate in file "
                        f"{Path(each_file).as_posix()}"
                    )
                    continue
                if nobs < 10:
                    logger.error(
                        f"Problem file (ObsRate={nobs}): {Path(each_file).as_posix()}"
                    )
                    continue
                product = nc_data.getncattr("product")
                if product == "L1":
                    logger.info(f"Processing {product} file {Path(each_file).name}")
                    # Slice out the time intervals during which we may have had MEM
                    # observations.
                    pass_indices = split_on_science_segments(nc_data)

                    for index_pair in pass_indices:
                        # Requires L1 product file - not populated in L2 product ATM
                        # FIXME: Passing tdiff=True will plot the temperature
                        # differences from one time step to the next, rather than the
                        # current value at that timestep (project scientist request).
                        # for t_diff in [True, False]:
                        for t_diff in [False]:
                            gw_plot_methods.plot_calibration(
                                nc_data=nc_data,
                                source=each_file,
                                indices=index_pair,
                                save_directory=out_dir_path,
                                t_diff=t_diff,
                                overwrite=overwrite,
                                dark_mode=dark_mode,
                                figure_dpi=figure_dpi,
                            )

                        gw_plot_methods.plot_geolocation(
                            nc_data=nc_data,
                            source=each_file,
                            indices=index_pair,
                            save_directory=out_dir_path,
                            overwrite=overwrite,
                            figure_dpi=figure_dpi,
                            dark_mode=dark_mode,
                        )  # Requires L0A or higher level product file

                        gw_plot_methods.plot_ancillary(
                            nc_data=nc_data,
                            source=each_file,
                            indices=index_pair,
                            save_directory=out_dir_path,
                            overwrite=overwrite,
                            figure_dpi=figure_dpi,
                            dark_mode=dark_mode,
                        )  # Requires L1 or higher level product files

                else:
                    logger.error(f"{product} misidentified as L1-skipping")

        elif "l2" in each_file.stem.lower():
            # NOTE: L2 files are already segmented into individual science passes based
            # on the algorithm in split_on_science_segments(). No further distinction
            # needs to be made.

            with Dataset(each_file, mode="r") as nc_data:
                nobs = nc_data.dimensions["ObsRate"].size
                if nobs < 10:
                    logger.error(
                        f"Problem file (ObsRate={nobs}): {Path(each_file).as_posix()}"
                    )
                    continue
                if nc_data.getncattr("product") == "L2":
                    logger.info(f"Processing L2 file {Path(each_file).name}")

                    gw_plot_methods.plot_retrieved_bd_only(
                        nc_data=nc_data,
                        source=each_file,
                        save_directory=out_dir_path,
                        version=version,
                        mode="Corrected",
                        south_inverted=True,  # Use heliospheric community mapping style
                        overwrite=overwrite,
                        figure_dpi=300,
                        dark_mode=dark_mode,
                    )  # Requires L2 or higher level products files

                    gw_plot_methods.plot_retrieved_bd_only(
                        nc_data=nc_data,
                        source=each_file,
                        save_directory=out_dir_path,
                        version=version,
                        mode="Uncorrected",
                        south_inverted=True,
                        overwrite=overwrite,
                        figure_dpi=300,
                        dark_mode=dark_mode,
                    )  # Requires L2 or higher level products files

                else:
                    logger.error(
                        f"{nc_data.getncattr('product')} misidentified as L2-skipping"
                    )

        elif "l3" in each_file.stem.lower():
            # FIXME - pipeline L3 file format does not match Brent's latest?
            # L3 files do not yet seem to have any of the common metadata or variables?
            with Dataset(each_file, mode="r") as nc_data:
                try:
                    nobs = nc_data["l2_data/time_utc"].size
                except Exception as exc:
                    logger.error(
                        f"Exception: {exc} field (time_utc) missing in "
                        f"file {each_file.as_posix()}"
                    )
                    continue
                if nobs < 10:
                    logger.error(
                        f"Problem file (time={nobs}): {Path(each_file).as_posix()}"
                    )
                    continue
                plot_L3_Bdown.plot_b_1D_maps_with_time(
                    nc_data=nc_data,
                    source=each_file,
                    save_directory=out_dir_path,
                    south_inverted=True,
                    overwrite=overwrite,
                    dark_mode=dark_mode,
                    figure_dpi=200,  # FIXME - What Brent was using, keep for now
                )

                # if nc_data.getncattr("product") == "L3":
                #     logger.info(f"Processing L3 file {Path(each_file).name}")
                # else:
                #     logger.error(
                #         f"{nc_data.getncattr('product')} misidentified as L3-skipping"
                #     )

    # Finished - log elapsed execution time
    spiceypy.kclear()
    t_end = datetime.datetime.now(UTC_TZ)
    logger.info(f"Elapsed time (seconds): {(t_end - t_bgn).total_seconds():7.3f}")
    return 0


if __name__ == "__main__":
    main(max_content_width=120)
