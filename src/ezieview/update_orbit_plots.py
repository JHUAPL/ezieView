"""
Generate EZIE single-orbit summary plots.

For each per-orbit product file in the requested date range, produces calibration,
geolocation, ancillary, and retrieved plasma parameter plots, as appropriate for the
product level (L1, L2, or L3).

    Example command line--run from repo root, in an activated python venv:

    view_orbits \\
        -d0 2025-06-20 \\
        -d1 2025-06-22 \\
        -fd '/project/ezie/data/l2' \\
        -fp 'ezie_l2_*.nc4'  \\
        -pd "/project/ezie/plots/single-orbit"
"""

# region imports
import datetime
import logging
import multiprocessing
import os
from pathlib import Path

import click
import spiceypy
from netCDF4 import Dataset

from ezieview.ezvislib import gw_logger, gw_plot_methods
from ezieview.ezvislib.gw_plot_params import (
    CLI_DATE_FORMAT,
    DFLT_RES,
    UTC_TZ,
)
from ezieview.ezvislib.gw_plot_utils import (
    filter_files_by_version,
    parse_ezie_product_name,
)
from ezieview.ezvislib.kernel_mgr import spice_kernel_mgr

# endregion imports


def file_handler(
    each_file: Path,
    logger: logging.Logger,
    out_dir_path: Path,
    overwrite: bool,
    dark_mode: bool,
    figure_dpi: int,
    merged_l2_l3: bool,
):

    prsd = parse_ezie_product_name(each_file)
    if str(prsd.vrsn) == "nan":
        logger.error(f"Corrupted file name? {each_file}")
        return

    # NOTE: Kernels are furnished per file, so each worker process furnishes its own
    # copy when multiprocessing is enabled.
    spk, lsk, pck = spice_kernel_mgr()
    spiceypy.furnsh([spk.as_posix(), lsk.as_posix(), pck.as_posix()])

    if "l1" in each_file.stem.lower():
        with Dataset(each_file, mode="r") as nc_data:
            try:
                nobs = nc_data.dimensions["ObsRate"].size
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.error(
                    f"Exception: {exc} obtaining ObsRate in file {each_file.as_posix()}"
                )
                return
            if nobs < 10:
                logger.error(f"Problem file (ObsRate={nobs}): {each_file.as_posix()}")
                return

            product = nc_data.getncattr("product")
            if product == "L1":
                logger.info(f"Processing {product} file {each_file.name}")
                # Calibration plots requires L1 product file - not populated in L2
                # product ATM.

                # Slice out the time intervals during which we may have had MEM
                # observations.
                # pass_indices = split_on_science_segments(nc_data)  # daily L1
                pass_indices = [[0, -1]]  # When using per-orbit L1 files as inputs

                for index_pair in pass_indices:
                    # NOTE: Passing tdiff=True will plot the temperature differences
                    # from one time step to the next, rather than the current value
                    # at that timestep (project scientist request).
                    common = {
                        "source": each_file,
                        "indices": index_pair,
                        "save_directory": out_dir_path,
                        "overwrite": overwrite,
                        "figure_dpi": figure_dpi,
                        "dark_mode": dark_mode,
                    }
                    gw_plot_methods.plot_calibration(
                        nc_data=nc_data,
                        t_diff=False,
                        **common,
                    )
                    gw_plot_methods.plot_geolocation(
                        nc_data=nc_data,  # Requires L0A or higher level product file
                        **common,
                    )
                    gw_plot_methods.plot_ancillary(
                        nc_data=nc_data,  # Requires L1 or higher level product files
                        **common,
                    )

            else:
                logger.error(f"{product} misidentified as L1-skipping")

    elif "l2" in each_file.stem.lower():
        # NOTE: L2 files are already segmented into individual science passes based
        # on the algorithm in split_on_science_segments(). No further distinction
        # needs to be made.
        # version = f"{prsd.vrsn}_{prsd.rvsn}"

        try:
            nc_data = Dataset(each_file, mode="r")
        except OSError as exc:
            logger.error(f"Unable to open file {each_file.as_posix()}: {exc}")
            return

        with nc_data:
            nobs = nc_data.dimensions["ObsRate"].size
            if nobs < 10:
                logger.error(f"Problem file (ObsRate={nobs}): {each_file.as_posix()}")
                return

            product = nc_data.getncattr("product")
            if product == "L2" and not merged_l2_l3:
                logger.info(f"Processing L2 file {each_file.name}")

                gw_plot_methods.plot_retrieved_bd_only(
                    nc_data=nc_data,
                    source=each_file,
                    save_directory=out_dir_path,
                    # version=version,
                    south_inverted=True,  # Use heliospheric community mapping style
                    overwrite=overwrite,
                    figure_dpi=figure_dpi,
                    dark_mode=dark_mode,
                )  # Requires L2 or higher level products files

            elif product == "L2" and merged_l2_l3:
                gw_plot_methods.plot_retrieved_B_and_J(
                    nc2_data=nc_data,
                    nc2_sorc=each_file,
                    save_directory=out_dir_path,
                    # version=version,
                    south_inverted=True,  # Use heliospheric community mapping style
                    overwrite=overwrite,
                    figure_dpi=figure_dpi,
                    dark_mode=dark_mode,
                )  # Requires L2 or higher level products files

            else:
                logger.error(f"{product} misidentified as L2-skipping")

    elif "l3" in each_file.stem.lower():
        # TODO Version 2 L3 files do not yet have any much common metadata/attributes.
        # Grab pipeline version info from corresponding L2 file if it is not yet
        # incorporated in the supplied product. REMOVE this once ALL product files have
        # been updated.
        source_l2 = each_file.with_name(each_file.name.replace("l3", "l2", 1))
        git_branch = None
        git_commit = None
        for src in (each_file, source_l2):
            try:
                with Dataset(src, mode="r") as data:
                    git_branch = str(data["Configuration/git_branch"][:])
                    git_commit = str(data["Configuration/git_hash"][:])
                break
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.warning(f"Unable to get git branch/commit from {src}: {exc}")

        with Dataset(each_file, mode="r") as nc_data:
            try:
                nobs = nc_data["l2_data/time_utc"].size
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.error(
                    f"Exception: {exc} field (time_utc) missing in "
                    f"file {each_file.as_posix()}"
                )
                return
            if nobs < 10:
                logger.error(f"Problem file (time={nobs}): {each_file.as_posix()}")
                return

            gw_plot_methods.plot_b_1D_maps_with_time(
                nc_data=nc_data,
                source=each_file,
                save_directory=out_dir_path,
                south_inverted=True,
                git_branch=git_branch,  # Not yet in L3 product, borrowed from L2
                git_commit=git_commit,  # Not yet in L3 product, borrowed from L2
                overwrite=overwrite,
                dark_mode=dark_mode,
                figure_dpi=figure_dpi,
            )

    spiceypy.kclear()


# region cli
# Define command line options using 'click' package decorators.
@click.command(context_settings={"show_default": True})
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
    default="ezie_l*.nc4",
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
    "-merged",
    "--merged_l2_l3",
    type=bool,
    help="Generate a figure with both L2 and L3 results plotted",
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
    default=False,
    type=bool,
    is_flag=True,
    help="Generate plots using 'dark mode' format",
)
@click.option(
    "-dpi",
    "--figure_dpi",
    type=int,
    default=DFLT_RES,
    help=f"Generate plots at specified DPI rather than default value of {DFLT_RES}",
)
@click.option(
    "-mp",
    "--multiprocess",
    is_flag=True,
    help="Use a multiprocessing worker pool to process individual orbits in parallel",
)
# endregion cli
def main(
    start_date: datetime.datetime,
    stop_date: datetime.datetime,
    file_directory: str,
    file_pattern: str,
    plots_directory: str,
    merged_l2_l3: bool,
    overwrite: bool,
    dark_mode: bool,
    figure_dpi: int,
    multiprocess: bool,
) -> int:
    """
    Generate a set of EZIE single-orbit summary plots based on the specified date and
    supplied file types, as determined by the supplied file pattern input. Options to
    make plots at higher resolution (-dpi/--figure_dpi, default is 100 DPI)
    and in 'dark mode' are also available.

    Args:
        start_date: Start date of entries to be processed.
        stop_date: Stop date (inclusive) of entries to be processed.
        file_directory: Root directory path to be searched for EZIE input files.
        file_pattern: Pattern of file path to be searched for EZIE input files.
        plots_directory: Root directory in which to save generated plot files.
        merged_l2_l3: Generate a figure with both L2 and L3 results plotted.
        overwrite: Overwrite existing files. If not set, any existing files will
        be skipped.
        dark_mode: Generate plots using 'dark mode' format.
        figure_dpi: Resolution (DPI) at which to generate the plots.
        multiprocess: Use a multiprocessing worker pool to process individual
            orbits in parallel.

    Returns:
        int: 0 on success, 1 if no files were found matching the date specifications.
    """
    # Configure logging using method's default log output folder and log level (INFO)
    logger = gw_logger.initialize_logging(log_to_file=True, rotating=True)

    t_bgn = datetime.datetime.now(UTC_TZ)
    start_date = start_date.replace(tzinfo=UTC_TZ)
    stop_date = stop_date.replace(tzinfo=UTC_TZ).replace(hour=23, minute=59, second=59)

    # Log all supplied arguments
    logger.info("======= NEW RUN =======")
    context = click.get_current_context()
    for nam, val in context.params.items():
        logger.info(f"{nam:18s} : {val}")

    git_branch, git_hash = gw_logger.get_git_version()
    logger.info(f"git branch: {git_branch} commit hash: {git_hash}")

    # Create list of highest version/revision for a given spacecraft and date. For
    # products that require separation of full-day files into individual science passes,
    # that segmentation will be performed in the individual product method. We've now
    # switched to using per-orbit L1 files rather than dailies, matching corresponding
    # L2+L3 products.
    keep_files = filter_files_by_version(
        file_directory=file_directory,
        file_pattern=file_pattern,
        start_date=start_date,
        stop_date=stop_date,
        remove_near_dupes=True,  # NOTE: Set False for test files lacking revisions
    )

    if keep_files is None:
        logger.error("No files found matching date specifications. Exiting")
        return 1
    else:
        logger.info(
            f"Found {len(keep_files)} files matching specified pattern and date range"
        )

    # Re-sort from the lexicographical order (including the full path!) returned by
    # filter_files_by_version to one based on date/time primarily and SV secondarily,
    # processing oldest data first.
    keep_files.sort(
        key=lambda f: (
            parse_ezie_product_name(f).dttm,
            parse_ezie_product_name(f).spcv,
        )
    )
    for kept in keep_files:
        logger.debug(f"File: {kept.stem}")

    # Create output directory for storage of plot files if it does not already exist
    out_dir_path = Path(plots_directory)
    if not out_dir_path.exists():
        out_dir_path.mkdir(parents=True, exist_ok=True)

    # NOTE: Need to load these until such time as all relevant data is available
    # directly from .nc4 files. We are currently lacking:
    # 1) SV magnetic coordinates (lat, lon, MLT)
    # 2) Solar subpoint magnetic (and geodetic?) coordinates to correctly orient MLT
    #    plots. Geodetic probably not needed? TBD

    file_args = []
    if multiprocess:
        # Divide cpu count by 2 if you want to be nice to other users/activities.
        n_process = min(os.cpu_count() // 2, 15)  # max 15 orbits/day, should suffice
        with multiprocessing.Pool(processes=n_process) as pool:
            logger.info(f"Distributing jobs to {n_process} processes")
            # NOTE: Per-process logging is not currenrly supported when multiprocessing
            # is enabled. Only the main thread entries will be logged.
            file_args = [
                (
                    f,
                    logger,
                    out_dir_path,
                    overwrite,
                    dark_mode,
                    figure_dpi,
                    merged_l2_l3,
                )
                for f in keep_files
            ]
            pool.starmap(func=file_handler, iterable=file_args)

    else:
        for each_file in keep_files:
            file_handler(
                each_file=each_file,
                logger=logger,
                out_dir_path=out_dir_path,
                overwrite=overwrite,
                dark_mode=dark_mode,
                figure_dpi=figure_dpi,
                merged_l2_l3=merged_l2_l3,
            )

    # Finished - log elapsed execution time
    t_end = datetime.datetime.now(UTC_TZ)
    logger.info(f"Elapsed time (seconds): {(t_end - t_bgn).total_seconds():7.3f}")
    return 0


if __name__ == "__main__":
    main(max_content_width=120)
