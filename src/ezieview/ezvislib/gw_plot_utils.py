# region imports
import collections
import datetime
import functools
import logging
import math
import os
import re
import time
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import netCDF4
import numpy as np
from apexpy import Apex
from netCDF4 import Dataset
from PIL import Image

from ezieview.ezvislib.gw_plot_params import DFLT_RES, REFERENCE_ALTITUDE_KM

# endregion

# region globals
logger = logging.getLogger(__name__)

# endregion


def split_on_calibration_segments(
    nc_data: netCDF4.Dataset,
):
    """
    Build table of observations where the MEM acquisition mode flag is set to
    CALIBRATION (1) or ENGINEERING (3). Use the corresponding time segments as a proxy
    for orbit number, pending possible inclusion of ACTUAL orbit numbers in the EZIE
    products at some later date.

    Based on Rob Barnes' algorithm in SplitL1.py, but more pythonified.

    MEM acquisition mode flags (as defined in the L1 files from Sid/JPL):
    - 0 - Other
    - 1 - Calibration
    - 2 - Science
    - 3 - Engineering

    Args:
        nc_data (netCDF4.Dataset): EZIE L1 file

    Returns:
        2-D array of (integer) start and stop indices for multiple discrete science
        observations in an EZIE L1 file. There are currently ~15 orbits per day, with
        one (or perhaps more? TBD) science observations per orbit.
    """
    CALIBRATION: int = 1
    ENGINEERING: int = 3
    MAX_GAP: float = 60.0  # seconds
    obs_max: int = nc_data["Time/time_utc"].size

    # FIXME: Switch to using MEM footprint != NaN instead of mode == SCIENCE? TBD

    # Check acquisition_mode status for each time step.
    obs_flg = (nc_data["ChannelOrderedCounts/acquisition_mode"][:] == CALIBRATION) | (
        nc_data["ChannelOrderedCounts/acquisition_mode"][:] == ENGINEERING
    )

    # Set obsflg to False where time gap is > MAX_GAP between consecutive steps.
    t_jmp = np.diff(nc_data["Time/time_tai"][:]) > MAX_GAP
    obs_flg[np.argwhere(t_jmp).flatten()] = False

    # Get indices of time steps where obs_flg changes from its previous value.
    status_change = np.diff(obs_flg) > 0
    seg_lst = [int(1 + x) for x in np.argwhere(status_change).flatten()]

    # Check for segments in progress at start or end of file, enclose them as needed
    if obs_flg[0]:  # We're starting off already in a desired mode, "open" the segment
        seg_lst = [0] + seg_lst
    if obs_flg[-1]:  # We're ending while still in a desired mode, "close" the segment
        seg_lst = seg_lst + [obs_max - 1]

    # Place paired segment starts and stops in list of 2-tuples (1-D list of tuples)
    pair_iterator = iter(seg_lst)
    seg_lst = list(zip(pair_iterator, pair_iterator))

    return seg_lst


def split_on_science_segments(
    nc_data: netCDF4.Dataset,
):
    """
    Build table of observations where at least one MEM intersects the Earth and the
    MEM acquisition mode flag is set to SCIENCE (2) for at least one MEM. Use the
    corresponding time segments as a proxy for orbit number, pending possible
    inclusion of ACTUAL orbit numbers in the EZIE products at some later date.

    Based on Rob Barnes' algorithm in SplitL1.py, but more pythonified.

    MEM acquisition mode flags (as defined in the L1 files from Sid/JPL):
    - 0 - Other
    - 1 - Calibration
    - 2 - Science
    - 3 - Engineering

    Args:
        nc_data (netCDF4.Dataset): EZIE L1 file

    Returns:
        2-D array of (integer) start and stop indices for multiple discrete science
        observations in an EZIE L1 file. There are currently ~15 orbits per day,
        with one (or perhaps more? TBD) science observations per orbit.
    """
    SCIENCE: int = 2
    MAX_GAP: float = 60.0  # seconds
    obs_max: int = nc_data["Time/time_utc"].size

    # FIXME: Switch to using MEM footprint != NaN instead of mode == SCIENCE? TBD

    # Check for valid numeric results for each MEM obs_lat. iflg will be True for a
    # given MEM and timestep if the geolocated MEM footprint latitude is not NaN.
    # Q: Do we need np.isinf() check too? TBD
    # iflg = np.full((NUM_MEM, obs_max), fill_value=False)
    # for mem_ndx, mem_num in enumerate(MEM_NUMBERS):
    #     iflg[mem_ndx, :] = ~np.isnan(nc_data[f"Geolocation/obs_lat{mem_num}"][:])

    # Check acquisition_mode flags and MEM geolocation status for each time step.
    # obs_flg is True where one or more MEMs had a valid latitude geolocation value
    # (iflg[mem,obs] = True) and the acquisition mode value was SCIENCE (=2).
    # obs_flg = (np.sum(iflg, axis=0) > 0) & (
    #     nc_data["ChannelOrderedCounts/acquisition_mode"][:] == SCIENCE
    # )
    obs_flg = nc_data["ChannelOrderedCounts/acquisition_mode"][:] == SCIENCE

    # Set obsflg to False where time gap is > MAX_GAP between consecutive steps.
    t_jmp = np.diff(nc_data["Time/time_tai"][:]) > MAX_GAP
    obs_flg[np.argwhere(t_jmp).flatten()] = False

    # Get indices of time steps where obs_flg changes from its previous value.
    status_change = np.diff(obs_flg) > 0
    seg_lst = [int(1 + x) for x in np.argwhere(status_change).flatten()]

    # Check for segments in progress at start or end of file, enclose them as needed
    if obs_flg[0]:  # We're starting off already in SCIENCE mode, "open" the segment
        seg_lst = [0] + seg_lst
    if obs_flg[-1]:  # We're ending while still in SCIENCE mode, "close" the segment
        seg_lst = seg_lst + [obs_max - 1]

    # Place paired segment starts and stops in their own columns (2-D array)
    # seg_lst = np.array(seg_lst).reshape(len(seg_lst) // 2, 2)  # OR alternatively ...
    # Place paired segment starts and stops in list of 2-tuples (1-D list of tuples)
    pair_iterator = iter(seg_lst)
    seg_lst = list(zip(pair_iterator, pair_iterator))

    return seg_lst


def extract_science_passes(nc_data: netCDF4.Dataset):
    """
    For products that require separation of full-day files into individual science
    passes, break up observations into pairs of indices that start and finish at
    observation gap edges. Until we get actual orbit numbers, these will serve as a
    proxy for orbit boundaries.

    NOTE: This has been superseded by split_on_science_segments() above for L1 file
    processing, which uses the acquisition_mode flag from the L1 file to better bracket
    science passes. The L2 files do not yet have that flag, but have already been split
    into individual science passes based on that algorithm.

    Args:
        nc_data (Dataset, optional): Path to full-day netCDF file. Defaults to None.

    Returns:
        list of 2-tuples: Bracketing what we believe are "good" observation intervals.
    """
    timevals = nc_data["Time/time_tai"][:]
    timediff = np.diff(timevals)
    timejump = timediff > 1.0e3
    timerang = [
        0,
        *[int(x) for x in np.argwhere(timejump).flatten()],
        len(timevals),
    ]
    # Break up observations into pairs of indices that start and finish at gap edges.
    index_pairs = []
    for tt, t0 in enumerate(timerang[:-1]):
        t1 = timerang[tt + 1]
        i0, i1 = t0 + 1, t1 - 1  # Avoid gap indices themselves
        # Extract subrange where all MEM's have been geolocated
        mem_good = (
            (~np.isnan(nc_data["Geolocation/obs_lat1"][i0:i1]))
            & (~np.isnan(nc_data["Geolocation/obs_lat2"][i0:i1]))
            & (~np.isnan(nc_data["Geolocation/obs_lat3"][i0:i1]))
            & (~np.isnan(nc_data["Geolocation/obs_lat4"][i0:i1]))
        )
        if np.sum(mem_good) > 0:
            i0 += np.argmax(mem_good)
            i1 -= np.argmax(mem_good[::-1])
            # Trim off timesteps where we're slewing at start of observation
            i0 += 6
            # i1 -= 6  # Necessary at end of pass as well? TBD
            index_pairs.append((int(i0), int(i1)))
    # Return pairs bracketing what we believe are "good" observation intervals.
    if len(index_pairs) == 0:
        # KLUDGE to make partial L0B w/missing timestamps at start process properly
        logger.info("KLUDGE invoked-remove after processing partial L0B file!")
        index_pairs = [(91, timerang[-1])]
    return index_pairs


def jpl_safe_datetime(
    datestr: str,
    timestr: str,
) -> datetime.datetime:
    """
    Fix cases where we've received L1 files with a timestamp "seconds" value of 60.
    Presumably floating point seconds were rounded rather than truncated and not bound
    to 0-59 domain?

    Args:
        datestr (str): String representation of calendar date in YYYYMMDD format
        timestr (str): String representation of UTC time in HHMMSS format

    Returns:
        datetime.datetime: Adjusted valid datetime object
    """
    try:
        # Should work with any VALID date and time strings
        date_sanitized = datetime.datetime.strptime(
            f"{datestr}_{timestr}", "%Y%m%d_%H%M%S"
        ).replace(tzinfo=datetime.UTC)
    except Exception as exc:
        # Should fix cases where seconds was set to 60. Error handling should be made
        # more general, but this'll get us past the current roadblock.
        _datestr = datestr
        _timestr = timestr[0:4] + "59"
        logger.warning("Encountered exception parsing date or time string in filename:")
        logger.warning(f"====> {exc}")
        logger.warning(
            f"Date + time modified from {datestr}_{timestr} to {_datestr}_{_timestr}"
        )
        date_sanitized = datetime.datetime.strptime(
            f"{_datestr}_{_timestr}", "%Y%m%d_%H%M%S"
        ).replace(tzinfo=datetime.UTC)
    return date_sanitized


def parse_ezie_product_name(source: str | Path):
    parsed = None
    if not isinstance(source, Path):
        source_as_path = Path(source)
    else:
        source_as_path = source
    stem = source_as_path.stem
    # TODO: We will need to modify this once L2 and L3 products adopt naming convention
    # that includes the orbit number.
    if "l1" in stem and "orbit" in source.as_posix():
        pattern = (
            "^ezie_([a-zA-z0-9]{2,3})_(\\d{8})_(\\d{6})_(\\d{6})"
            "_sv([a-c])_v(\\d{2,3})_r(\\d{2,3})$"
        )
        match = re.compile(pattern).search(stem)
        if hasattr(match, "group"):
            parsed = {
                "prod": match.group(1),
                "date": match.group(2),
                "time": match.group(3),
                "orbt": match.group(4),
                "spcv": match.group(5),
                "vrsn": match.group(6),
                "rvsn": match.group(7),
                "dttm": jpl_safe_datetime(
                    datestr=match.group(2), timestr=match.group(3)
                ),
            }
    else:
        pattern = (
            "^ezie_([a-zA-z0-9]{2,3})_(\\d{8})_(\\d{6})"
            "_sv([a-c])_v(\\d{2,3})_r(\\d{2,3})$"
        )
        match = re.compile(pattern).search(stem)
        if match is None:
            logger.warning(f"Anomalous file name: {source_as_path.as_posix()}")
        elif hasattr(match, "group"):
            # JPL sent us a file with seconds=60?
            logger.debug(f"File name: {source_as_path.as_posix()}")
            logger.debug(
                f"Parsed values for date: {match.group(2)} time: {match.group(3)}"
            )
            parsed = {
                "prod": match.group(1),
                "date": match.group(2),
                "time": match.group(3),
                "spcv": match.group(4),
                "vrsn": match.group(5),
                "rvsn": match.group(6),
                "dttm": jpl_safe_datetime(
                    datestr=match.group(2), timestr=match.group(3)
                ),
            }
    if parsed is None:
        if stem != "":
            # We pass an empty string deliberately when the file filter fails to find a
            # match, and we deliberately return a 'parsed' dict with entries set to NaN
            # in this case. If we fail on a non-empty string, however, we want to know
            # what is was (and why it was passed here in the first place).
            logger.error(f"Encountered exception while parsing filename: {stem}")
        parsed = {
            "prod": math.nan,
            "date": math.nan,
            "time": math.nan,
            "orbt": math.nan,
            "spcv": math.nan,
            "vrsn": math.nan,
            "rvsn": math.nan,
            "dttm": math.nan,
        }
    return collections.namedtuple(
        "GenericDict",
        parsed.keys(),
    )(**parsed)


def j2000_to_utc(seconds_since_j2000):
    # Ensure type compatibility
    seconds = float(seconds_since_j2000)

    # J2000 epoch (UTC)
    j2000_epoch = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)

    # Add offset
    return j2000_epoch + datetime.timedelta(seconds=seconds)


def filter_daily_files_by_version(
    file_directory: str,
    file_pattern: str,
    start_date: datetime.datetime,
    stop_date: datetime.datetime,
):
    """
    Create list of highest version/revision for a given spacecraft and date.
    """
    log_mthd = logger.debug
    # log_mthd = logger.info
    # Use glob to find all files matching the high-level pattern and sort them
    id_path = Path(file_directory)
    ezie_files = sorted(id_path.rglob(pattern=file_pattern))
    if len(ezie_files) == 0:
        logger.warning(f"Returning, no files found matching {file_pattern}")
        return

    # Further filter the initial group of files by date, keeping only the latest version
    # for any given date and spacecraft.
    revisions = {}
    dttm_frmt = "%Y%m%d"
    for each_file in ezie_files:
        prsd = parse_ezie_product_name(each_file)
        if str(prsd.date) == "nan":
            logger.error(f"Invalid file name encountered: {each_file.stem}")
            continue  # file name was NOT valid
        file_dttm = datetime.datetime.strptime(f"{prsd.date}", dttm_frmt).replace(
            tzinfo=datetime.UTC
        )
        log_mthd(f"Examining file {each_file.name} with date {prsd.date}")
        # Will "backup" subdirectory regularly appear in pipeline data directory? TBD
        if (
            (file_dttm >= start_date)
            and (file_dttm <= stop_date)
            and (each_file.parent != "backup")
        ):
            file_key, revision = (
                f"{prsd.date}_sv{prsd.spcv}",
                f"v{prsd.vrsn}_r{prsd.rvsn}",
            )
            logger.debug(f"{file_key} {revision}")
            if file_key not in revisions:
                revisions[file_key] = (revision, each_file)
            else:
                # Only keep file with the highest version/revision
                revisions[file_key] = max(revisions[file_key], (revision, each_file))
        else:
            log_mthd(
                f"File {each_file.name} with date {prsd.date} did not meet "
                f"date specifications {start_date} <= {file_dttm} <= {stop_date}"
            )

    newest_versions = sorted([each_file[1] for each_file in revisions.values()])
    if len(newest_versions) > 0:
        log_mthd(f"Newest version[s]: {[str(vrsn) for vrsn in newest_versions]}")
        return newest_versions
    else:
        return  # return None, do NOT return an empty list


def filter_files_by_version(
    file_directory: str,
    file_pattern: str,
    start_date: datetime.datetime,
    stop_date: datetime.datetime,
    remove_near_dupes: bool = False,
):
    """
    Create list of highest version/revision for a given spacecraft and date.
    """
    log_mthd = logger.debug
    # log_mthd = logger.info
    # Use glob to find all files matching the high-level pattern and sort them
    id_path = Path(file_directory)
    ezie_files = sorted(id_path.rglob(pattern=file_pattern))
    if len(ezie_files) == 0:
        log_mthd(f"{file_pattern}")
        logger.warning(f"Returning, no files found matching {file_pattern}")
        return

    # Further filter the initial group of files by date, keeping only the latest version
    # for any given date and spacecraft.
    revisions = {}
    # dttm_frmt = "%Y%m%d_%H%M%S"
    for each_file in ezie_files:
        prsd = parse_ezie_product_name(each_file)
        if str(prsd.date) == "nan":
            logger.error(f"Invalid file name encountered: {each_file.stem}")
            continue  # file name was NOT valid
        file_dttm = jpl_safe_datetime(datestr=prsd.date, timestr=prsd.time)
        # Deal with invalid datetime formats
        # file_dttm = datetime.datetime.strptime(
        #     f"{prsd.date}_{prsd.time}", dttm_frmt
        # ).replace(tzinfo=datetime.UTC)
        log_mthd(f"Examining file {each_file.name} with date {prsd.date}_{prsd.time}")
        # Will "backup" subdirectory regularly appear in pipeline data directory? TBD
        if (
            (file_dttm >= start_date)
            and (file_dttm <= stop_date)
            and (each_file.parent != "backup")
        ):
            file_key, revision = (
                f"{prsd.date}_{prsd.time}_sv{prsd.spcv}",
                f"v{prsd.vrsn}_r{prsd.rvsn}",
            )
            log_mthd(f"{file_key} {revision}")
            if file_key not in revisions:
                revisions[file_key] = (revision, each_file)
            else:
                # Only keep file with the highest version/revision
                revisions[file_key] = max(revisions[file_key], (revision, each_file))
        else:
            log_mthd(
                f"File {each_file.name} with date {prsd.date}_{prsd.time} did not meet "
                f"date specifications {start_date} <= {file_dttm} <= {stop_date}"
            )

    newest_versions = sorted([each_file[1] for each_file in revisions.values()])
    log_mthd(f"Newest versions are {newest_versions}")

    # Do we need to prune files with the _same_ version and date but only slightly
    # differing timestamps (applies only to per-orbit L2 and L3 files)?
    if not remove_near_dupes:
        # No. We can just stop now and return results.
        if len(newest_versions) > 0:
            return newest_versions
        else:
            return  # return None, do NOT return an empty list
    else:
        # Yes. We need to do some more filtering (below).
        tstamps = {}
        for ff, each_file in enumerate(newest_versions):
            log_mthd(f"Input {ff:04d} {each_file.name}")
            prsd = parse_ezie_product_name(each_file)
            file_key, tstamp = (
                f"sv{prsd.spcv}_v{prsd.vrsn}_r{prsd.rvsn}",
                jpl_safe_datetime(datestr=prsd.date, timestr=prsd.time),
                # datetime.datetime.strptime(
                #     f"{prsd.date}_{prsd.time}", dttm_frmt
                # ).replace(tzinfo=datetime.UTC),
            )
            log_mthd(f"{file_key} {tstamp}")
            if file_key not in tstamps:
                tstamps[file_key] = []
            tstamps[file_key].append((tstamp, each_file))

        # Files should ALWAYS be of the same product type
        if "_l0" in each_file.stem or (
            ("_l1_" in each_file.stem) and ("daily" in each_file.parent.name)
        ):
            dup_time_limit = 86400  # seconds, for full day files
        else:
            dup_time_limit = 60  # seconds, for single orbit files

        all_pruned = []
        for each_key, each_val in tstamps.items():
            log_mthd(f"{each_key} {each_val}")
            pruned = [each_val[0]]
            pp = 0
            for each_time in each_val[1:]:
                log_mthd(
                    f"{each_time[0]} {pruned[pp][0]} "
                    f"{(each_time[0] - pruned[pp][0]).total_seconds()} "
                    f"{dup_time_limit}"
                )
                if (each_time[0] - pruned[pp][0]).total_seconds() > dup_time_limit:
                    pruned.append(each_time)
                    pp += 1
                else:
                    pruned[pp] = each_time
            all_pruned.extend(pruned)
            log_mthd(f"{pruned} {all_pruned}")

        pruned_versions = sorted([each_pair[1] for each_pair in all_pruned])
        log_mthd(f"Pruned versions are {pruned_versions}")
        if len(pruned_versions) > 0:
            for ff, each_file in enumerate(pruned_versions):
                log_mthd(f"Pruned: {ff:04d} {each_file.name}")
            return pruned_versions
        else:
            return  # do NOT return an empty list


def get_datetime_from_utc_string(
    nc_time_group,
    indices: tuple | None = None,
):
    """
    Generate a datetime object from the netcdf file UTC time field strings
    """
    if indices is not None:
        logger.debug(f"{indices!s}")
        tslc = np.s_[indices[0] : indices[1]]
    else:
        tslc = np.s_[:]
    try:
        datetimes_utc = [
            # datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ")
            datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f").replace(
                tzinfo=datetime.UTC
            )
            for t in nc_time_group.variables["time_utc"][tslc]
        ]
    except ValueError:
        try:
            datetimes_utc = [
                datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                    tzinfo=datetime.UTC
                )
                for t in nc_time_group.variables["time_utc"][tslc]
            ]
        except ValueError:
            logger.error(
                f"Unable to process UTC time strings: "
                f"{nc_time_group.variables['time_utc'][tslc]}"
            )
            return None
    # n_times: int = len(datetimes_utc)
    # return np.array(datetimes_utc), datetimes_utc[n_times // 2]
    return np.array(datetimes_utc), datetimes_utc[0]


def moving_average(x, w):
    return np.convolve(a=x, v=np.ones(w), mode="same") / w


def convert_to_mag(
    geom,
    year4mag: int,
    alt4mag: float = REFERENCE_ALTITUDE_KM,
):
    """
    Converts cartopy/shapely features, e.g., continents, from geographic to geomagnetic
    (APEX).

    Based on fabulously thorough code at https://gis.stackexchange.com/a/291293

    TODO: Do we need to determine whether APEX assumes geodetic or geocentric latitude?
    """
    if geom.is_empty:
        return geom

    apex = Apex(date=year4mag, refh=0)
    if geom.has_z:

        def convert_to_mag_doer(
            coords,
            time4mag,
            _,
        ):
            """
            For features that have a z coordinate.
            QUERY: Are there any that we use? TBD
            TODO: THIS METHOD NOT YET TESTED FOR EZIE APPLICATIONS!
            """
            for long, lat, _alt4mag in coords:
                # Sanitize longitude range for APEX
                if long > 180.0:
                    long -= 360.0
                if long < -180.0:
                    long += 360.0
                [lat_mag, long_mag] = apex.geo2apex(
                    lat,
                    long,
                    _alt4mag,
                )
                # [lat_mag, long_mag, alt_mag] = apex.gg_gm_apex(
                #     int(year4mag),
                #     lat,
                #     long,
                #     _alt4mag,
                #     apex.GEO_TO_MAG,
                # )
                yield (long_mag, lat_mag, _alt4mag)
            # END FOR long, lat, alt_mag

    else:

        def convert_to_mag_doer(
            coords,
            year4mag,
            alt4mag,
        ):
            """
            For surface features that don't have a z coordinate. Validated for EZIE use.
            """
            for long, lat in coords:
                # # Sanitize longitude range for APEX
                # if long > 180.0:
                #     long -= 360.0
                # if long < -180.0:
                #     long += 360.0
                [lat_mag, long_mag] = apex.geo2apex(
                    lat,
                    long,
                    alt4mag,
                )
                yield (long_mag, lat_mag)

    # end if geom.has_Z

    # Process coordinates from each supported geometry type
    if geom.geom_type in ("Point", "LineString", "LinearRing"):
        logger.debug(
            "Converting simple (Point, LineString, linearRing) feature from geodetic "
            "to magnetic coords"
        )
        return type(geom)(list(convert_to_mag_doer(geom.coords, year4mag, alt4mag)))
    elif geom.geom_type == "Polygon":
        raise RuntimeError("Polygon case not yet supported")
        logger.debug("Converting polygonal feature from geodetic to magnetic coords")
        ring = geom.exterior
        shell = type(ring)(list(convert_to_mag_doer(ring.coords, year4mag, alt4mag)))
        holes = list(geom.interiors)
        for pos, ring in enumerate(holes):
            holes[pos] = type(ring)(
                list(convert_to_mag_doer(ring.coords, year4mag, alt4mag))
            )
        return type(geom)(shell, holes)
    elif geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
        # Recursive call
        logger.debug("Converting geometry collection from geodetic to magnetic coords")
        return type(geom)(
            [convert_to_mag(part, year4mag, alt4mag) for part in geom.geoms]
        )
    else:
        raise ValueError(f"Type {geom.geom_type!r} not recognized")


def reverse_lon(
    geom,
):
    """
    Inverts longitude in a set of geographic coordinates. Used to make cartopy plot the
    southern hemisphere in 'heliospheric' mode, viewing it as if looking down through a
    transparent earth from above the north pole.
    """
    if geom.is_empty:
        return geom

    def reverse_lon_doer(coords):
        """
        For surface features that don't have a z coordinate. Validated for EZIE use.
        """
        for long, lat in coords:
            [lat_inv, long_inv] = [lat, -1 * long]
            yield (long_inv, lat_inv)

    # Process coordinates from each supported geometry type
    if geom.geom_type in ("Point", "LineString", "LinearRing"):
        logger.debug(
            "Inverting latitude of simple (Point, LineString, linearRing) feature"
        )
        return type(geom)(list(reverse_lon_doer(geom.coords)))
    elif geom.geom_type == "Polygon":
        logger.debug("Inverting latitude of polygonal feature")
        ring = geom.exterior
        shell = type(ring)(list(reverse_lon_doer(ring.coords)))
        holes = list(geom.interiors)
        for pos, ring in enumerate(holes):
            holes[pos] = type(ring)(list(reverse_lon_doer(ring.coords)))
        return type(geom)(shell, holes)
    elif geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
        # Recursive call
        logger.debug("Inverting latitude in geometry collection")
        return type(geom)([reverse_lon_doer(part) for part in geom.geoms])
    else:
        raise ValueError(f"Type {geom.geom_type!r} not recognized")


@functools.cache
def _inverted_continents():
    """
    Generate a list of the geometries with longitude coordinate values inverted, i.e.,
    plot southern hemisphere as if we were looking down on it from above the north pole
    through a transparent earth.
    """

    # Get feature geometries to transform
    continents_geo = cfeature.NaturalEarthFeature("physical", "coastline", "110m")

    geom_inv = []  # prep a list
    for geom in continents_geo.geometries():
        geom_inv.append(reverse_lon(geom))  # +lat/+lon to +lat/-lon

    # Return new Shapelyfeature object with the transformed coordinate values
    return cfeature.ShapelyFeature(geom_inv, ccrs.PlateCarree())


def map_inverted_continents(
    ax: plt.Axes,
    zorder: int = 3,
    color: str = "xkcd:black",
    linewidth: float = 0.5,
):
    continents_inv = _inverted_continents()

    for geom in continents_inv.geometries():
        inv_mlt, inv_lat = np.array(geom.coords.xy[0]), np.array(geom.coords.xy[1])
        ax.plot(
            inv_mlt,
            inv_lat,
            color=color,
            linewidth=linewidth,
            zorder=zorder,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )


@functools.cache
def _magnetic_continents(year4mag: int, alt4mag=REFERENCE_ALTITUDE_KM):
    """
    Performs magnetic lat/lon coordinate transformation of continent geometries.

    Args:
    year4mag (float): [Decimal] year
    alt4mag (float): Height in km
    """
    logger.info("Performing magnetic lat/lon coordinate transformation of continents")
    bgn = time.perf_counter()

    # Get feature geometries to transform
    continents_geo = cfeature.NaturalEarthFeature("physical", "coastline", "110m")

    # Generate a list of the geometries with coordinate values transformed from geo
    # lat/lon to mag lat/lon
    geom_mag = []  # prep a list
    for geom in continents_geo.geometries():
        geom_mag.append(convert_to_mag(geom, year4mag, alt4mag))

    end = time.perf_counter()
    logger.info(f"Elapsed time: {end - bgn:.2f} seconds")

    # Generate new Shapelyfeature object with the transformed coordinate values
    return cfeature.ShapelyFeature(geom_mag, ccrs.PlateCarree())


def map_magnetic_continents(
    ax: plt.Axes,
    time4mag: datetime.datetime,
    sun_mlon: float,
    alt4mag: float,
    south_inverted: bool = False,
    linewidth: float = 1.0,  # Use thicker lines for L2 plots that don't overlay quivers
    zorder: int = 3,
    color: str = "xkcd:black",
):

    # This will return cached results if the arguments are the same as a previous call.
    continents_mag = _magnetic_continents(
        int(time4mag.year), alt4mag=REFERENCE_ALTITUDE_KM
    )

    # Convert magnetic longitudes to MLT and then plot features.
    # When run at 50M resolution instead of 110, the following error is encountered:
    #   NotImplementedError: Sub-geometries may have coordinate sequences, but
    #   multi-part geometries do not
    logger.info("Performing MLT coordinate transformation of continents")
    bgn = time.perf_counter()
    delta = 180.0 - sun_mlon
    MLT_sign = -1 if south_inverted else +1
    for geom in continents_mag.geometries():
        mag_mlt, mag_lat = np.array(geom.coords.xy[0]), np.array(geom.coords.xy[1])
        mag_mlt = (mag_mlt + delta + 180.0) % 360.0 - 180.0

        logger.debug("Closing large longitude differences, e.g., bad dateline wrapping")
        for ndx, mlndx in enumerate(mag_mlt[:-1]):
            mp_diff = mag_mlt[ndx + 1] - mlndx
            if mp_diff > 180.0:
                mag_mlt[ndx + 1] -= 360
            if mp_diff < -180.0:
                mag_mlt[ndx + 1] += 360

        ax.plot(
            MLT_sign * mag_mlt,
            mag_lat,
            color=color,
            linewidth=linewidth,
            zorder=zorder,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )
    end = time.perf_counter()
    logger.info(f"Elapsed time: {end - bgn:.2f} seconds")


# def watermark(ax):
#     """
#     Overlay EZIE logo (img) in bottom right corner of plot axes (ax)
#     """
#     from matplotlib.offsetbox import AnchoredOffsetbox, OffsetImage
#     from scipy.ndimage import zoom
#     ezie_logo_img = plt.imread(
#         Path(__file__).parent.parent / "binary-assets" / "ezie_logo_no_bg.png"
#     )
#     scaled_logo = zoom(ezie_logo_img, (1.0, 1.0, 1))
#     imagebox = OffsetImage(scaled_logo, zoom=0.25, alpha=1.0)
#     imagebox.image.axes = ax
#     ao = AnchoredOffsetbox(4, pad=0.05, borderpad=0, child=imagebox)
#     ao.patch.set_alpha(0)
#     ax.add_artist(ao)


def overlay_ezie_logo(
    fig: plt.Figure,
    dark_mode: bool = False,
):
    """
    Overlay EZIE logo in top left corner of figure (fig)
    """
    # Overlay EZIE logo
    img_name = "ezie_logo_no_bg.png" if not dark_mode else "ezie_logo_light.png"
    ezie_logo_img = plt.imread(
        Path(__file__).parent.parent / "binary-assets" / img_name
    )
    img_aspect_ratio = ezie_logo_img.shape[1] / ezie_logo_img.shape[0]
    hi = 0.07
    wi = hi * img_aspect_ratio * fig.get_figheight() / fig.get_figwidth()
    axs_pos = fig.add_axes([0.001, 1 - hi - 0.002, wi, hi])
    axs_pos.imshow(ezie_logo_img, aspect="auto")
    axs_pos.set_xticklabels("")
    axs_pos.axis("off")


def add_product_metadata(
    fig: plt.Figure,
    nc_data: Dataset,
    source: Path,
    second: bool = False,
    size: str = "small",
):
    """
    Add figure text with source file metadata, dealing with files that have different
    formats for version and revision or even no version or revision designation at all.
    """
    # prsd = parse_ezie_product_name(source)
    # ver_str = f"v{prsd.vrsn}" if str(prsd.vrsn) != "nan" else "TEST"
    # rev_str = f"r{prsd.rvsn}" if str(prsd.rvsn) != "nan" else "TEST"

    # Time specification differs from one product level to another (JPL vs APL?), and
    # currently L3 products have no internal creation date metadata whatsoever. Deal
    # with it.
    try:
        created = nc_data["Metadata/CreationTimeString"][:]  # True creation date
    except Exception as _exc:
        created = datetime.datetime.fromtimestamp(
            os.path.getmtime(source), tz=datetime.UTC
        ).isoformat()  # Get date from OS. Match JPL microseconds+TZ ISO format

    # Deal with product-specific time formatting differences
    truncate = -4  # No TZ, has milliseconds
    try:
        _has_tz = created.index("+")
        truncate -= 9  # cut more for microseconds + TZ specification
    except Exception as _exc:
        logger.warning(f"Problem handling timestamp formatting: {created!s}")
    created = created[0:truncate]

    # src_str = (
    #     f"Source Directory: {source.parent!s}" if source.parent is not None else ""
    # )
    fig.text(
        0.005 if not second else 0.60,
        0.005,
        # f"Data Version: {ver_str} Revision: {rev_str} Created: {created}   {src_str}",
        f"Data Source: {source.name}  Source created at: {created}",
        ha="left" if not second else "center",
        va="bottom",
        rotation=0,
        weight="bold",
        size=size,
    )


def add_pipeline_metadata(
    fig: plt.Figure,
    nc_data: Dataset,
    git_branch: str | None = None,
    git_commit: str | None = None,
    size: str = "small",
):
    EMPTY_FIELD: str = "Empty"
    MISSING_FIELD: str = "Unavailable"
    try:
        pipeline_branch = f"{nc_data['Configuration/git_branch'][:]}"
        commit_hash = f"{nc_data['Configuration/git_hash'][:]}"
        if pipeline_branch == "":
            pipeline_branch = EMPTY_FIELD
        if commit_hash == "":
            commit_hash = EMPTY_FIELD
    except KeyError:
        pipeline_branch = MISSING_FIELD
        commit_hash = MISSING_FIELD
    # Allow override of values in data product, e.g., for using L2 product values for
    # currently undefined L3 product pipeline info fields.
    if (git_branch is not None) and (
        (pipeline_branch == EMPTY_FIELD) or (pipeline_branch == MISSING_FIELD)
    ):
        pipeline_branch = git_branch
    if (git_commit is not None) and (
        (commit_hash == EMPTY_FIELD) or (commit_hash == MISSING_FIELD)
    ):
        commit_hash = git_commit
    fig.text(
        0.995,
        0.005,
        # f"Pipeline Software Version: {nc_data['Metadata/SoftwareVersion'][:]}",
        # f" / Calibration Version: {nc_data['Metadata/CalibrationVersion'][:]}",
        f"Pipeline Branch: {pipeline_branch}  Commit Hash: {commit_hash}",
        ha="right",
        va="bottom",
        rotation=0,
        weight="bold",
        size=size,
    )


def set_yaxis_tick_format(axs: plt.Axes, max_ticks=10):
    """
    Set time format used for x axis on individual orbit plots
    """
    axs.yaxis.set_major_locator(mdate.MinuteLocator(interval=1))
    axs.yaxis.set_major_formatter(mdate.DateFormatter("%H:%M"))
    # axs.yaxis.set_major_formatter(
    #     plt.FuncFormatter(lambda x, _: mpl.dates.num2date(x).strftime("%H:%M"))
    # )
    # Limit the number of x-axis ticks as needed
    # axs.yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks, prune=None))


def set_xaxis_tick_format(
    axs: plt.Axes, max_ticks=10, rotation=0, use_seconds: bool = False
):
    # Set time format used for x axis on individual orbit plots, adjusting as needed for
    # overall duration.
    if use_seconds:
        # logger.info("Setting tick locations with use_seconds flag:")
        # axs.xaxis.set_major_locator(mdate.SecondLocator(interval=5))
        axs.xaxis.set_major_locator(mdate.AutoDateLocator())
        # logger.info(f"==> {axs.get_xticks()!r}")
        axs.xaxis.set_major_formatter(mdate.DateFormatter("%H:%M:%S"))
    else:
        axs.xaxis.set_major_locator(mdate.MinuteLocator(interval=1))
        axs.xaxis.set_major_formatter(mdate.DateFormatter("%H:%M"))
    # plt.MaxNLocator(nbins=max_ticks, prune="both")
    # plt.FuncFormatter(lambda x, _: mpl.dates.num2date(x).strftime("%H:%M:%S"))
    # axs.xaxis.set_minor_locator(plt.MaxNLocator(nbins=2 * max_ticks, prune="both"))
    axs.xaxis.set_minor_locator(mtick.AutoMinorLocator())
    if rotation != 0:
        axs.set_xticks(
            axs.get_xticks(),
            axs.get_xticklabels(),
            rotation=rotation,
            ha="right",
        )


def magnetic_parallel(
    year: int,
    mag_lat: float = 0.0,
    alt: float = REFERENCE_ALTITUDE_KM,
):
    """
    Return geographic latitude and longitude vectors corresponding to a parallel of
    APEX-defined magnetic latitude points, at the specified altitude, for the specified
    year.
    """
    res_deg = 1.0  # make keyword?
    mag_lon = np.arange(-180.0, 180.0, res_deg)
    _tmp_lat = np.full_like(mag_lon, mag_lat)
    geo_lat, geo_lon, _err_deg = Apex(date=year, refh=alt).apex2geo(
        mag_lat,
        mag_lon,
        alt,
    )

    # FIXME: write a general solution to this glitch
    if mag_lat == -80:
        # Geodetic south pole is outside parallel, sorting breaks plot
        return geo_lat, geo_lon
    else:
        # Prevent line wrapping around on global projections
        srt_ndx = np.argsort(geo_lon)
        return geo_lat[srt_ndx], geo_lon[srt_ndx]


def get_north_magnetic_pole(year4mag: float):
    """
    Compute location of north magnetic pole using APEX magnetic coordinate model's
    qd2geo method.

    Args:
        year4mag (float): Decimal year for which pole location should be computed

    Returns:
        tuple(float,float): Geodetic latitude of pole, geodetic longitude of pole
    """
    return Apex(date=year4mag, refh=0).qd2geo(+90, 0, 0)


def plot_geomagnetic_references(
    ax: plt.Axes,
    at_time: float,
    latitudes: list[float] | None = None,
    south_inverted: bool = False,
    lat_clr: str = "black",
    ls: str = "dashed",
    lw: int = 1,
):
    """
    Overplot location of magnetic poles (tilted dipole) and magnetic latitude parallels
    (currently Apex, might want to shift to tilted dipole instead?)

    Args:
        ax (plt.Axes): Plot axes on which to draw the specified markings.
        at_time (float):  Datetime for which magnetic model is computed.
        latitudes (list[float] | None, optional): Parallel latitudes. Defaults to None.
        south_inverted (bool, optional): Plot southern hemisphere as if looking down
            through earth from above north pole. Defaults to False.
        lat_clr (str, optional): _description_. Defaults to "black".
        ls (str, optional): _description_. Defaults to "dashed".
        lw (int, optional): _description_. Defaults to 1.

    Returns:
        Line2D: The artist corresponding to the overlaid magnetic parallels, e.g., for
        use in a plot legend.
    """

    # Compute location of north magnetic pole using APEX magnetic coordinate model's
    # qd2geo method.
    n_mag_pole_lat, n_mag_pole_lon, _err = Apex(date=at_time.year, refh=0).qd2geo(
        +90, 0, 0
    )
    MLT_sign = -1 if south_inverted else +1

    mag_lat_artist = None
    if latitudes is not None:
        for mag_lat in latitudes:
            glat, glon = magnetic_parallel(mag_lat=mag_lat, year=int(at_time.year))
            mag_lat_artist = ax.plot(
                MLT_sign * glon,
                glat,
                color=lat_clr,
                ls=ls,
                lw=lw,
                transform=ccrs.PlateCarree(),
            )[0]
    else:  # Just mark magnetic poles (tilted dipole approximation)
        plt.text(
            np.degrees(n_mag_pole_lon),
            np.degrees(n_mag_pole_lat),
            "N",
            fontsize="x-small",
            color="green",
            ha="center",
            va="center",
            transform=ccrs.PlateCarree(),
        )
        plt.text(
            (180.0 + MLT_sign * np.degrees(n_mag_pole_lon)) % 360.0,
            -1.0 * np.degrees(n_mag_pole_lat),
            "S",
            fontsize="x-small",
            color="green",
            ha="center",
            va="center",
            transform=ccrs.PlateCarree(),
        )
    return mag_lat_artist


def read_png_metadata(filename: Path | str):
    """
    Read PNG metadata using PIL/Pillow

    Args:
        filename (Path | str): Path to a PNG image

    Returns:
        dict: Dictionary containing any text fields found in the PNG file's metadata
    """

    img = Image.open(filename)
    metadata = {}
    if hasattr(img, "text"):  # Get all text chunks
        metadata = img.text
    return metadata


def hash_ezie_data_product(source: str | Path):
    import hashlib

    with open(source, "rb") as fp:
        return hashlib.file_digest(fp, "md5").hexdigest()


def save_close_figure(
    plot_type: str,
    obs_date: datetime.datetime,
    save_directory: Path,
    name_only: bool = False,
    source: Path | None = None,
    figure: plt.Figure | None = None,
    spacecraft: str | None = None,
    orbit: int | None = None,
    tstmp: int | None = None,
    version: str | None = None,
    dpi: int = DFLT_RES,
    dark_mode: bool = False,
    old_format: bool = False,
) -> Path:
    """
    Save and then close figure, constructing name and path from supplied parameters.
    There are currently three cases:
        1) Daily summary plots have no source, spacecraft or tstmp/orbit.
        2) Calibration, ancillary, and geolocation plots have an L0B or L1 source file
           from which spacecraft and obs_date may be parsed, but the pass tstmp/orbit
           must be determined from data internal to the file, and there are typically 15
           orbits per file. NOTE: This may change to a single orbit per file soon!
        3) L2 and L3 plots are generated on a single orbit basis, and all required plot
           naming parameters can be extracted from the source file name.
    """

    old_hash = None
    if source is not None:
        new_hash = hash_ezie_data_product(source=source)
    else:
        new_hash = None

    if old_format:
        od_path = save_directory / obs_date.strftime("%Y-%j-%m-%d")
        if not od_path.exists():
            od_path.mkdir(parents=True, exist_ok=True)
        spacecraft_str = f"{spacecraft!s}_" if spacecraft is not None else ""
        # FIXME: Need to transition from "timestamp" orbit number to actual value
        orbit_str = f"Orbit-{int(orbit):06d}_" if orbit is not None else ""
        tstmp_str = f"Orbit-{int(tstmp):06d}_" if tstmp is not None else orbit_str
        version_str = f"{version}_" if version is not None else ""
        if dpi is None:
            dpi = DFLT_RES
        fig_path = od_path / "_".join(
            [
                obs_date.strftime("%Y-%j-%m-%d"),
                "DM" if dark_mode else "LM",
                f"{dpi:03d}_DPI{spacecraft_str}{tstmp_str}{version_str}{plot_type}.png",
            ]
        )

    else:
        od_path = save_directory / obs_date.strftime("%Y%m%d")
        if spacecraft is not None:
            od_path = od_path / str(spacecraft)[-1].lower()
        if not od_path.exists():
            od_path.mkdir(parents=True, exist_ok=True)
        orbit_str = f"{int(orbit):06d}_" if orbit is not None else ""
        # QUERY: Do we need to transition from a "timestamp" orbit number to the actual
        # value?
        spacecraft_str = (
            f"sv{str(spacecraft)[-1].lower()}_" if spacecraft is not None else ""
        )
        version_str = f"{version}_" if version is not None else ""
        if source is not None:
            prsd = parse_ezie_product_name(source)
            if prsd.prod == "l2" or prsd.prod == "l3":
                tstmp_str = prsd.time
            else:
                tstmp_str = f"{int(tstmp):06d}" if tstmp is not None else ""
            fig_path = od_path / (
                f"ezie_{prsd.prod}_{prsd.date}_{tstmp_str}_sv{prsd.spcv}_"
                f"{plot_type}.png"
            )
        else:
            tstmp_str = f"{int(tstmp):06d}_" if tstmp is not None else orbit_str
            fig_path = od_path / (
                f"ezie_{obs_date.strftime('%Y%m%d')}_"
                f"{tstmp_str}{spacecraft_str}{version_str}{plot_type}.png"
            )
        if fig_path.exists():
            # Open image and access metadata (PNG text chunks)
            img = Image.open(fig_path)
            try:
                metadata = img.text
                logger.debug(f"{type(metadata)}")
                logger.debug(f"{metadata!s}")
                if "EZIE Source Hash" in metadata:
                    old_hash = metadata["EZIE Source Hash"]
            except Exception as exc:
                logger.warning(f"{exc}: Unable to read image metadata, will regenerate")

    if name_only:
        logger.debug(f"Figure path would be: {fig_path.name}")
        return fig_path, old_hash, new_hash

    # Plot creation and modification times are already in the standard PNG metadata:
    #   Properties:
    #     date:create: 2026-03-07T02:38:41+00:00
    #     date:modify: 2026-03-07T02:38:41+00:00
    #     date:timestamp: 2026-03-07T02:41:33+00:00
    if source is not None:
        # "EZIE Plot Creation Time": datetime.datetime.now(tz=datetime.UTC).isoformat(
        #     timespec="seconds"
        # ),
        metadata = {"EZIE Source File": source.name, "EZIE Source Hash": new_hash}
    else:
        metadata = None
    try:
        figure.savefig(
            fig_path,
            dpi=dpi,
            metadata=metadata,
        )  # ty:ignore[possibly-missing-attribute]
        logger.info(f"Saved figure: {fig_path.name}")
    except Exception as exc:
        logger.error(f"Exception encountered: {exc}")
        logger.error("Processing skipped--truncated or corrupted data file?")
        logger.info(f"Failed to save figure: {fig_path.name}")
        logger.error(f"Problem file (figure): {source.as_posix()}")
    plt.close(figure)  # Necessary if using non-interactive backend? TBD
    mpl.rcParams.update(mpl.rcParamsDefault)
    return fig_path
