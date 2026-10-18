"""
Smoke tests for ezieview.update_coverage_plots.

The plotting pipeline is exercised end-to-end on small synthetic data, so no L1
input files are required.

Requires: uv add --group dev pytest  (pytest is not yet a declared dependency)
"""

import datetime
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import ezieview.update_coverage_plots as ucp
from ezieview.ezvislib.gw_plot_params import (
    COVERAGE_TYPES,
    EQUATORIAL,
    MAGLAT,
    MEM_NUMBERS,
    MLT,
    NORTH,
    NUM_MEM,
    REGIONS,
    SOUTH,
    SPACECRAFT,
    SZA,
    UTC_TZ,
)

OBS_DATE = datetime.datetime(2025, 6, 20, tzinfo=UTC_TZ)


def _make_sample_data(n: int = 200, seed: int = 7) -> dict:
    """Random per-sample spacecraft data and (MEM, sample) stacked MEM data."""
    rng = np.random.default_rng(seed)
    data = {
        "sc_id": np.array([SPACECRAFT[i % len(SPACECRAFT)] for i in range(n)]),
        "orbit": rng.integers(1, 15, n),
        "sclat": rng.uniform(-89.0, 89.0, n),
        "sclon": rng.uniform(-180.0, 180.0, n),
        "sc_mlat": rng.uniform(-89.0, 89.0, n),
        "sc_mlt": rng.uniform(0.0, 24.0, n),
    }
    data["mem_lat"] = rng.uniform(-89.0, 89.0, (NUM_MEM, n))
    data["mem_lon"] = rng.uniform(-180.0, 180.0, (NUM_MEM, n))
    data["mem_mlat"] = rng.uniform(-89.0, 89.0, (NUM_MEM, n))
    data["mem_mlt"] = rng.uniform(0.0, 24.0, (NUM_MEM, n))
    data["mem_sza"] = rng.uniform(0.0, 180.0, (NUM_MEM, n))
    data["mem_mode"] = rng.random((NUM_MEM, n)) > 0.3
    return data


def _make_fdd(data: dict) -> dict:
    """Convert _make_sample_data output into ingest_full_day_all_sv dict format."""
    fdd = {
        "SpaceVehicle": data["sc_id"],
        "orbit_number": data["orbit"],
        "sat_lat": data["sclat"],
        "sat_lon": data["sclon"],
        "sat_mlat": data["sc_mlat"],
        "sat_MLT": data["sc_mlt"],
    }
    for m in MEM_NUMBERS:
        fdd[f"obs_lat{m}"] = data["mem_lat"][m - 1]
        fdd[f"obs_lon{m}"] = data["mem_lon"][m - 1]
        fdd[f"obs_maglat{m}"] = data["mem_mlat"][m - 1]
        fdd[f"obs_magLT{m}"] = data["mem_mlt"][m - 1]
        fdd[f"obs_solar_zen{m}"] = data["mem_sza"][m - 1]
    return fdd


# ===== Pure helper tests ==========================================================


def test_in_region():
    lat = np.array([-60.0, -45.0, 0.0, +45.0, +60.0])
    np.testing.assert_array_equal(
        ucp.in_region(lat, NORTH), [False, False, False, True, True]
    )
    np.testing.assert_array_equal(
        ucp.in_region(lat, SOUTH), [True, True, False, False, False]
    )
    np.testing.assert_array_equal(
        ucp.in_region(lat, EQUATORIAL), [False, False, True, False, False]
    )


def test_coverage_bin_edges():
    mlat_south = ucp._coverage_bin_edges(MAGLAT, SOUTH)
    assert mlat_south[0] == pytest.approx(-91.0)
    mlt = ucp._coverage_bin_edges(MLT, NORTH)
    assert mlt[0] == pytest.approx(-0.25)
    assert mlt[-1] == pytest.approx(23.75)
    sza = ucp._coverage_bin_edges(SZA, EQUATORIAL)
    assert sza[0] == pytest.approx(-2.5)
    assert sza[-1] == pytest.approx(182.5)


def test_orbit_edges():
    edg = ucp._orbit_edges(np.array([5, 9, 12]))
    assert edg[0] == pytest.approx(4.5)
    assert len(edg) == ucp.MAX_ORBITS + 1  # 17 edges -> 16 unit bins


def test_coverage_histograms_mlt_wraparound():
    # A single sample with MLT just short of 24 h must wrap into the first bin
    # and must NOT mutate the input array.
    n = 1
    sc_id = np.array([SPACECRAFT[0]])
    orbit = np.array([3])
    mem_mlat = np.full((NUM_MEM, n), 60.0)  # inside NORTH
    mem_mlt = np.full((NUM_MEM, n), 23.9)
    mem_mode = np.ones((NUM_MEM, n), dtype=bool)
    bin_edg = ucp._coverage_bin_edges(MLT, NORTH)
    orb_edg = ucp._orbit_edges(orbit)

    cvrg, cmax, sampled = ucp._coverage_histograms(
        cvrg_type=MLT,
        regn=NORTH,
        sc_id=sc_id,
        orbit=orbit,
        mem_mlat=mem_mlat,
        mem_qty=mem_mlt,
        mem_mode=mem_mode,
        bin_edg=bin_edg,
        orb_edg=orb_edg,
    )

    assert np.sum(sampled) == NUM_MEM
    assert np.nansum(cvrg[0][0]) == 1
    assert cvrg[0][0][0, 0] == 1  # wrapped value in first MLT bin, first orbit
    np.testing.assert_array_equal(mem_mlt, 23.9)  # input unmutated


def test_valid_files_and_dates():
    good = Path("orbit_files/ezie_l1_20250620_120000_000001_sv_a_v01_r01.nc4")
    bad = Path("orbit_files/garbage.nc4")
    valid, dates = ucp._valid_files_and_dates([good, bad])
    assert valid == [good]
    assert dates == ["20250620"]


def test_stack_mem_arrays():
    fdd = {}
    for m in MEM_NUMBERS:
        fdd[f"obs_lat{m}"] = np.full(5, float(m))
        fdd[f"obs_lon{m}"] = np.full(5, 10.0 * m)
        fdd[f"obs_maglat{m}"] = np.array([np.nan, np.nan, float(m), float(m), float(m)])
        fdd[f"obs_magLT{m}"] = np.full(5, 1.0 * m)
        fdd[f"obs_solar_zen{m}"] = np.full(5, 10.0 * m)

    mem_lat, mem_lon, mem_mlat, mem_mlt, mem_sza, mem_mode = ucp._stack_mem_arrays(fdd)
    assert mem_lat.shape == (NUM_MEM, 5)
    np.testing.assert_array_equal(mem_lat[0], [1.0, 1.0, 1.0, 1.0, 1.0])
    # NaN magnetic latitude means sky-look
    assert not mem_mode[:, :2].any()
    assert mem_mode[:, 2:].all()


def test_regions_with_observations():
    fdd = {
        "SpaceVehicle": np.array([SPACECRAFT[0], SPACECRAFT[0], SPACECRAFT[1]]),
        "orbit_number": np.array([1, 1, 2]),
        "sat_mlat": np.array([60.0, 62.0, -55.0]),  # A -> NORTH, B -> SOUTH
    }
    regns = ucp._regions_with_observations(fdd, np.array([1, 2]))
    assert regns == [NORTH, SOUTH]


# ===== End-to-end plot tests ==========================================================


def test_plot_mlt_sza_coverage_smoke(tmp_path):
    d = _make_sample_data()
    ucp.plot_mlt_sza_coverage(
        obs_date=OBS_DATE,
        sc_id=d["sc_id"],
        orbit=d["orbit"],
        mem_mlat=d["mem_mlat"],
        mem_mlt=d["mem_mlt"],
        mem_sza=d["mem_sza"],
        mem_mode=d["mem_mode"],
        output_dir=tmp_path,
        regions=list(REGIONS),
    )
    # One figure per region per coverage type
    assert len(list(tmp_path.rglob("*.png"))) == len(REGIONS) * len(COVERAGE_TYPES)


def test_plot_daily_maps_smoke(tmp_path):
    d = _make_sample_data()
    ucp.plot_daily_maps(
        obs_date=OBS_DATE,
        sc_id=d["sc_id"],
        orbit=d["orbit"],
        sclat=d["sclat"],
        sclon=d["sclon"],
        sc_mlat=d["sc_mlat"],
        sc_mlt=d["sc_mlt"],
        mem_lat=d["mem_lat"],
        mem_lon=d["mem_lon"],
        mem_mlat=d["mem_mlat"],
        mem_mlt=d["mem_mlt"],
        mem_mode=d["mem_mode"],
        output_dir=tmp_path,
        regions=list(REGIONS),
        south_inverted=True,
    )
    # 2 polar regions x 4 map types + 2 equatorial Mollweide maps
    assert len(list(tmp_path.rglob("*.png"))) == 2 * 4 + 2


def test_main_smoke(tmp_path, monkeypatch):
    d = _make_sample_data()
    fdd = _make_fdd(d)
    fake_file = Path("orbit_files/ezie_l1_20250620_120000_000001_sv_a_v01_r01.nc4")

    monkeypatch.setattr(
        ucp,
        "filter_files_by_version",
        lambda **kw: [fake_file],
    )
    monkeypatch.setattr(
        ucp,
        "ingest_full_day_all_sv",
        lambda **kw: (fdd, np.array([SPACECRAFT[0]]), np.unique(d["orbit"])),
    )
    monkeypatch.chdir(tmp_path)  # keep the log file inside tmp_path

    runner = CliRunner()
    result = runner.invoke(
        ucp.main,
        [
            "-d0",
            "2025-06-20",
            "-d1",
            "2025-06-20",
            "-fd",
            str(tmp_path),
            "-fp",
            "ezie_l1*.nc4",
            "-pd",
            str(tmp_path / "plots"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "plots").rglob("*.png"))) >= 9
