"""
Manage the NAIF/SPICE kernel files required for Earth/Sun geometry calculations.

Downloads the SPK, LSK, and PCK kernels into a local cache on first use (so
that subsequent runs work offline) and returns their paths.
"""

import logging
from pathlib import Path

import requests


def spice_kernel_mgr() -> tuple[Path, Path, Path]:
    """
    Download NAIF/SPICE kernel files required to calculate relative Earth/Sun geometry,
    if not present. Otherwise just return paths to the cached kernels.

    Returns:
        tuple[Path, Path, Path]: Paths to SPK,LSK,PCK kernels needed for plot creation.
    """
    logger = logging.getLogger(__name__)

    # cache_dir = Path.home() / ".ezieview"
    cache_dir = Path(__file__).parent.parent / "binary-assets"

    spk_cache_dir = cache_dir / "spk"
    lsk_cache_dir = cache_dir / "lsk"
    pck_cache_dir = cache_dir / "pck"

    spk_cache_dir.mkdir(exist_ok=True)
    lsk_cache_dir.mkdir(exist_ok=True)
    pck_cache_dir.mkdir(exist_ok=True)

    spk_path = spk_cache_dir / "de432s.bsp"
    lsk_path = lsk_cache_dir / "naif0012.tls"
    pck_path = pck_cache_dir / "pck00011.tpc"

    spk_url = (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp"
    )
    lsk_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
    pck_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc"

    if not spk_path.exists():
        logger.info("Downloading and caching SPK file...")
        response = requests.get(spk_url)
        response.raise_for_status()
        spk_path.write_bytes(response.content)

    if not lsk_path.exists():
        logger.info("Downloading and caching LSK file...")
        response = requests.get(lsk_url)
        response.raise_for_status()
        lsk_path.write_bytes(response.content)

    if not pck_path.exists():
        logger.info("Downloading and caching PCK file...")
        response = requests.get(pck_url)
        response.raise_for_status()
        pck_path.write_bytes(response.content)

    logger.info(f"Using SPK file {spk_path.as_posix()}")
    logger.info(f"Using LSK file {lsk_path.as_posix()}")
    logger.info(f"Using PCK file {pck_path.as_posix()}")

    return spk_path, lsk_path, pck_path
