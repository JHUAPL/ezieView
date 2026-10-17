"""
Manage the NAIF/SPICE kernel files required for Earth/Sun geometry calculations.

Downloads the SPK, LSK, and PCK kernels into a local cache on first use (so
that subsequent runs work offline) and returns their paths.
"""

import logging
from pathlib import Path

import requests


def _download_kernel(url: str, dest: Path, timeout: float = 300.0) -> None:
    """
    Stream a kernel file from `url` to `dest`, atomically.

    The download is written to a temporary sibling file first and moved into
    place only after it completes, so an interrupted download can never leave
    a corrupt (but apparently present) cache entry behind.

    Args:
        url (str): Remote URL of the kernel file.
        dest (Path): Local path at which the kernel should be cached.
        timeout (float, optional): Request timeout in seconds.
            Defaults to 300.
    """
    tmp_path = dest.with_name(dest.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as fp:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    fp.write(chunk)
        tmp_path.replace(dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def spice_kernel_mgr() -> tuple[Path, Path, Path]:
    """
    Download NAIF/SPICE kernel files required to calculate relative Earth/Sun geometry,
    if not present. Otherwise just return paths to the cached kernels.

    Returns:
        tuple[Path, Path, Path]: Paths to SPK,LSK,PCK kernels needed for plot creation.
    """
    logger = logging.getLogger(__name__)

    cache_dir = Path(__file__).parent.parent / "binary-assets"

    # (subdirectory, file name, remote URL)
    kernels = (
        (
            "spk",
            "de432s.bsp",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp",
        ),
        (
            "lsk",
            "naif0012.tls",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
        ),
        (
            "pck",
            "pck00011.tpc",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
        ),
    )

    paths = []
    for subdir, name, url in kernels:
        kernel_path = cache_dir / subdir / name
        kernel_path.parent.mkdir(parents=True, exist_ok=True)
        if not kernel_path.exists():
            logger.info(f"Downloading and caching {subdir.upper()} file...")
            _download_kernel(url, kernel_path)
        logger.info(f"Using {subdir.upper()} file {kernel_path.as_posix()}")
        paths.append(kernel_path)

    return paths[0], paths[1], paths[2]
