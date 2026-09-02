"""
Logging setup for the EZIE science gateway plotting codes.

Provides a single helper to configure the root logger for standard scripts or
Jupyter notebooks, with optional file output, rotation, and expanded message
formats.
"""

# region imports
import datetime
import logging
import logging.handlers
import platform
import subprocess
import sys
import time
from pathlib import Path

# endregion


def initialize_logging(
    log_to_file: bool = False,
    log_folder: str = "./logs",
    log_level: int = logging.INFO,
    rotating: bool = False,
    timestamp: bool = True,
    open_console: bool = False,
    jupyter: bool = False,
    multiproc: bool = False,
    using_imports: bool = False,
    bare: bool = False,
) -> logging.Logger:
    """
    Configure the root logger for standard python scripts or Jupyter notebooks,
    with or without an expanded format for logging messages from imported
    modules.

    The root logger is used (rather than a module-named logger) so that
    messages from imported modules are visible. Existing handlers are cleared
    before adding the new one to avoid duplicate output.

    Args:
        log_to_file (bool, optional): Log to a file, or stdout?
            Defaults to False (stdout).
        log_folder (str, optional): Folder in which to write log files (created
            if missing). Defaults to "./logs".
        log_level (int, optional): Logging level for the root logger.
            Defaults to logging.INFO.
        rotating (bool, optional): Use a log file that rotates at midnight UTC
            (one file per day). Defaults to False.
        timestamp (bool, optional): Prefix the log file name with a timestamp.
            Defaults to True.
        open_console (bool, optional): Open the log file in Console.app
            (macOS only, and only when logging to a file). Defaults to False.
        jupyter (bool, optional): Use a slimmer format suited to notebooks
            (omits filename and function name). Defaults to False.
        multiproc (bool, optional): Use a format that includes thread and
            process identifiers, for multi-threaded/multi-process use.
            Defaults to False.
        using_imports (bool, optional): Include file and function name in the
            log format, for use with imported (local) code. Defaults to False.
        bare (bool, optional): Message-only logging, like print.
            Defaults to False.

    Returns:
        logging.Logger: The configured root logger.
    """

    # Format logging messages to suit application.
    if bare:
        log_fmt_str = "%(message)s"
    elif jupyter and not using_imports:
        log_fmt_str = (
            "%(asctime)s "
            "%(levelname)8s "
            "[%(lineno)4i] "
            "%(message)s"
        )  # skip filename and funcName fields for self-contained notebooks
    elif multiproc:
        # These fields are only useful for multi-threaded or multi-process applications
        # -- and logging multi-process applications opens its own can of worms.
        log_fmt_str = (
            "%(asctime)s "
            "%(levelname)8s "
            "[%(filename)24s: %(lineno)4i] "
            "%(funcName)-29s "
            "%(threadName)-20s "
            "%(process)-6d "
            "%(processName)-18s "
            "%(message)s"
        )
    else:
        log_fmt_str = (
            "%(asctime)s "
            "%(levelname)8s "
            "[%(filename)24s: %(lineno)4i] "
            "%(funcName)-29s "
            "%(message)s"
        )

    # Set up log path, with the option to log to a single daily log file that rotates at
    # midnight UTC, e.g., for ongoing data pipeline use.
    if log_to_file:
        lf_path = Path(log_folder)
        if not lf_path.exists():
            lf_path.mkdir(parents=True, exist_ok=True)

        # Old version
        # try:
        #     basename = Path(sys.modules["__main__"].__file__).stem
        # except AttributeError:
        #     basename = "Jupyter_Notebook"

        # New version
        main_script = sys.modules["__main__"]
        if hasattr(main_script, "__file__"):
            # If there is a __file__ attribute, use it for the log file name
            basename = Path(main_script.__file__).stem
        else:
            # No __file__ attribute for Jupyter notebooks (or iPython sessions?)
            basename = "Jupyter_Notebook"

        if rotating:
            log_file = (
                lf_path
                / f"{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d')}-"
                f"{basename}.log"
            )
            handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when="midnight",
                utc=True,
                backupCount=0,
            )
        elif timestamp:
            log_file = (
                lf_path
                / f"{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d_%H%M%S')}-"
                f"{basename}.log"
            )
            handler = logging.FileHandler(filename=log_file, mode="w")
        else:
            log_file = lf_path / f"{basename}.log"
            handler = logging.FileHandler(filename=log_file, mode="w")
    else:
        handler = logging.StreamHandler(sys.stdout)
        # handler = StreamHandler()  # different for Jupyter notebooks? TBD

    # Tweak logging format
    # Get _root_ logger so we can see other modules.
    # Do NOT use __name__ in call to getLogger.
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()  # Do NOT add duplicate handlers!
    logger.setLevel(log_level)
    logger.propagate = False
    formatter = logging.Formatter(log_fmt_str)
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # This option is invoked to automatically open Console.app (or bring log file to
    # front) ONLY on macOS, e.g., while development is in progress, or if you just want
    # to see logging outputs. I find it quite helpful––YMMV. On linux systems it's
    # easiest to just run tail -f on the log file from a terminal window.
    if (
        open_console
        and platform.system() == "Darwin"
        and log_to_file
        and log_file.exists()
    ):
        subprocess.Popen(("open", log_file))

    return logger
