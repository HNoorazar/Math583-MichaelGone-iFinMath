"""Importing this module will insert HGROOT to the start of sys.path."""
import logging
import subprocess


def log(msg, level=logging.INFO):
    logging.getLogger(__name__).log(level=level, msg=msg)


try:
    HGROOT = subprocess.check_output(["hg", "root"]).strip().decode("utf-8")
    paths = [HGROOT]
    import mmf_setup

    mmf_setup.ROOT = HGROOT
    mmf_setup.HGROOT = HGROOT

    # Now add any paths specified in system.cfg
    import mmf_setup.set_path

    mmf_setup.set_path.set_path_from_file()

except subprocess.CalledProcessError:
    # Could not run hg or not in a repo.
    log("Could not set HGROOT", level=logging.WARNING)
