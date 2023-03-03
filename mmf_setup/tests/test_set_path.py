import copy

try:
    from importlib import reload
except ImportError:
    pass

import os.path
import shutil
import subprocess
import sys
import tempfile

import mmf_setup.set_path.hgroot

import pytest


@pytest.fixture()
def tmpdir():
    """Provides a temporary directory for testing."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_set_path_hgroot():
    HGROOT = subprocess.check_output(["hg", "root"]).strip().decode("utf8")
    while HGROOT in sys.path:
        sys.path.remove(HGROOT)

    reload(mmf_setup.set_path.hgroot)
    assert HGROOT == sys.path[0]
    assert mmf_setup.HGROOT == HGROOT


def test_set_path_from_file_empty_path(tmpdir):
    import mmf_setup.set_path

    mmf_setup.ROOT = tmpdir

    assert tmpdir not in sys.path

    original_path = copy.deepcopy(sys.path)

    def run_test(
        paths=[],
        section="mmf_setup",
        filename="setup.cfg",
        expected_paths=None,
        check=False,
    ):
        sys.path = copy.deepcopy(original_path)
        with open(os.path.join(tmpdir, filename), "w") as f:
            f.write("[{}]\n".format(section))
            if paths:
                f.write("paths = " + paths[0] + "\n")
                for path in paths[1:]:
                    f.write("        " + path + "\n")

        vargs = ()
        if filename != "setup.cfg":
            vargs = (filename,)

        mmf_setup.set_path.set_path_from_file(*vargs, check=check)

        if expected_paths is not None:
            paths = expected_paths

        expected_paths = []
        for path in paths:
            if not os.path.isabs(path):
                path = os.path.join(tmpdir, path)
            expected_paths.append(os.path.abspath(path))

        assert sys.path == expected_paths + original_path

    run_test(paths=["src1"], filename="setup1.cfg")
    run_test(paths=[], expected_paths=["."])
    run_test(paths=["."])
    run_test(paths=[".", "src"])
    run_test(paths=[".", "src"], section="mmf_setup_mispelled", expected_paths=[tmpdir])
    run_test(paths=[".   # comment ignored"], expected_paths=["."])
