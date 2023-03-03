"""Regression test for issue #17: add HGROOT variable."""

import mmf_setup
import subprocess


def test_issue_17():
    mmf_setup.nbinit(hgroot=True, debug=True)
    assert mmf_setup.HGROOT == (
        subprocess.check_output(["hg", "root"]).strip().decode("utf8")
    )
