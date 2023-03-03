"""Different matplotlib style options for customizing plots.

To allow for more flexibility, we provide these as dictionaries.  (This allows one to
use hashes for example, which are not valid in an mplstyle file.)

Use as you would styles:

>>> import matplotlib.pyplot as plt
>>> import mpl_styles
>>> plt.style.use(mpl_styles.cycles)

Or, better, use it locally:

>>> with plt.style.context(mpl_styles.cycles, after_reset=True):
...     plt.plot([1,2], [1,2])

etc.
"""
import contextlib, importlib
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# https://miguendes.me/the-best-way-to-compare-two-dictionaries-in-python
# from deepdiff import DeepDiff


def diff_styles(style1, style2):
    rc1 = {}
    rc2 = {}

    with plt.style.context(style1, after_reset=True):
        rc1.update(plt.rcParams)

    with plt.style.context(style2, after_reset=True):
        rc2.update(plt.rcParams)

    return DeepDiff(rc1, rc2)


# For plots in the Cycles paper
constants = dict(
    pt_per_inch=72.27,
    fullwidth_pt=480.0,  # \begin{figure*}\showthe\textwidth\end{figure*}
    textwidth_pt=312.0,  # \begin{figure}\showthe\textwidth\end{figure}
    marginwidth_pt=144.0,  # \begin{marginfigure}\showthe\textwidth\end{marginfigure}
    golden_mean=2 / (1 + np.sqrt(5)),
)
constants["fullwidth"] = constants["fullwidth_pt"] / constants["pt_per_inch"]
constants["textwidth"] = constants["textwidth_pt"] / constants["pt_per_inch"]
constants["marginwidth"] = constants["marginwidth_pt"] / constants["pt_per_inch"]


cycles = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathpazo}",
    "font.family": "serif",
    # "font.serif": "Palatino",
    "axes.titlesize": 10.0,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
}


@contextlib.contextmanager
def cycles_style(
    figuretype="figure",
    height="golden_mean*width",
    base="seaborn-darkgrid",
    after_reset=True,
):
    style = dict(cycles)
    d = dict(constants)
    width = {
        "figure": "textwidth",
        "figure*": "fullwidth",
        "marginfigure": "marginwidth",
    }[figuretype]

    d["width"] = eval(str(width), d)
    d["height"] = eval(str(height), d)
    style["figure.figsize"] = (d["width"], d["height"])

    with plt.style.context([base, style], after_reset=after_reset):
        yield
