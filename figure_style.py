"""Plotting tools.

This module includes some additional tools for making publication
quality figures for LaTeX documents.  The main goal is for good
control over sizing, fonts, etc.

Usage
-----
For a paper, one should typically create a subclass of the :cls:`Paper` class,
and then define various `fig_*` methods.

To Do
-----
.. todo:: Fix this bug!

   >>> from mmf.utils.mmf_plot import Figure
   >>> fig = Figure()
   >>> plt.plot([0, 1], [1, 1.118])
   >>> fig.adjust()

   This has something to do with the yticks.  If they are set, then
   things work fine:

   >>> plt.clf()
   >>> plt.plot([0, 1], [1, 1.118])
   >>> plt.yticks(plt.yticks()[0])
   >>> fig.adjust()

"""
from __future__ import division

import copy
import inspect
import logging
import os.path
from pathlib import Path

import numpy as np
import scipy.stats
import scipy.interpolate
import scipy as sp
import matplotlib.collections
import matplotlib.artist
import matplotlib.colors
import matplotlib.gridspec
import matplotlib.pyplot as plt


from mmfutils.containers import Object

# import mmf.utils.mac

_FINFO = np.finfo(float)
_EPS = _FINFO.eps


class Paper(object):
    """Subclass this to generate a set of figures of a paper.  Each figure
    should have a corresponding `fig_*` method that returns an appropriate
    :cls:`Figure` object.

    Parameters
    ----------
    style : ['aps', 'arXiv', 'nature']
       Font style etc.  Figures will also be saved to `<fig_dir>/style`.
    final : bool
       If `True`, then make plots better typographically, but less useful
       interactively.
    save : bool
       If `True`, then save the figures.  This can take some time, so while
       fiddling you might like this to be `False`
    figdir : str
       Figures will be save here organized by `style`.
    """

    figdir = "./_build/figures/%(style)s"
    style = "arXiv"
    final = True
    save = True

    # figure() caches the last myfig so we can save even if the user
    # forgets to return it.
    _cached_myfig = None

    def __init__(self, **kw):
        """Initialize plotting to use correct fonts and figure size.

        Additional kw arguments are passed to :cls:`LaTeXPlotProperties`.
        """
        for _attr in ["final", "save", "style", "figdir"]:
            setattr(self, _attr, kw.pop(_attr, getattr(self, _attr)))
        self.plot_properties = LaTeXPlotProperties(style=self.style, **kw)
        self.plot_properties.initialize_matplotlib()

    def savefig(self, myfig, _meth_name=None, dpi=None, ext=".pdf", suffix="", **kw):
        dir = self.figdir % self.__dict__
        if self.save:
            if not os.path.exists(dir):
                os.makedirs(dir)

            if not isinstance(myfig, Figure):
                logger = logging.getLogger("figure_style.Paper.savefig")
                if _meth_name is not None:
                    msg = "Method {} did not return Figure instance.".format(_meth_name)
                else:
                    msg = (
                        "savefig called without Figure instance "
                        + "(got myfig={})".format(myfig)
                    )

                msg = "\n".join([msg, "Using cached Figure (may be incorrect)."])
                logger.warn(msg)
                myfig = self._cached_myfig

            filename = myfig.filename
            if filename is None:
                if myfig.ext is not None:
                    ext = myfig.ext
                if _meth_name.startswith("fig_"):
                    _meth_name = _meth_name[4:]
                filename = f"{_meth_name}{'_'+dpi if dpi else ''}{suffix}{ext}"
            filename = Path(dir, filename)
            myfig.savefig(filename, dpi=dpi, **kw)

    def draw(self, meth, dpi=None, suffix="", backend=None, *v, **kw):
        """Draw and save the specified figure.

        Arguments
        ---------
        meth : str, method
           Method or name of method to draw figure.
        dpi : int
           Resolution to use if rasterizing.
        suffix : str, None
           Appended to the filename before saving: `{filename}{suffix}{ext}`.
        """
        if isinstance(meth, str):
            name = meth
            meth = getattr(self, name)
        elif inspect.ismethod(meth):
            name = meth.im_func.func_name
        print("Drawing figure: %s()" % (name,))
        myfig = meth(*v, **kw)
        # if fig.tight_layout:
        #     plt.tight_layout(pad=0.0)
        self.savefig(
            myfig=myfig, _meth_name=name, dpi=dpi, suffix=suffix, backend=backend
        )

    def draw_all(self):
        r"""Draw (and save) all figures."""
        # Close all plots so that the figures can be opened at the appropriate
        # size.
        plt.close("all")
        for meth in [
            _meth
            for _name, _meth in inspect.getmembers(self)
            if _name.startswith("fig_") and inspect.ismethod(_meth)
        ]:
            self.draw(meth)

    def figure(self, num=None, **kw):
        r"""Call this to get a new :cls:`Figure` object."""
        myfig = Figure(num=num, plot_properties=self.plot_properties, **kw)
        self._cached_myfig = myfig
        return myfig

    def _fig_template(self):
        r"""Sample figure."""
        fig = self.figure(
            num=1,  # If you want to redraw in the same window
            width="columnwidth",  # For two-column documents, vs. 'textwidth'
        )
        x = np.linspace(-1, 1, 100)
        y = np.sin(4 * np.pi * x)
        plt.plot(x, y, "-")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        return fig


class Defaults(object):
    r"""Default values.  Change the values here to affect all plots.
    (Defaults are set when :cls:`Figure` instances are created.)"""

    rc = {
        "axes": dict(linewidth=1.0, edgecolor="grey", grid=True, axisbelow=True),
        "grid": dict(ls="-", lw=1.0, c="WhiteSmoke", alpha=0.3),
        "ytick": dict(direction="out"),
        "xtick": dict(direction="out"),
        "xtick.major": dict(size=2),
        "xtick.minor": dict(size=1),
        "ytick.major": dict(size=2),
        "ytick.minor": dict(size=1),
        "lines": dict(linewidth=1),
        # 'xtick': dict(color='k'),
        # 'ytick': dict(color='k'),
    }

    @classmethod
    def set_rc(cls, **kw):
        rc = dict(cls.rc)
        for key in kw:
            rc.setdefault(key, kw[key]).update(kw[key])
        for _name in rc:
            plt.rc(_name, **rc[_name])


def plot_errorbars(
    x,
    y,
    dx=None,
    dy=None,
    colour="",
    linestyle="",
    pointstyle=".",
    barwidth=0.5,
    **kwargs,
):

    if pointstyle != "" or linestyle != "":
        plt.plot(x, y, pointstyle + linestyle + colour, **kwargs)
    elif dx is None and dy is None:
        # Plot points if both error bars are not drawn.
        plt.plot(x, y, "." + colour, **kwargs)

    if dx is not None:
        xmax = x + dx
        xmin = x - dx
    if dy is not None:
        ymax = y + dy
        ymin = y - dy

    for n in xrange(len(x)):
        if dx is not None:
            plt.plot([xmin[n], xmax[n]], [y[n], y[n]], "-|" + colour, lw=barwidth)
        if dy is not None:
            plt.plot([x[n], x[n]], [ymin[n], ymax[n]], "-_" + colour, lw=barwidth)


def plot_err(x, y, yerr=None, xerr=None, **kwarg):
    """Plot x vs. y with errorbars.

    Right now we support the following cases:
    x = 1D, y = 1D
    """
    if 1 == len(x.shape) and 1 == len(y.shape):
        plt.errorbar(x, y, yerr=yerr, xerr=xerr, **kwarg)
    elif 1 == len(x.shape) and 1 < len(y.shape):
        plot_axis = np.where(np.array(y.shape) == len(x))[0][0]
        y = y.swapaxes(0, plot_axis)
        Nx, Ny = y.shape
        for n in Ny:
            plt.errorbar(x, y[:, n], **kwarg)
    elif max(x.shape) == np.prod(x.shape):
        plot_axis = np.argmax(x.shape)
        x = x.ravel()
        y = y.swapaxes(0, plot_axis)
        if yerr is not None:
            yerr = yerr.swapaxes(0, plot_axis)
        Nx, Ny = y.shape
        for n in xrange(Ny):
            if yerr is None:
                plt.errorbar(x, y[:, n], xerr=xerr, **kwarg)
            else:
                plt.errorbar(x, y[:, n], xerr=xerr, yerr=yerr[:, n], **kwarg)
    else:
        plt.plot(x, y, **kwarg)


def error_line(x, y, dy, fgc="k", bgc="w", N=20, fill=True):
    """Plots a curve (x, y) with gaussian errors dy represented by
    shading out to 5 dy."""
    yp0 = y
    ym0 = y
    pdf = sp.stats.norm().pdf
    to_rgb = plt.matplotlib.colors.ColorConverter().to_rgb

    bg_colour = np.array(to_rgb(bgc))
    fg_colour = np.array(to_rgb(fgc))
    if fill:
        patches = []
    else:
        lines = []

    ax = plt.gca()
    for sigma in np.linspace(0, 5, N)[1:]:
        yp = y + dy * sigma
        ym = y - dy * sigma
        c = pdf(sigma) / pdf(0.0)
        # colour = fg_colour*c + (1.0-c)*bg_colour
        colour = fg_colour

        if fill:
            X = np.hstack((x, np.flipud(x)))
            Y = np.hstack((yp0, np.flipud(yp)))
            patches.extend(ax.fill(X, Y, fc=colour, ec=colour, lw=0, alpha=c))
            X = np.hstack((x, np.flipud(x)))
            Y = np.hstack((ym0, np.flipud(ym)))
            patches.extend(ax.fill(X, Y, fc=fg_colour, ec=fg_colour, lw=0, alpha=c))
        else:
            lines.extend(
                ax.plot(x, yp, color=colour, alpha=c)
                + ax.plot(x, ym, color=fg_colour * c + (1.0 - c) * bg_colour)
            )

        ym0 = ym
        yp0 = yp

    if fill:
        artists = [matplotlib.collections.PatchCollection(patches)]
    else:
        if False:
            # Can't add alphas to LineCollection unfortunately.
            args = dict(
                zip(
                    ["segments", "linewidths", "colors", "antialiaseds", "linestyles"],
                    zip(
                        *[
                            (
                                _l.get_xydata(),
                                _l.get_linewidth(),
                                _l.get_color(),
                                _l.get_antialiased(),
                                _l.get_linestyle(),
                            )
                            for _l in lines
                        ]
                    ),
                )
            )
            artists = [matplotlib.collections.LineCollection(**args)]
        else:
            artists = [ListCollection(lines)]

        # Remove individual lines from the axis...
        for _l in lines:
            _l.remove()

        # ... and add back as a collection.
        ax.add_collection(artists[0])

    return artists


class LaTeXPlotProperties(object):
    r"""Instances of this class provide a description of properties of
    a plot based on numbers extracted from a LaTeX file.  Insert the
    following code into the section where the plot is to appear in
    order to extract the appropriate parameters and then use the
    reported values to initialize this class::

       \showthe\textwidth
       \showthe\textheight
       \showthe\columnwidth
       \showthe\baselineskip

    .. note:: We assume that the document is typeset using the
       Computer Modern fonts.
    """
    textwidth_pt = 510.0  # From LaTeX \showthe\textwidth
    textheight_pt = 672.0  # From LaTeX \showthe\textheight
    columnwidth_pt = 246.0  # From LaTeX \showthe\columnwidth
    baselineskip_pt = 12.0  # From LaTeX \showthe\baselineskip
    tick_fontsize = "footnotesize"  # Ticks etc. will be typeset in this font

    usetex = True
    # If `True`, then LaTeX will be used to typeset labels
    # etc.  Otherwise, labels etc. will be left as plain text
    # that can be replaced with the ``\psfrag{}{}`` command in
    # the LaTeX file.
    #
    # As of matplotlib version 1.0.1, psfrag replacements do not
    # work, so the default is now to use LaTeX.

    style = None  # Pick a style.  One of 'aps', 'arXiv', or 'nature'
    grid = True  # Draw gridlines.  Turn this off for PRC

    # The following are "constants" that you should typically not
    # have to adjust unless you use a different font package.
    font_info = {
        "times": ("ptm", r"\usepackage{mathptm}"),
        "helvetica": ("phv", r"\usepackage[scaled]{helvet}"),
        "euler": (
            "zeur",
            r"\usepackage[sc]{mathpazo}" + r"\usepackage[euler-digits, small]{eulervm}",
        ),
    }
    font = {
        "family": "serif",
        "serif": ["computer modern roman"],
        "sans-serif": ["computer modern sans serif"],
        "monospace": ["computer modern typewriter"],
    }
    font = {
        "family": "serif",
        "serif": ["euler"],
        "sans-serif": ["bera sans serif"],
        "monospace": ["computer modern typewriter"],
    }

    font = {
        "family": "sans-serif",
        "sans-serif": ["Helvetica"],
        "monospace": ["computer modern typewriter"],
    }

    latex_preamble = [r"\usepackage{mmfmath}\usepackage{amsmath}\usepackage{siunitx}"]
    latex_preamble = [
        r"\usepackage{amsmath}\providecommand{\mathdefault}[1]{#1}\usepackage{siunitx}"
    ]
    # List of strings to add to LaTeX preamble.  Add any
    # ``\usepackage{}`` commands here.
    #
    # .. note:: Don't forget to use raw strings to prevent
    #           escaping of characters.  Thus use something like the
    #           default value: `[r"\usepackage{amsmath}"]`"""),

    golden_mean = (np.sqrt(5) - 1) / 2
    font_size_pt = 10
    font_factors = {  # Font size reduction factors for latex fonts.
        "small": 9 / 10,
        "footnotesize": 8 / 10,
    }

    # Some units.  These can appear in expressions.
    inches_per_pt = 1.0 / 72.27
    inches = 1.0
    pt = inches_per_pt

    def __init__(self, **kw):
        self.__dict__.update(**kw)
        super().__init__()
        self.init()

    def init(self):
        self.textwidth = self.textwidth_pt * self.inches_per_pt
        self.textheight = self.textheight_pt * self.inches_per_pt
        self.columnwidth = self.columnwidth_pt * self.inches_per_pt
        self.baselineskip = self.baselineskip_pt * self.inches_per_pt
        self.tick_font = self.font_size_pt * self.font_factors[self.tick_fontsize]

    def initialize_matplotlib(self):
        """:class:`Figure` calls this."""
        if self.style == "none":
            return
        elif "aps" == self.style:
            # For APS journals: use times and no smallcaps!
            self.font = {
                "family": "serif",
                #'serif': ['times'],
                #'sans-serif': ['computer modern sans serif'],
                #'monospace': ['computer modern typewriter']
            }
            self.latex_preamble.extend(
                [
                    r"\usepackage{siunitx}",
                    r"\usepackage[varg]{newtxmath}",
                    r"\usepackage[scaled]{helvet}",
                    r"\usepackage{newtxtext}",
                    r"\sisetup{mode=text, detect-all=true, detect-mode=false}",
                    r"\def\textsc#1{\text{\MakeUppercase{#1}}}",
                ],
            )
            self.textwidth_pt = 510.0
            self.textheight_pt = 672.0
            self.columnwidth_pt = 246.0

            # Makes smallest font 7pt
            self.font_size_pt = 7.0 / self.font_factors["footnotesize"]

            self.baselineskip_pt = 12.0

        elif "arXiv" == self.style:
            # My style for the arXiv.  Use Palatino and Euler.
            self.font = {
                "family": "serif",
                #'serif': ['euler'],
                #'sans-serif': ['bera sans serif'],
                #'monospace': ['bera mono']
            }
            self.latex_preamble.extend(
                [
                    r"\usepackage{siunitx}",
                    r"\usepackage[sc]{mathpazo}",
                    r"\usepackage[euler-digits,small]{eulervm}",
                    r"\usepackage[scaled]{berasans}",
                    r"\usepackage[scaled]{beramono}",
                    # Sans-serif for figures... if we go for sf captions.
                    r"\renewcommand{\rmdefault}{\sfdefault}",
                    r"\sisetup{mode=text, detect-all=true, detect-mode=false}",
                    # r"\sisetup{mode=math, math-rm=\usefont{U}{zeur}{m}{n}{}"
                    # + r"\selectfont}",
                ]
            )
            self.textwidth_pt = 510.0
            self.textheight_pt = 672.0
            self.columnwidth_pt = 246.0
            self.font_size_pt = 10.0
            self.baselineskip_pt = 12.0

        elif "nature" == self.style:
            # For Nature journals: use times and helvetica.
            # There is no official class, so we assume revtex here (for column
            # width etc.)
            self.font = {
                "family": "sans-serif",
                #'serif': ['times'],
                #'sans-serif': ['Helvetica'],
                #'monospace': ['computer modern typewriter']
            }
            self.latex_preamble.extend(
                [
                    r"\usepackage{newtxmath}",
                    # http://tex.stackexchange.com/a/197874/6903
                    r"\usepackage{sansmathfonts}",
                    # r"\usepackage{helvet}",
                    r"\renewcommand{\rmdefault}{\sfdefault}",
                    r"\usepackage{amsmath}",
                    r"\usepackage{siunitx}",
                    r"\sisetup{mode=text, detect-all=true, detect-mode=false}",
                ]
            )
            self.textwidth_pt = 510.0
            self.textheight_pt = 672.0
            self.columnwidth_pt = 246.0

            # Makes smallest font 7pt
            self.font_size_pt = 7.0 / self.font_factors["footnotesize"]

            self.baselineskip_pt = 12.0

        matplotlib.rc("text", usetex=self.usetex)
        matplotlib.rc("font", **self.font)
        matplotlib.rc("text.latex", preamble="\n".join(self.latex_preamble))
        matplotlib.rc("font", size=self.font_size_pt)
        matplotlib.rc("axes", titlesize=self.font_size_pt, labelsize=self.font_size_pt)
        # Use TT fonts
        matplotlib.rc("ps", fonttype=42)

        if not self.grid:
            # Disable grid-lines.  This only disables major gridlines: the
            # minor gridlines must be controlled separately...
            # see Figure.__init__()
            matplotlib.rc("axes", grid=False)


# Default global instance.
_PLOT_PROPERTIES = LaTeXPlotProperties()


def xticks(ticks):
    """Replace ticks with real text so psfrag works.  There
    was an API change somewhere along the line that broke this..."""
    plt.xticks(ticks, ticks)


def yticks(ticks):
    plt.yticks(ticks, ticks)


class Figure(object):
    r"""This class represents a single figure and allows customization
     of properties, as well as providing plotting facilities.

     Notes
     -----
     Units are either pts (for fonts) or inches (for linear
     measurements).

     Examples
     --------
     Here is an example of a figure suitable for a half of a page in a
     normal LaTeX book.  First we run the following file through
     LaTeX::

        \documentclass{book}
        \begin{document}
        \showthe\textwidth
        \showthe\columnwidth
        \showthe\baselineskip
        \end{document}

    This gives::

       > 345.0pt.
       l.3 \showthe\textwidth

       ?
       > 345.0pt.
       l.4 \showthe\columnwidth

       ?
       > 12.0pt.
       l.5 \showthe\baselineskip

     .. plot::
        :include-source:

        x = np.linspace(0, 1.01, 100)
        y = np.sin(x)
        plot_prop = LaTeXPlotProperties(textwidth_pt=345.0,
                                        columnwidth_pt=345.0,
                                        baselineskip_pt=12.0)
        fig = Figure(filename='tst_book.eps',
                     width='0.5*textwidth',
                     plot_properties=plot_prop)
        plt.plot(x, y, label="r'\sin(x)'")
        plt.axis([-0.02, 1.02, -0.02, 1.02])
        plt.ylabel(
            r'$\int_{0}^{x}\left(\frac{\cos(\tilde{x})}{1}\right)d{\tilde{x}}$')
        #fig.savefig()

     Here is another example using a two-column article::

        \documentclass[twocolumn]{article}
        \begin{document}
        \showthe\textwidth
        \showthe\columnwidth
        \showthe\baselineskip
        \end{document}

     This gives::
        > 469.0pt.
        l.3 \showthe\textwidth

        ?
        > 229.5pt.
        l.4 \showthe\columnwidth

        ?
        > 12.0pt.
        l.5 \showthe\baselineskip

     .. plot::
        :include-source:

        x = np.linspace(0, 1.01, 100)
        y = np.sin(x)
        plot_prop = LaTeXPlotProperties(textwidth_pt=489.0,
                                        columnwidth_pt=229.5,
                                        baselineskip_pt=12.0)
        fig = Figure(filename='tst_article.eps',
                     plot_properties=plot_prop)
        plt.plot(x, y, label="r'\sin(x)'")
        plt.axis([-0.02, 1.02, -0.02, 1.02])
        plt.ylabel(
            r'$\int_{0}^{x}\left(\frac{\cos(\tilde{x})}{1}\right)d{\tilde{x}}$')
        #fig.savefig()

    """
    num = None  # Figure number
    filename = None  # Filename for figure.
    ext = ".pdf"  # Figure extension
    width = "columnwidth"  # Expression with 'columnwidth' and/or 'textwidth'
    height = (
        None  # Fraction of `width`, str i.e. '0.5*columnwidth') or None (golden mean)
    )
    plot_properties = None
    axes_dict = dict(labelsize="medium")
    tick_dict = dict(labelsize="small")
    legend_dict = dict(
        fontsize="small",
        handlelength=4.0,
        frameon=True,
        # lw=0.5, c='k'
    )
    constrained_layout = True

    # I cannot figure out how to set the size of the axes and allow
    # tight_layout() to work.  If you want tight_layout() to work, then you
    # should set this to be `True` and do not provide `margin_factors`.
    margin_factors = dict(  # These allocate extra space for labels etc.
        top=0.5, left=2.8, bot=3, right=0.5
    )

    # If you need to specify the margins in terms of fractions of the
    # figure, use these instead.
    subplot_args = None

    autoadjust = False
    # Attempt to autoadjust for labels, otherwise you can do this manually by
    # calling :meth:`adjust`.

    figures = {}  # Dictonary of computed figures.
    on_draw_id = None  # Id associated with 'on_draw' event
    dpi = 600  # Resolution for saved figure (affects images)

    fig_box = False  # If True, then draw a box around the figure

    def __init__(self, **kw):
        self.margin_factors = dict(
            [(_k, kw.pop(_k, self.margin_factors[_k])) for _k in self.margin_factors]
        )
        self._kw = kw
        for _key in kw:
            if not hasattr(self, _key):
                raise AttributeError("Figure has no attribute '{}'".format(_key))

        self.__dict__.update(**kw)
        super().__init__()
        self.init()

    def init(self):
        if self.plot_properties is None:
            self.plot_properties = _PLOT_PROPERTIES
        pp = self.plot_properties
        self._inset_axes = set()
        for _size in pp.font_factors:
            self._size = pp.font_size_pt * pp.font_factors[_size]

        if "num" in self._kw or "filename" in self._kw:
            width = eval(self.width, pp.__dict__)
            if self.height is None:
                self.height = pp.golden_mean
            elif isinstance(self.height, str):
                height = eval(self.height, dict(pp.__dict__, width=width))
            else:
                height = self.height * width

            fig_width = width
            fig_height = height

            size = pp.font_size_pt * pp.inches_per_pt

            if self.subplot_args:
                # If suplot_args are provided, then disable
                # constrained_layout as the user is managing the layout.
                subplot_args = self.subplot_args
                self.constrained_layout = False
            else:
                # top space = 1/2 font
                space_top = self.margin_factors["top"] * size
                space_left = self.margin_factors["left"] * size
                space_bottom = self.margin_factors["bot"] * size
                space_right = self.margin_factors["right"] * size

                # Compute subplot_spec.  These are fractional positions of
                # the final figure size used to locate the main axis or
                # subplot in the figure.
                subplot_args = dict(
                    left=space_left / fig_width,
                    bottom=space_bottom / fig_height,
                    right=1.0 - (space_left + space_right) / fig_width,
                    top=1.0 - (space_bottom + space_top) / fig_height,
                )

            if False:
                Defaults.set_rc(
                    **{
                        "font": dict(size=pp.font_size_pt),
                        "axes": self.axes_dict,
                        "xtick": self.tick_dict,
                        "ytick": self.tick_dict,
                        "legend": self.legend_dict,
                    }
                )

            self.fig = plt.figure(
                num=self.num,
                figsize=(fig_width, fig_height),
                constrained_layout=self.constrained_layout,
            )

            self.subplot_spec = matplotlib.gridspec.GridSpec(
                1, 1, figure=self.fig, **subplot_args
            )[0, 0]

            # Always create an axis... users can remove this with
            # ax.remove()
            self.ax = plt.subplot(self.subplot_spec)

            self.figure_manager = plt.get_current_fig_manager()

            """
            if mmf.utils.mac.has_appkit:
                # Check for screen info and position the window.
                screens = mmf.utils.mac.get_screen_info()
                if 1 < len(screens):
                    # More than one screen.  Put this on the second screen.
                    screen = screens[-1]
                    self.figure_manager.window.geometry(
                        "+%i+%i" % (screen.x, screen.y))
            """
            self.num = self.figure_manager.num
            self.figures[self.num] = self.figure_manager

            if pp.grid:
                plt.grid(True, which="both")
                plt.grid(True, "minor", lw=0.2)

        if self.autoadjust and False:
            # This makes the axis full frame.  Use adjust to shrink.
            plt.gca().set_position([0, 0, 1, 1])
            self.start_adjusting()
        elif False:
            self.stop_adjusting

        if self.fig_box:
            rect = matplotlib.patches.Rectangle(
                xy=(0, 0), width=1, height=1, lw=self.fig_box, fill=None
            )
            self.fig.add_artist(rect)

    def activate(self):
        return plt.figure(self.num)

    def start_adjusting(self):
        if self.on_draw_id:
            self.figure_manager.canvas.mpl_disconnect(self.on_draw_id)
        self.on_draw_id = self.figure_manager.canvas.mpl_connect(
            "draw_event", self.on_draw
        )

    def stop_adjusting(self):
        if self.on_draw_id:
            self.figure_manager.canvas.mpl_disconnect(self.on_draw_id)
        self.on_draw_id = 0

    def new_inset_axes(self, rect):
        r"""Return a new axes set inside the main axis.

        Parameters
        ----------
        rect : [left, bottom, width or right, height or top]
           This is the rectangle for the new axes (the labels etc. will be
           outside).  Coordinates may be either floating point numbers which
           specify the location of the inset in terms of a fraction between 0
           and 1 of the current axis.

           One may also specify the coordinates in the data units of the actual
           corners by specifying the data as an imaginary number.  This will be
           transformed into relative axis coordinates using the current axis
           limits (the subplot will not subsequently move).  (Not implemented
           yet.)
        """
        ax = plt.axes(rect)
        self._inset_axes.add(ax)
        return ax

    def axis(self, *v, **kw):
        r"""Wrapper for :func:`pyplot.axis` function that applies the
        transformation to each axis (useful if :func:`pyplot.twinx` or
        :func:`pyplot.twiny` has been used)."""
        fig = self.figure_manager.canvas.figure
        for _a in fig.axes:
            _a.axis(*v, **kw)

    def adjust(self, full=True, padding=0.05):
        r"""Adjust the axes so that all text lies withing the figure.
        Optionally, add some padding."""
        plt.ioff()
        plt.figure(self.num)
        if full:
            # Reset axis to full size.
            fig = self.figure_manager.canvas.figure
            for _a in fig.axes:
                _a.set_position([0, 0, 1, 1])
        on_draw_id = self.figure_manager.canvas.mpl_connect("draw_event", self.on_draw)
        try:
            plt.ion()
            plt.draw()
        except:
            raise
        finally:
            pass
        self.figure_manager.canvas.mpl_disconnect(on_draw_id)

        adjustable_axes = [_a for _a in fig.axes if _a not in self._inset_axes]

        if 0 < padding:
            for _a in adjustable_axes:
                bb_a = _a.get_position()
                dx = bb_a.width * padding / 2
                dy = bb_a.height * padding / 2
                bb_a.x0 += dx
                bb_a.x1 -= dx
                bb_a.y0 += dy
                bb_a.y1 -= dy
                bb_a = _a.set_position(bb_a)

    @staticmethod
    def _shrink_bb(bb, factor=_EPS):
        r"""Shrink the bounding box bb by factor in order to prevent unneeded
        work due to rounding."""
        p = bb.get_points()
        p += factor * (np.diff(p) * np.array([1, -1])).T
        bb.set_points(p)
        return bb

    def _adjust(self, logger=logging.getLogger("mmf.utils.mmf_plot.Figure._adjust")):
        r"""Adjust the axes to make sure all text is inside the box."""
        fig = self.figure_manager.canvas.figure
        bb_f = fig.get_window_extent().inverse_transformed(fig.transFigure)
        logger.debug("Fig  bb %s" % (" ".join(str(bb_f).split()),))

        texts = []
        adjustable_axes = [_a for _a in fig.axes if _a not in self._inset_axes]
        for _a in adjustable_axes:
            texts.extend(_a.texts)
            texts.append(_a.title)
            texts.extend(_a.get_xticklabels())
            texts.extend(_a.get_yticklabels())
            texts.append(_a.xaxis.get_label())
            texts.append(_a.yaxis.get_label())

        bboxes = []
        for t in texts:
            if not t.get_text():
                # Ignore empty text!
                continue
            bbox = t.get_window_extent()
            # the figure transform goes from relative
            # coords->pixels and we want the inverse of that
            bboxi = bbox.inverse_transformed(fig.transFigure)
            bboxes.append(bboxi)

        # this is the bbox that bounds all the bboxes, again in
        # relative figure coords
        bbox = self._shrink_bb(matplotlib.transforms.Bbox.union(bboxes))
        adjusted = False
        if not np.all([bb_f.contains(*c) for c in bbox.corners()]):
            # Adjust axes position
            for _a in adjustable_axes:
                bb_a = _a.get_position()
                logger.debug("Text bb   %s" % (" ".join(str(bbox).split()),))
                logger.debug("Axis bb   %s" % (" ".join(str(bb_a).split()),))
                bb_a.x0 += max(0, bb_f.xmin - bbox.xmin)
                bb_a.x1 += min(0, bb_f.xmax - bbox.xmax)
                bb_a.y0 += max(0, bb_f.ymin - bbox.ymin)
                bb_a.y1 += min(0, bb_f.ymax - bbox.ymax)
                logger.debug("New  bb   %s" % (" ".join(str(bb_a).split()),))
                _a.set_position(bb_a)
            adjusted = True
        return adjusted

    def on_draw(self, event, _adjusting=[False]):
        """We register this to perform processing after the figure is
        drawn, like adjusting the margins so that the labels fit."""
        fig = self.figure_manager.canvas.figure

        logger = logging.getLogger("mmf.utils.mmf_plot.Figure.on_draw")
        if _adjusting[0]:
            # Don't recurse!
            return

        if event is None:
            # If called interactively...
            import pdb

            pdb.set_trace()
        _adjusting[0] = True

        try:
            _max_adjust = 10
            adjusted = False
            for _n in xrange(_max_adjust):
                adjusted = self._adjust(logger=logger)
                if adjusted:
                    fig.canvas.draw()
                else:
                    break
            if adjusted:
                # Even after _max_adjust steps we still needed adjusting:
                logger.warn("Still need adjustment after %i steps" % (_max_adjust,))
        finally:
            _adjusting[0] = False

    def adjust_axis(
        self,
        extents=None,
        xl=None,
        xh=None,
        yl=None,
        yh=None,
        extend_x=0.0,
        extend_y=0.0,
    ):
        if extents is not None:
            plt.axis(extents)
        xl_, xh_, yl_, yh_ = plt.axis()
        if xl is not None:
            xl_ = xl
        if xh is not None:
            xh_ = xh
        if yl is not None:
            yl_ = yl
        if yh is not None:
            yh_ = yh
        plt.axis([xl_, xh_, yl_, yh_])
        dx = extend_x * (xh_ - xl_)
        dy = extend_y * (yh_ - yl_)
        return plt.axis([xl_ - dx, xh_ + dx, yl_ - dy, yh_ + dy])

    def savefig(self, filename=None, dpi=None, backend=None):
        if not filename:
            filename = self.filename
        print("Saving plot as {}...".format(filename))
        fig = self.fig
        plt.ion()  # Do this to ensure autoadjustments
        plt.draw()  # are made!
        if dpi is None:
            dpi = self.dpi
        plt.savefig(filename, dpi=dpi, backend=backend)
        print("Saving plot as {}. Done.".format(filename))

    def __del__(self):
        """Destructor: make sure we unregister the autoadjustor."""
        self.autoadjust = False


# Here we monkeypath mpl_toolkits.axes_grid.inset_locator to allow for
# independent x and y zoom factors.
def monkey_patch_inset_locator():
    from mpl_toolkits.axes_grid.inset_locator import AnchoredZoomLocator
    import matplotlib.transforms

    def get_extent(self, renderer):
        bb = matplotlib.transforms.TransformedBbox(
            self.axes.viewLim, self.parent_axes.transData
        )
        x, y, w, h = bb.bounds

        xd, yd = 0, 0

        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        pad = self.pad * fontsize

        wh = np.array([w, h])
        return tuple((wh * self.zoom + 2 * pad).tolist() + [xd + pad, yd + pad])

    AnchoredZoomLocator.get_extent = get_extent


class ListCollection(matplotlib.collections.Collection):
    r"""Provide a simple :class:`matplotlib.collections.Collection` of a list of
    artists.  Provided so that this collection of artists can be simultaneously
    rasterized.  Used by my custom :func:`contourf` function."""

    def __init__(self, collections, **kwargs):
        matplotlib.collections.Collection.__init__(self, **kwargs)
        self.set_collections(collections)

    def set_collections(self, collections):
        self._collections = collections

    def get_collections(self):
        return self._collections

    @matplotlib.artist.allow_rasterization
    def draw(self, renderer):
        for _c in self._collections:
            _c.draw(renderer)


def contourf(*v, **kw):
    r"""Replacement for :func:`matplotlib.pyplot.contourf` that supports the
    `rasterized` keyword."""
    was_interactive = matplotlib.is_interactive()
    matplotlib.interactive(False)
    contour_set = plt.contourf(*v, **kw)
    for _c in contour_set.collections:
        _c.remove()
    collection = ListCollection(
        contour_set.collections, rasterized=kw.get("rasterized", None)
    )
    ax = plt.gca()
    ax.add_artist(collection)
    matplotlib.interactive(was_interactive)
    return contour_set


def imcontourf(x, y, z, contours=None, interpolate=True, *v, **kw):
    r"""Like :func:`matplotlib.pyplot.contourf` but does not actually find
    contours.  Just displays `z` using :func:`matplotlib.pyplot.imshow` which is
    much faster and uses exactly the information available.

    Parameters
    ----------
    x, y, z : array-like
       Assumes that `z` is ordered as `z[x, y]`.  If `x` and `y` have the same
       shape as `z`, then `x = x[:, 0]` and `y = y[0, :]` are used.  Otherwise,
       `z.shape == (len(x), len(y))`.  `x` and `y` must be equally spaced.
    interpolate : bool
       If `True`, then interpolate the function onto an evenly spaced set of
       abscissa using cublic splines.
    """
    x, y, z = map(np.asarray, (x, y, z))
    if x.shape == z.shape:
        x = x[:, 0]
    else:
        x = x.ravel()
    if y.shape == z.shape:
        y = y[0, :]
    else:
        y = y.ravel()
    assert z.shape[:2] == (len(x), len(y))

    if interpolate and not (
        np.allclose(np.diff(np.diff(x)), 0) and np.allclose(np.diff(np.diff(y)), 0)
    ):
        spl = scipy.interpolate.RectBivariateSpline(x, y, z)
        x = np.linspace(x.min(), x.max(), len(x))
        y = np.linspace(y.min(), y.max(), len(y))
        z = spl(x, y)

    assert np.allclose(np.diff(np.diff(x)), 0)
    assert np.allclose(np.diff(np.diff(y)), 0)
    kwargs = dict(**kw)
    kwargs.setdefault("cmap", "gist_heat")
    kwargs.setdefault("aspect", "auto")
    img = plt.imshow(
        np.rollaxis(z, 0, 2),
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        *v,
        **kwargs,
    )

    # Provide a method for updating the data properly for quick plotting.
    def set_data(z, img=img, sd=img.set_data):
        sd(np.rollaxis(z, 0, 2))

    img.set_data = set_data
    return img


def phase_contour(x, y, z, N=10, **kw):
    r"""Specialized contour plot for plotting the contours of constant phase for
    the complex variable z.  Plots `4*N` contours in total.  Note: two sets of
    contours are returned, and, due to processing, these do not have the correct
    values.

    The problem this solves is that plotting the contours of `np.angle(z)` gives
    a whole swath of contours at the discontinuity between `-pi` and `pi`.  We
    get around this by doing two things:

    1) We plot the contours of `abs(angle(z))`.  This almost fixes the problem,
       but can give rise to spurious closed contours near zero and `pi`.  To
       deal with this:
    2) We plot only the contours between `pi/4` and `3*pi/4`.  We do this twice,
       multiplying `z` by `exp(0.5j*pi)`.
    3) We carefully choose the contours so that they have even spacing.
    """
    levels = 0.5 * np.pi * (0.5 + (np.arange(N) + 0.5) / N)
    c1 = plt.contour(x, y, abs(np.angle(z)), levels=levels, **kw)
    c2 = plt.contour(x, y, abs(np.angle(z * np.exp(0.5j * np.pi))), levels=levels, **kw)
    c2.levels = c2.levels + 0.5 * np.pi
    c2.levels = np.where(c2.levels <= np.pi, c2.levels, c2.levels - 2.0 * np.pi)
    return c1, c2


def plot3d(x, y, z, wireframe=False, **kw):
    r"""Wrapper to generate 3d surface plots."""
    # Move these out once fixed.
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    if 1 == len(x.shape):
        x = x[:, None]
    if 1 == len(y.shape):
        y = y[None, :]
    if x.shape != z.shape:
        x = x + 0 * y
    if y.shape != z.shape:
        y = y + 0 * x

    assert z.shape == x.shape
    assert z.shape == y.shape
    assert np.allclose(np.diff(np.diff(x)), 0)
    assert np.allclose(np.diff(np.diff(y)), 0)
    kwargs = dict(**kw)
    fig = plt.gcf()
    ax = fig.gca(projection="3d")
    kw.setdefault("cmap", cm.jet)
    if wireframe:
        kw.setdefault("rstride", 10)
        kw.setdefault("cstride", 10)
        surf = ax.plot_wireframe(x, y, z, **kw)
    else:
        kw.setdefault("rstride", 1)
        kw.setdefault("cstride", 1)
        kw.setdefault("antialiased", False)
        surf = ax.plot_surface(x, y, z, **kw)

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.draw_if_interactive()
    return surf


def plot3dmpl(
    X, Y, Z, zmin=-np.inf, zmax=np.inf, xlabel=None, ylabel=None, abs_parts=False, **kw
):
    r"""Use MayaVi2 to plot the surface.

    Parameters
    ----------
    abs_parts : bool
       If `True`, the plot `abs(real)` and `-abs(imag)`.
    """
    from mayavi import mlab

    def draw(z, kw=dict(kw), **_kw):
        _kw.update(kw)
        return mlab.surf(X, Y, np.maximum(np.minimum(z, zmax), zmin), **_kw)

    if np.any(np.iscomplex(Z)):
        if abs_parts:
            s = (
                draw(abs(Z.real), colormap="Greens", opacity=1.0),
                draw(-abs(Z.imag), colormap="Reds", opacity=1.0),
            )
        else:
            s = (
                draw(Z.real, colormap="Greens", opacity=0.5),
                draw(Z.imag, colormap="Reds", opacity=0.5),
            )
    else:
        s = draw(Z)
    mlab.axes()
    if xlabel:
        mlab.xlabel(xlabel)
    if ylabel:
        mlab.xlabel(ylabel)
    return s


class SparklineHandler(matplotlib.legend_handler.HandlerBase):
    r"""Custom legend handler that supports "sparkline" legend entries (reduced
    copies of the actual data.

    Examples
    --------
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> y = np.sin(2*x) + x
    >>> l1, = plt.plot(x, y, 'b', label='sin(2x)+x')
    >>> l2, = plt.plot(x, -y/2, 'g--', label='-(sin(2x)+x)/2')
    >>> l2, = plt.plot(x, -y/2, 'g--', label='-(sin(2x)+x)/2')
    >>> plt.legend(handler_map={l1.__class__: SparklineHandler()})

    """

    def __init__(self, use_clip_box=False, *v, **kw):
        r"""
        Parameters
        ----------
        use_clip_box : bool
           If `True`, then use the axes clip_box for scaling,
           otherwise try to scale the actual data.  The advantage of using the
           clip_box is that the relative positions of the elements is
           maintained. This is not a good idea if the objects occupy only a
           small region of the plot.
        """
        self.use_clip_box = use_clip_box
        matplotlib.legend_handler.HandlerBase.__init__(self, *v, **kw)

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        spark = copy.copy(orig_handle)
        ax = spark.get_axes()
        clip_bbox = orig_handle.get_clip_box()
        # clip_bbox = ax.get_clip_box()
        clip_box = ax.transData.inverted().transform(clip_bbox)

        # trans currently takes the input data in the
        # Rectangle((xdescent, ydescent), width, height) to the appropriate
        # location in the legend.  We need to apply a transform from the
        # original data into this rectangle.  The first problem is finding a
        # generic way to get the bounding box of the data.  Here is a hack that
        # works with lines... this should be generalized
        if self.use_clip_box:
            boxin = matplotlib.transforms.Bbox(clip_box)
        else:
            boxin = self.get_extents(spark)
        boxout = matplotlib.transforms.Bbox.from_extents(
            xdescent, ydescent, width, height
        )
        _trans = matplotlib.transforms.BboxTransform(boxin=boxin, boxout=boxout)

        self.update_prop(spark, orig_handle, legend)
        spark.set_clip_box(
            matplotlib.transforms.TransformedBbox(
                matplotlib.transforms.Bbox(clip_box), _trans + trans
            )
        )
        spark.set_transform(_trans + trans)
        spark.set_clip_on(True)
        return [spark]


class Line2DHandler(SparklineHandler):
    def get_extents(self, spark):
        _d = spark.get_xydata()
        xy_min = _d.min(axis=0)
        xy_max = _d.max(axis=0)
        return matplotlib.transforms.Bbox([xy_min, xy_max])


class PolyCollectionHandler(SparklineHandler):
    def get_extents(self, spark):
        return plt.matplotlib.transforms.Bbox.union(
            [_p.get_extents() for _p in spark.get_paths()]
        )


spark_handler_map = {
    matplotlib.collections.PolyCollection: PolyCollectionHandler(use_clip_box=True),
    matplotlib.lines.Line2D: Line2DHandler(use_clip_box=True),
}


def cmap_subset(c, min=0, max=1):
    r"""Return a new colormap that is a subset of the colormap `c`"""
    assert min >= 0
    assert max <= 1
    assert min < max
    name = "%s_%g_%g" % (c.name, min, max)

    spec = dict(
        red=np.vectorize(lambda x: c(x * (max - min) + min)[0]),
        green=np.vectorize(lambda x: c(x * (max - min) + min)[1]),
        blue=np.vectorize(lambda x: c(x * (max - min) + min)[2]),
        alpha=np.vectorize(lambda x: c(x * (max - min) + min)[3]),
    )
    return matplotlib.colors.LinearSegmentedColormap(name, spec)


# Patch TexManager for a couple of issues.
# https://github.com/matplotlib/matplotlib/issues/9118
#
# We redefine the TexManager class so we have control over the LaTeX
# preamble.  As described in issue 9118, there are issues using
# newtxtext because of option clashes.
if not hasattr(matplotlib.texmanager.TexManager, "_my_sentinal"):
    _TexManager = matplotlib.texmanager.TexManager


class TexManager(_TexManager):
    _my_sentinal = True

    def __init__(self, *v, **kw):
        _TexManager.__init__(self, *v, **kw)

    # The following two functions set the latex template.  We overload
    # them using the code from version 3.0.2 and put custom_preamble first.
    def make_tex(self, tex, fontsize):
        """
        Generate a tex file to render the tex string at a specific font size.

        Return the file name.
        """
        basefile = self.get_basefile(tex, fontsize)
        texfile = "%s.tex" % basefile
        custom_preamble = self.get_custom_preamble()
        fontcmd = {"sans-serif": r"{\sffamily %s}", "monospace": r"{\ttfamily %s}"}.get(
            self.font_family, r"{\rmfamily %s}"
        )
        tex = fontcmd % tex

        s = r"""
\documentclass{article}
%s
%s
\usepackage[utf8]{inputenc}
\usepackage[papersize={72in,72in},body={70in,70in},margin={1in,1in}]{geometry}
\pagestyle{empty}
\begin{document}
\fontsize{%f}{%f}%s
\end{document}
""" % (
            custom_preamble,
            self._font_preamble,
            fontsize,
            fontsize * 1.25,
            tex,
        )
        with open(texfile, "wb") as fh:
            fh.write(s.encode("utf8"))
        return texfile

    def make_tex_preview(self, tex, fontsize):
        """
        Generate a tex file to render the tex string at a specific font size.

        It uses the preview.sty to determine the dimension (width, height,
        descent) of the output.

        Return the file name.
        """
        basefile = self.get_basefile(tex, fontsize)
        texfile = "%s.tex" % basefile
        custom_preamble = self.get_custom_preamble()
        fontcmd = {"sans-serif": r"{\sffamily %s}", "monospace": r"{\ttfamily %s}"}.get(
            self.font_family, r"{\rmfamily %s}"
        )
        tex = fontcmd % tex

        # newbox, setbox, immediate, etc. are used to find the box
        # extent of the rendered text.

        s = r"""
\documentclass{article}
%s
%s
\usepackage[utf8]{inputenc}
\usepackage[active,showbox,tightpage]{preview}
\usepackage[papersize={72in,72in},body={70in,70in},margin={1in,1in}]{geometry}

%% we override the default showbox as it is treated as an error and makes
%% the exit status not zero
\def\showbox#1%%
{\immediate\write16{MatplotlibBox:(\the\ht#1+\the\dp#1)x\the\wd#1}}

\begin{document}
\begin{preview}
{\fontsize{%f}{%f}%s}
\end{preview}
\end{document}
""" % (
            custom_preamble,
            self._font_preamble,
            fontsize,
            fontsize * 1.25,
            tex,
        )
        with open(texfile, "wb") as fh:
            fh.write(s.encode("utf8"))
        return texfile


## matplotlib.texmanager.TexManager = TexManager


# Patch TexManager
## matplotlib.texmanager.TexManager.font_info.update(LaTeXPlotProperties.font_info)
