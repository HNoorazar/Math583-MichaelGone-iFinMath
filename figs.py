"""Functions to draw figures for paper.
"""
from collections import namedtuple, OrderedDict
from functools import partial
import itertools

import numpy as np
import scipy as sp
from scipy.stats.mstats import gmean

from matplotlib import pyplot as plt
import matplotlib.path
import matplotlib.patches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import ticker

from mmfutils import plot as mmfplt

from figure_style import Paper

from gpe.plot_utils import MPLGrid
from hypersonic import expt_data, u

u.microK = u.nK * 1000


class TestFigures(Paper):
    style = "nature"
    # style = 'arXiv'
    style = "none"

    def fig_columnwidth(self):
        myfig = self.figure(
            width="columnwidth",
        )

        x = np.linspace(0, 1, 100)

        V0 = 1.234
        label = r"\textbf{a}"
        ax = myfig.ax
        ax.plot(x, x ** 2, label=r"$x^2$ [\SI{10}{μm}]")
        ax.plot(x, x ** 3 - x, label="$x^3-x$ [μm]")
        ax.set(
            xlabel="x$x$ viverra Vivamus",
            ylabel="y",
            title=fr"{label} $U/k_B=\SI{{{V0:.2f}}}{{\micro K}}$",
        )
        self._l = ax.legend()
        return myfig

    def fig_textwidth(self):
        myfig = self.figure(
            width="textwidth",
        )

        x = np.linspace(0, 1, 100)
        ax = myfig.ax
        ax.plot(x, x ** 2, label="$x^2$ [micron]")
        ax.plot(x, x ** 3 - x, label="$x^3-x$ [micron]")
        ax.set(xlabel="x", ylabel="y")
        ax.legend()


class PaperFigures(Paper):
    style = "nature"
    # style = 'arXiv'
    style = "none"

    def fig_34(
        self,
        data=None,
        cmaps=["gray", "gray"],
        c0="C0",
        n_percentiles=(0, 99.95),
        n_power=0.5,
        draft=False,
    ):
        # Parameters customizing physical display of data.
        Nx0 = 100
        x0_max = 20 * u.micron

        # Allow different xlims
        xlims = [
            [(-120, 120), (-250, 250), (-250, 250), (-250, 250)],
            [(-120, 120), (-250, 250), (-250, 250), (-250, 250)],
        ]

        keys = [
            [("fig3", 0), ("fig3", 1), ("fig3", 2), ("fig3", 3)],
            [("fig4", 0), ("fig4", 1), ("fig4", 2), ("fig4", 3)],
        ]
        theory_row = [False, False]

        plt.rcParams["grid.alpha"] = 0.05

        ruler_color = "w"

        # Assume all ylims are the same.
        ylims = [
            (-600, 20),
            (-600, 20),
        ]

        # Combined full-width figure with data.  These data will consist of 2x4 frames
        # which will be presented without spaces at an aspect ratio of 1:1.  We will
        # calculate the rectangle needed for this, manually taking into account the
        # space needed for margins.

        dxs = [[np.diff(_lim)[0] for _lim in _xlims] for _xlims in xlims]
        dys = [np.diff(_lim)[0] for _lim in ylims]
        data_width = max(sum(_dxs) for _dxs in dxs)
        _dxs = list(sum(_dxs) for _dxs in dxs)
        if not np.allclose(_dxs[0], _dxs):
            print(
                "Warning: xlims not the same - different panels will have different scales"
                + f": {_dxs}"
            )
        data_height = sum(dys)

        # Margins and spacing
        left, right = 0.005, 0.005
        top, bottom = 0.03, 0.05
        hspace, wspace = 0.0, 0
        cax_frac = 0.1

        # The width ratios should be [1 - cax_frac, cax_frac] which is equivalent to
        # [1, cax_frac/(1-cax_frac)] =  [1, cax_factor]
        cax_factor = 1 / (1 / cax_frac - 1)

        # Optional - put labels in between rows
        bottom = 0.005
        hspace = 0.07

        # Calculate figure height, width, etc.
        Ncols, Nrow = list(map(len, keys)), len(ylims)
        width = data_width * (1 + cax_frac) / (1 - left - right - wspace * (max(Ncols)))
        height = data_height / (1 - top - bottom - hspace * (Nrow - 1))

        myfig = self.figure(
            num=0,  # If you want to redraw in the same window
            width="textwidth",  # For two-column documents, vs. 'textwidth'
            height=height / width,  # Fraction of width
            constrained_layout=False,
            margin_factors=dict(top=0, left=0, bot=0, right=0),
        )
        myfig.ax.remove()  # Don't use base axis.

        use_gridspec = True
        if use_gridspec:
            gs_rows = GridSpec(
                Nrow,
                1,
                figure=myfig.fig,
                height_ratios=dys,
                left=left,
                right=1 - right,
                bottom=bottom,
                top=1 - top,
                hspace=0,
                wspace=wspace,
            )
            gs = [
                GridSpecFromSubplotSpec(
                    1,
                    Ncols[_ir] + 1,  # Extra space for colorbars
                    subplot_spec=gs_rows[_ir],
                    width_ratios=dxs[_ir] + [data_width * cax_factor],
                    hspace=hspace,
                    wspace=0,
                )
                for _ir in range(Nrow)
            ]

            # The color-bar will go in the right panel
            gs_cax = GridSpecFromSubplotSpec(
                1,
                2,
                width_ratios=[1, cax_factor],
                subplot_spec=gs_rows[:2],
            )[0, 1]

            # Prepare axes.
            axs = []
            for ir in range(Nrow):
                axs.append([])
                for ic in range(Ncols[ir]):
                    ax = myfig.fig.add_subplot(gs[ir][ic])
                    axs[ir].append(ax)
        else:
            grid0 = MPLGrid(direction=down, space=hspace)
            row_grids = [
                grid0.next(dys[ir], direction="right", space=wspace)
                for ir in range(Nrow)
            ]
            axs = []
            for ir in range(Nrow):
                axs.append([])
                grid = row_grid[ir]
                for ic in range(Ncols[ir]):
                    axs[-1].append(grid.next(dxs[ir]))

        # Indices to iterate over rows and columns in plot
        ircs = [
            (ir, ic)
            for ir in range(Nrow)
            for ic in range(Ncols[ir])
            if keys[ir][ic] is not None
        ]

        for ir, ic in ircs:
            ax = axs[ir][ic]
            xlim, ylim = xlims[ir][ic], ylims[ir]
            ax.set(xlim=xlim, ylim=ylim, aspect=1)

            tick_spacing = 100
            # Each axis needs its own locator: https://stackoverflow.com/a/24744337
            ax.xaxis.set_major_locator(ticker.MultipleLocator(base=tick_spacing))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(base=tick_spacing))

            plt.setp(
                ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels(), visible=False
            )

            # ax.spines["top"].set_visible(False)  # ax.is_first_row())
            # ax.spines["bottom"].set_visible(False)  # ax.is_last_row())
            # ax.spines["left"].set_visible(False)  # ax.is_first_col())
            # ax.spines["right"].set_visible(False)  # ax.is_last_col())

        cax = myfig.fig.add_subplot(gs_cax)
        cax.set_visible(False)
        aspect = height * cax_frac
        plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmaps[0]),
            aspect=aspect / 2,
            ax=cax,
            shrink=0.9,
            fraction=0.9,
            extend="both",
            pad=0.1,
            label=r"$\sqrt{\textsf{optical depth}}$ (arbitrary units)"
            if n_power == 0.5
            else rf"$\textsf{{optical depth}}^{n_power:.2f}$ (arbitrary units)",
        )

        # Add ruler for length in specified subplots
        rulers = [(-1, -1)]
        for (ir, ic) in rulers:
            ax = axs[ir][ic]
            xlim, ylim = xlims[ir][ic], ylims[ir]
            x0, y0 = 0.15 * xlim[0] + 0.85 * xlim[-1], 0.15 * ylim[0] + 0.85 * ylim[-1]
            ax.plot(
                [x0 + 50, x0],
                [y0, y0],
                "-k",
                lw=2.0,
                solid_capstyle="butt",
                c=ruler_color,
            )
            ax.text(
                x0 + 25,
                y0 + 10,
                "50 μm",
                ha="center",
                va="bottom",
                fontsize=7.0,
                c=ruler_color,
            )

        # Compute the data.
        Data = namedtuple("Data", ["d", "xyn_ex"])
        if data is None:
            data = []

            for ir in range(Nrow):
                data.append([])
                for ic in range(Ncols[ir]):
                    if keys[ir][ic] is None:
                        continue
                    key1, key2 = keys[ir][ic]
                    d = expt_data.DataMaren(key1=key1, key2=key2)
                    xyn_ex = d.load()
                    data[ir].append(Data(d=d, xyn_ex=xyn_ex))

        self._data34 = data

        # Add lables a, b, c, etc. with U/k_B values:
        labels = [[rf"\textbf{{{_l}}}" for _l in _lrow] for _lrow in ["abcd", "efgh"]]
        for ir in range(Nrow):
            for ic in range(Ncols[ir]):
                if keys[ir][ic] is None:
                    continue
                ax = axs[ir][ic]
                d = data[ir][ic].d
                V0 = d.experiment._Vs[0].V0
                KE = -d.experiment.m * d.experiment.V_accel * d.experiment.V_z0
                epsilon = V0 / KE

                ax.text(
                    0.0,
                    1.03 if (ir == 0 or hspace > 0) else -0.04,
                    labels[ir][ic],
                    transform=ax.transAxes,
                    ha="left",
                    va="baseline",
                )
                ax.text(
                    0.5,
                    1.03 if (ir == 0 or hspace > 0) else -0.04,
                    rf"$\varepsilon=\num{{{epsilon:.2f}}}$",
                    transform=ax.transAxes,
                    fontsize=7.0,
                    ha="center",
                    va="baseline",
                )

        if draft:
            # Short circuit if we are are adjusting layout.
            return myfig

        ns = []
        for _data in data:
            for (d, (x, y, n)) in _data:
                ns.extend(n.ravel())
        self._ns = np.array(ns)
        nps = np.maximum(ns, 0) ** n_power
        np_min = 0
        np_max = np.percentile(nps, n_percentiles[1])

        for ir in range(Nrow):
            for ic in range(Ncols[ir]):
                if keys[ir][ic] is None:
                    continue
                if draft and (ir, ic) != (0, 3):
                    continue
                (d, (x, y, n)), ax = data[ir][ic], axs[ir][ic]
                xlim, ylim = xlims[ir][ic], ylims[ir]
                plt.sca(ax)

                # Experimental Data
                V0 = d.experiment._Vs[0].V0
                np_ = np.maximum(n, 0) ** n_power
                mmfplt.imcontourf(
                    y,
                    x,
                    np_.T / np_max,
                    cmap=cmaps[ir],
                    vmin=np_min / np_max,
                    vmax=np_max / np_max,
                    aspect=1,
                )

                ax.set(xlim=xlim, ylim=ylim, aspect=1)

        return data

    def fig_34_(
        self,
        data=None,
        cmaps=["gray", "gray", "gray_r"],
        c0="C0",
        n_percentiles=(0, 99.95),
        n_power=0.5,
        draft=False,
    ):
        # Parameters customizing physical display of data.
        Nx0 = 100
        x0_max = 20 * u.micron

        # Allow different xlims
        xlims = [
            [(-120, 120), (-220, 220), (-220, 220), (-220, 220)],
            [(-120, 120), (-220, 220), (-220, 220), (-220, 220)],
            [(-220, 220), (-220, 220), (-220, 220), (0, 240)],
        ]

        keys = [
            [("fig3", 0), ("fig3", 1), ("fig3", 2), ("fig3", 3)],
            [("fig4", 0), ("fig4", 1), ("fig4", 2), ("fig4", 3)],
            [("fig4", 2), ("fig3", 2), ("fig3", 3), None],
        ]
        theory_row = [False, False, True]

        plt.rcParams["grid.alpha"] = 0.05

        ruler_color = "k"

        # Assume all ylims are the same.
        ylims = [
            (-600, 20),
            (-600, 20),
            (-600, 20),
        ]

        # Combined full-width figure with data.  These data will consist of 2x4 frames
        # which will be presented without spaces at an aspect ratio of 1:1.  We will
        # calculate the rectangle needed for this, manually taking into account the
        # space needed for margins.

        dxs = [[np.diff(_lim)[0] for _lim in _xlims] for _xlims in xlims]
        dys = [np.diff(_lim)[0] for _lim in ylims]
        data_width = max(sum(_dxs) for _dxs in dxs)
        _dxs = list(sum(_dxs) for _dxs in dxs)
        if not np.allclose(_dxs[0], _dxs):
            print(
                "Warning: xlims not the same - different panels will have different scales"
                + f": {_dxs}"
            )
        data_height = sum(dys)

        # Margins and spacing
        left, right = 0.005, 0.005
        top, bottom = 0.03, 0.05
        hspace, wspace = 0.0, 0
        cax_frac = 0.1

        # The width ratios should be [1 - cax_frac, cax_frac] which is equivalent to
        # [1, cax_frac/(1-cax_frac)] =  [1, cax_factor]
        cax_factor = 1 / (1 / cax_frac - 1)

        # Optional - put labels in between rows
        bottom = 0.005
        hspace = 0.07

        # Calculate figure height, width, etc.
        Ncols, Nrow = list(map(len, keys)), len(ylims)
        width = data_width * (1 + cax_frac) / (1 - left - right - wspace * (max(Ncols)))
        height = data_height / (1 - top - bottom - hspace * (Nrow - 1))

        myfig = self.figure(
            num=0,  # If you want to redraw in the same window
            width="textwidth",  # For two-column documents, vs. 'textwidth'
            height=height / width,  # Fraction of width
            constrained_layout=False,
            margin_factors=dict(top=0, left=0, bot=0, right=0),
        )
        myfig.ax.remove()  # Don't use base axis.

        use_gridspec = True
        if use_gridspec:
            gs_rows = GridSpec(
                Nrow,
                1,
                figure=myfig.fig,
                height_ratios=dys,
                left=left,
                right=1 - right,
                bottom=bottom,
                top=1 - top,
                hspace=0,
                wspace=wspace,
            )
            gs = [
                GridSpecFromSubplotSpec(
                    1,
                    Ncols[_ir] + 1,  # Extra space for colorbars
                    subplot_spec=gs_rows[_ir],
                    width_ratios=dxs[_ir] + [data_width * cax_factor],
                    hspace=hspace,
                    wspace=0,
                )
                for _ir in range(Nrow)
            ]

            # The color-bar will go in the right margin
            gs_cax1 = GridSpecFromSubplotSpec(
                1,
                2,
                width_ratios=[1, cax_factor],
                subplot_spec=gs_rows[:2],
            )[0, 1]

            gs_cax2 = GridSpecFromSubplotSpec(
                1,
                2,
                width_ratios=[1, cax_factor],
                subplot_spec=gs_rows[2:],
            )[0, 1]

            # Prepare axes.
            axs = []
            for ir in range(Nrow):
                axs.append([])
                for ic in range(Ncols[ir]):
                    ax = myfig.fig.add_subplot(gs[ir][ic])
                    axs[ir].append(ax)
        else:
            grid0 = MPLGrid(direction=down, space=hspace)
            row_grids = [
                grid0.next(dys[ir], direction="right", space=wspace)
                for ir in range(Nrow)
            ]
            axs = []
            for ir in range(Nrow):
                axs.append([])
                grid = row_grid[ir]
                for ic in range(Ncols[ir]):
                    axs[-1].append(grid.next(dxs[ir]))

        # Indices to iterate over rows and columns in plot
        ircs = [
            (ir, ic)
            for ir in range(Nrow)
            for ic in range(Ncols[ir])
            if keys[ir][ic] is not None
        ]

        for ir, ic in ircs:
            ax = axs[ir][ic]
            xlim, ylim = xlims[ir][ic], ylims[ir]
            ax.set(xlim=xlim, ylim=ylim, aspect=1)

            tick_spacing = 100
            # Each axis needs its own locator: https://stackoverflow.com/a/24744337
            ax.xaxis.set_major_locator(ticker.MultipleLocator(base=tick_spacing))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(base=tick_spacing))

            plt.setp(
                ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels(), visible=False
            )

            # ax.spines["top"].set_visible(False)  # ax.is_first_row())
            # ax.spines["bottom"].set_visible(False)  # ax.is_last_row())
            # ax.spines["left"].set_visible(False)  # ax.is_first_col())
            # ax.spines["right"].set_visible(False)  # ax.is_last_col())

        cax = myfig.fig.add_subplot(gs_cax1)
        cax.set_visible(False)
        aspect = height / (right * width)
        plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmaps[0]),
            aspect=4 * aspect,
            ax=cax,
            shrink=0.9,
            fraction=0.9,
            extend="max",
            pad=0.1,
            label=r"$\sqrt{\textsf{optical depth}}$ (arbitrary units)"
            if n_power == 0.5
            else rf"$n^{n_power:.2f}$ (arbitrary units)",
        )

        # Add ruler for length in specified subplots
        rulers = [(-1, -1)]
        for (ir, ic) in rulers:
            ax = axs[ir][ic]
            xlim, ylim = xlims[ir][ic], ylims[ir]
            x0, y0 = 0.15 * xlim[0] + 0.85 * xlim[-1], 0.15 * ylim[0] + 0.85 * ylim[-1]
            ax.plot(
                [x0 + 50, x0], [y0, y0], "-k", lw=4.0, solid_capstyle="butt", c=ruler_c
            )
            ax.text(
                x0 + 25,
                y0 + 10,
                "50 μm",
                ha="center",
                va="bottom",
                fontsize=7.0,
                c=ruler_c,
            )

        # Compute the data.
        Data = namedtuple("Data", ["d", "xyn_ex", "zv_th"])
        if data is None:
            data = []

            for ir in range(Nrow):
                data.append([])
                for ic in range(Ncols[ir]):
                    if keys[ir][ic] is None:
                        continue
                    key1, key2 = keys[ir][ic]
                    d = expt_data.DataMaren(key1=key1, key2=key2)
                    xyn_ex = d.load()
                    zv_th = None
                    if theory_row[ir]:
                        z0s = 0.0 + np.linspace(0, x0_max, Nx0) * 1j
                        zv_th = d.wkb.get_classical_sol(z0=z0s)[:2]
                    else:
                        zv_th = (None, None)
                    data[ir].append(Data(d=d, xyn_ex=xyn_ex, zv_th=zv_th))

        self._data34 = data

        # Add lables a, b, c, etc. with U/k_B values:
        labels = [
            [rf"\textbf{{{_l}}}" for _l in _lrow] for _lrow in ["abcd", "efgh", "hijk"]
        ]
        for ir in range(Nrow):
            for ic in range(Ncols[ir]):
                if keys[ir][ic] is None:
                    continue
                ax = axs[ir][ic]
                d = data[ir][ic].d
                V0 = d.experiment._Vs[0].V0
                KE = -d.experiment.m * d.experiment.V_accel * d.experiment.V_z0

                ax.text(
                    0.0,
                    1.03 if (ir == 0 or hspace > 0) else -0.04,
                    labels[ir][ic],
                    transform=ax.transAxes,
                    ha="left",
                    va="baseline",
                )
                ax.text(
                    0.5,
                    1.03 if (ir == 0 or hspace > 0) else -0.04,
                    rf"$\varepsilon=\num{{{V0/KE:.2f}}}$",
                    transform=ax.transAxes,
                    fontsize=7.0,
                    ha="center",
                    va="baseline",
                )

        if draft:
            # Short circuit if we are are adjusting layout.
            return myfig

        ns = []
        for _data in data:
            for (d, (x, y, n), (z, v)) in _data:
                ns.extend(n.ravel())
        self._ns = np.array(ns)
        nps = np.maximum(ns, 0) ** n_power
        np_min = np.percentile(nps, n_percentiles[0])
        np_max = np.percentile(nps, n_percentiles[1])

        for ir in range(Nrow):
            for ic in range(Ncols[ir]):
                if keys[ir][ic] is None:
                    continue
                if draft and (ir, ic) != (0, 3):
                    continue
                (d, (x, y, n), (z, v)), ax = data[ir][ic], axs[ir][ic]
                xlim, ylim = xlims[ir][ic], ylims[ir]
                plt.sca(ax)

                # Experimental Data
                V0 = d.experiment._Vs[0].V0
                np_ = np.maximum(n, 0) ** n_power
                mmfplt.imcontourf(
                    y,
                    x,
                    np_.T / np_max,
                    cmap=cmaps[ir],
                    vmin=np_min / np_max,
                    vmax=1,
                    aspect=1,
                )

                if theory_row[ir]:
                    # Trajectories
                    lines = ax.plot(z.imag, z.real, c=c0, lw=0.1, alpha=0.5)

                ax.set(xlim=xlim, ylim=ylim, aspect=1)

        return data

    def fig_3(self, data=None, cmap="bone", n_fact=1.0):
        key1 = "fig3"
        Ncols = 4
        Nx0 = 100
        x0_max = 20 * u.micron

        xlims = [(-200, 200), (-200, 200), (-200, 200), (-200, 200)]
        xlims = [(-120, 120), (-140, 140), (-220, 220), (-320, 320)]
        ylim = (-600, 20)

        myfig = self.figure(
            num=3,  # If you want to redraw in the same window
            width="textwidth",  # For two-column documents, vs. 'textwidth'
            height=0.62,
            constrained_layout=False,
            margin_factors=dict(top=-0.5, left=4, bot=3.1, right=-0.5),
        )

        Data = namedtuple("Data", ["d", "xyn_ex", "zv_th"])

        if data is None:
            data = []
            for key2 in range(Ncols):
                d = expt_data.DataMaren(key1=key1, key2=key2)
                xyn_ex = d.load()
                z0s = 0.0 + np.linspace(0, x0_max, Nx0) * 1j
                zv_th = d.wkb.get_classical_sol(z0=z0s)[:2]
                data.append(Data(d=d, xyn_ex=xyn_ex, zv_th=zv_th))

        self._data3 = data

        myfig.ax.remove()  # Don't use base axis.
        master_grid = MPLGrid(
            fig=myfig.fig,
            subplot_spec=myfig.subplot_spec,
            direction="right",
            space=0.05,
            share=False,
        )
        grid = master_grid.grid(1, direction="right", share=True, space=0.02)
        cax = master_grid.next(0.02)

        axs = []
        n_max = max(d.xyn_ex[-1].max() for d in data)

        for i, (d, (x, y, n), (z, v)) in enumerate(data):
            label = rf"\textbf{{{'abcd'[i]})}}"
            ax = grid.next(np.diff(xlims[i]))
            axs.append(ax)
            ax.grid(False)

            # Experimental Data
            V0 = d.experiment._Vs[0].V0
            mmfplt.imcontourf(
                y, x, n_fact * n.T / n_max, cmap=cmap, vmin=0, vmax=1, aspect=1
            )

            ax.set(
                xlim=xlims[i],
                ylim=ylim,
                # xticks=[-100, 0, 100],
                # title=fr"\textbf{{{'abcd'[i]})}} $U/k_B={V0/u.microK:.2f}\;\mu$K",
                title=fr"{label} $U/k_B=\SI{{{V0/u.microK:.2f}}}{{\micro K}}$",
                xlabel="$x$ ($\mu$m)",
            )

            # Trajectories
            lines = ax.plot(z.imag, z.real, c="w", lw=0.1, alpha=0.5)

        plt.colorbar(cax=cax, pad=0.1, label=r"Density (arbitrary units)")
        axs[0].set(ylabel="$z$ ($\mu$m)")

        # If you need to see the background for adjusting margins etc.
        # myfig.fig.set_facecolor([0.9, 0.9, 0.9])
        return data

    def fig_4(self, data=None, cmap="bone", n_fact=1.0):
        key1 = "fig4"
        Ncols = 4
        Nx0 = 100
        x0_max = 30 * u.micron

        xlims = [(-120, 120), (-140, 140), (-220, 220), (-320, 320)]
        ylim = (-600, 20)

        myfig = self.figure(
            num=4,  # If you want to redraw in the same window
            width="textwidth",  # For two-column documents, vs. 'textwidth'
            height=0.6,
            constrained_layout=False,
            margin_factors=dict(top=-0.5, left=4, bot=3.1, right=-0.5),
        )

        Data = namedtuple("Data", ["d", "xyn_ex", "zv_th"])

        if data is None:
            data = []
            for key2 in range(Ncols):
                d = expt_data.DataMaren(key1=key1, key2=key2)
                xyn_ex = d.load()
                z0s = 0.0 + np.linspace(0, x0_max, Nx0) * 1j
                zv_th = d.wkb.get_classical_sol(z0=z0s)[:2]
                data.append(Data(d=d, xyn_ex=xyn_ex, zv_th=zv_th))

        self._data4 = data

        myfig.ax.remove()  # Don't use base axis.
        master_grid = MPLGrid(
            fig=myfig.fig,
            subplot_spec=myfig.subplot_spec,
            direction="right",
            space=0.1,
            share=False,
        )
        grid = master_grid.grid(1, direction="right", share=True, space=0.1)
        cax = master_grid.next(0.02)

        axs = []
        n_max = max(d.xyn_ex[-1].max() for d in data)

        for i, (d, (x, y, n), (z, v)) in enumerate(data):
            label = rf"\textbf{{{'abcd'[i]})}}"

            ax = grid.next(np.diff(xlims[i]))
            axs.append(ax)

            # Experimental Data
            V0 = d.experiment._Vs[0].V0
            mmfplt.imcontourf(
                y, x, n_fact * n.T / n_max, cmap=cmap, vmin=0, vmax=1, aspect=1
            )
            ax.set(
                xlim=xlims[i],
                ylim=ylim,  # xticks=[-100, 0, 100]
                title=fr"{label} $U/k_B=\SI{{{V0/u.microK:.2f}}}{{\micro K}}$",
                xlabel="$x$ ($\mu$m)",
            )
            ax.grid(False)

            # Trajectories
            lines = ax.plot(z.imag, z.real, c="w", lw=0.1, alpha=0.5)

        plt.colorbar(cax=cax, pad=0.1, label=r"Density (arbitrary units)")
        axs[0].set(ylabel="[microns]")
        return data


'''
        
    
    def get_experiment(self,
                       tube=False,
                       axial=True,
                       single_band=False,
                       small=False,
                       **kw):
        """"Return an experiment to get parameters."""
        if small:
            Experiment = ExperimentCatchAndReleaseSmall
        else:
            Experiment = ExperimentCatchAndRelease

        args = dict(
            tube=tube,
            basis_type='axial' if axial else '1D',
            single_band=single_band,
            barrier_depth_nK=-90,
            dx=0.05*u.micron,
            # IPG parameters
            x_TF=150*u.micron,
            trapping_frequencies_Hz=(3.49, 278, 278),
            species=((1, 0), (1, 0)),
            B_gradient_mG_cm=0.0,
            detuning_kHz=2.0,
            rabi_frequency_E_R=1.5,
            cells_x=200)

        args.update(kw)
        
        return Experiment(**args)
        
    def fig_DSW(self, data_dsw=None,
                tube=True,
                axial=False,
                single_band=True):
        """Draw the DSW figure."""

        myfig = self.figure(
            num=1,                # If you want to redraw in the same window
            width='columnwidth',  # For two-column documents, vs. 'textwidth'
            height=1.3,
            margin_factors = dict(
                top=-2.9,
                left=3.9,
                bot=3.1,
                right=-3.4)
            )

        if axial:
            tube = True
        if data_dsw is None:
            e = self.get_experiment(
                tube=tube, axial=axial, single_band=single_band, small=True)
            if False:
                s0 = s1 = s2 = s3 = s4 = s5 = e.get_state()
            else:
                s0 = e.get_initial_state()
                s1 = evolve_to(s0, t=1*e.t_unit)
                s2 = evolve_to(s1, t=2*e.t_unit)
                s3 = evolve_to(s2, t=3*e.t_unit)
                s4 = evolve_to(s3, t=4*e.t_unit)
                s5 = evolve_to(s4, t=5*e.t_unit)
            states = [s0, s1, s2, s3, s4, s5]
            data_dsw = dict(e=e, states=states)

        self._data_dsw = data_dsw
        e = data_dsw['e']
        s0, s1, s2, s3, s4, s5 = states = data_dsw['states']

        states = [s0, s1, s2, s3, s4]

        x = s0.xyz[0].ravel()
        x_unit = u.micron
        if tube:
            n_unit = 1./u.nm
        else:
            n_unit = 1./u.micron**3
            
        ns = [s.get_density_x().sum(axis=0) for s in states]
        n_min = 0*np.min(ns)
        n_max = 1.1*np.max(ns)

        myfig.ax.remove()       # Don't use base axis.
        grid = MPLGrid(fig=myfig.fig, subplot_spec=myfig.subplot_spec,
                       direction='down', space=0.2, share=True)

        if True:
            # Grid of separate evolution.
            for _n, s in enumerate(states):
                n = s.get_density_x().sum(axis=0)
                n_max = 1.1*n.max()
                dn = n_max - n_min
                ax = grid.next(dn)
                ax.plot(x/x_unit, n/n_unit)
                ax.set_yticks([0, 100, 200, 300, 400])
                ax.set_ylim(n_min/n_unit, n_max/n_unit)
                ax.text(24, 10,
                        rf'$t_{{\mathrm{{wait}}}}=\SI{{{s.t/e.t_unit:.0f}}}{{ms}}$',
                        fontsize='small',
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        # transform=ax.transAxes,
                )
                if _n == len(states)//2:
                    # Put the y-label on the middle axis.
                    ax_mid = ax
        else:
            # Single plot with graduated evolved states.
            ax = ax_mid = grid.next()
            
            alpha0 = 0.25
            for _n, s in enumerate(states):
                n = s.get_density_x().sum(axis=0)
                alpha = alpha0 + (1-alpha0)*(_n/(len(states)-1))
                label = None
                if _n in set([0, len(states)-1]):
                    label = r'$t=\SI{{{:.0f}}}{{ms}}$'.format(s.t/e.t_unit)
                ax.plot(x/x_unit, n/n_unit, c='C0', alpha=alpha,
                        label=label)
                
            plt.legend()
            if tube:
                pass
            else:
                plt.yticks([100, 200, 300, 400])
                
        # Put labels on the bottom axis only
        plt.setp(ax.yaxis.get_ticklabels(), visible=True)
        plt.xlim(-25, 25)
        plt.xlabel(r'$x$ [$\si{\micro\meter}$]')
        if tube:
            ylabel = r'$n_{1D} = \int n(\vec{r})d{y}d{z}$\quad [$\si{nm^{-1}}$]'
        else:
            ylabel = r'total density $n$\quad [$\si{\micro\meter^{-3}}$]'
        ax_mid.set_ylabel(ylabel)
                    
    def fig_DSW_1D(self, **kw):
        return self.fig_DSW(tube=False, axial=False, **kw)

    def fig_DSW_tube(self, **kw):
        return self.fig_DSW(tube=True, axial=False, **kw)
    
    def fig_DSW_axial(self, **kw):
        return self.fig_DSW(tube=True, axial=True, **kw)

    def fig_phonon_dispersion(self):
        """Draw the phonon dispersion."""

        fig = self.figure(
            num=1,                # If you want to redraw in the same window
            width='columnwidth',  # For two-column documents, vs. 'textwidth'
            height=1.0,
        )

        e = self.get_experiment()
        s = e.get_state()

        m = s.ms.mean()
        
        # Plot with phonon dispersion for several Omegas.
        k_R = e.k_r
        E_R = e.E_R
        ks_ = np.linspace(-2, 1.5, 100)
        ks = ks_ * k_R
        qs = u.hbar * ks

        E_ = e.get_dispersion()
        k0 = E_.get_k0()
        E0_ = E_(k0)[0]
        p0 = u.hbar * k0 * k_R
        g = gmean(e.gs)
        n = e.fiducial_V_TF/g

        phonon = PhononDispersion(
            gs=e.gs, ms=e.ms, hbar=u.hbar, k_R=k_R,
            Omega=e.Omega, delta=e.delta)

        @np.vectorize
        def Eph(q, _Eph=phonon.get_phonon_dispersion(p=p0, n=n)):
            """Return the lowest phonon branch (vectorized)."""
            Es_ = _Eph(q)
            return min(Es_[np.where(Es_ > 0)])

        @np.vectorize
        def get_na_n(q):
            mu, ns = phonon.get_homogeneous_state(p=q, n=n)
            return ns[0]/n

        Es_phonon = Eph(qs-p0)
        Es_particle = (E_(ks_)[0] - E0_)*2*E_R
        
        x, y = ks_, Es_particle/2/E_R

        color_according_to_species = False

        ax = plt.gca()
        
        if color_according_to_species:        
            na_n = get_na_n(qs)
            nb_n = 1 - na_n
            ca = np.array([0, 0, 1, 1])[None, :]
            cb = np.array([1, 0, 0, 1])[None, :]
            colors = na_n[:, None]*ca + nb_n[:, None]*cb

            # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/
            #         multicolored_line.html
            from matplotlib.collections import LineCollection
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=colors, lw=3)
            #lc.set_array(na_n-nb_n)
            line = ax.add_collection(lc)
            #plt.colorbar(line)
            c0 = 'k'
        else:        
            l, = plt.plot(x, y, c='C0', alpha=0.5, lw=2,
                          ls='--', label='Single Particle')
            c0 = 'k'
            
        ix = np.where(E_(ks_, d=2)[0] < 0)[0]
        plt.fill_betweenx(x, x[ix[0]], x[ix[-1]],
                          alpha=0.3)
        
        plt.plot(ks_, Es_phonon/2/E_R, label="Phonon",
                 c=c0, ls='-')
            
        plt.title(r'$\Omega={:.2f}E_R$, $\delta={:.2f}E_R$'.format(
            e.Omega/e.E_R, e.delta/e.E_R),
                  fontsize='small')

        plt.xlim(-2, 1.5)
        plt.xticks([-2,-1,0,1])
        plt.ylim(0, 0.7)
        plt.xlabel(r'$q/\hbar k_R$')
        plt.ylabel(r'Energy ($2E_R$)')
        plt.legend()
        #plt.title("Phonon Dispersion")
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def fig_stages(self, data=None, stream=True,
                   y_tilt=0.01, column=False):
        """Figure showing the stages of evolution of the DSW.

        Arguments
        ---------
        data : None, dict
           Pre-computed data can be passed in.  Get this from
           self._data_stages.
        column : bool
           If True, then typeset the figure as a single column,
           otherwise go for full width.
        
        The grid is constructed as a 4x4 super-grid each with a 2x1
        grid of density on top and contours below.
        """
        t_s = [1.0, 5.0, 12.0, 10+14.0]
        t_s = [1.0, 5.0, 12.0, 10+10.0]

        xlim1 = (-12, 12)
        xlim2 = (-38, 38)
        ylim = (-3, 3)
        
        # Margins and padding - relative to full size
        left = 0.06
        right = 0.06
        wspace = 0.02

        top = 0.05
        bottom = 0.06
        hspace = 0.1

        # Height and width of various frames.
        w1, w2 = np.diff(xlim1)[0], np.diff(xlim2)[0]
        w3 = 0.1*w1
        h2 = h3 = np.diff(ylim)[0]
        h1 = 1*h2
        width =  w1 + w2 + w3
        height = 2*(h1 + h2 + h3)

        fig_width = width/(1 - left - right - wspace)
        fig_height = height/(1 - top - bottom - hspace)
        fig_aspect = fig_height/fig_width

        myfig = self.figure(
            num=3,              # If you want to redraw in the same window
            width='textwidth',  # For two-column documents, vs. 'textwidth'
            height=f"{fig_aspect}*width",
            #fig_box=0.1,        # Shows outline for placement of text
            constrained_layout=False,  # We manage our own margins
            subplot_args=dict(left=left, bottom=bottom, 
                              right=1.0-right, top=1.0-top,
                              hspace=hspace, wspace=wspace),
        )

        if data is None:
            import runs_car_fast_final
            #run = runs_car_fast_final.Run_IPG_small_xTF150()

            # Axial data
            run = runs_car_fast_final.Run_IPG_xTF150()
            sim = run.simulations[1]
            print(f"Loading data from {sim._dir_name}")
            states = [sim.get_state(t_=t_, image=False)
                      for t_ in t_s]

            # 3D data
            run = runs_car_fast_final.Run_IPG_small_3D_big()
            attrs = ['barrier_depth_nK']
            sims = [_sim for _sim in run.simulations
                    if (_sim.experiment.y_tilt == y_tilt
                        and all(
                            getattr(sim.experiment, _attr) ==
                            getattr(_sim.experiment, _attr)
                            for _attr in attrs))]
            assert len(sims) == 1
            sim3D = sims[0]
            print(f"Loading data from {sim._dir_name}")
            states3D = [sim3D.get_state(t_=t_, image=False)
                        for t_ in t_s]

            # Store for future use.
            data = dict(states=states, sim=sim,
                        states3D=states3D, sim3D=sim3D)


        states = list(zip(data['states'], data['states3D']))
        sim, sim3D = data['sim'], data['sim3D']
        self._data_stages = data

        myfig.ax.remove()       # Don't use base axis.
        # The right-most frame is the colorbar.
        gs0 = GridSpec(2, 3,
                       figure=myfig.fig,
                       height_ratios=[h1+h2+h3]*2,
                       width_ratios=[w1, w2, w3],
                       left=left, right=1-right,
                       bottom=bottom, top=1-top,
                       hspace=hspace,
                       wspace=wspace)
        gs0s =[gs0[0, 0], gs0[1, 0], gs0[0, 1], gs0[1, 1]]

        # Colorbar axis
        cax = plt.subplot(gs0[:, 2])
        #ax0 = (grid1.next(h1), grid1.next(h2))
        #ax1 = (grid1.next(h1), grid1.next(h2))
        #ax2 = (grid2.next(h1), grid2.next(h2))
        #ax3 = (grid2.next(h1), grid2.next(h2))

        ax0s = []
        for gs in gs0s:
            # Steal a little space between the figures without mucking
            # up the perfect alignment of the lowest frame which has
            # fixed aspect ratio.
            gs1 = GridSpecFromSubplotSpec(
                5, 1, gs,
                height_ratios=(0.8*h1, 0.1*h1, h2, 0.1*h1, h3),
                hspace=0.0, wspace=0.0)
            ax0s.append((plt.subplot(gs1[0, 0]),
                         plt.subplot(gs1[2, 0]),
                         plt.subplot(gs1[-1, 0])))

        n1_max = max(s.get_density_x(mean=False).sum(axis=0).max()
                     for (s, s3) in states)
        n_max = max(s.get_density().sum(axis=0).max() for (s, s3) in states)
        xlims = [xlim1, xlim1, xlim2, xlim2]
        for axs, xlim, (s, s3) in zip(ax0s, xlims, states):
            n1_unit = 1./u.nm
            n_unit = 100./u.micron**3
            x_unit = u.micron

            ax_a, ax_b, ax_c = axs
            psi = psi_a, psi_b = s.get_psi()
            n1_a, n1_b = s.get_density_x(mean=False)
            n1 = n1_a + n1_b
            n_a, n_b = s.get_density()
            n0 = n_a + n_b
            x, r0 = s.xyz

            # Compute the axial velocities for the streamline plot.
            # We first deal with the x-direction
            kx = s.basis.kx
            px = (
                psi.conj()*s.basis.ifft(s.hbar*kx*s.basis.fft(psi))
            ).sum(axis=0).real #/ n0
            pr = (-1j*s.hbar*(
                psi_a.conj()*np.gradient(
                    psi_a, x.ravel(), r0.ravel())[1]
                +
                psi_b.conj()*np.gradient(
                    psi_b, x.ravel(), r0.ravel())[1])
            ).real #/ n0

            r = np.hstack([-r0[:,::-1], r0])
            x_, r_ = x.ravel(), r.ravel()

            psi = np.hstack([psi_a[:,::-1], psi_a])
            n = np.hstack([n0[:,::-1], n0])
            px = np.hstack([px[:,::-1], px])
            pr = np.hstack([-pr[:,::-1], pr])

            r_new = np.linspace(0.9*ylim[0], 0.9*ylim[1],
                                len(r.ravel()))[1:-1][None, :]
            px = sp.interpolate.RegularGridInterpolator(
                (x.ravel(), r.ravel()), px)((x, r_new))
            pr = sp.interpolate.RegularGridInterpolator(
                (x.ravel(), r.ravel()), pr)((x, r_new))

            # Plot 1D integrated densities
            n1 = sum(s.get_density_x(mean=False))
            n13 = sum(s3.get_density_x(mean=False))
            x = s.xyz[0].ravel()
            x3 = s3.xyz[0].ravel()

            sim_axial_color = '#377eb8'
            sim_3d_color = '#ff7f00'
            ax = ax_a
            plt.sca(ax)
            plt.plot(x/x_unit, n1/n1_unit, c=sim_axial_color, ls='--',
                     alpha=0.8,
                     label='Axial')
            plt.plot(x3/x_unit, n13/n1_unit, c=sim_3d_color, ls='-',
                     alpha=0.8,
                     label='3D')
            plt.ylim(-0.1*n1_max/n1_unit, n1_max*1.1/n1_unit)
            ax.set_yticks([0, 2])

            # Add phantom + so that the label aligns with lower plot.
            ax.set_yticklabels([r"$\phantom{+}0$", r"$\phantom{+}2$"])

            # Draw 3D axial contours
            ax = ax_c
            plt.xlim(xlim)            
            plt.sca(ax)
            #z = mmfplt.colors.color_complex(psi)
            x3, y3 = s3.xyz[:2]
            x3, y3 = x3[..., 0], y3[..., 0]
            n3 = sum(s3.get_density())[:, :, s3.shape[2]//2]
            mmfplt.imcontourf(x3, y3, n3/n_unit, vmin=0, vmax=n_max/n_unit,
                              aspect=1)
            plt.colorbar(cax=cax, pad=0.1,
                         label=r"$n_{3D}$ [$\SI{100}{\micro m^{-3}}$]")
            #mmfplt.imcontourf(x, r_new, px**2+pr**2,
            #                  aspect=1)
            #mmfplt.phase_contour(x, r, psi, linewidths=0.01,
            #                     alpha=0.2)

            ax.grid(lw=0.2, alpha=0.5, c='WhiteSmoke')

            plt.xlim(xlim)
            plt.ylim(ylim)
            ax.set_yticks([-2, 0, 2])
            #ax.set_yticklabels(["$-2$", "$\phantom{+}0$", "$\phantom{+}2$"])

            # Draw axial contours
            ax = ax_b
            plt.xlim(xlim)            
            plt.sca(ax)
            #z = mmfplt.colors.color_complex(psi)
            mmfplt.imcontourf(x, r, n/n_unit, vmin=0, vmax=n_max/n_unit,
                              aspect=1)
            plt.colorbar(cax=cax, pad=0.1,
                         label=r"$n_{3D}$ [$\SI{100}{\micro m^{-3}}$]")
            #mmfplt.imcontourf(x, r_new, px**2+pr**2,
            #                  aspect=1)
            #mmfplt.phase_contour(x, r, psi, linewidths=0.01,
            #                     alpha=0.2)

            ax.grid(lw=0.2, alpha=0.5, c='WhiteSmoke')

            plt.xlim(xlim)
            plt.ylim(ylim)
            ax.set_yticks([-2, 0, 2])
            #ax.set_yticklabels(["$-2$", "$\phantom{+}0$", "$\phantom{+}2$"])

            if not stream:
                continue
            dx = 0.1
            dr = 0.1
            x0 = np.linspace(xlim[0], xlim[1], int(np.diff(xlim)/dx))[1:-1]
            r0 = np.linspace(r_new.min(), r_new.max(), int(np.diff(ylim)/dr))[1:-1]
            X0, R0 = np.meshgrid(x0, r0, sparse=False, indexing='ij')
            #plt.plot(X0.ravel(), R0.ravel(), 'x', c='w', ms=0.1)
            start_points = np.array([X0.ravel(), R0.ravel()]).T
            #start_points = np.array([(0.5, 0.5), (1.1, 1.1), (-1.1, 1.1)])
            #plt.plot(start_points[:, 0], start_points[:, 1], 'x', c='w', ms=0.1)
            ix = np.where(abs(x)<xlim[1])[0]
            res = plt.streamplot(x.ravel()[ix], r_new.ravel(),
                                 px[ix,:].T, pr[ix,:].T,
                                 density=(np.diff(xlim)/10/2, 1),
                                 #minlength=1,
                                 #maxlength=5,
                                 #integration_direction='both',
                                 #start_points=start_points,
                                 linewidth=0.1, arrowsize=0.1)

            print("Done")

        # Hide all tick labels for now:
        for axs in ax0s:
            for ax in axs:
                ax.tick_params(direction='in',
                               left=True, right=True,
                               bottom=True, top=True,
                               labelbottom=False, labelleft=False,
                               # length=6, width=2, colors='r',
                               # grid_color='r', grid_alpha=0.5)
                               )

        ax0s[0][0].tick_params(labelleft=True)
        ax0s[1][1].tick_params(labelleft=True)
        ax0s[1][-1].tick_params(labelleft=True, labelbottom=True)
        ax0s[3][-1].tick_params(labelbottom=True)

        # Labels
        x_label = r"$x$ [$\si{\micro m}$]"
        z_label = r"$z$ [$\si{\micro m}$]"
        n_label = r"$n_{1D}$ [$\si{nm^{-1}}$]$\qquad$"

        ax0s[0][0].set_ylabel(n_label)
        ax0s[1][1].set_ylabel(z_label)
        for ax in [ax0s[1][1], ax0s[3][1]]:
            ax.set_xlabel(x_label)

        for ax_, t_, xlim_, _label in zip(ax0s, t_s, xlims, "abcd"):
            ax = ax_[0]
            ax.text(xlim_[1]-0.5, 0.7*n1_max/n1_unit,
                    rf"$t=\SI{{{t_-10:g}}}{{ms}}$",
                    fontsize='small',
                    horizontalalignment='right')
            ax.text(xlim_[0]+1, 2.02,
                    rf"$\textbf{{{_label}}}$")
            
        return myfig
        
    def fig_comparison(self, t__waits=[6, 16, 26, 36], x_th_shift=1.0,
                       data=None):
        """Main comparison figure with experimental and numerical
        data."""

        # First calculate size so that images with fixed aspect ratio
        # fit correctly
        xlim = (-180, 180)
        ylim = (-50, 50)

        aspect_ratio = abs(np.diff(ylim)/np.diff(xlim))[0]
        rows = 4
        cols = 4
        space = 0.1             # Space between elements

        # Margins
        left = 0.1
        right = 0.04
        top = 0.05
        bottom = 0.1

        width = cols * 1 + (cols-1)*space
        height = (3*rows) * aspect_ratio + (cols-1)*space

        fig_width = width/(1 - left - right)
        fig_height = height/(1 - top - bottom)
        fig_aspect = fig_height/fig_width
        
        myfig = self.figure(
            num=2,              # If you want to redraw in the same window
            width='textwidth',  # For two-column documents, vs. 'textwidth'
            height=f"{fig_aspect}*width",
            subplot_args=dict(left=left, bottom=bottom, 
                              right=1.0-right, top=1.0-top),
        )

        x_unit = u.micron
        n_unit = 1./u.micron**2
        
        if data is None:
            # Experimental data
            from DATA import data_190424 as expt_nosoc
            from DATA import data_190425 as expt_soc
            expt_soc = {30: dict(data=expt_soc.data['C_R_fast_15uW.txt']),
                        60: dict(data=expt_soc.data['C_R_fast_30uW.txt']),
                        90: dict(data=expt_soc.data['C_R_fast_45uW.txt'])}
            expt_nosoc = {30: dict(data=expt_nosoc.data['C_R_fast_noSOC_15uW.txt']),
                          60: dict(data=expt_nosoc.data['C_R_fast_noSOC_30uW.txt']),
                          90: dict(data=expt_nosoc.data['C_R_fast_noSOC_45uW.txt'])}

            for key in expt_soc:
                _expt = expt_soc[key]
                ns = []
                for t__wait in t__waits:
                    x_, y_, n_ = _expt['data'].load(t_wait=t__wait, t_exp=10.1)
                    n_ = n_[0]

                    
                    #####################
                    # Process experimental images - manually selecting
                    # regions of interest for plots, in pixels
                    image_extent = [250, 750, 50, 400] #should be same as ROI below
                    
                    # Filters and detrends data
                    n_, (x_, y_) = data_init_base.process_image(n_.filled(-4000), 
                                                                skip=20, 
                                                                image_extent=image_extent,
                                                                x=x_, y=y_,
                                                                sentinel=-4000,
                                                                registration=False,
                                                                return_registration=False,
                                                                experimental_method=True,
                                                                detrend=True,)
                     
                    x_offset = 4    # shifts left or right, in micron
                    y_0 = 106       # Midpoint of plot

                    # Region of Interest (ROI) box, in pixels
                    iL, iR, iD, iU = image_extent #250, 750, 50, 400
                    y_off = 500   # Uses a new ROI shifter up by y_off to
                                  # subtract off background (pixels)

                    # Select ROI
                    x = x_[iL:iR] - (x_[iL:iR][-1] + x_[iL:iR][0])/2 - x_offset
                    y = y_[iD:iU] - y_[iD:iU][0] - y_0

                    # Also for density, and subtract background
                    n_background = n_[iL:iR, iD+y_off:iU+y_off].mean(axis=1)[:, None]
                    n = n_[iL:iR, iD:iU] - n_background
                    
                    # Subtract of thermal background
                    # CURRENTLY NOT IMPLMENTED, only works in 1D
                    #p0 = [260, 175, 100000, 0, 260, 300, 25000, 0]
                    #x, p_ = Moss.bimodalfit(n.sum(axis=1), p0, show=False)
                    #n -= Moss.gaussian(x,p_[4:8])[:,None]
                    #####################

                    ns.append(n)
                _expt['x'] = x
                _expt['y'] = y
                _expt['ns'] = ns

            for key in expt_nosoc:
                _expt = expt_nosoc[key]
                ns = []
                for t__wait in t__waits:
                    try:
                        x_, y_, n_ = _expt['data'].load(t_wait=t__wait, t_exp=10.1)
                        n_ = n_[0]
                    except KeyError:
                        n_ = 0*n_

                    #####################
                    # Process experimental images - manually selecting
                    # regions of interest for plots, in pixels
                    image_extent = [250, 750, 50, 400] #should be same as ROI below
                    
                    # Filters and detrends data
                    n_, (x_, y_) = data_init_base.process_image(n_.filled(-4000), 
                                                                skip=20, 
                                                                image_extent=image_extent,
                                                                x=x_, y=y_,
                                                                sentinel=-4000,
                                                                registration=False,
                                                                return_registration=False,
                                                                experimental_method=True,
                                                                detrend=True,)
                    
                    
                    x_offset = 50    # shifts left or right, in micron
                    y_0 = 106       # Midpoint of plot

                    # Region of Interest (ROI) box, in pixels
                    iL, iR, iD, iU = image_extent
                    y_off = 500   # Uses a new ROI shifter up by y_off to
                                  # subtract off background (pixels)

                    # Select ROI
                    x = x_[iL:iR] - (x_[iL:iR][-1] + x_[iL:iR][0])/2 - x_offset
                    y = y_[iD:iU] - y_[iD:iU][0] - y_0

                    # Also for density, and subtract background
                    n_background = n_[iL:iR, iD+y_off:iU+y_off].mean(axis=1)[:, None]
                    n = n_[iL:iR, iD:iU] - n_background
                    #####################

                    ns.append(n)
                _expt['x'] = x
                _expt['y'] = y
                _expt['ns'] = ns
    
            # Simulations
            import runs_car_fast_final
            run = runs_car_fast_final.Run_IPG_xTF150()
            sims_soc = {
                30: dict(sim=run.simulations[0]),
                60: dict(sim=run.simulations[1]),
                90: dict(sim=run.simulations[2])
            }

            for key in [30, 60, 90]:
                sims = sims_soc[key]
                sim = sims['sim']
                e = sim.experiment
                assert sim.experiment.barrier_depth_nK == -key
                s0 = sim.get_state(t_=t__waits[0] + e.t__barrier, image=True)
                x, r = s0.xyz[:2]

                ns = []
                for t__wait in t__waits:
                    s = sim.get_state(t_=t__wait + e.t__barrier, image=True)
                    assert np.allclose(s.xyz[0], x) 
                    assert np.allclose(s.xyz[1], r)

                    # Density of majority component
                    n_ = s.get_density()[0]

                    # Compute line-of-sight integral (Abel transform)
                    y = np.linspace(-r.max(), r.max(), 2*r.size)

                    # Remember - the basis does not know about lambda
                    n = s.basis.integrate2(n_, y/s.get_lambda())

                    # Join
                    #y = np.concatenate((-r[..., ::-1], r), axis=-1)
                    #n = np.concatenate((n_[..., ::-1], n_), axis=-1)

                    ns.append(n)
                sims['x'] = x
                sims['y'] = y
                sims['ns'] = ns
                
            data = dict(expt_nosoc=expt_nosoc, expt_soc=expt_soc, sims_soc=sims_soc)
            
        self._data_comparison = data
        expt_nosoc, expt_soc, sims_soc = (
            data['expt_nosoc'], data['expt_soc'], data['sims_soc'])

        myfig.ax.remove()       # Don't use base axis.
        grid0 = MPLGrid(fig=myfig.fig, subplot_spec=myfig.subplot_spec,
                        direction='right', share=True, space=space)
        keys = ['NoSOC', 30, 60, 90]
        x_label_key = 60

        for _k, key in enumerate(keys):
            grid1 = grid0.grid(direction='down', share=True, space=space)
            if key == "NoSOC":
                # Hack for now until we have NoSOC results
                key = keys[1]
                
            sims = sims_soc[key]
            expts = expt_soc[key]
            expts0 = expt_nosoc[key]
            x_t = sims['x']
            y_t = sims['y']
            ns_t = sims['ns']

            x_e = expts['x']
            y_e = expts['y']
            ns_e = expts['ns']

            x_e0 = expts0['x']
            y_e0 = expts0['y']
            ns_e0 = expts0['ns']
            
            for _t, t__wait in enumerate(t__waits):
                grid2 = grid1.grid(direction='down', space=0, share=True)

                ####################
                # Experiment: No SOC
                ax = ax_e0 = grid2.next()
                plt.sca(ax)

                if _t == 0:
                    # Top axis, label bucket depth
                    plt.title(fr"$U_{{\mathrm{{b}}}} = -\SI{{{key}}}{{nK}}$")
                ax.grid(False)
                ax.set_yticks([-40, 0, 40])
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                mmfplt.imcontourf(x_e0, y_e0, ns_e0[_t],
                                  aspect=1)

                # Add times
                if _k == len(keys)-1:
                    ax_e1.yaxis.set_label_position('right')
                    ax_e1.set_ylabel(
                        fr"$t_{{\mathrm{{wait}}}}=\SI{{{t__wait}}}{{ms}}$",
                        horizontalalignment='right', y=0.8,
                    )

                ####################
                # Experiment: SOC
                ax = ax_e1 = grid2.next()
                plt.sca(ax)

                ax.grid(False)
                ax.set_yticks([-40, 0, 40])
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                mmfplt.imcontourf(x_e, y_e, ns_e[_t],
                                  aspect=1)

                ####################
                # Theory: SOC
                ax = ax_t = grid2.next()
                plt.sca(ax)
                ax.grid(False)                
                ax.set_yticks([-40, 0, 40])
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                mmfplt.imcontourf(x_t/x_unit, y_t/x_unit,
                                  ns_t[_t]/n_unit,
                                  aspect=1)

                    
                # Hide inner tick-lables
                for _ax in [ax_e0, ax_e1, ax_t]:
                    if _k > 0:
                        plt.setp(_ax.yaxis.get_ticklabels(), visible=False)
                    plt.setp(_ax.xaxis.get_ticklabels(), visible=False)
                        
            # Last axis, include labels
            plt.setp(ax.xaxis.get_ticklabels(), visible=True)
            if key == x_label_key:
                plt.xlabel(r'$x$ [\si{\micro\meter}]')

        return myfig

        # Data organized by t_exp key here.
        data_exs = {}
        for key in data_180517.data:
            d = data_180517.data[key]
            t_exp = d.t_exps[0]
            if t_exps is not None and t_exp not in t_exps:
                continue
            if np.allclose(d.t_exps, t_exp):
                data_exs.setdefault(t_exp, []).append(d)
        for t_exp in data_exs:
            if not len(data_exs[t_exp]) == 1:
                print("t_exp={} has {} runs - using first"
                      .format(t_exp, len(data_exs[t_exp])))
            data_exs[t_exp] = data_exs[t_exp][0]

        data_exs = OrderedDict([(_t, data_exs[_t]) for _t in
                                sorted(data_exs)])
        
        import runs_final
        run = runs_final.RunSOCPaperSLOW()
        sims = {_s.experiment.rabi_frequency_E_R: _s for _s in run.simulations}

        # Get the no-SOC data.  The key here is rabi_frequency_E_R
        data_th = sims[0]

        # Again, keys are t_exp, data is (t_wait, x_ex, n_ex, x_th, n_th)
        txns = {_t: [] for _t in data_exs}
        
        for t_exp in data_exs:
            data_ex = data_exs[t_exp]
            ts_ex = data_ex.t_waits
            ns_ex = data_ex.get_1d_data().ns
            x_ex = data_ex.get_1d_data().x
            for _n, t_ in enumerate(ts_ex):
                try:
                    state = data_th.get_state(t_, image=t_exp)
                except IndexError:
                    print("t_={} not found".format(t_))
                    continue
                txns[t_exp].append(
                    (t_,
                     x_ex, ns_ex[_n],
                     state.xyz[0], state.get_density_x(mean=False).sum(axis=0)))

        grid = MPLGrid(fig=fig.fig, ax=fig.ax, direction='right', space=0.1)
        xlim = 100

        for t_exp in txns:
            subgrid = grid.grid(direction='down', share=True, space=0)
            for _n, (t_wait, x_ex, n_ex, x_th, n_th) in enumerate(txns[t_exp]):
                ax = subgrid.next()
                if _n == 0:
                    plt.title(r"t\_exp={}".format(t_exp))
                factor = np.trapz(n_th, x_th)/np.trapz(n_ex, x_ex)
                print(factor)
                ax.plot(x_th+x_th_shift, n_th, 'k-', lw=0.5, alpha=0.5)
                ax.plot(x_ex, n_ex*factor, 'r:', lw=0.5, alpha=0.5)
                ax.set_xlim(-xlim, xlim)
                ax.text(0.05, 0.7, 't={:g}ms'.format(t_wait),
                        transform=ax.transAxes)
        self._l = locals()
        return fig

    def fig_test(self):
        fig = self.figure(
            num=2,              # If you want to redraw in the same window
            width='textwidth',  # For two-column documents, vs. 'textwidth'
            height=1.0,
            tight_layout=True,
        )
        
        grid = MPLGrid(fig=fig.fig, ax=fig.ax,
                       space=0)
        ax = grid.next()
        ax.plot([0,1],[0,1])
        ax = grid.next()
        ax.plot([0,1],[0,2])
        
        return fig

    def fig_comparison_no_soc(self, t_exps=[10], x_th_shift=1.0):
        fig = self.figure(
            num=2,              # If you want to redraw in the same window
            width='textwidth',  # For two-column documents, vs. 'textwidth'
            height=1.0,
            tight_layout=False,
        )

        from DATA import data_180517

        # Data organized by t_exp key here.
        data_exs = {}
        for key in data_180517.data:
            d = data_180517.data[key]
            t_exp = d.t_exps[0]
            if t_exps is not None and t_exp not in t_exps:
                continue
            if np.allclose(d.t_exps, t_exp):
                data_exs.setdefault(t_exp, []).append(d)
        for t_exp in data_exs:
            if not len(data_exs[t_exp]) == 1:
                print("t_exp={} has {} runs - using first"
                      .format(t_exp, len(data_exs[t_exp])))
            data_exs[t_exp] = data_exs[t_exp][0]

        data_exs = OrderedDict([(_t, data_exs[_t]) for _t in
                                sorted(data_exs)])
        
        import runs_final
        run = runs_final.RunSOCPaperSLOW()
        sims = {_s.experiment.rabi_frequency_E_R: _s for _s in run.simulations}

        # Get the no-SOC data.  The key here is rabi_frequency_E_R
        data_th = sims[0]

        # Again, keys are t_exp, data is (t_wait, x_ex, n_ex, x_th, n_th)
        txns = {_t: [] for _t in data_exs}
        
        for t_exp in data_exs:
            data_ex = data_exs[t_exp]
            ts_ex = data_ex.t_waits
            ns_ex = data_ex.get_1d_data().ns
            x_ex = data_ex.get_1d_data().x
            for _n, t_ in enumerate(ts_ex):
                try:
                    state = data_th.get_state(t_, image=t_exp)
                except IndexError:
                    print("t_={} not found".format(t_))
                    continue
                txns[t_exp].append(
                    (t_,
                     x_ex, ns_ex[_n],
                     state.xyz[0], state.get_density_x().sum(axis=0)))

        grid = MPLGrid(fig=fig.fig, ax=fig.ax, direction='right', space=0.1)
        xlim = 100

        for t_exp in txns:
            subgrid = grid.grid(direction='down', share=True, space=0)
            for _n, (t_wait, x_ex, n_ex, x_th, n_th) in enumerate(txns[t_exp]):
                ax = subgrid.next()
                if _n == 0:
                    plt.title(r"t\_exp={}".format(t_exp))
                factor = np.trapz(n_th, x_th)/np.trapz(n_ex, x_ex)
                print(factor)
                ax.plot(x_th+x_th_shift, n_th, 'k-', lw=0.5, alpha=0.5)
                ax.plot(x_ex, n_ex*factor, 'r:', lw=0.5, alpha=0.5)
                ax.set_xlim(-xlim, xlim)
                ax.text(0.05, 0.7, 't={:g}ms'.format(t_wait),
                        transform=ax.transAxes)
        self._l = locals()
        return fig


    def fig_comparison_soc1(self, dx=-15, dOmega=0.0):
        fig = self.figure(
            num=3,              # If you want to redraw in the same window
            width='textwidth',  # For two-column documents, vs. 'textwidth'
            height=1.0,
            tight_layout=False,
        )

        from DATA import data_180613

        # Data organized by t_exp key here.
        data_ex = {}
        key = 'C&R_slow.txt'
        d = data_180613.data[key]
        t_exp = 10.1

        
        Omega_E_R = 1.5
        
        assert np.allclose(d.t_exps, t_exp)
        E_R = data_180613.data_init_base.Units.E_R
        info = d.get_info()
        assert np.allclose(info.Omega, Omega_E_R*E_R)
        assert np.allclose(info.t_exp, t_exp)

        data_exs = OrderedDict([(Omega_E_R,  d)])
        
        import runs_final
        run = runs_final.RunSOCPaperSLOW()
        sims = {_s.experiment.rabi_frequency_E_R: _s for _s in run.simulations}

        # Key is Omega_E_R
        txns = {Omega_E_R: []}
        
        for Omega_E_R in data_exs:
            data_th = sims[Omega_E_R+dOmega]
            data_ex = data_exs[Omega_E_R]
            ts_ex = data_ex.t_waits
            ns_ex = data_ex.get_1d_data().ns
            x_ex = data_ex.get_1d_data().x
            for _n, t_ in enumerate(ts_ex):
                try:
                    state = data_th.get_state(t_, image=t_exp)
                except IndexError:
                    print("t_={} not found".format(t_))
                    continue
                txns[Omega_E_R].append(
                    (t_,
                     x_ex, ns_ex[_n],
                     state.xyz[0], state.get_density_x().sum(axis=0)))

        grid = MPLGrid(fig=fig.fig, ax=fig.ax, direction='right', space=0.1)
        xlim = 100

        for Omega_E_R in txns:
            subgrid = grid.grid(direction='down', share=True, space=0)
            for _n, (t_wait, x_ex, n_ex, x_th, n_th) in enumerate(txns[Omega_E_R]):
                ax = subgrid.next()
                if _n == 0:
                    plt.title(r"$\Omega={}E_R$".format(Omega_E_R))
                factor = np.trapz(n_th, x_th)/np.trapz(n_ex, x_ex)
                print(factor)
                ax.plot(x_th, n_th, 'k-', lw=0.5, alpha=0.5)
                ax.plot(x_ex-dx, n_ex*factor, 'r:', lw=0.5, alpha=0.5)
                ax.set_xlim(-xlim, xlim)
        self._l = locals()
        return fig
    
    def fig_comparison_soc2(self, dx=0):
        fig = self.figure(
            num=4,              # If you want to redraw in the same window
            width='textwidth',  # For two-column documents, vs. 'textwidth'
            height=1.0,
            tight_layout=False,
        )

        from DATA import data_181104, data_181106

        # Data organized by t_exp key here.
        data_exs = {}
        key = 'C&R_slow.txt'
        data_exs[1.5] = data_181104.data['MMF.txt']
        data_exs[0.75] = data_181106.data['1106_0.75_CandR.txt']
        data_exs[2.25] = data_181106.data['1106_2.25_CandR.txt']

        t_exp = 10.1

        for Omega_E_R in data_exs:
            d = data_ex = data_exs[Omega_E_R]       
            assert np.allclose(d.t_exps, t_exp)
            E_R =  data_181106.data_init_base.Units.E_R
            info = d.get_info()
            assert np.allclose(info.Omega, Omega_E_R*E_R)
            assert np.allclose(info.t_exp, t_exp)

        data_exs = OrderedDict([(Omega_E_R,  data_exs[Omega_E_R]) for
                                Omega_E_R in sorted(data_exs)])
        
        import runs_final
        if detuning:
            run = runs_final.RunSOCPaperSLOWdetuning()
            sims = {_s.experiment.rabi_frequency_E_R: _s for _s in run.simulations}
            sims[0] = runs_final.RunSOCPaperSLOW().simulations[0]
        else:
            run = runs_final.RunSOCPaperSLOW()
            sims = {_s.experiment.rabi_frequency_E_R: _s for _s in run.simulations}

        # Key is Omega_E_R
        txns = {Omega_E_R: [] for Omega_E_R in data_exs}
        
        for Omega_E_R in data_exs:
            data_th = sims[Omega_E_R]
            data_ex = data_exs[Omega_E_R]
            ts_ex = data_ex.t_waits
            ns_ex = data_ex.get_1d_data().ns
            x_ex = data_ex.get_1d_data().x
            for _n, t_ in enumerate(ts_ex):
                try:
                    state = data_th.get_state(t_, image=t_exp)
                except IndexError:
                    print("t_={} not found".format(t_))
                    continue
                txns[Omega_E_R].append(
                    (t_,
                     x_ex, ns_ex[_n],
                     state.xyz[0], state.get_density_x().sum(axis=0)))

        grid = MPLGrid(fig=fig.fig, ax=fig.ax, direction='right', space=0.1)
        xlim = 100

        for Omega_E_R in txns:
            subgrid = grid.grid(direction='down', share=True, space=0)
            for _n, (t_wait, x_ex, n_ex, x_th, n_th) in enumerate(txns[Omega_E_R]):
                ax = subgrid.next()
                if _n == 0:
                    plt.title(r"$\Omega={}E_R$".format(Omega_E_R))
                factor = np.trapz(n_th, x_th)/np.trapz(n_ex, x_ex)
                print(factor)
                ax.plot(x_th, n_th, 'k-', lw=0.5, alpha=0.5)
                ax.plot(x_ex-dx, n_ex*factor, 'r:', lw=0.5, alpha=0.5)
                ax.set_xlim(-xlim, xlim)
        self._l = locals()
        return fig
    

    def fig_comparison_fast(self, dx=0):
        fig = self.figure(
            num=4,              # If you want to redraw in the same window
            width='textwidth',  # For two-column documents, vs. 'textwidth'
            height=1.0,
            tight_layout=False,
        )

        from DATA import data_180129

        # Data organized by t_exp key here.
        data_exs = {}
        keys = ['C&R_FastLoad_24uW_1.txt',
                'C&R_FastLoad_24uW_2.txt',
                'C&R_FastLoad_24uW_3.txt']
        data_exs[1.5] = data_180129.data[keys[0]]

        t_exp = 10.1

        for Omega_E_R in data_exs:
            d = data_ex = data_exs[Omega_E_R]       
            assert np.allclose(d.t_exps, t_exp)
            E_R =  data_180129.data_init_base.Units.E_R
            info = d.get_info()
            assert np.allclose(info.Omega, Omega_E_R*E_R)
            assert np.allclose(info.t_exp, t_exp)

        data_exs = OrderedDict([(Omega_E_R,  data_exs[Omega_E_R]) for
                                Omega_E_R in sorted(data_exs)])
        
        import runs_final
        run = runs_final.RunSOCPaperFAST()
        sims = {_s.experiment.rabi_frequency_E_R: _s for _s in run.simulations}

        # Key is Omega_E_R
        txns = {Omega_E_R: [] for Omega_E_R in data_exs}
        
        for Omega_E_R in data_exs:
            data_th = sims[Omega_E_R]
            data_ex = data_exs[Omega_E_R]
            ts_ex = data_ex.t_waits
            ns_ex = data_ex.get_1d_data().ns
            x_ex = data_ex.get_1d_data().x
            for _n, t_ in enumerate(ts_ex):
                try:
                    state = data_th.get_state(t_, image=t_exp)
                except IndexError:
                    print("t_={} not found".format(t_))
                    continue
                txns[Omega_E_R].append(
                    (t_,
                     x_ex, ns_ex[_n],
                     state.xyz[0], state.get_density_x().sum(axis=0)))

        grid = MPLGrid(fig=fig.fig, ax=fig.ax, direction='right', space=0.1)
        xlim = 100

        for Omega_E_R in txns:
            subgrid = grid.grid(direction='down', share=True, space=0)
            for _n, (t_wait, x_ex, n_ex, x_th, n_th) in enumerate(txns[Omega_E_R]):
                ax = subgrid.next()
                if _n == 0:
                    plt.title(r"$\Omega={}E_R$".format(Omega_E_R))
                factor = np.trapz(n_th, x_th)/np.trapz(n_ex, x_ex)
                print(factor)
                ax.plot(x_th, n_th, 'k-', lw=0.5, alpha=0.5)
                ax.plot(x_ex-dx, n_ex*factor, 'r:', lw=0.5, alpha=0.5)
                ax.set_xlim(-xlim, xlim)
        self._l = locals()
        return fig
    
    
'''
