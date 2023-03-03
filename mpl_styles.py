"""Different matplotlib style options for customizing plots.

To allow for more flexibility, we provide these as dictionaries.  (This allows one to
use hashes for example, which are not valid in an mplstyle file.)

Use as you would styles:

>>> import matplotlib.pyplot as plt
>>> import mpl_styles
>>> plt.style.use(mpl_styles.nature)

Or, better, use it locally:

>>> with plt.style.context(mpl_styles.nature, after_reset=True):
...     plt.plot([1,2], [1,2])

etc.
"""

common = {
    # ***************************************************************************
    # * LINES                                                                   *
    # ***************************************************************************
    # See https://matplotlib.org/api/artist_api.html#module-matplotlib.lines
    # for more information on line properties.
    "lines.solid_capstyle": "round",
    "figure.facecolor": (1, 1, 1, 0),  # Transparent background.
    "figure.facecolor": (1, 1, 1, 1),  # Solid background.
    #
    # ***************************************************************************
    # * GRIDS                                                                   *
    # ***************************************************************************
    "grid.linestyle": "-",
    "grid.linewidth": 1.0,
    "grid.color": "WhiteSmoke",
    "grid.alpha": 0.3,
    #
    # ***************************************************************************
    # * AXES                                                                    *
    # ***************************************************************************
    # Following are default face and edge colors, default tick sizes,
    # default fontsizes for ticklabels, and so on.  See
    # https://matplotlib.org/api/axes_api.html#module-matplotlib.axes
    "axes.grid": True,
    "axes.grid.axis": "both",
    "axes.grid.which": "major",
    "axes.edgecolor": "grey",
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.minor.size": 1,
    "ytick.minor.size": 1,
    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
    'xtick.bottom': True,
    'xtick.top': True,    
    'ytick.left': True,
    'ytick.right': True,    
    
    "lines.linewidth": 1,
    #
    # ***************************************************************************
    # * LEGEND                                                                  *
    # ***************************************************************************
    "legend.handlelength": 4.0,
    "legend.frameon": True,
    "legend.loc": "best",
    #
    # ***************************************************************************
    # * FONT                                                                    *
    # ***************************************************************************
    # The font properties used by `text.Text`.
    # See https://matplotlib.org/api/font_manager_api.html for more information
    # on font properties.  The 6 font properties used for font matching are
    # given below with their default values.
    #
    # The font.family property has five values:
    #     - 'serif' (e.g., Times),
    #     - 'sans-serif' (e.g., Helvetica),
    #     - 'cursive' (e.g., Zapf-Chancery),
    #     - 'fantasy' (e.g., Western), and
    #     - 'monospace' (e.g., Courier).
    # Each of these font families has a default list of font names in decreasing
    # order of priority associated with them.  When text.usetex is False,
    # font.family may also be one or more concrete font names.
    #
    # The font.style property has three values: normal (or roman), italic
    # or oblique.  The oblique style will be used for italic, if it is not
    # present.
    #
    # The font.variant property has two values: normal or small-caps.  For
    # TrueType fonts, which are scalable fonts, small-caps is equivalent
    # to using a font size of 'smaller', or about 83%% of the current font
    # size.
    #
    # The font.weight property has effectively 13 values: normal, bold,
    # bolder, lighter, 100, 200, 300, ..., 900.  Normal is the same as
    # 400, and bold is 700.  bolder and lighter are relative values with
    # respect to the current weight.
    #
    # The font.stretch property has 11 values: ultra-condensed,
    # extra-condensed, condensed, semi-condensed, normal, semi-expanded,
    # expanded, extra-expanded, ultra-expanded, wider, and narrower.  This
    # property is not currently implemented.
    #
    # The font.size property is the default font size for text, given in pts.
    # 10 pt is the standard value.
    #
    # Note that font.size controls default text sizes.  To configure
    # special text sizes tick labels, axes, labels, title, etc, see the rc
    # settings for axes and ticks.  Special text sizes can be defined
    # relative to font.size, using the following values: xx-small, x-small,
    # small, medium, large, x-large, xx-large, larger, or smaller
    "font.family": "serif",
    "font.size": 10.0,
    "font.size": 9.24774,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 8.0,
    "legend.title_fontsize": None,  # None sets to the same as the default axes.
    "font.serif": [
        "DejaVu Serif",
        "Bitstream Vera Serif",
        "Computer Modern Roman",
        "New Century Schoolbook",
        "Century Schoolbook L",
        "Utopia",
        "ITC Bookman",
        "Bookman",
        "Nimbus Roman No9 L",
        "Times New Roman",
        "Times",
        "Palatino",
        "Charter",
        "serif",
    ],
    "font.sans-serif": [
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "Computer Modern Sans Serif",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Arial",
        "Helvetica",
        "Avant Garde",
        "sans-serif",
    ],
    "font.cursive": [
        "Apple Chancery",
        "Textile",
        "Zapf Chancery",
        "Sand",
        "Script MT",
        "Felipa",
        "cursive",
    ],
    "font.fantasy": [
        "Comic Neue",
        "Comic Sans MS",
        "Chicago",
        "Charcoal",
        "ImpactWestern",
        "Humor Sans",
        "xkcd",
        "fantasy",
    ],
    "font.monospace": [
        "DejaVu Sans Mono",
        "Bitstream Vera Sans Mono",
        "Computer Modern Typewriter",
        "Andale Mono",
        "Nimbus Mono L",
        "Courier New",
        "Courier",
        "Fixed",
        "Terminal",
        "monospace",
    ],
}


arXiv = dict(common)
arXiv.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Neo Euler",
            "Palatino",
            "Times New Roman",
            "Times",
            "serif",
        ],
    }
)

latex_preamble = r"""
\usepackage[greek, english]{babel}
\usepackage[euler-digits,euler-hat-accent]{eulervm}
\usepackage[round-mode=off,mode=text,detect-all=true,detect-mode=false,
            list-units=single,unit-mode=text,number-mode=math]{siunitx}
"""

nature = dict(common)
nature.update(
    {
        "text.latex.preamble": "\n".join(
            [
                latex_preamble,
            ]
        ),
        "pgf.preamble": "\n".join(
            [
                latex_preamble,
                # r"\usepackage{unicode-math}",  # unicode math setup
                # r"\setmainfont{DejaVu Serif}",
            ]
        ),
        "mathtext.fontset": "custom",
        "mathtext.default": "rm",
        "font.family": "sans-serif",
        "font.style": "normal",
        "font.variant": "normal",
        "font.weight": "normal",
        "font.stretch": "normal",
        "font.serif": [
            "Palatino Linotype",
            "Palatino",
            "serif",
        ],
        "font.sans-serif": [
            "DejaVu Sans",
            "Helvetica",
            "Bitstream Vera Sans",
            "Computer Modern Sans Serif",
            "Lucida Grande",
            "Verdana",
            "Geneva",
            "Lucid",
            "Avant Garde",
            "sans-serif",
        ],
        "ps.fonttype": 42,
    }
)

"""

RcParams({'_internal.classic_mode': False,
    "lines.antialiased": True,
    "lines.color": "C0",
    "lines.dash_capstyle": "butt",
    "lines.dash_joinstyle": "round",
    "lines.dashdot_pattern": [6.4, 1.6, 1.0, 1.6],
    "lines.dashed_pattern": [3.7, 1.6],
    "lines.dotted_pattern": [1.0, 1.65],
    "lines.linestyle": "-",
    "lines.linewidth": 1.5,
    "lines.marker": "None",
    "lines.markeredgecolor": "auto",
    "lines.markeredgewidth": 1.0,
    "lines.markerfacecolor": "auto",
    "lines.markersize": 6.0,
    "lines.scale_dashes": True,
    "lines.solid_capstyle": "projecting",
    "lines.solid_joinstyle": "round",
          'agg.path.chunksize': 0,
          'animation.avconv_args': [],
          'animation.avconv_path': 'avconv',
          'animation.bitrate': -1,
          'animation.codec': 'h264',
          'animation.convert_args': [],
          'animation.convert_path': 'convert',
          'animation.embed_limit': 20.0,
          'animation.ffmpeg_args': [],
          'animation.ffmpeg_path': 'ffmpeg',
          'animation.frame_format': 'png',
          'animation.html': 'none',
          'animation.html_args': [],
          'animation.writer': 'ffmpeg',
          'axes.autolimit_mode': 'data',
          'axes.axisbelow': 'line',
          'axes.edgecolor': 'black',
          'axes.facecolor': 'white',
          'axes.formatter.limits': [-5, 6],
          'axes.formatter.min_exponent': 0,
          'axes.formatter.offset_threshold': 4,
          'axes.formatter.use_locale': False,
          'axes.formatter.use_mathtext': False,
          'axes.formatter.useoffset': True,
          'axes.grid': False,
          'axes.grid.axis': 'both',
          'axes.grid.which': 'major',
          'axes.labelcolor': 'black',
          'axes.labelpad': 4.0,
          'axes.labelsize': 'medium',
          'axes.labelweight': 'normal',
          'axes.linewidth': 0.8,
          'axes.prop_cycle': cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
          'axes.spines.bottom': True,
          'axes.spines.left': True,
          'axes.spines.right': True,
          'axes.spines.top': True,
          'axes.titlecolor': 'auto',
          'axes.titlelocation': 'center',
          'axes.titlepad': 6.0,
          'axes.titlesize': 'large',
          'axes.titleweight': 'normal',
          'axes.titley': None,
          'axes.unicode_minus': True,
          'axes.xmargin': 0.05,
          'axes.ymargin': 0.05,
          'axes3d.grid': True,
          'backend': 'cairo',
          'backend_fallback': False,
          'boxplot.bootstrap': None,
          'boxplot.boxprops.color': 'black',
          'boxplot.boxprops.linestyle': '-',
          'boxplot.boxprops.linewidth': 1.0,
          'boxplot.capprops.color': 'black',
          'boxplot.capprops.linestyle': '-',
          'boxplot.capprops.linewidth': 1.0,
          'boxplot.flierprops.color': 'black',
          'boxplot.flierprops.linestyle': 'none',
          'boxplot.flierprops.linewidth': 1.0,
          'boxplot.flierprops.marker': 'o',
          'boxplot.flierprops.markeredgecolor': 'black',
          'boxplot.flierprops.markeredgewidth': 1.0,
          'boxplot.flierprops.markerfacecolor': 'none',
          'boxplot.flierprops.markersize': 6.0,
          'boxplot.meanline': False,
          'boxplot.meanprops.color': 'C2',
          'boxplot.meanprops.linestyle': '--',
          'boxplot.meanprops.linewidth': 1.0,
          'boxplot.meanprops.marker': '^',
          'boxplot.meanprops.markeredgecolor': 'C2',
          'boxplot.meanprops.markerfacecolor': 'C2',
          'boxplot.meanprops.markersize': 6.0,
          'boxplot.medianprops.color': 'C1',
          'boxplot.medianprops.linestyle': '-',
          'boxplot.medianprops.linewidth': 1.0,
          'boxplot.notch': False,
          'boxplot.patchartist': False,
          'boxplot.showbox': True,
          'boxplot.showcaps': True,
          'boxplot.showfliers': True,
          'boxplot.showmeans': False,
          'boxplot.vertical': True,
          'boxplot.whiskerprops.color': 'black',
          'boxplot.whiskerprops.linestyle': '-',
          'boxplot.whiskerprops.linewidth': 1.0,
          'boxplot.whiskers': 1.5,
          'contour.corner_mask': True,
          'contour.linewidth': None,
          'contour.negative_linestyle': 'dashed',
          'date.autoformatter.day': '%Y-%m-%d',
          'date.autoformatter.hour': '%m-%d %H',
          'date.autoformatter.microsecond': '%M:%S.%f',
          'date.autoformatter.minute': '%d %H:%M',
          'date.autoformatter.month': '%Y-%m',
          'date.autoformatter.second': '%H:%M:%S',
          'date.autoformatter.year': '%Y',
          'date.epoch': '1970-01-01T00:00:00',
          'docstring.hardcopy': False,
          'errorbar.capsize': 0.0,
          'figure.autolayout': False,
          'figure.constrained_layout.h_pad': 0.04167,
          'figure.constrained_layout.hspace': 0.02,
          'figure.constrained_layout.use': False,
          'figure.constrained_layout.w_pad': 0.04167,
          'figure.constrained_layout.wspace': 0.02,
          'figure.dpi': 72.0,
          'figure.edgecolor': (1, 1, 1, 0),
          'figure.facecolor': (1, 1, 1, 0),
          'figure.figsize': [6.0, 4.0],
          'figure.frameon': True,
          'figure.max_open_warning': 20,
          'figure.raise_window': True,
          'figure.subplot.bottom': 0.125,
          'figure.subplot.hspace': 0.2,
          'figure.subplot.left': 0.125,
          'figure.subplot.right': 0.9,
          'figure.subplot.top': 0.88,
          'figure.subplot.wspace': 0.2,
          'figure.titlesize': 'large',
          'figure.titleweight': 'normal',
          'font.cursive': ['Apple Chancery',
                           'Textile',
                           'Zapf Chancery',
                           'Sand',
                           'Script MT',
                           'Felipa',
                           'cursive'],
          'font.family': ['sans-serif'],
          'font.fantasy': ['Comic Neue',
                           'Comic Sans MS',
                           'Chicago',
                           'Charcoal',
                           'ImpactWestern',
                           'Humor Sans',
                           'xkcd',
                           'fantasy'],
          'font.monospace': ['DejaVu Sans Mono',
                             'Bitstream Vera Sans Mono',
                             'Computer Modern Typewriter',
                             'Andale Mono',
                             'Nimbus Mono L',
                             'Courier New',
                             'Courier',
                             'Fixed',
                             'Terminal',
                             'monospace'],
          'font.sans-serif': ['DejaVu Sans',
                              'Bitstream Vera Sans',
                              'Computer Modern Sans Serif',
                              'Lucida Grande',
                              'Verdana',
                              'Geneva',
                              'Lucid',
                              'Arial',
                              'Helvetica',
                              'Avant Garde',
                              'sans-serif'],
          'font.serif': ['DejaVu Serif',
                         'Bitstream Vera Serif',
                         'Computer Modern Roman',
                         'New Century Schoolbook',
                         'Century Schoolbook L',
                         'Utopia',
                         'ITC Bookman',
                         'Bookman',
                         'Nimbus Roman No9 L',
                         'Times New Roman',
                         'Times',
                         'Palatino',
                         'Charter',
                         'serif'],
          'font.size': 10.0,
          'font.stretch': 'normal',
          'font.style': 'normal',
          'font.variant': 'normal',
          'font.weight': 'normal',
          'grid.alpha': 1.0,
          'grid.color': '#b0b0b0',
          'grid.linestyle': '-',
          'grid.linewidth': 0.8,
          'hatch.color': 'black',
          'hatch.linewidth': 1.0,
          'hist.bins': 10,
          'image.aspect': 'equal',
          'image.cmap': 'viridis',
          'image.composite_image': True,
          'image.interpolation': 'antialiased',
          'image.lut': 256,
          'image.origin': 'upper',
          'image.resample': True,
          'interactive': True,
          'keymap.all_axes': ['a'],
          'keymap.back': ['left', 'c', 'backspace', 'MouseButton.BACK'],
          'keymap.copy': ['ctrl+c', 'cmd+c'],
          'keymap.forward': ['right', 'v', 'MouseButton.FORWARD'],
          'keymap.fullscreen': ['f', 'ctrl+f'],
          'keymap.grid': ['g'],
          'keymap.grid_minor': ['G'],
          'keymap.help': ['f1'],
          'keymap.home': ['h', 'r', 'home'],
          'keymap.pan': ['p'],
          'keymap.quit': ['ctrl+w', 'cmd+w', 'q'],
          'keymap.quit_all': [],
          'keymap.save': ['s', 'ctrl+s'],
          'keymap.xscale': ['k', 'L'],
          'keymap.yscale': ['l'],
          'keymap.zoom': ['o'],
          'legend.borderaxespad': 0.5,
          'legend.borderpad': 0.4,
          'legend.columnspacing': 2.0,
          'legend.edgecolor': '0.8',
          'legend.facecolor': 'inherit',
          'legend.fancybox': True,
          'legend.fontsize': 'medium',
          'legend.framealpha': 0.8,
          'legend.frameon': True,
          'legend.handleheight': 0.7,
          'legend.handlelength': 2.0,
          'legend.handletextpad': 0.8,
          'legend.labelspacing': 0.5,
          'legend.loc': 'best',
          'legend.markerscale': 1.0,
          'legend.numpoints': 1,
          'legend.scatterpoints': 1,
          'legend.shadow': False,
          'legend.title_fontsize': None,
          'markers.fillstyle': 'full',
          'mathtext.bf': 'sans:bold',
          'mathtext.cal': 'cursive',
          'mathtext.default': 'it',
          'mathtext.fallback': 'cm',
          'mathtext.fallback_to_cm': None,
          'mathtext.fontset': 'dejavusans',
          'mathtext.it': 'sans:italic',
          'mathtext.rm': 'sans',
          'mathtext.sf': 'sans',
          'mathtext.tt': 'monospace',
          'mpl_toolkits.legacy_colorbar': True,
          'patch.antialiased': True,
          'patch.edgecolor': 'black',
          'patch.facecolor': 'C0',
          'patch.force_edgecolor': False,
          'patch.linewidth': 1.0,
          'path.effects': [],
          'path.simplify': True,
          'path.simplify_threshold': 0.111111111111,
          'path.sketch': None,
          'path.snap': True,
          'pcolor.shading': 'flat',
          'pdf.compression': 6,
          'pdf.fonttype': 3,
          'pdf.inheritcolor': False,
          'pdf.use14corefonts': False,
          'pgf.preamble': '',
          'pgf.rcfonts': True,
          'pgf.texsystem': 'xelatex',
          'polaraxes.grid': True,
          'ps.distiller.res': 6000,
          'ps.fonttype': 3,
          'ps.papersize': 'letter',
          'ps.useafm': False,
          'ps.usedistiller': None,
          'savefig.bbox': None,
          'savefig.directory': '~',
          'savefig.dpi': 'figure',
          'savefig.edgecolor': 'auto',
          'savefig.facecolor': 'auto',
          'savefig.format': 'png',
          'savefig.jpeg_quality': 95,
          'savefig.orientation': 'portrait',
          'savefig.pad_inches': 0.1,
          'savefig.transparent': False,
          'scatter.edgecolors': 'face',
          'scatter.marker': 'o',
          'svg.fonttype': 'path',
          'svg.hashsalt': None,
          'svg.image_inline': True,
          'text.antialiased': True,
          'text.color': 'black',
          'text.hinting': 'force_autohint',
          'text.hinting_factor': 8,
          'text.kerning_factor': 0,
          'text.latex.preamble': '',
          'text.latex.preview': False,
          'text.usetex': False,
          'timezone': 'UTC',
          'tk.window_focus': False,
          'toolbar': 'toolbar2',
          'webagg.address': '127.0.0.1',
          'webagg.open_in_browser': True,
          'webagg.port': 8988,
          'webagg.port_retries': 50,
          'xaxis.labellocation': 'center',
          'xtick.alignment': 'center',
          'xtick.bottom': True,
          'xtick.color': 'black',
          'xtick.direction': 'out',
          'xtick.labelbottom': True,
          'xtick.labelsize': 'medium',
          'xtick.labeltop': False,
          'xtick.major.bottom': True,
          'xtick.major.pad': 3.5,
          'xtick.major.size': 3.5,
          'xtick.major.top': True,
          'xtick.major.width': 0.8,
          'xtick.minor.bottom': True,
          'xtick.minor.pad': 3.4,
          'xtick.minor.size': 2.0,
          'xtick.minor.top': True,
          'xtick.minor.visible': False,
          'xtick.minor.width': 0.6,
          'xtick.top': False,
          'yaxis.labellocation': 'center',
          'ytick.alignment': 'center_baseline',
          'ytick.color': 'black',
          'ytick.direction': 'out',
          'ytick.labelleft': True,
          'ytick.labelright': False,
          'ytick.labelsize': 'medium',
          'ytick.left': True,
          'ytick.major.left': True,
          'ytick.major.pad': 3.5,
          'ytick.major.right': True,
          'ytick.major.size': 3.5,
          'ytick.major.width': 0.8,
          'ytick.minor.left': True,
          'ytick.minor.pad': 3.4,
          'ytick.minor.right': True,
          'ytick.minor.size': 2.0,
          'ytick.minor.visible': False,
          'ytick.minor.width': 0.6,
          'ytick.right': False})
"""
