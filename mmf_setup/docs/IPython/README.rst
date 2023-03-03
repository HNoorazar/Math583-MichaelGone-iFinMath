.. -*- rst -*- -*- restructuredtext -*-

.. This file should be written using the restructure text
.. conventions.  It will be displayed on the bitbucket source page and
.. serves as the documentation of the directory.

.. .. include:: .links.rst

.. |virtualenv.py| replace:: ``virtualenv.py``
.. _virtualenv.py: https://raw.github.com/pypa/virtualenv/master/virtualenv.py

.. |EPD| replace:: Enthough Python Distribution
.. _EPD: http://www.enthought.com/products/epd.php
.. _Anaconda: https://store.continuum.io/cshop/anaconda
.. _Conda: http://docs.continuum.io/conda

.. _Enthought: http://www.enthought.com
.. _Continuum Analytics: http://continuum.io

.. _mercurial: http://mercurial.selenic.com/
.. _virtualenv: http://www.virtualenv.org/en/latest/
.. _IPython: http://ipython.org/
.. _IPython notebook: http://ipython.org/notebook.html
.. _IPython Notebook Viewer: http://nbviewer.ipython.org
.. _qtconsole: http://ipython.org/ipython-doc/stable/interactive/qtconsole.html
.. _Qt: http://qt-project.org
.. |pip| replace:: ``pip``
.. _pip: http://www.pip-installer.org/
.. _git: http://git-scm.com/
.. _github: https://github.com
.. _RunSnakeRun: http://www.vrplumber.com/programming/runsnakerun/
.. _GSL: http://www.gnu.org/software/gsl/
.. _pygsl: https://bitbucket.org/mforbes/pygsl
.. _Sphinx: http://sphinx-doc.org/
.. _SciPy: http://www.scipy.org/
.. _NumPy: http://numpy.scipy.org/
.. _Numba: https://github.com/numba/numba#readme
.. _Python: http://www.python.org/
.. _matplotlib: http://matplotlib.org/
.. _Matlab: http://www.mathworks.com/products/matlab/
.. _MKL: http://software.intel.com/en-us/intel-mkl
.. _Intel compilers: http://software.intel.com/en-us/intel-compilers
.. _Bento: http://cournape.github.com/Bento/
.. _pyaudio: http://people.csail.mit.edu/hubert/pyaudio/
.. _PortAudio: http://www.portaudio.com/archives/pa_stable_v19_20111121.tgz
.. _MathJax: http://www.mathjax.org/
.. _LaTeX: http://www.latex-project.org
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Emacs: http://www.gnu.org/software/emacs/
.. _Pymacs: https://github.com/pinard/Pymacs
.. _Ropemacs: http://rope.sourceforge.net/ropemacs.html
.. _Julia: http://julialang.org
.. _R: http://www.r-project.org

.. default-role:: math

.. This is so that I can work offline.  It should be ignored on bitbucket for
.. example.

.. raw:: html

   <script type="text/javascript"
    src="/Users/mforbes/.mathjax/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
   </script>

.. sidebar:: Sidebar

   .. contents::

==========
 IPython_
==========

IPython_ provides various different ways of interfacing with python_.  At its
simplest, it can be thought of as an improved command line, with some goodies
such as saving input and output for future use (you can access previous results
with a syntax like ``Out[3]``) and some magics like ``%debug`` and ``%paste``.
It also provides access to shell commands such as ``!ls`` that can then be used
in python.  When launched with the ``--pylab`` flag, IPython_ also manages
various GUI backends for matplotlib_, allowing one to interact with plots.  This
has traditionally been a challenge since the GUI event looks generally expect to
be run in the main thread.

At a more sophisticated level, IPython_ provides a mechanism for managing
computations.  For example, it provides machinery for launching and interacting
with multiple compute engines to facilitate parallel computing.  Once the hurdle
of configuring a cluster is overcome, this allows one to easily leverage
multiple computers or a *bona fide* cluster from a single python instance.
Tools are provided for launching multiple compute jobs, and then collecting the
results for analysis etc.  A simple version of this idea allows one to turn a
python session into a "kernel" to which additional python processes can attach.
One can thus have multiple interfaces to a single kernel (useful when debugging
notebooks for example – see below).

Finally, IPython_ provides a notebook interface. The `IPython notebook`_ allows
one to interact with python via a web-browser, either locally or remotely.  The
output is stored in the notebook for offline viewing (see for example the
`IPython Notebook Viewer`_).  These notebooks can contain embedded documentation
using embedded HTML objects such as movies, and have MathJax_ support for
LaTeX_.  Although not ideal for editing (these suffer from common problems of
browser editing, such as poor search and replace functionality, poor
performance, focus issues etc.), it is rapidly becoming the *de facto* tool for
interactive computing and recording the steps required to reproduce a
calculation.  The `IPython notebook`_ is expanding its support beyond python_ to
support other languages such as Julia_ and R_.  Once these bridges are complete
and stable, the `IPython notebook`_ will surpass any other tools of which I am
aware for managing complex calculations, providing a unified interface for
interacting with multiple languages, and parallel computing.

This set of notes helps describe how to get around some rough edges with
IPython_, mostly when dealing with remote access issues.


Quick Start
===========

I highly recommend that you install a recent development snapshot.  IPython_
development is proceeding rapidly and the latest releases often have critical
functionality.  To do this, I recommend using git_:

.. code:: bash

   git clone https://github.com/ipython/ipython.git
   cd ipython
   conda pip .  # OR pip install . OR python setup.py install

IPython_ is a pure python package, so this should not have any problems.

Configurations
==============

IPython_ configurations live in your home directory ``~/.ipython``.  In
particular, you should create a profile for each computing situation you need to
use.  I recommend version controlling these profiles so you can see what
configurations you needed to make.


`IPython Notebook`_
===================

One can simply launch a notebook instance as follows:

.. code:: bash

   $ cd ~/my_notebooks
   $ ipython notebook --pylab=inline
   ... [NotebookApp] Using existing profile dir: ...
   ... [NotebookApp] Using local MathJax from .../MathJax.js
   ... [NotebookApp] Serving notebooks from local directory: ~/my_notebooks
   ... [NotebookApp] The IPython Notebook is running at: http://127.0.0.1:8888/
   ... [NotebookApp] Use Control-C to stop this server and shut down all kernels.

You can connect to this notebook server from your local machine by pointing your
web-browser to http://127.0.0.1:8888/.  This will show you any notebooks (and
allow you to create new notebooks) in the directory from which you ran the
command (``~/my_notebooks`` in this case).  From your browser you can create new
notebooks, start editing existing notebooks, and launch clusters.  The
``--pylab=inline`` option does a ``from pylab import *`` as well as starts
matplotlib_ in a mode where the plots will appear and be saved in the notebook.

Note: if you use a recent development version of IPython_, then you can change
the backend with ``%pylab osx`` or ``%pylab qt``.  After doing this, plotting
will open a new window which will allow you to interact with the plot.

Notebooks are no good (yet) for debugging.  If you want to debug a notebook, you
can launch a qtconsole_ with the magic ``%qtconsole``.  This will not work if
you do not have Qt_ setup (in particular, it will fail with the default
Anaconda_ installation on Mac OS X), so the other option is to turn the process
into a kernel as follows, then connect with another IPython_ application (such
as the ``console``).  To do this, look at the output from the ``ipython
notebook`` command given above after you open a notebook:

.. code ::bash

   ... [NotebookApp] Connecting to: tcp://127.0.0.1:57100
   ... [NotebookApp] Kernel started: 3e4f0001-41c9-47ac-8c7a-94162bd41ec3
   ... [NotebookApp] Connecting to: tcp://127.0.0.1:57097
   ... [NotebookApp] Connecting to: tcp://127.0.0.1:57098
   ... [NotebookApp] Connecting to: tcp://127.0.0.1:57099

This tells you that a kernel with id ``3e4f0001-41c9-47ac-8c7a-94162bd41ec3``
has started.  You can connect to this with a console as follows:

.. code:: bash

   $ ipython console --existing=3e4f0001-41c9-47ac-8c7a-94162bd41ec3

(If you only have one kernel running, you can usually omit the argument and just
use ``ipython console --existing``.)  This will now allow you to interact with
the same kernel.  I use this quite often to play with things (outside of the
notebook) before saving the good commands in the notebook for posterity.  Note:
you can also run a ``qtconsole`` or possibly even another ``notebook`` rather
than a ``console`` if you like.

Remote Notebook Access
++++++++++++++++++++++

One very nice feature is the ability to connect to an `IPython notebook`_
running on a different machine.  This generally requires that you setup a
profile as `described here`__.  The basic idea is to:

1) Create a profile called ``nbserver``: ``ipython profile create nbserver``.
2) Add a password, SSL certificate, and some configurations to the
   ``~/.ipython/profile_nbserver/ipython_notebook_config.py``.  See
   `how to create the encrypted password and SSL certificates`__.  This is
   important for security because otherwise your password will be sent across
   the network in plain text.  It can be somewhat inconvenient, however, as
   discussed below.
3) Copy the SSL certificate to your local computer and accept it. (This might
   only be needed for Mac OS X and Safari, in which case you can import it into
   the ``Keychain.app``.)
4) Start the notebook server on your host.
5) Connect to the host remotely on the specified port.  Of course, for this to
   work, you will need to be able to connect to the host, so you will need to
   know its remote IP address, not the ``127.0.0.1`` from above which only works
   locally.  Try looking at ``ifconfig`` or a similar command.

   .. warning:: If you enabled SSL (which you should), then you must specify the
      address as ``https://my.host.com:<port>`` otherwise the server will return
      nothing (just a blank page) and will display something like the following
      as an error (in the terminal)::

        [tornado.general] WARNING | SSL Error on 7 ('<IP>', <port>): \
        [Errno 1] _ssl.c:504: error:1407609C:SSL \
        routines:SSL23_GET_CLIENT_HELLO:http request

__ http://ipython.org/ipython-doc/dev/interactive/
   htmlnotebook.html#running-a-public-notebook-server 
__ http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html#security

