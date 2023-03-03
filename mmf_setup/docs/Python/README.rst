.. -*- rst -*- -*- restructuredtext -*-

.. This file should be written using the restructure text
.. conventions.  It will be displayed on the bitbucket source page and
.. serves as the documentation of the directory.

.. |virtualenv.py| replace:: ``virtualenv.py``
.. _virtualenv.py: https://raw.github.com/pypa/virtualenv/master/virtualenv.py

.. |EPD| replace:: Enthough Python Distribution
.. _EPD: http://www.enthought.com/products/epd.php
.. _Anaconda: https://store.continuum.io/cshop/anaconda
.. _Conda: http://docs.continuum.io/conda
.. _Miniconda: http://conda.pydata.org/miniconda.html

.. _Enthought: http://www.enthought.com
.. _Continuum Analytics: http://continuum.io

.. _Spyder: https://code.google.com/p/spyderlib/
.. _Wakari: https://www.wakari.io
.. _Canopy: https://www.enthought.com/products/canopy/

.. _mercurial: http://mercurial.selenic.com/
.. _virtualenv: http://www.virtualenv.org/en/latest/
.. _IPython: http://ipython.org/
.. _Ipython notebook: \
   http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html
.. _NBViewer: http://nbviewer.ipython.org
.. |pip| replace:: ``pip``
.. _pip: http://www.pip-installer.org/
.. _git: http://git-scm.com/
.. _github: https://github.com
.. _RunSnakeRun: http://www.vrplumber.com/programming/runsnakerun/
.. _GSL: http://www.gnu.org/software/gsl/
.. _pygsl: https://bitbucket.org/mforbes/pygsl
.. _Sphinx: http://sphinx-doc.org/
.. _SciPy: http://www.scipy.org/
.. _Mayavi: http://code.enthought.com/projects/mayavi/
.. _NumPy: http://numpy.scipy.org/
.. _Numba: https://github.com/numba/numba#readme
.. _NumbaPro: http://docs.continuum.io/numbapro/
.. _Blaze: http://blaze.pydata.org
.. _Python: http://www.python.org/
.. _matplotlib: http://matplotlib.org/
.. _Matlab: http://www.mathworks.com/products/matlab/
.. _MKL: http://software.intel.com/en-us/intel-mkl
.. _Intel compilers: http://software.intel.com/en-us/intel-compilers
.. _Bento: http://cournape.github.com/Bento/
.. _pyaudio: http://people.csail.mit.edu/hubert/pyaudio/
.. _PortAudio: http://www.portaudio.com/archives/pa_stable_v19_20111121.tgz
.. _MathJax: http://www.mathjax.org/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Emacs: http://www.gnu.org/software/emacs/
.. _Pymacs: https://github.com/pinard/Pymacs
.. _Ropemacs: http://rope.sourceforge.net/ropemacs.html
.. _PyPI: https://pypi.python.org/pypi

.. _FFTW: http://www.fftw.org
.. _EC2: http://aws.amazon.com/ec2/
.. _QT: http://qt.digia.com

.. |site.USER_BASE| replace:: ``site.USER_BASE``
.. _site.USER_BASE: https://docs.python.org/2/library/site.html#site.USER_BASE


.. default-role:: math

.. This is so that I can work offline.  It should be ignored on bitbucket for
.. example.

.. sidebar:: Sidebar

   .. contents::

==============
 Python Setup
==============
This meta-project provides an easy way to install all of the python
tools I typically use.  It also serves as a fairly minimal example of
setting up a package the |pip|_ can install, and specifying
dependencies.

In particular, I structure it for the following use-cases:

1. Rapid installation and configuration of the tools I need.  For
   example, I often use [Sage Mathcloud](cloud.sagemath.com).
   Whenever I create a new project, I need to perform some
   initialization.  With this project, it is simply a matter of using
   |pip|_ to install this package, and then using some of the tools.
2. Initial setup of a python distribution on a new computer.  This is
   a little more involved since one needs to first install python (I
   recommend using Miniconda_) and then updating the tools.
3. A place to document various aspects of installing and setting up
   python and related tools.  Some of this is old, but kept here for
   reference.

============
 Full Setup
============

Summary
=======
One can use and install python in several ways, but I strongly suggest using one
of the complete package managers.  These provide a convenient way of quickly
getting up and started, and can simplify dependency issues.  Presently
I recommend using Anaconda_ in the form of a minimal Miniconda_
installation with custom environments.

===========
 Old Notes
===========
Here are some old notes: I don't think this information is relevant
any more, but have not checked carefully, so I am keeping it here.

Anaconda_:
   Anaconda_, provided by `Continuum Analytics`_, is a complete python.
   distribution. There are free and professional versions with the
   latter being free for academic use.  Here are some features:

   * Conda_ package manager allows you to quickly install and remove components
     with commands like ``conda install openmpi``.  This also manages binary
     files, representing an improvement over the standard python package
     managing tools.
   * High performance computing tools like Numba_, NumbaPro_ (supports GPU
     programming), and Blaze_ (not yet ready for use).
   * Spyder_ is a complete development environment (IDE) if you prefer this type
     of thing.  I do not use it though.
   * Wakari_ is an online service that allows you to use and share Anaconda_
     environments.  It is free for 500MB and low-performance computing resources
     (you pay once you need more space or higher performance, which you can
     purchase through cloud computing services like Amazon's EC2_.)  Wakari_ is
     almost an ideal tool for developing an hosting course notes in the form of
     an `IPython notebook`_.  You can share this with others, who can view it (
     as with NBViewer_) but users can also click on a link to setup a Wakari_
     account where they can download and execute the notebook in a mater of
     minutes. An impressive feature is that, in the Wakari_ environment, you
     have access to a complete bash shell so that you can do whatever setup you
     need.

   One caveat is that Anaconda_ does not work all systems (it did not work on 32
   bit Mac for example) and lacks some of the GUI tools like Mayavi_ provided
   with |EPD|_. (It seems to me that lack of QT_ support is the issue.)

|EPD|_:
   The |EPD|_ provided by Enthought_ is another  complete python installation
   including NumPy_, SciPy_, matplotlib_, and many other useful tools.  There is
   both a free version, and a professional version -- the latter is free for
   academic use and includes some additional tools for analysis and
   visualization.

   * Canopy_ is a complete development environment (IDE) and package management
     system that seems to be Enthought_'s focus. If you prefer this type of
     thing, it is probably the way to go (I.e. as a MATLAB_ replacement).  I
     have not used it though.
   * The |EPD|_ seems be supported for more platforms (such as 32
     bit OS X) than Anaconda_.
   * The |EPD|_ provides visualizations tools like Mayavi_ that depend on QT_.

I do not know the details, but it seems that `Continuum Analytics`_ was setup by
some developers that left Enthought_ and aims to provide a set of high
performance tools, where as Enthought_ is moving more towards convenience tools
with their Canopy_ platform.

It is perfectly reasonably to install both Anaconda_ and |EPD|_: just adjust
your path to make sure you execute the appropriate version of python when you
want to switch between the two.

My suggestion is to install one (or both) of these rather than a custom python
installation -- they smooth over difficulties with compiling various libraries
required especially by matplotlib_.  The only real difficulty with maintaining
both is that you will need to install your needed libraries in each.

If you can, I would recommend creating appropriate virtual environments for
managing any additional packages you need to install.  In this way, you keep
your system python clean, have access to the latest tools in a complete
distribution (which is also kept clean) and can play with various package
combinations: Important, for example, if you want to make sure you understand
all of the dependencies of your code.

Note that Anaconda_ has its own environment manager instead of virtualenv_
called Conda_ so the setups are different.  We describe both flavours here.


1: Quick Start
==============

1.1: Anaconda_
--------------

I install Anaconda_ in ``/data/apps/anaconda/1.3.1`` which I symlink to
``/data/apps/anaconda/current``.  Add ``/data/apps/anaconda/current/bin`` to
your path.  Then use Conda_ to manage the equivalents of virtual environments,
but for now I am just using a "global" environment.  I needed to do the
following to get to a working state::

   conda update anaconda conda ipython pip sympy numexpr
   conda pip ipdb winpdb zope.interface mercurial
   conda pip psutil memory_profiler
   conda pip scikits.bvp1lg theano
   conda pip pp
   conda pip


1.2: |EPD|_
-----------

Here is the executive summary based on the |EPD|_:

* Install |EPD|_ (or Anaconda_), git_, and GSL_.
* Install virtualenv_ (and pip_ which is not provided by |EPD|_)::

   sudo easy_install pip
   sudo pip install virtualenv

  or down load virtualenv.py_ and replace `virtualenv` with
  `python virtualenv.py` below if you want to keep your base python installation
  pure.
* Install the virtual environments setup some aliases::

   virtualenv --system-site-packages --distribute ~/.python_environments/epd
   virtualenv --no-site-packages --distribute ~/.python_environments/clean
   virtualenv -p /usr/bin/python --system-site-packages --distribute \
              ~/.python_environments/sys
   virtualenv -p ~/usr/apps/anaconda/Current/bin/python \
              --system-site-packages --distribute \
              ~/.python_environments/anaconda

   cat >> ~/.bashrc <<EOF
   alias v.epd=". ~/.python_environments/epd/bin/activate"
   alias v.sys=". ~/.python_environments/sys/bin/activate"
   alias v.clean=". ~/.python_environments/clean/bin/activate"
   v.epd
   EOF

* Install Mercurial_::

   pip install hg

* If on a Mac, then fix ``pythonw``::

   mkdir -p ~/src/python/git
   cd ~/src/python/git
   #git clone http://github.com/gldnspud/virtualenv-pythonw-osx.git
   git clone http://github.com/nicholsn/virtualenv-pythonw-osx.git
   cd virtualenv-pythonw-osx
   deactivate; v.epd        # Make sure you use the appropriate virtualenv
   python install_pythonw.py /Users/mforbes/.python_environments/epd

* Activate your desired virtual environment and choose the set of requirements
  to install::

   v.epd
   pip install -r all.txt


2: Requirements
===============
Here is a list of various requirements obtained by running ``pip --freeze``.
These were intended to be used with virtualenv_ so I am not sure yet about
there relevance when using Anaconda_  These are all disjoint, so you can
pick and choose.

``doc.txt`` :
   Various documentation tools like Sphinx_ and associated packages.  I use this
   for both my code documentation and for things like my website.
``emacs.txt`` :
   Various tools for setting up my development environment (I use emacs)
   including checking tools.
``debug.txt`` :
   Debugging tools, including remote debuggers.
``profile.txt`` :
   Profiling tools for optimizing code.
``testing.txt`` :
   Testing tools including code coverage.
``vc.txt`` :
   Version control tools like mercurial and extensions
``misc.txt`` :
   Odds and ends.
``mmf.txt`` :
   My source packages for projects.  These will be installed as source
   distributions.
``all.txt`` :
   All of the above.

Here are some additional requirement files:

``EPD.txt`` :
   The list of requirements frozen from a fresh EPD_ install.
``freeze.txt`` :
   Snapshot of my system by running ``pip freeze > freeze.txt``
``bleeding-edge.txt`` :
   Installs NumPy_, SciPy_, and matplotlib_ from source.  Note: this does not
   work for some reason because |pip|_ fails to install some compiled
   libraries.  (The NumPy_ install will look fine, but SciPy_ will then fail.)
   Here is `a discussion.`__  To deal with this, first use |pip| to install this
   developmental version of NumPy_.  This will install the source.  Then go into
   the source directory and run ``python setup.py install
   --prefix=/path/to/virtualenv``.  I.e.::

      pip install --upgrade -r bleading-edge.txt
      cd ~/.python_environments/epd/src/numpy
      python setup.py install --prefix=~/.python_environments/epd
``mac.txt`` :
   Specific packages for Mac's.

__ http://stackoverflow.com/questions/12574604/scipy-install-on-mountain-lion-failing


3: Details (EPD)
================
Here are detailed instructions using |EPD|_:

1) Install a version of python.  Many systems have a version preinstalled, so
   this step is optional.  However, if you plan to do serious development, then
   I strongly recommend installing the |EPD|_.  There is a free version, and an
   almost full featured free version for academic use: You can also pay for a
   comercial version and recieve support.  The EPD_ is very complete, and just
   works on most common platforms and I highly recommend it.  Make sure you can
   run the version of python you desire.

   If you install the EPD_, then it will typically add something like the
   following to your ``~/.bash_login`` or ``~/.profile`` files::

      # Setting PATH for EPD-7.3-2
      # The orginal version is saved in .bash_login.pysave
      PATH="/Library/Frameworks/Python.framework/Versions/Current/bin:${PATH}"
      export PATH

      MKL_NUM_THREADS=1
      export MKL_NUM_THREADS

   (If you want to use a multithreaded version of ``numpy``, you will need to
   change the value of ``MKL_NUM_THREADS``.  See `this discussion`__.)

2) Create a virtualenv_.  This will allow you to install new packages in a
   controlled manner that will not mess with the system version (or the EPD_
   version).  You can create multiple virtual environments for different
   projects or associated with different versions of python.  Again, this is
   highly recommended.  There are several ways of doing this.

   .. note:: Methods 1) and 2) will install virtualenv_ to the location
      specified by the current version of python.  This means that you might
      need root access, and it will slightly "muck up" you pristine system
      install. This is generally not a problem, but if it bothers you see step
      3).

   1) If you have |pip|_ (the new python packageing system), then you can use it
      to install virtualenv_ as follows::

         pip install virtualenv

   2) If you do not have |pip|_, you might have ``easy_install``::

         easy_install virtualenv

   3) If you do not want to muck up your system version of python at all, then
      you can simply download the file |virtualenv.py|_.  In the commands that
      follow, replace ``virtualenv`` with ``python virtualenv.py``.

3) Setup a virtual environment for your work.  You can have many differen
   environments, so you will need to choose a meaningful name.  I use "epd" for
   the EPD_ version of python, "sys" for the system version of python, and
   "clean" for a version using EPD_ but without the site-packages::

       virtualenv --system-site-packages --distribute ~/.python_environments/epd
       virtualenv --no-site-packages --distribute ~/.python_environments/clean
       virtualenv -p /usr/bin/python --system-site-packages --distribute \
                  ~/.python_environments/sys

   Once this virtualenv_ is activated, install packages with pip_ will place all
   of the installed files in the ``~/.python_environments/epd`` directory.  (You
   can change this to any convenient location).  The ``--system-site-packages``
   option allows the virtualenv_ access to the system libraries (in my case, all
   of the EPD_ goodies).  If you want to test a system for deployment, making
   sure that it does not have any external dependencies, then you would use the
   ``--no-site-packages`` option instead.  Run ``virtualenv --help`` for more
   information.

4) Add some aliases to help you activate virtualenv_ sessions.  I include the
   following in my ``.bashrc`` file::

      # Some virtualenv related macros
      alias v.epd=". ~/.python_environments/epd/bin/activate"
      alias v.sys=". ~/.python_environments/sys/bin/activate"
      alias v.clean=". ~/.python_environments/clean/bin/activate"
      v.epd

   You can activate your chosen environment with one of the commands ``v.epd``,
   ``v.clean``, or ``v.sys``.  The default activation script will insert "(epd)"
   etc. to your prompt::

      ~ mforbes$ v.epd
      (epd)~ mforbes$ v.sys
      (sys)~ mforbes$ deactivate
      ~ mforbes$

   To get out of the environments, just type ``deactivate`` as shown above.

   .. note:: If you have an older version of IPython_ (pre 0.13), then you may
      need to call ``ipython`` from a `function like this`__::

         # Remap ipython if VIRTUAL_ENV is defined
         function ipython {
           if [ -n "${VIRTUAL_ENV}" -a -x "${VIRTUAL_ENV}/bin/python" ]; then
             START_IPYTHON='\
               import sys; \
               from IPython.frontend.terminal.ipapp import launch_new_instance;\
               sys.exit(launch_new_instance())'
              "${VIRTUAL_ENV}/bin/python" -c "${START_IPYTHON}" "$@"
            else
              command ipython "$*"
            fi
         }


      This deals with issues that IPython_ was not virtualenv_ aware.  The
      recommended solution is still to install IPython_ in the virtualenv_ using
      ``pip install ipython``, but then you will need one in each environment.
      As of IPython_ 0.13, this support is included. (See `this PR`__.)

      If you have not used IPython_ before, then you should have a look.  It has
      some fantastic features like ``%paste`` and the `IPython notebook`_
      interface.

5) Install mercurial_.  You may already have this (try ``hg --version``).  If
   not, either install a native distribution (which might have some GUI tools)
   or install with::

      pip install hg

6) Install git_.  This may not be as easy, but some packages are only available
   from github_.

7) On Mac OS X you may need to install ``pythonw`` for some GUI applications
   (like RunSnakeRun_).  You an do this using `this solution`__::

      mkdir -p ~/src/python/git
      cd ~/src/python/git
      git clone http://github.com/gldnspud/virtualenv-pythonw-osx.git
      cd virtualenv-pythonw-osx
      python install_pythonw.py /Users/mforbes/.python_environments/epd

   You will have to do this in each virtualenv_ you want to use the GUI apps
   from.

8) Non-python prerequisites.  These need to be installed outside of the python
   environment for some of the required libraries to work.

   * GSL_: This is needed for pygsl_.


9) Install various requirements as follows::

      pip install -r requirements/all.txt

.. These must not appear in the list because PyPI is stupid.

__ http://stackoverflow.com/q/5260068/1088938
__ http://igotgenes.blogspot.fr/2010/01/interactive-sandboxes-using-ipython.html
__ https://github.com/ipython/ipython/pull/1388/
__ https://github.com/gldnspud/virtualenv-pythonw-osx


4: Using |pip|_
===============
Here are some notes about using |pip|_ that I did not find obvious.

Version Control
---------------
It is clear from the `documentation about requirements`__ that you can specify
version controlled repositories with |pip|_, however, the exact syntax for
specifying revisions etc. is not so clear.  Examining `the source`__ shows that
you can specify revisions, tags, etc. as follows::

   # Get the "tip"
   hg+http://bitbucket.org/mforbes/pymmf#egg=pymmf

   # Get the revision with tag "v1.0" or at the tip of branch "v1.0"
   hg+https://bitbucket.org/mforbes/pymmf@v1.0#egg=pymmf

   # Get the specified revision exactly
   hg+https://bitbucket.org/mforbes/pymmf@633be89a#egg=pymmf

What appears after the "@" sign is any valid revision (for mercurial see ``hg
help revision`` for various options).  Unfortunately, I see no way of specifying
something like ">=1.1", or ">=633be89a" (i.e. a descendent of a particular
revision).  (See `issue 782`__)

__ http://www.pip-installer.org/en/latest/requirements.html
__ https://github.com/pypa/pip/blob/develop/pip/vcs/mercurial.py
__ https://github.com/pypa/pip/issues/728

5: Using the MKL
================
The EPD_ is built using the Intel MKL_.  Here are some instructions on how to
compile your own version of `NumPy and SciPy with the MKL`__.

__ http://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl

* Checkout the source code::

     pip install --no-install -e git+http://github.com/numpy/numpy#egg=numpy-dev
     pip install --no-install -e git+http://github.com/scipy/scipy#egg=scipy-dev

* Setup the environment to use the `Intel compilers`_::

     . /usr/local/bin/intel64.sh
     . /opt/intel/Compiler/11.1/069/mkl/tools/environment/mklvarsem64t.sh

* Edit the ``site.cfg`` file in the NumPy_ source directory.  I am not sure
  exactly which libraries to include. See these discussions:

     * http://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl
     * Check the ``site.cfg`` in your EPD_ installation.

  .. code::

     cd ~/.python_environments/epd/src/numpy
     cp site.cfg.example site.cfg
     vi site.cfg

  Here is what I used::

     [mkl]
     library_dirs = /opt/intel/Compiler/11.1/069/mkl/lib/em64t/
     include_dirs = /opt/intel/Compiler/11.1/069/mkl/include
     lapack_libs = mkl_lapack95_lp64
     mkl_libs = mkl_def, mkl_intel_lp64, mkl_intel_thread, mkl_core, mkl_mc

  I also needed to modify ``numpy/distutils/intelccompiler.py`` as follows::

          cc_args = "-fPIC"
          def __init__ (self, verbose=0, dry_run=0, force=0):
              UnixCCompiler.__init__ (self, verbose,dry_run, force)
     -        self.cc_exe = 'icc -m64 -fPIC'
     +        self.cc_exe = 'icc -O3 -g -openmp -m64 -fPIC'
              compiler = self.cc_exe
              self.set_executables(compiler=compiler,
                                   compiler_so=compiler,

* Build both NumPy_ and SciPy_ with the following::

     cd ~/.python_environments/epd/src/numpy
     python setup.py config --compiler=intelem --fcompiler=intelem\
                 build_clib --compiler=intelem --fcompiler=intelem\
                 build_ext --compiler=intelem --fcompiler=intelem\
                 install
     cd ~/.python_environments/epd/src/scipy

* Run and check the build configuration::

     $ python -c "import numpy;print numpy.__file__;print numpy.show_config()"
     /phys/users/mforbes/.python_environments/epd/lib/python2.7/site-packages/numpy/__init__.pyc
     lapack_opt_info:
         libraries = ['mkl_lapack95_lp64', 'mkl_def', 'mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'mkl_mc', 'pthread']
         library_dirs = ['/opt/intel/Compiler/11.1/069/mkl/lib/em64t/']
         define_macros = [('SCIPY_MKL_H', None)]
         include_dirs = ['/opt/intel/Compiler/11.1/069/mkl/include']
     blas_opt_info:
         libraries = ['mkl_def', 'mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'mkl_mc', 'pthread']
         library_dirs = ['/opt/intel/Compiler/11.1/069/mkl/lib/em64t/']
         define_macros = [('SCIPY_MKL_H', None)]
         include_dirs = ['/opt/intel/Compiler/11.1/069/mkl/include']
     lapack_mkl_info:
         libraries = ['mkl_lapack95_lp64', 'mkl_def', 'mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'mkl_mc', 'pthread']
         library_dirs = ['/opt/intel/Compiler/11.1/069/mkl/lib/em64t/']
         define_macros = [('SCIPY_MKL_H', None)]
         include_dirs = ['/opt/intel/Compiler/11.1/069/mkl/include']
     blas_mkl_info:
         libraries = ['mkl_def', 'mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'mkl_mc', 'pthread']
         library_dirs = ['/opt/intel/Compiler/11.1/069/mkl/lib/em64t/']
         define_macros = [('SCIPY_MKL_H', None)]
         include_dirs = ['/opt/intel/Compiler/11.1/069/mkl/include']
     mkl_info:
         libraries = ['mkl_def', 'mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'mkl_mc', 'pthread']
         library_dirs = ['/opt/intel/Compiler/11.1/069/mkl/lib/em64t/']
         define_macros = [('SCIPY_MKL_H', None)]
         include_dirs = ['/opt/intel/Compiler/11.1/069/mkl/include']
     None

  .. note:: You will need to setup the environment to run with the MKL_
     libraries.  The EPD_ avoids this by distributing the libraries.  I suggest
     that you add the following to the activation script::

        cat >> ~/.python_environments/epd/bin/activate <<EOF

        # This adds the MKL libraries to the path for use with my custom numpy
        # and scipy builds.
        . /usr/local/bin/intel64.sh
        . /opt/intel/Compiler/11.1/069/mkl/tools/environment/mklvarsem64t.sh
        EOF


See also:

  * http://math.nju.edu.cn/help/mathhpc/doc/intel/mkl/mklgs_lnx.htm
  * http://blog.sun.tc/2010/11/numpy-and-scipy-with-intel-mkl-on-linux.html
  * http://www.scipy.org/Installing_SciPy/Linux

    This suggests maybe using the runtime libraries instead (just ``mkl_libs =
    mkl_rt``).  I have not yet tried this.

  * http://cournape.github.com/Bento/

    It looks like it might be easier to use Bento_ rather than distutils

================
 Other Software
================
This section describes various other pieces of software that I use that interact
with python.

1: pyaudio_
===========
pyaudio_ is a python interface to the PortAudio_ library for generating sounds
and sound files.  To do real-time sound generation, one really needs to
non-blocking interface (otherwise, the delay between blocking calls will affect
the signal in a manner that is difficult to compensate for).  Unfortunately, the
default builds require Mac OS X 10.7 or higher.

2: reStructuredText_
====================
I like to write my local documentation in reStructuredText_ (such as this
file).  As I often use math, I make the default role ``:math:```` and use
MathJax_.  Here is an example:

.. code:: rst

   .. default-role:: math

   Now I can type math like this: `E=mc^2` or in an equation line this

   .. math::
      \int_0^1 e^{x} = e - 1

.. note::
   Now I can type math like this: `E=mc^2` or in an equation line this

   .. math::
      \int_0^1 e^{x} = e - 1

In order to work offline, I install MathJax_ locally using the IPython_ as
`described here`__:

.. code:: python

   from IPython.external.mathjax import install_mathjax
   install_mathjax()

__ https://github.com/ipython/ipython/pull/714

This installs it in
``~/.python_environments/epd/lib/python2.7/site-packages/IPython/frontend/html/notebook/static/mathjax``
which can be used locally.  I symlink it to ``~/.mathjax``, but you must find a
way to inject the stylesheet into your HTML.  One way is with the ``.. raw::
html`` directive:

.. code:: html

   .. raw:: html

      <script type="text/javascript"
       src="/Users/mforbes/.mathjax/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
      </script>

3: Profiling
============
This page has a great discussion of line and memory profiling:

* http://scikit-learn.org/dev/developers/performance.html


4: Emacs_
=========

I use Emacs_ as my principle editor and like to have access to syntax
highlighting, auto-completion etc. Thus, I typically install the following
packages, but these are not completely straightforward.

4.1: Pymacs_
------------

Pymacs_ allows Emacs_ to access the python interpreter and is used by Ropemacs_
to provide some nice features like code checking. The source appears not to be
pip_ installable, so you must download it and run ``make`` as follows:

.. code:: bash

   git clone http://github.com/pinard/Pymacs.git
   cd Pymacs
   make
   pip install -e .


5: Anaconda_
============
Anaconda_ provides a very nice python system, especially with the Conda_ package
management tool, but there are a few problems:

1) No installation for 32-bit Mac OS X systems.  (No longer an issue for me
   since I finally have a 64 bit machine.)
2) No Mayavi_.  This means that I must maintain an EPD_ 32-bit installation as
   well (with all my required packages) in order to visualize.

Creating Packages
-----------------
As an example, here we create a Conda_ package for installing the FFTW_ and
related software.  We start with a fresh Anaconda_ installation: (this command
would show if we have any packages installed that are not managed by Conda_)

   $ conda package --untracked
   prefix: /data/apps/anaconda/1.3.1

Now we manually install the FFTW_ etc.::

   cd ~/src
   wget http://www.fftw.org/fftw-3.3.3.tar.gz
   wget http://www.fftw.org/fftw-3.3.3.tar.gz.md5sum
   md5 fftw-3.3.3.tar.gz           # Check that this is okay
   tar -zxvf fftw-3.3.3.tar.gz
   cd fftw-3.3.3

   # Build and install the single, double, long-double
   # and quad-precision versions
   PREFIX=/data/apps/anaconda/current/
   for opt in " " "--enable-sse2 --enable-single" \
                  "--enable-long-double" "--enable-quad-precision"; do
     ./configure --prefix="${PREFIX}"\
                 --enable-threads\
                 --enable-shared\
                 $opt
     make -j8 install
   done

   # Note: this needs a patch to work on Mac OS X
   # https://code.google.com/p/anfft/issues/detail?id=4
   export FFTW_PATH=/data/apps/anaconda/current/lib/
   pip install --upgrade anfft pyfftw


These are untracked::

   $ conda package --untracked
   prefix: /data/apps/anaconda/1.3.1
   bin/fftw-wisdom
   ...
   include/fftw3.f
   ...
   lib/libfftw3.3.dylib
   ...
   lib/pkgconfig/fftw3.pc
   ...
   lib/python2.7/site-packages/Mako-0.7.3-py2.7.egg-info/PKG-INFO
   ...
   lib/python2.7/site-packages/anfft-0.2-py2.7.egg-info/PKG-INFO
   ...
   lib/python2.7/site-packages/pyFFTW-0.9.0-py2.7.egg-info/PKG-INFO
   ...
   lib/python2.7/site-packages/pyfftw/__init__.py
   ...
   share/info/fftw3.info
   ...
   share/man/man1/fftw-wisdom-to-conf.1
   ...

These can be bundled into a new package that can later be installed directly::

   $ conda package --pkg-name=fftw --pkg-version=3.3.3
   prefix: /data/apps/anaconda/1.3.1
   Number of files: 82
   fftw-3.3.3-py27_0.tar.bz2 created successfully

==========
 Problems
==========

I had problems installing a virtual environment with Anaconda_.  *Don't do
this!*  Use Conda_ instead.
