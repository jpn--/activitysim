
Getting Started
===============

This page describes how to get started with ActivitySim.

.. note::
   ActivitySim is under active development


.. index:: installation


Pre-packaged Installer
----------------------

Beginning with version 1.2, ActivitySim is now available for Windows via a
pre-packaged installer.  This installer provides everything you need to run
ActivitySim, including Python, all the necessary supporting packages, and
ActivitySim itself.  You should only choose this installation process if you
plan to use ActivitySim but you don't need or want to do other Python
development.  Note this installer is provided as an "executable" which (of course)
installs a variety of things on your system, and it is quite likely to be flagged by
Windows, anti-virus, or institutional IT policies as "unusual" software, which
may require special treatment to actually install and use.

Download the installer from GitHub `here <https://github.com/ActivitySim/activitysim/releases/download/v1.3.1/Activitysim-1.3.1-Windows-x86_64.exe>`_.
It is strongly recommended to choose the option to install "for me only", as this
should not require administrator privileges on your machine.  Pay attention
to the *complete path* of the installation location. You will need to know
that path to run ActivitySim in the future, as the installer does not modify
your "PATH" and the location of the `ActivitySim.exe` command line tool will not
be available without knowing the path to where the install has happened.

Once the install is complete, ActivitySim can be run directly from any command
prompt by running `<install_location>/Scripts/ActivitySim.exe`.


Installation
------------

1. It is recommended that you install and use a *conda* package manager
for your system. One easy way to do so is by using
`Mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`__.
Conda is a free open source cross-platform package manager that runs on
Windows, OS X and Linux. Modern versions of conda have significantly improved
performance compared to older versions.

Alternatively, if you prefer a package installer backed by corporate tech
support available (for a fee) as necessary, you can install
`Anaconda 64bit Python 3 <https://www.anaconda.com/distribution/>`__,
although you should consult the `terms of service <https://www.anaconda.com/terms-of-service>`__
for this product and ensure you qualify since businesses and
governments with over 200 employees do not qualify for free usage.
You can use `conda` for all the commands below, as it now has the same
performance and capabilities that were previously only available in mamba.

2. If you access the internet from behind a firewall, then you may need to
configure your proxy server. To do so, create a `.condarc` file in your
home installation folder, such as:

::

  proxy_servers:
    http: http://myproxy.org:8080
    https: https://myproxy.org:8080
  ssl_verify: false

3. Create a conda environment (basically a Python install just for this project)
using conda Prompt (on Windows) or the terminal (macOS or Linux)::

  conda create -n asim python=3.10 activitysim -c conda-forge --override-channels

This command will create the environment and install all the dependencies
required for running ActivitySim.  It is only necessary to create the environment
once per machine, you do not need to (re)create the environment for each session.
If you would also like to install other tools or optional dependencies, it is
possible to do so by adding additional libraries to this command.  For example::

  conda create -n asim python=3.10 activitysim jupyterlab larch -c conda-forge --override-channels

This example installs a specific version of Python, version 3.10.  A similar
approach can be used to install specific versions of other libraries as well,
including ActivitySim, itself. For example::

  conda create -n asim python=3.10 activitysim=1.0.2 -c conda-forge --override-channels

Additional libraries can also be installed later.  You may want to consider these
tools for certain development tasks::

  # packages for testing
  conda install pytest pytest-cov coveralls black flake8 pytest-regressions -c conda-forge --override-channels -n asim

  # packages for building documentation
  conda install sphinx numpydoc sphinx_rtd_theme==0.5.2 -c conda-forge --override-channels -n asim

  # packages for estimation integration
  conda install larch -c conda-forge --override-channels -n asim

  # packages for example notebooks
  conda install jupyterlab matplotlib geopandas descartes -c conda-forge --override-channels -n asim

To create an environment containing all these optional dependencies at once, you
can run the shortcut command

::

  conda env create activitysim/ASIM -n asim

4. To use the **asim** environment, you need to activate it

::

  conda activate asim

The activation of the correct environment needs to be done every time you
start a new session (e.g. opening a new conda Prompt window).

Alternative Installation Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to install ActivitySim without conda, it is possible to
do so with pip, although you may find it more difficult to get all of the
required dependencies installed correctly.  If you can use conda for
the dependencies, you can get most of the libraries you need from there::

  # required packages for running ActivitySim
  conda install cytoolz numpy pandas psutil pyarrow numba pytables pyyaml openmatrix requests -c conda-forge

  # required for ActivitySim version 1.0.1 and earlier
  pip install zbox

And then simply install just activitysim with pip.

::

  python -m pip install activitysim

If you are using a firewall you may need to add ``--trusted-host pypi.python.org --proxy=myproxy.org:8080`` to this command.

For development work, can also install ActivitySim directly from source. Clone
the ActivitySim repository, and then from within that directory run::

  python -m pip install . -e

The "-e" will install in editable mode, so any changes you make to the ActivitySim
code will also be reflected in your installation.

Installing from source is easier if you have all the necessary dependencies already
installed in a development conda environment.  Developers can create an
environment that has all the optional dependencies preinstalled by running::

  conda env create activitysim/ASIM-DEV

If you prefer to use a different environment name than `ASIM-DEV`, just
append `--name OTHERNAME` to the command. Then all that's left to do is install
ActivitySim itself in editable mode as described above.

.. note::

  ActivitySim is a 64bit Python 3 library that uses a number of packages from the
  scientific Python ecosystem, most notably `pandas <http://pandas.pydata.org>`__
  and `numpy <http://numpy.org>`__.

  As mentioned above, the recommended way to get your own scientific Python installation is to
  install 64 bit Anaconda, which contains many of the libraries upon which
  ActivitySim depends + some handy Python installation management tools.

  Anaconda includes the ``conda`` command line tool, which does a number of useful
  things, including creating `environments <http://conda.pydata.org/docs/using/envs.html>`__
  (i.e. stand-alone Python installations/instances/sandboxes) that are the recommended
  way to work with multiple versions of Python on one machine.  Using conda
  environments keeps multiple Python setups from conflicting with one another.

  You need to activate the activitysim environment each time you start a new command
  session.  You can remove an environment with ``conda remove -n asim --all`` and
  check the current active environment with ``conda info -e``.

  For more information on Anaconda, see Anaconda's `getting started
  <https://docs.anaconda.com/anaconda/user-guide/getting-started>`__ guide.

Run the Primary Example
-----------------------

ActivitySim includes a :ref:`cli` for creating examples and running the model.

To setup and run the primary example (see :ref:`examples`), do the following:

* Open a command prompt
* If you installed ActivitySim using conda environments, activate the conda
  environment with ActivitySim installed (i.e. ``conda activate asim``)
* Or, if you used the :ref:`pre-packaged installer<Pre-packaged Installer>`,
  replace all the commands below that call ``activitysim ...`` with the complete
  path to your installed location, which is probably something
  like ``c:\programdata\activitysim\scripts\activitysim.exe``.
* Type ``activitysim create -e prototype_mtc -d test_prototype_mtc`` to copy
  the very small prototype_mtc example to a new test_prototype_mtc directory
* Change to the test_prototype_mtc directory
* Type ``activitysim run -c configs -o output -d data`` to run the example
* Review the outputs in the output directory

.. note::
   Common configuration settings can be overridden at runtime.  See ``activitysim -h``, ``activitysim create -h`` and ``activitysim run -h``.
   ActivitySim model runs can be configured with settings file inheritance to avoid duplicating settings across model configurations.  See :ref:`cli` for more information.

Additional examples, including the full scale prototype MTC regional demand model, estimation integration examples, multiple zone system examples,
and examples for agency partners are available for creation by typing ``activitysim create -l``.  To create these examples, ActivitySim downloads the (large) input files from
the `ActivitySim resources <https://github.com/rsginc/activitysim_resources>`__ repository.  See :ref:`examples` for more information.

Try the Notebooks
-----------------

ActivitySim includes a `Jupyter Notebook <https://jupyter.org>`__ recipe book with interactive examples.  To run a Jupyter notebook, do the following:

* Open a conda prompt and activate the conda environment with ActivitySim installed
* If needed, ``conda install jupyterlab`` so you can run jupyter notebooks
* Type ``jupyter notebook`` to launch the web-based notebook manager
* Navigate to the ``examples/prototype_mtc/notebooks`` folder and select a notebook to learn more:

  * `Getting started <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/notebooks/getting_started.ipynb/>`__
  * `Summarizing results <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/notebooks/summarizing_results.ipynb/>`__
  * `Testing a change in auto ownership <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/notebooks/change_in_auto_ownership.ipynb/>`__
  * `Adding TNCs <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/notebooks/adding_tncs.ipynb/>`__
  * `Memory usage <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/notebooks/memory_usage.ipynb/>`__

Hardware
--------

The computing hardware required to run a model implemented in the ActivitySim framework generally depends on:

* The number of households to be simulated for disaggregate model steps
* The number of model zones (for each zone system) for aggregate model steps
* The number and size of network skims by mode and time-of-day
* The number of zone systems, see :ref:`multiple_zone_systems`
* The desired runtimes

ActivitySim framework models use a significant amount of RAM since they store data in-memory to reduce
data access time in order to minimize runtime.  For example, the prototype MTC example model has 2.7 million
households, 7.5 million people, 1475 zones, 826 network skims and has been run between one hour and one day depending
on the amount of RAM and number of processors allocated.  See :ref:`multiprocessing` and :ref:`chunk_size` for more information.

.. note::
   ActivitySim has been run in the cloud, on both Windows and Linux using
   `Microsoft Azure <https://azure.microsoft.com/en-us/>`__.  Example configurations,
   scripts, and runtimes are in the ``other_resources\example_azure`` folder.

.. _mkl_settings :

MKL Settings
~~~~~~~~~~~~

Anaconda Python on Windows uses the `Intel Math Kernel Library <https://software.intel.com/en-us/mkl>`__ for
many of its computationally intensive low-level C/C++ calculations.  By default, MKL threads many of its routines
in order to be performant out-of-the-box.  However, for ActivitySim multiprocessing, which processes households in
parallel since they are largely independent of one another, it can be advantageous to override threading within
processes and instead let ActivitySim run each process with one computing core or thread.  In order to do so,
override the MKL number of threads setting via a system environment variable that is set before running the model.
In practice, this means before running the model, first set the MKL number of threads variable via the command
line as follows: ``SET MKL_NUM_THREADS=1``
