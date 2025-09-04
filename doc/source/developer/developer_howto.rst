####################################
How to contribute
####################################

Python Versions
^^^^^^^^^^^^^^^

The target version is currently Python 3.10 minimum as of 2025 and will increase at the pace of Python's end of life.

Conda is probably the easiest way to install Python, especially when several versions are needed.

Install an editable version
---------------------------

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/smrt-model/smrt.git
    cd smrt
    ```

2.  **Create and activate a virtual environment**
    
    This can be done for example with `venv` but please refer to https://docs.python.org/3/library/venv.html if this is new to you. Most IDE have their own way of generating virtual environments, which may be easier than using `venv`.

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```
    

3.  **Install the project in editable mode**:
    ```bash
    pip install -e '.[dev]'
    ```
    This command installs the package in "editable" mode, meaning any changes you make to the source code will be immediately reflected without needing to reinstall. Additional dependencies for building the documentation are also installed.


Git
---

- The development by the core developers is done on the master branch (as of 2025), although this practice may change in the future. Other developers can either use branches (if rights granted) or use pull requests (for all users). The latter is recommended.
- When using branches, name branches explicitly, e.g., 'feature/changes-being-made-JD', to avoid name conflicts with other developers.

Bug Correction
--------------

Every bug found and corrected should result in writing a unit test to prevent the bug from reappearing (regression).

Documentation Generation with Sphinx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation is generated automatically on `ReadTheDocs <https://smrt.readthedocs.io/>`_ for the last version and the ``master`` branch. It is also possible to generate the documentation locally using `Sphinx <http://www.sphinx-doc.org/en/stable/>`_. If no new module is added, it is simple to generate the rst and HTML documentation by typing (from the smrt/doc directory)::

    make fullhtml

The documentation can then be accessed via the index.html page in the smrt/doc/build/html folder. It is requested to check that the documentation is well rendered after every major change.

If you have math symbols to be displayed, this can be done with the imgmath extension (already used), which generates a PNG and inserts the image at the appropriate place. You may need to set the path to LaTeX and dvipng on your system. From the source directory, this can be done with, e.g.::

    sphinx-build -b html -D imgmath_latex=/sw/bin/latex -D imgmath_dvipng=/sw/bin/dvipng . ../build/html

Or to continue to use ``make html`` or ``make fullhtml``, by setting your path (C-shell), e.g.::

    set path = ($path /sw/bin)

Or for bash::

    PATH=$PATH:/sw/bin

.. note::

    Math symbols will need double backslashes in place of the single backslash used in LaTeX.

To generate a list of undocumented elements, while in the *source* directory::

    sphinx-build -b coverage . coverage

The files will be listed in the *coverage/python.txt* file.