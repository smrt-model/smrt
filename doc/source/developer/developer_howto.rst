####################################
How to contribute
####################################

Python Versions
^^^^^^^^^^^^^^^

The target version is currently Python 3.10 minimum as of 2025 and will increase at the pace of Python's end of life.

Conda is probably the easiest way to install Python, especially when several versions are needed.



Git
---

- The development by the core developers is done on the master branch (as of 2025), although this practice may change in the future. Other developers can either use branches (if rights granted) or use pull requests (for all users). The latter is recommended.
- If using branches, name branches explicitly, e.g., 'feature/changes-being-made-JD', to avoid name conflicts with other developers.

Bug Correction
--------------

Every bug found and corrected should result in writing a unit test to prevent the bug from reappearing (regression).

Documentation Generation with Sphinx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation is generated automatically after each push to GitHub by Read the Docs. It is requested to check that the online documentation is well rendered after every major change.

However, it is also possible to generate the documentation locally using `Sphinx <http://www.sphinx-doc.org/en/stable/>`_. If no new module is added, it is simple to generate the rst and HTML documentation by typing (from the smrt/doc directory)::

    make fullhtml

The documentation can then be accessed via the index.html page in the smrt/doc/build/html folder.

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