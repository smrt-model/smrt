####################################
Guidelines for Developers
####################################

At the moment this is an organic document to collect all the model design and developer style decisions. This will also
include information on how to get started with useful developer tools. At the moment, it contains personal experience of installing and using these tools although these may be removed if they do not appear to be useful to others.

These guidelines will be turned into a formal document towards the end of the project.

Use of import statements
-------------------------

`Good rules for python imports <http://stackoverflow.com/questions/193919/what-are-good-rules-of-thumb-for-python-imports>`_

In short:

* use fully qualified names
* :code:`from blabla import *` should never be used.
* :code:`from blabla import passive` should be avoided in SMRT but can be used in user code.
* keep at least the module e.g. "from smrt import sensor_list" is the best compromise.
* use "as" with moderation and everyone should agree to use it.
* but :code:`import numpy as np` is good.
* to start, we will use an explicit import at the top of the driver file, making the code more cumbersome, but may later consider a plugin framework to do the import and introspection in a nice way.


Note: it's part of the Google Python style guides that
all imports must import a module, not a class or function from that
module. There are way more classes and functions than there are modules,
so recalling where a particular thing comes from is much easier if it is
prefixed with a module name. Often multiple modules happen to define
things with the same name -- so a reader of the code doesn't have to go
back to the top of the file to see from which module a given name is
imported.


Python
----------------------------------------
Python was chosen because of its growing use in the scientific community and higher flexibility than compiled legacy languages like FORTRAN. This enables the model to be modular much more easily, which is a main constraint of the project,  allows faster development and an easier exploration of new ideas. The performance should not be an issue as the time consuming part of the model should be localized in the RT solver and numerical integrations which uses the highly optimized scipy module facility that basically uses BLAS, LAPACK and MINPACK libraries as would be done in FORTRAN. Compilation of the Python code with Numba or Pypy will be considered in case of performance issues later in the project or even more probably after. Parallelization could be done later e.g. through joblib module.

The model in the framework of the current project mainly aims at exploring new ideas involving the microstructure and tests various modelling solutions. It is quite likely that operational needs (especially very intensive ones) will require rewritting a selected subset of the model.

Python versions
^^^^^^^^^^^^^^^^
The target version is Python 3.4+ which is better optimized and is the only supported version in the future (after 2020) with the use of a subset syntax to ensure compatibility with the lastest 2.7.x and PyPy. It means in practice that the model will be compatible with the last 2.7.x version but is “ready” for Python 3 and later. For this  "__future__" directives and six module will be used. The tests must pass the two versions. This choice is overall a weak constraint for developers and big asset for users.

Anaconda is probably the easiest way to install python, especially when several versions are needed. See also `Installing multiple versions of python <http://stackoverflow.com/questions/2547554/official-multiple-python-versions-on-the-same-machine>`_ is system dependent and also `depends on your preferred install method <http://stackoverflow.com/questions/2812520/pip-dealing-with-multiple-python-versions>`_.

Perhaps it's not strictly necessary to follow all steps, but I followed these `instructions for Mac OSX <https://iainhunter.wordpress.com/2012/11/08/howto-install-python3-pip3-tornado-on-mac/>`_ to install python 3.5. Then ``pip`` installs 
packages into python 2.7 and ``pip3`` installs packages into python 3.5. `Note on Tcl/Tk for Mac OSX <https://www.python.org/download/mac/tcltk/>`_. I have installed ActiveTcl 8.6.4 and am keeping my fingers crossed that these changes have not broken anything...I have subsequently installed python 3.4.3. This means that ``python3`` will run version 3.4.3 by default. It doesn't seem trivial to get ``python3`` to point back to python 3.5, but that's probably ok as the target version is 3.4, and it will be worth testing for 3.5 alongside.

tox: testing multiple python versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`The tox package <https://tox.readthedocs.org/en/latest/>`_ allows multiple versions of python to be tested. Although not clear whether this needs to be installed in python 2 or 3, I installed with ``pip`` rather than ``pip3`` and trust that it will take care of everything. This seems to work fine.

The setup to run tox is contained in the tox.ini file. At the moment this is setup for nosetests against python versions 2.7, 3.4 and 3.5. Also, at present `tox.ini does not require a setup.py to run <http://stackoverflow.com/questions/18962403/how-do-i-run-tox-in-a-project-that-has-no-setup-py>`_. Once the model is fully operational the line ``skipsdist = True`` should be deleted, or this parameter set to False. Note that all modules to be imported need to be listed in the dependencies (deps) in the tox.ini file. An ImportError may indicate that the module it is trying to import has not been included in the tox.ini.

To run the nosetests for all the different versions, using the installed tox package, simply type::

    tox

If you want to test for only one python version, type e.g::

    tox -e py27


setup.py
------------------------

This is needed in order to build, install and distribute the model through Distutils (`instructions <https://docs.python.org/2/distutils/setupscript.html>`_). To be done for the public release.


bug correction
-----------------

Every bug should result in writing a test.


Classes
---------

If the compulsary argument list becomes too long (say 4?), use optional arguments to make things easier to read.

`Guidelines on number of parameters a function should take <http://programmers.stackexchange.com/questions/145055/are-there-guidelines-on-how-many-parameters-a-function-should-accept>`_.

`Merge two objects in python <http://byatool.com/lessons/simple-property-merge-for-python-objects/>`_.

PEP008
---------

Code must conform to `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ - with the exception that lines of up to 140 characters are allowed and extra space are allowed in long formula for readability. Particular points of note:

- 4 spaces for the indentation.
- one space after comma and around operators.
- all names (variable, function, …) are meaningful. Abbreviations are used in a very limited number of cases.
- function names are lowercase only and word a spaced by underscore.
- Constants are usually defined on a module level and written in all capital letters with underscores separating words

You can check for PEP8 compliance automatically with nosetests. To do this, install `tissue <https://code.activestate.com/pypm/tissue/>`_ and pep8. Then type::

    nosetests --with-tissue --tissue-ignore=E501

or::

    nosetests --with-tissue --tissue-ignore=E501 **specific filename**

to run nosetests with the pep8 checks. As we have allowed 140 characters per line, the E501 longer line warning needs to be suppressed.

Sphinx
---------
Documentation is done in-code, and is automatically generated with `Sphinx <http://www.sphinx-doc.org/en/stable/>`_. If no new modules are added, generate the rst and html documentation from the in-code Sphinx comments, by typing (whilst in smrt/doc directory)::

    make fullhtml

The documentation can be accessed via the index.html page in the smrt/doc/build/html folder.

If you have math symbols to be displayed, this can be done with the imgmath extension (already used), which generates a png and inserts the image at the appropriate place. You may need to set the path to latex and dvipng on your system. From the source directory, this can be done with e.g.::

    sphinx-build -b html -D imgmath_latex=/sw/bin/latex -D imgmath_dvipng=/sw/bin/dvipng . ../build/html

or to continue to use ::code:`make html` or :code:`make fullhtml`, by setting your path (C-shell) e.g.::

    set path = ($path /sw/bin)

or bash::

    PATH=$PATH:/sw/bin

.. note::

    Math symbols will need double backslashes in place of the single backslash used in latex.

To generate a list of undocumented elements, whilst in the *source* directory::

    sphinx-build -b coverage . coverage

The files will be listed in the *coverage/python.txt* file
