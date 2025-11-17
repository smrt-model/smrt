####################################
Coding rules and style
####################################

Coding Style, PEP 8, and Ruff
-----------------------------

Coding style in SMRT is strict in order to reduce the maintenance burden, facilitate developer onboarding, and ensure long-term compatibility by minimizing changes.

At a minimum, code must conform to `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ with the exception that lines of up to 140 characters are allowed and extra spaces are allowed in long formulas for readability. Particular points of note:

- 4 spaces for indentation.
- One space after commas and around operators.

From 2025, SMRT is moving to Ruff formatting to conform with the Python community's best practices. This eliminates the need to manually format the code and learn all the rules. This avoids arguments about the best format; everyone uses Ruff.

Exceptions to Ruff formatting are only allowed for equations because their writing often carries physical meaning that is important to keep in the code. The developers are responsible for adding tags (fmt: off / fmt: on) to prevent Ruff from automatically formatting these sections. Any developper is allowed to apply ruff to all the code base and commit the changes, this is why protecting equations must be done at the first implementation.

Use pre-commit is strongly recommended to automatically check the code before each commit. To install pre-commit, run the following commands in your development environment:

    pip install pre-commit  # or conda install pre-commit or uvm install pre-commit
    pre-commit install
    pre-commit run --all-files   # test-run the hooks on the whole repo


Local Rules
^^^^^^^^^^^

- All names of files, public functions, public methods, classes, and arguments must be meaningful and carefully crafted. SMRT's usability by beginners and long-term maintainability strongly depend on the quality of these names, especially those used daily by the users (e.g., those in the smrt/inputs directory). For scientific code, it is recommended to suffix the file and function names with the reference to the main publication describing this formulations (first author + year with 2 digits). This avoids future name conflicts when new formulations are added.
- Abbreviations should never be used except in a very limited number of cases. They must be discussed with the community to balance the pros and cons and evaluate the readibility and long-term maintainability.
- The plural form in names is not recommended unless the ambiguity is too strong without it. In any case, it is not allowed to name two variables, one with a plural and the other singular, in the same file. E.g. use the prefix _list (or _set or _array) to designate multiple elements.
- Function, method, and file names are lowercase only, and words are separated by underscores. Even abbreviations are lowercase. In contrast, class names use CamelCase.
- Constants are usually defined at the module level (or in smrt.core.globalconstants) and written in all capital letters with underscores separating words.
- It is recommended to use optional arguments with sensitive default values for beginners rather than positional arguments. The primary mode of extension in SMRT is by offering options in the functions (see inputs/make_medium.py). If the positional argument list becomes too long (say >4), optional arguments must be used. `Guidelines on the number of parameters a function should take <http://programmers.stackexchange.com/questions/145055/are-there-guidelines-on-how-many-parameters-a-function-should-accept>`_. `Merge two objects in Python <http://byatool.com/lessons/simple-property-merge-for-python-objects/>`_.
- Naming of local variables in functions is left to the developer's preference. However, it is recommended to use the same symbols as in the publications for scientific code and to use meaningful names in other parts instead of long comments.
- Use f-strings (see also the Python tutorial) to format strings with variables from the code. Only use concatenation (+) between strings in exceptional cases.

Type Hinting
------------

Type hinting is an objective for 2025. The first objective is to define high-level types to handle the high flexibility of SMRT functions. Indeed, many functions accept arguments with a diversity of type; often NumPy arrays, scalars, or sequences and sometimes even strings, SMRT objects, and dictionaries. While this provides flexibility to the user (and in 2015 this was considered a big advantage of the dynamic language Python, today less...), this makes type hinting more difficult today. Work in progress.

Use of Import Statements
------------------------

`Good rules for Python imports <http://stackoverflow.com/questions/193919/what-are-good-rules-of-thumb-for-python-imports>`_

In short:

- Use fully qualified names as much as possible.
- It is recommend to use an explicit import at the top of the driver file, making the code more cumbersome, but may later consider a plugin framework to handle imports and introspection in a better way. Lazy import would be ideal... but does not exist in Python standard library.
- Wildcard import as ``from blabla import *`` are never be used.
- ``from blabla import passive`` should be avoided in SMRT but can be used in user code. Keep at least the module, e.g., "from smrt import sensor_list" is the best compromise.
- Use "as" with moderation, and everyone should agree to use it.
- Except for well-known libraries such as  ``import numpy as np``.
- ruff is used to sort imports (usually: standard library, external imports, internal imports)


.. note::

    It's part of the Google Python style guides that all imports must import a module, not a class or function from that module. There are way more classes and functions than there are modules, so recalling where a particular thing comes from is much easier if it is prefixed with a module name. Often, multiple modules happen to define things with the same name, so a reader of the code doesn't have to go back to the top of the file to see from which module a given name is imported.

Documentation
-------------

All functions must be documented with a docstring, enabling automatic documentation generation.

The Google style is used from 2025. This is not the case yet for all files, but new code should be written with respect to this rule.

Docstrings must start on the line following the triple quotes with a capital letter and just below (i.e. with indentation). For functions and methods, the first word is an imperative verb (without ending 's'). The first sentence must be direct and simple, with a blank line separating it from the rest of the docstring (see the NumPy documentation as an example).

Sentences end with a full stop (.), even in the argument section! Types in the argument should not be added, as type hinting will provide an automatic and more robust information.

Private functions and methods must start with an underscore and be documented. This was not the case before 2025, and the transition of old code will be slow, but new code must be written following this rule.

Sphinx does not generate documentation for functions and methods starting with an underscore, but docstrings are still compulsory to allow other developers to understand the code. Small technical functions considered "private" (i.e., only used in their module) can remain undocumented at this stage.

Check that bullet lists (- symbols) are well rendered in the documentation on Read the Docs. It seems that ":" in the previous sentence triggers bullet list formatting. Another option seems to be to indent the "-". Don't use * or other symbols for the bullet lists.

For argument names in the middle of a sentence, it is more readable to add backticks as ``parameters``. This renders the parameters in bold. Use with moderation.

Functions and modules are referred to with a ReST link as follows: :py:func:`~smrt.inputs.make_medium.make_snow_layer` or :py:mod:`smrt.rtsolver`. This makes a link in the documentation, which is very convenient for the user to browse the documentation. The tilde "~" means that only "make_snow_layer" is rendered (but the link uses the full path). Removing the ~ renders the full path. This can be useful in some cases, especially for modules.

Units with supercripts use the ReST syntax. For example, density must be written as kg m :sup:`-3`.

References to publications should be complete with hyperlinks for DOI: doi:10.1007/s10236-018-1166-4 ==> https://doi.org/10.1007/s10236-018-1166-4. For now, references can be written inline or in a reference section. Only one of these options will remain in the future.

For any other points, ask the community or make a decision following the practice of NumPy documentation (except that we use Google style, not NumPy style).
