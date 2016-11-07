
"""

This directory contains different solvers of the radiative transfer equation. Based on the electromagnetic properties of
each layer computed by the EM model, these RT solvers compute the emission and propagation of energy in the medium up to the surface (the atmosphere is usually
dealt with independently in dedicated modules in :py:mod:`smrt.atmosphere`).

The solvers differ by the approximations and numerical methods. :py:mod:`~smrt.rtsolver.dort` is currently the most accurate and recommended
in most cases unless the computation time is a constraint.

The selection of the solver is done with the :py:func:`~smrt.core.model.make_model` function.

.. admonition:: **For Developers**

    To experiment with DORT, we recommand to copy the file dort.py to e.g. dort_mytest.py so it is immediately available through
    :py:func:`~smrt.core.model.make_model`.

    To develop a new solver that will be accessible by the :py:func:`~smrt.core.model.make_model` function, you need to add
    a file in this directory, give a look at dort.py which is not simple but the only one at the moment. Only the method solve needs
    to be implemented. It must return a :py:class:`~smrt.core.result.Result` instance with the results. Contact the core developers to have more details.

 """
