"""
This package contains several solvers of the radiative transfer equation. Based on the electromagnetic properties of
each layer computed by the EM model, these RT solvers compute the emission and propagation of energy in the medium up to
the surface (the atmosphere is usually dealt with independently in dedicated modules in :py:mod:`smrt.atmosphere`).

The solvers differ by the approximations and numerical methods. :py:mod:`~smrt.rtsolver.dort` is currently the most accurate and recommended
in most cases unless the computation time is a constraint or for altimetric simulations.

Selection of the solver is done with the :py:mod:`smrt.core.model.make_model` function.

Here are some recommendations to chose an appropriate solver:

.. list-table:: SMRT Solvers Overview
   :header-rows: 1
   :widths: 20 50 30 20

   * - Solver Name
     - Description
     - Use Case
     - Performance
   * - ``dort``
     - The default original solver in SMRT for passive microwave (brightness temperature) and radar (sigma0). General and robust.
     - General applications for both passive microwave and radar.
     - Robust but slow
   * - ``iterative_first_order``
     - Radar-only solver using an iterative method up to first order. Fast and provides contributions of interaction mechanisms.
     - Radar studies, especially to analyze interaction mechanisms.
     - Fast
   * - ``multifresnel_thermalemission``
     - Passive microwave-only solver assuming no scattering in the layers and flat interfaces.
     - Radiometry at low frequency when scattering is negligible.
     - Very fast
   * - ``nadir_lrm_altimetry``
     - Specialized solver for altimeter in Low Rate Mode. Solve the first order interaction.
     - Altimetry in Low Rate Mode (LRM).
     -

Note: **For Developers**
    To experiment with DORT, it is recommended to copy the file dort.py to e.g. dort_mytest.py so it is immediately available through
    :py:mod:`smrt.core.model.make_model`.

    To develop a new solver that will be accessible by the :py:mod:`smrt.core.model.make_model` function, add
    a file in this directory. Refer to dort.py as an example. Only the `solve` method needs
    to be implemented. It must return a :py:mod:`smrt.core.result.Result` instance with the results. Contact the core developers for more details.
"""
