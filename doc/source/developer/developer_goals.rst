####################################
Goals
####################################

The initial goal of the model was to explore new ideas related to the microstructure and test various modeling solutions. This led to a highly modular architecture.

As of 2025, the goal is to provide the community with:

- A model that is easy to use for beginners and easy to extend for advanced users.
- Good documentation, including not only technical information but also scientific recommendations (compatibilities, limitations) and traceability in the literature.
- Many options to choose from at all levels of the RT calculation. There is no single best solution in electromagnetic snow modeling, and the authors have tried to minimize opinionated choices, leaving the responsibility and possibility to the user.
- Historical models and formulations to serve as a memory of our electromagnetic and snow history. If you read an "old" paper with some formulations, create a function and commit to SMRT. Even if no one uses it, it will be there forever in a digital format, avoiding future generation to reinvent the wheel.
- New models and formulations not available in any other snow RT models. These models are unique to SMRT.

Python
------

Python was chosen in 2015 because of its growing popularity in the scientific community and its higher flexibility compared to compiled legacy languages like FORTRAN. This choice is even more valid in 2025. Python enables the model to be extremely modular, much easier to use, which is a main constraint of the project, allows faster development, and facilitates the exploration of new ideas. Performance should not be an issue as the time-consuming part of the model is usually localized in the RT solver, where numerical integrations and eigenvalue problems are solved with the highly optimized SciPy module, backed by BLAS, LAPACK, and MINPACK libraries as would be done if using FORTRAN. In addition, we use Numba in some small portion of the code. The compilation of the full code base with PyPy will be considered in case of performance issues later. Parallelization is provided through the joblib and dask modules, and extensions are made easy with "runners".