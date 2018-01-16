"""
Microstructure models are different representations of the snow microstructure. Because these representations are different, the
parameters to describe actual snow micro-structure depends on the model. For instance, the Sticky Hard Spheres medium is implemented
in :py:mod:`~smrt.microstructure_model.sticky_hard_spheres` and its parameters are: the `radius` (required) and
the `stickiness` (optional, default value is non-sticky, even though we do recommend to use a stickiness of ~0.1-0.3 in practice).

Because IBA is one of the important electromagnetic theories provided by SMRT, the first/main role of microstructure models is to provide
the Fourier transform of the autocorrelation functions. Hence most microstructure models are named after the autocorrelation function.
For instance, the :py:mod:`~smrt.microstructure_model.exponential` autocorrelation function is that used in MEMLS. Its only parameter is the 
`corr_length`.

To use microstructure models, it is only required to read the documentation of each model to determine
the required and optional parameters. Selecting the microstructure model is usually done with make_snowpack which only requires the name of the
module (the filename with .py). The import of the module is automatic. For instance::

    from smrt import make_snowpack

    sp = make_snowpack([1, 1000], "exponential", density=[200, 300], corr_length=[0.2e-3, 0.5e-3])

This snippet creates a snowpack with the exponential autocorrelation function for all (2) layers. Import of the :py:mod:`~smrt.microstructure_model.exponential`
is automatic and creation of instance of the class :py:mod:`~smrt.microstructure_model.exponential.Exponential` is done by the model
:py:meth:`smrt.core.model.Model.run` method.


 """  
