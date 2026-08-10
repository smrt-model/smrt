####################################
Comparison of electromagnetic models
####################################

**Goal**: run and compare SMRT for different electromagnetic theories.

**Learning**:

- Understand which electromagnetic models can be used with what kind of microstructure.
- Learn how to compute scattering coefficient, without running the full model.

Some of the theories can be used only with sphere microstructures (QCA, QCA-CP, Rayleigh), others only with exponential
microstructures (SFT) and others can be combined with any microstructure model (IBA and the different variants of SCE).

This guide helps you create two ensembles of snowpacks with varying size parameter and then run SMRT for the different
snowpacks and compare the results of the different electromagnetic theories.

- One snowpack made of a sticky_hard_spheres microstructure with varying radius. Here it is possible to compare IBA,
  DMRT_QCA_shortrange, DMRT_QCA_shortrange, Rayleigh and the different variants of SCE.
- Another snowpack made of an exponential microstructure with varying correlation length (also know as Porod length).
  Here only IBA, SFT and the variants of SCE can be compared.

Electromagnetic models compatible with spheres
----------------------------------------------

First create an ensemble of snowpacks initialized with the sticky hard sphere (SHS) microstructure of different radii:

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt

    from smrt import make_model, make_snowpack, sensor_list

    # prepare the snowpack

    thickness = [10]
    density = 350
    temperature = 270
    stickiness = 0.15
    radius_list = np.arange(50, 400, 10) * 1e-6

    snowpack_list_shs = [make_snowpack(thickness=thickness,
                         microstructure_model='sticky_hard_spheres',
                         radius=r,
                         density=density,
                         temperature=temperature,
                         stickiness=stickiness) for r in radius_list]

Then, create electromagnetic models (compatible with sphere microstructures):

.. code:: ipython3

    # prepare several models

    models = dict(
        m_dmrt_qca = make_model("dmrt_qca_shortrange", "dort"),
        m_dmrt_qcacp = make_model("dmrt_qcacp_shortrange", "dort"),
        m_dmrt_qcacp = make_model("dmrt_rayleigh", "dort"),
    )


Run the models as usual, and plot the results:

.. code:: ipython3

    radiometer = sensor_list.amsre("37")

    # m is a dictionary of models, so it is possible to loop over using dict comprehension
    res = {m: models[m].run(radiometer, snowpack_list_shs) for m in models}



    plt.figure()
    for m in models:
        plt.plot(radius, res[m].TbV(), label=m)
    plt.show()


Electromagnetic models compatible with exponential microstructure
-----------------------------------------------------------------

The same approach can be used for the exponential microstructure, with different correlation lengths:

.. code:: ipython3

    # prepare the snowpack

    thickness = [1000.0]
    density = 350
    temperature = 270

    corr_length_list = np.arange(20, 200, 10) * 1e-6

    snowpack_list_exp = [make_snowpack(thickness=thickness,
                         microstructure_model='exponential',
                         corr_length=c,
                         density=density,
                         temperature=temperature) for c in corr_length_list]

    # prepare several models

    models = dict(
        m_iba = make_model("iba", "dort"),
        m_sce = make_model("sce_rechtsman08", "dort"),
        m_sft = make_model("sft_rayleigh", "dort"),
    )

    radiometer = sensor_list.amsre("37")

    res = {m: models[m].run(radiometer, snowpack_list_shs) for m in models}

    plt.figure()
    for m in models:
        plt.plot(radius, res[m].TbV(), label=m)
    plt.show()


Computing the scattering coefficient
------------------------------------

These models differ mainly by the scattering coefficient. It is often useful to investigate the
scattering coefficient.

There are three ways to get the scattering coefficient.

1) access the 'emmodel' attribute of the model and run it on one of the layer (not on the full snowpack):

.. code:: ipython3

    firstlayer = snowpack_list_exp[0].layers[0]  # first layer of the first snowpack

    m_iba = make_model("iba", "dort")

    m_iba.emmodel(sensor, firstlayer).ks


2) without the overhead of make_model. It is simpler when the full model is not needed:

.. code:: ipython3

    # new import needed
    from smrt import make_emmodel

    # make the EM model
    em_iba = make_emmodel("iba")(sensor, firstlayer)
    # get ks
    em_iba.ks

The double parenthesis ()() is not a typo, it is correct! `make_emmodel("iba")` returns a
:py:class:`~smrt.core.sensor.Sensor` class (here of type :py:class:`~smrt.emmodel.iba.IBA`) that needs to be called
with the sensor and the layer as argumentsto get a proper EM model instance.

3) when the full model has run as usual. In this case, the :py:class:`~smrt.core.result.Result` object contains the
   scattering coefficient for each layer, as well as other information such as the 'optical_depth' or the
   'single_scattering_albedo':

.. code:: ipython3

    m_iba = make_model("iba", "dort")
    res = m_iba.run(radiometer, snowpack_list_exp)

    res.ks

This latter approach is the most convenient, but the RTSolver is run which requires significantly more
computation than just computing the scattering coefficient. The former approaches are more efficient if you just want to
compare the scattering coefficient of different models.

Comparing the scattering coefficient from different formulations
----------------------------------------------------------------

Most of the theories can be compared for the SHS snowpack. Let's compute
the scattering coefficient and assess the radius dependence.

.. code:: ipython3

    ks_iba = [m_iba.emmodel(sensor, sp.layers[0]).ks for sp in snowpack_list_shs]
    ks_sce = [m_sce.emmodel(sensor, sp.layers[0]).ks for sp in snowpack_list_shs]
    ks_qca = [m_dmrt_qca.emmodel(sensor, sp.layers[0]).ks for sp in snowpack_list_shs]


Now compare the radius dependence:

.. code:: ipython3

    plt.figure()
    plt.plot(radius_list*1e6, ks_iba, label="IBA")
    plt.plot(radius_list*1e6, ks_qca, label="QCA")
    plt.plot(radius_list*1e6, ks_sce, label="SCE RT08")

    plt.legend()
    plt.xlabel("Radius ($\\mu$m)")
    plt.ylabel("Scattering coefficient (m$^{-1}$)")
