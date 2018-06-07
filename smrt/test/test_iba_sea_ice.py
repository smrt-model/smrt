# coding: utf-8

import numpy as np
from nose.tools import ok_

# local import
from smrt import make_model, sensor_list, PSU
from smrt.inputs.make_medium import make_ice_column, bulk_ice_density

#test if this configuration gives values as originally produced by examples/iba_sea_ice.py
#same structure as test_integration_iba.py


def setup_seaice():

    l = 9  # 9 ice layers
    n_max_stream = 64

    thickness = np.array([1.5 / l] * l)  # ice is 1.5m thick
    p_ex = np.array([500e-6] * (l))  # correlation length
    temperature = np.linspace(273.15 - 20., 273.15 - 1.8,
                              l + 1)  # temperature gradient in the ice from -20 deg C at top to freezing temperature of water at bottom (-1.8 deg C)
    temperature = temperature[:-1]
    salinity = np.linspace(2., 10., l + 1) * PSU  # salinity profile ranging from salinity=2 at the top to salinity=10 at the bottom of the ice
    salinity = salinity[:-1]

    return l, n_max_stream, thickness, p_ex, temperature, salinity


def test_oneconfig_for_firstyear_sea_ice():

    # prepare inputs

    l, n_max_stream, thickness, p_ex, temperature, salinity = setup_seaice()

    # create an ice column with assumption of spherical brine inclusions (inclusion_shape="spheres"):
    ice_column = make_ice_column("firstyear",
                                 thickness=thickness,
                                  temperature=temperature,
                                  microstructure_model="exponential",
                                  brine_inclusion_shape="spheres", #inclusion_shape can be "spheres" or "random_needles"
                                  salinity=salinity, #either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice
                                  corr_length=p_ex,
                                  add_water_substrate="ocean"
                                  )

    # create the sensor
    sensor = sensor_list.passive(1.4e9, 40.)
    
    m = make_model("iba", "dort")

    # run the model
    res = m.run(sensor, ice_column)

    print(res.TbV(), res.TbH())
    #absorption with effective permittivity
    ok_(abs(res.TbV() - 245.36044436424572) < 1e-4)
    ok_(abs(res.TbH() - 219.2042855456296) < 1e-4)


def test_oneconfig_for_multiyear_sea_ice():
    # CAREFULL, THIS TEST IS NOT REALISTIC FOR SEA ICE !!!!!!!!!!!!!!!!!

    # prepare inputs
    l, n_max_stream, thickness, p_ex, temperature, salinity = setup_seaice()

    porosity = 0.08  # ice porosity, in fraction

    # create an ice column with assumption of spherical brine inclusions (inclusion_shape="spheres"):
    ice_column = make_ice_column("multiyear",
                                 thickness=thickness,
                                 temperature=temperature,
                                 microstructure_model="exponential",
                                 brine_inclusion_shape="spheres",
                                 salinity=salinity * PSU,
                                 porosity=porosity,
                                 corr_length=p_ex,
                                 add_water_substrate="ocean"
                                 )

    # create the sensor
    sensor = sensor_list.passive(1.4e9, 40.)

    m = make_model("iba", "dort")

    # run the model
    res = m.run(sensor, ice_column)

    print(res.TbV(), res.TbH())
    # absorption with effective permittivity
    ok_(abs(res.TbV() - 144.58043633826188) < 1e-4)
    ok_(abs(res.TbH() - 124.54838130267804) < 1e-4)

    # CAREFULL, THIS TEST IS NOT REALISTIC FOR SEA ICE !!!!!!!!!!!!!!!!!





def test_equivalence_porosity_density():

    l, n_max_stream, thickness, p_ex, temperature, salinity = setup_seaice()

    ice_type = 'multiyear'  # first-year (FY) or multi-year (MY) sea ice
    porosity = 0.08  # ice porosity, in fraction

    ice_column1 = make_ice_column(ice_type,
                                  thickness=thickness,
                                  temperature=temperature,
                                  microstructure_model="exponential",
                                  brine_inclusion_shape="spheres",
                                  # brine_inclusion_shape can be "spheres", "random_needles" or "mix" (a mix of spheres and needles)
                                  salinity=salinity,
                                  # either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice
                                  porosity=porosity,
                                  # either density or 'porosity' should be set for sea ice. If porosity is given, density is calculated in the model. If none is given, ice is treated as having a porosity of 0% (no air inclusions)
                                  #                            density = density,
                                  corr_length=p_ex,
                                  add_water_substrate="ocean"  # see comment below
                                  )

    # Same, but giving the density instead:
    density = [bulk_ice_density(temp, salt , porosity) for temp, salt in zip(temperature, salinity)]

    ice_column2 = make_ice_column(ice_type,
                                  thickness=thickness,
                                  temperature=temperature,
                                  microstructure_model="exponential",
                                  brine_inclusion_shape="spheres",
                                  # brine_inclusion_shape can be "spheres", "random_needles" or "mix" (a mix of spheres and needles)
                                  salinity=salinity,
                                  # either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice
                                  #                            porosity = porosity, # either density or 'porosity' should be set for sea ice. If porosity is given, density is calculated in the model. If none is given, ice is treated as having a porosity of 0% (no air inclusions)
                                  density=density,
                                  corr_length=p_ex,
                                  add_water_substrate="ocean"  # see comment below
                                  )

    sensor = sensor_list.passive(1.4e9, 40.)

    n_max_stream = 128  # TB calculation is more accurate if number of streams is increased (currently: default = 32);
    # needs to be increased when using > 1 snow layer on top of sea ice!
    m = make_model("iba", "dort", rtsolver_options={"n_max_stream": n_max_stream})

    # run the model for sea ice with porosity / density
    res1 = m.run(sensor, ice_column1)
    res2 = m.run(sensor, ice_column2)
    print(res1.TbV(), res1.TbH())
    print(res2.TbV(), res2.TbH())

    # The two should be similar
    ok_(abs(res1.TbV() - res2.TbV()) < 1e-4)
    ok_(abs(res1.TbH() - res2.TbH()) < 1e-4)
