# coding: utf-8

import numpy as np
from nose.tools import ok_

# local import
from smrt import make_model, sensor_list, PSU
from smrt.inputs.make_medium import make_ice_column

#test if this configuration gives values as originally produced by examples/iba_sea_ice.py
#same structure as test_integration_iba.py


def test_iba_sea_ice_oneconfig():

    # prepare inputs

    l = 9 #9 ice layers
    n_max_stream = 64

    thickness = np.array([1.5/l] * l) #ice is 1.5m thick
    p_ex = np.array([5.e-4] * (l)) #correlation length
    temperature = np.linspace(273.15-20., 273.15 - 1.8, l+1) #temperature gradient in the ice from -20 deg C at top to freezing temperature of water at bottom (-1.8 deg C)
    temperature=temperature[:-1]
    salinity = np.linspace(2., 10., l+1) #salinity profile ranging from salinity=2 at the top to salinity=10 at the bottom of the ice
    salinity=salinity[:-1]

    # create an ice column with assumption of spherical brine inclusions (inclusion_shape="spheres"):
    ice_column = make_ice_column(thickness=thickness,
                                  temperature=temperature,
                                  microstructure_model="exponential",
                                  inclusion_shape="spheres", #inclusion_shape can be "spheres" or "random_needles"
                                  salinity=salinity*PSU, #either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice
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
    ok_(abs(res.TbV() - 244.9305497649834) < 1e-4)
    ok_(abs(res.TbH() - 216.86708460886817) < 1e-4)




