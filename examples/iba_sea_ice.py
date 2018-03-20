# coding: utf-8

import numpy as np
import sys

# local import
from smrt import make_ice_column, make_model, sensor_list
from smrt import PSU
#from smrt.inputs.make_medium import make_ice_column

# prepare inputs
l = 9 #9 ice layers
n_max_stream = 64

thickness = np.array([1.5/l] * l) #ice is 1.5m thick
p_ex = np.array([5.e-4] * (l)) #correlation length
temperature = np.linspace(273.15-20., 273.15 - 1.8, l) #temperature gradient in the ice from -20 deg C at top to freezing temperature of water at bottom (-1.8 deg C)
salinity = np.linspace(2., 10., l)*PSU #salinity profile ranging from salinity=2 at the top to salinity=10 at the bottom of the ice

# create an ice column with assumption of spherical brine inclusions (inclusion_shape="spheres"):
ice_column = make_ice_column(thickness=thickness,
                            temperature=temperature,
                            microstructure_model="exponential",
                            inclusion_shape="spheres", #inclusion_shape can be "spheres" or "random_needles" 
                            salinity=salinity, #either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice 
                            corr_length=p_ex,
                            add_water_substrate="ocean" #see comment below
                            )

#add_water_substrate: adds an semi-infinite layer of water below the ice column. Possible arguments are True (default, looks for salinity or
#brine volume fraction input to determine if a saline or fresh water layer is added), False (no water layer is added), 'ocean' (adds saline
#water), 'fresh' (adds fresh water layer).

#Optional arguments for function make_ice_column() are 'water_temperature' (default = -1.8degC),
#'water_salinity' (default = 32.) and 'water_depth' (default = 10m, i.e. infinitely thick for microwave radiation) of the water layer.

# create the sensor
sensor = sensor_list.passive(1.4e9, 40.)

m = make_model("iba", "dort")

# run the model
res = m.run(sensor, ice_column)

# print TBs at horizontal and vertical polarization:
print(res.TbH(), res.TbV())
