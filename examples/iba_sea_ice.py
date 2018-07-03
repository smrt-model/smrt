# coding: utf-8

import numpy as np
import sys

# local import
from smrt import make_ice_column, make_snowpack, make_model, sensor_list
from smrt import PSU
#from smrt.inputs.make_medium import make_ice_column

# prepare inputs
l = 9 #9 ice layers
thickness = np.array([1.5/l] * l) #ice is 1.5m thick
p_ex = np.array([1.0e-3] * (l)) #correlation length
temperature = np.linspace(273.15-20., 273.15 - 1.8, l) #temperature gradient in the ice from -20 deg C at top to freezing temperature of water at bottom (-1.8 deg C)
salinity = np.linspace(2., 10., l)*PSU #salinity profile ranging from salinity=2 at the top to salinity=10 at the bottom of the ice

# create a multi-year sea ice column with assumption of spherical brine inclusions (brine_inclusion_shape="spheres"), and 10% porosity:
ice_type = 'multiyear' # first-year or multi-year sea ice
porosity = 0.08 # ice porosity in fractions, [0..1]

ice_column = make_ice_column(ice_type=ice_type,thickness=thickness,
                            temperature=temperature,
                            microstructure_model="exponential",
                            brine_inclusion_shape="spheres", #brine_inclusion_shape can be "spheres", "random_needles" or "mix_spheres_needles"
                            salinity=salinity, #either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice 
                            porosity = porosity, # either density or 'porosity' should be set for sea ice. If porosity is given, density is calculated in the model. If none is given, ice is treated as having a porosity of 0% (no air inclusions)
                            corr_length=p_ex,
                            add_water_substrate="ocean" #see comment below
                            )

#add_water_substrate: adds an semi-infinite layer of water below the ice column. Possible arguments are True (default, looks for salinity or
#brine volume fraction input to determine if a saline or fresh water layer is added), False (no water layer is added), 'ocean' (adds saline
#water), 'fresh' (adds fresh water layer).

#Optional arguments for function make_ice_column() are 'water_temperature' (default = -1.8degC),
#'water_salinity' (default = 32.) and 'water_depth' (default = 10m, i.e. infinitely thick for microwave radiation) of the water layer.

################################################################################################

# create snowpack with two snow layers:
l_s=2 #2 snow layers
thickness_s = np.array([0.05, 0.2])
p_ex_s = np.array([5e-5]*l_s)
temperature_s = np.linspace(273.15-25., 273.15 - 20, l_s)
density_s = [200, 340]

# create the snowpack
snowpack = make_snowpack(thickness=thickness_s,
                         microstructure_model="exponential",
                         density=density_s,
                         temperature=temperature_s,
                         corr_length=p_ex_s)

#add snowpack on top of ice column:
medium = snowpack + ice_column

# create the sensor
sensor = sensor_list.passive(1.4e9, 40.)

n_max_stream = 128 #TB calculation is more accurate if number of streams is increased (currently: default = 32);
#needs to be increased when using > 1 snow layer on top of sea ice! 
m = make_model("iba", "dort", rtsolver_options ={"n_max_stream": n_max_stream})

# run the model for bare sea ice:
res1 = m.run(sensor, ice_column)
# run the model for snow-covered sea ice:
res2 = m.run(sensor, medium)

# print TBs at horizontal and vertical polarization:
print(res1.TbH(), res1.TbV())
print(res2.TbH(), res2.TbV())