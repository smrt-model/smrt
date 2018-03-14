# coding: utf-8

import numpy as np
import sys

# local import
from smrt import make_model, sensor_list
from smrt.inputs.make_medium import make_ice_column

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
ice_column1 = make_ice_column(thickness=thickness,
                            temperature=temperature,
                            microstructure_model="exponential",
                            inclusion_shape="spheres", #inclusion_shape can be "spheres" or "random_needles" 
                            salinity=salinity, #either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice 
                            corr_length=p_ex,
                            add_water_substrate="ocean" #see comment below
                            )

#add_water_substrate: adds an semi-infinite layer of water below the ice column. Possible arguments are True (default, looks for salinity or brine volume fraction input to determine if a saline or fresh water layer is added), False (no water layer is added), 'ocean' (adds saline water), 'fresh' (adds fresh water layer).

# create the sensor
sensor = sensor_list.passive(1.4e9, 40.)

m1 = make_model("iba", "dort")

# run the model
res1 = m1.run(sensor, ice_column1)

# print TBs at horizontal and vertical polarization:
print(res1.TbH(), res1.TbV())
