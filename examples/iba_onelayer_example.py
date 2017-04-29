#!/usr/bin/env python

from smrt import make_snowpack, make_model, sensor_list

# prepare inputs
thickness = [100]
corr_length = [5e-5]
temperature = [270]
density = [320]

# create the snowpack
snowpack = make_snowpack(thickness=thickness,
                         microstructure_model="exponential",
                         density=density,
                         temperature=temperature,
                         corr_length=corr_length)

# create the sensor
radiometer = sensor_list.amsre('37V')

# create the model
m = make_model("iba", "dort")

# run the model
result = m.run(radiometer, snowpack)

# outputs
print(result.TbV(), result.TbH())
