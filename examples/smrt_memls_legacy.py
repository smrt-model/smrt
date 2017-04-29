
import matplotlib.pyplot as plt

# general import for smrt
from smrt import make_snowpack, make_model, sensor_list

# import for memls
from smrt.utils import memls_legacy

# prepare snowpack
pc = 0.2e-3
snowpack = make_snowpack(thickness=[10], microstructure_model="exponential",
                         density=[300], temperature=[265], corr_length=pc)

# create the sensor
theta = range(10, 80, 5)
radiometer = sensor_list.passive(37e9, theta)

# create the EM Model
m = make_model("iba", "dort")

# run the model
sresult = m.run(radiometer, snowpack)

# run MEMLS matlab code
mresult = memls_legacy.run(radiometer, snowpack)


# outputs
plt.plot(theta, sresult.TbV(), 'r-', label='SMRT V')
plt.plot(theta, sresult.TbH(), 'r--', label='SMRT H')
plt.plot(theta, mresult.TbV(), 'b-', label='MEMLS V')
plt.plot(theta, mresult.TbH(), 'b--', label='MEMLS H')
plt.legend(loc='best')
plt.show()