
Snow Microwave Radiative Transfer model
=============================================

[SMRT](https://www.smrt-model.science/) is a radiative transfer model to compute emission and backscatter from snowpack.

Getting started is easy, follow the [instructions](https://www.smrt-model.science/getstarted.html) and explore the other repositories
with examples in the ['smrt-model' github organization](https://github.com/smrt-model) or read the detailed ['documentation'](https://smrt.readthedocs.io/en/latest/).

If you want to try without installing anything on your computer, use free mybinder.org notenooks: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/smrt-model/smrt/master?filepath=examples/iba_onelayer.ipynb)

Quick Installation
--------------------

To install the latest stable release:

```console
pip install smrt
```

Alternatively, the latest developments are available using:

```console
pip install git+https://github.com/smrt-model/smrt.git
```

or by ['manual installation'](https://smrt-model.science/getstarted.html).


A simple example
--------------------

An example to calculate the brightness temperature from a one-layer snowpack.

```python

from smrt import make_snowpack, sensor_list, make_model

# create a snowpack
snowpack = make_snowpack(thickness=10.,   # snowpack depth in m
                         microstructure_model="sticky_hard_spheres",
                         density=320.0,   # density in kg/m3
                         temperature=260, # temperature in Kelvin
                         radius=100e-6)   # scatterers raidus in m

# create the sensor (AMSRE, channel 37 GHz vertical polarization)
radiometer = sensor_list.amsre('37V')

# create the model including the scattering model (IBA) and the radiative transfer solver (DORT)
m = make_model("iba", "dort")

# run the model
result = m.run(radiometer, snowpack)

print(result.TbV())

```


License information
--------------------

See the file ``LICENSE.txt`` for terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.

DISCLAIMER: This version of SMRT is under peer review. Please use this software with caution, ask for assistance if needed, and let us know any feedback you may have.

Copyright (c) 2016-2022 Ghislain Picard, Melody Sandells, Henning Löwe.


Other contributions
--------------------

 - Nina Maass
 - Ludovic Brucker
 - Marion Leduc-Leballeur
 - Mai Winstrup
 - Carlo Marin
 - Justin Murfitt
 - Julien Meloche

