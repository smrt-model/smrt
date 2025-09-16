from smrt.inputs.make_medium import make_snowpack
from smrt import sensor_list, make_model
from smrt.rtsolver.dort import DORT
from smrt.inputs.sensor_list import amsre
from smrt.core.model import Model
import numpy as np
from smrt.microstructure_model.sticky_hard_spheres import StickyHardSpheres
from smrt.core.error import SMRTWarning
import warnings

class MultipleSemiInfinite:
    """
    Benchmark time and peak memory for creating multiple snowpacks and running model in parrallel

    Inspired from smrt/test/test_model.py
    """
    param_names = ["n_snowpack"]
    params = [500] #Add values for multiple benchmarks
    def setup(self, n_snowpack):
        """
        Initiates model and sensor for the benchmarks
        """
        self.temperature_list = np.linspace(200, 270, n_snowpack)
        self.radius_list = np.linspace(5e-5, 1e-3, n_snowpack) 
        self.m = Model('iba', "dort")
        self.sensor = amsre()
    
    def set_and_run(self, n_snowpack):
        self.snowpack_list = [make_snowpack([2000], StickyHardSpheres, density=[250],
                                            temperature=self.temperature_list[i],
                                            radius= self.radius_list[i], stickiness=0.2) for i in range(n_snowpack)]
        self.m.run(self.sensor, self.snowpack_list, parallel_computation=True)

    def time_setrun(self, n_snowpacks):
        """
        Benchmark time for creating and running snowpack list
        """
        self.set_and_run(n_snowpacks)
        
    def peakmem_setrun(self, n_snowpacks):
        """
        Benchmark memory usage for creating snowpack list
        """
        self.set_and_run(n_snowpacks)

class SemiInfiniteMultipleLayer:
    """
    Benchmark time and peak memory for creating a large snowpack and running model

    Inspired from smrt/test/test_coherent_layer.py
    """
    param_names = ["n_layer"]
    params = [300] #Add values for multiple benchmarks
    def setup(self, n_layer):
        """
        Initiates snowpack, model and sensor for the benchmarks
        """
        self.density = np.linspace(300, 916.7, n_layer)
        self.thickness = np.ones(n_layer) * 3000 / n_layer
        self.temperature = np.linspace(200, 270, n_layer)
        self.corr_length = np.ones(n_layer) *200e-6
        theta = 60
        self.radiometer = sensor_list.passive(5e9, theta)
        # create the EM Model - Equivalent DMRTML
        self.m = make_model("iba", "dort", rtsolver_options={"n_max_stream": 64, "process_coherent_layers": True})
        warnings.simplefilter("ignore", category=SMRTWarning)

    def set_run(self, n_layer):
        self.sp = make_snowpack(self.thickness, "exponential", density=self.density,
                                temperature=self.temperature, corr_length=self.corr_length)
        self.m.run(self.radiometer, self.sp, parallel_computation=True)
    def time_setrun(self, n_layer):
        """
        Benchmark time for creating snowpack list
        """
        self.set_run(n_layer)
        
    def peakmem_setrun(self, n_layer):
        """
        Benchmark memory usage for creating snowpack list
        """
        self.set_run(n_layer)