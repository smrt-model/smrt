from smrt.inputs.make_medium import make_snowpack
from smrt import sensor_list, make_model
from smrt.rtsolver.dort import DORT
from smrt.inputs.sensor_list import amsre
from smrt.core.model import Model
import numpy as np
from smrt.microstructure_model.sticky_hard_spheres import StickyHardSpheres
from smrt.core.error import SMRTWarning
import warnings

class Multiple_Snowpacks:
    """
    Benchmark time and peak memory for creating multiple snowpacks and running model in parrallel

    Inspired from smrt/test/test_model.py
    """
    param_names = ["snowpack_amount"]
    params = [1000] #Add values for multiple benchmarks
    def setup(self, n):
        """
        Initiates snowpacks, model and sensor for the benchmarks
        """
        self.temperatures = np.linspace(200, 270, n)
        self.snowpack_list = [make_snowpack([2000], StickyHardSpheres, density=[250], temperature=t,
                                            radius=0.3e-3, stickiness=0.2) for t in self.temperatures]
        self.m = Model('dmrt_qcacp_shortrange', DORT)
        self.sensor = amsre()

    def time_setup(self, n):
        """
        Benchmark time for creating snowpack list
        """
        [make_snowpack([2000], StickyHardSpheres, density=[250], temperature=t,
                                            radius=0.3e-3, stickiness=0.2) 
                                            for t in self.temperatures]
        
    def peakmem_setup(self, n):
        """
        Benchmark memory usage for creating snowpack list
        """
        [make_snowpack([2000], StickyHardSpheres, density=[250], temperature=t,
                                            radius=0.3e-3, stickiness=0.2)
                                            for t in self.temperatures]

    def time_parrallel_run(self, n):
        """
        Benchmark time for running model
        """
        self.m.run(self.sensor, self.snowpack_list, parallel_computation=True)

    def peakmem_parrallel_run(self, n):
        """
        Benchmark memory usage for running model
        """
        self.m.run(self.sensor, self.snowpack_list, parallel_computation=True)

class Large_Snowpack:
    """
    Benchmark time and peak memory for creating a large snowpack and running model

    Inspired from smrt/test/test_coherent_layer.py
    """
    param_names = ["layer_amount"]
    params = [600] #Add values for multiple benchmarks
    def setup(self, layer_amount):
        """
        Initiates snowpack, model and sensor for the benchmarks
        """
         # this test is only here to avoid regression, it is not scientifically validated

        self.density = np.linspace(300, 916.7, layer_amount)
        self.thickness = np.ones(layer_amount) * 3000 / layer_amount
        self.temperature = np.linspace(200, 270, layer_amount)
        self.corr_length = np.ones(layer_amount) *200e-6
        theta = 60
        self.radiometer = sensor_list.passive(5e9, theta)
        self.sp = make_snowpack(self.thickness, "exponential", density=self.density,
                                temperature=self.temperature, corr_length=self.corr_length)
        # create the EM Model - Equivalent DMRTML
        self.m = make_model("iba", "dort", rtsolver_options={"n_max_stream": 64, "process_coherent_layers": True})
        warnings.simplefilter("ignore", category=SMRTWarning)

    def time_setup(self, layer_amount):
        """
        Benchmark time for creating snowpack list
        """
        make_snowpack(self.thickness, "exponential", density=self.density, temperature=self.temperature, corr_length=self.corr_length)
        
    def peakmem_setup(self, layer_amount):
        """
        Benchmark memory usage for creating snowpack list
        """
        make_snowpack(self.thickness, "exponential", density=self.density,
                      temperature=self.temperature, corr_length=self.corr_length)

    def time_parrallel_run(self, layer_amount):
        """
        Benchmark time for running model
        """
        self.m.run(self.radiometer, self.sp, parallel_computation=True)

    def peakmem_parrallel_run(self, layer_amount):
        """
        Benchmark memory usage for running model
        """
        self.m.run(self.radiometer, self.sp, parallel_computation=True)