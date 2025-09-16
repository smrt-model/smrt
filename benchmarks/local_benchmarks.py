from smrt.inputs.make_medium import make_snowpack
from smrt import sensor_list, make_model
from smrt.inputs.sensor_list import amsre
from smrt.core.model import Model
import numpy as np
from smrt.microstructure_model.sticky_hard_spheres import StickyHardSpheres
from smrt.core.error import SMRTWarning
import warnings
from abc import ABC, abstractmethod

class Benchmark(ABC):
    param_names = ["parameters"]
    @abstractmethod
    def setup(self, *args):
        pass
    @abstractmethod
    def set_run(self, *args):
        pass
    def time_setrun(self, *args):
        self.set_run(*args)
    def peakmem_setrun(self, *args):
        self.set_run(*args)

class MultipleSemiInfiniteSnowpacks(Benchmark):
    """
    Benchmark time and peak memory for creating multiple snowpacks and running model in parrallel

    Inspired from smrt/test/test_model.py
    """
    params = [{"solver":"dort", "emmodel":"iba", "size":500}] #Add values for multiple benchmarks
    def setup(self, params):
        """
        Initiates model and sensor for the benchmarks
        """
        self.temperature_list = np.linspace(200, 270, params["size"])
        self.radius_list = np.linspace(5e-5, 1e-3, params["size"]) 
        self.m = Model(params["emmodel"], params["solver"])
        self.sensor = amsre()
    
    def set_run(self, params):
        self.snowpack_list = [make_snowpack([2000], StickyHardSpheres, density=[250],
                                            temperature=self.temperature_list[i],
                                            radius= self.radius_list[i], stickiness=0.2) for i in range(params["size"])]
        self.m.run(self.sensor, self.snowpack_list, parallel_computation=True)

class SemiInfiniteMultipleLayer(Benchmark):
    """
    Benchmark time and peak memory for creating a large snowpack and running model

    Inspired from smrt/test/test_coherent_layer.py
    """
    
    params = [{"solver":"dort", "emmodel":"iba", "size":500}] #Add values for multiple benchmarks
    def setup(self, params):
        """
        Initiates snowpack, model and sensor for the benchmarks
        """
        self.density = np.linspace(300, 916.7, params["size"])
        self.thickness = np.ones(params["size"]) * 3000 / params["size"]
        self.temperature = np.linspace(200, 270, params["size"])
        self.corr_length = np.ones(params["size"]) *200e-6
        theta = 60
        self.radiometer = sensor_list.passive(5e9, theta)
        # create the EM Model - Equivalent DMRTML
        self.m = make_model(params["emmodel"], params["solver"], rtsolver_options={"n_max_stream": 64, "process_coherent_layers": True})
        warnings.simplefilter("ignore", category=SMRTWarning)

    def set_run(self, params):
        self.sp = make_snowpack(self.thickness, "exponential", density=self.density,
                                temperature=self.temperature, corr_length=self.corr_length)
        self.m.run(self.radiometer, self.sp, parallel_computation=True)