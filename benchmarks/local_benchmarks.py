from smrt.inputs.make_medium import make_snowpack
from smrt import sensor_list, make_model
from smrt.inputs.sensor_list import amsre
from smrt.core.model import Model
import numpy as np
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


class ManySemiInfiniteSnowpacks(Benchmark):
    """
    Benchmark time and peak memory for creating multiple snowpacks and running model in parrallel

    Inspired from smrt/test/test_model.py
    """

    params = [{"solver": "dort", "emmodel": "iba", "n_snowpacks": 200, "parallel_computation": True},
              {"solver": "dort", "emmodel": "iba", "n_snowpacks": 20, "parallel_computation": False}]  # Add values for multiple benchmarks

    def setup(self, params):
        """
        Initiates model and sensor for the benchmarks
        """
        self.temperature_list = np.linspace(200, 270, params["n_snowpacks"])
        self.radius_list = np.linspace(50e-6, 1000e-6, params["n_snowpacks"])
        self.model = Model(params["emmodel"], params["solver"])
        self.sensor = amsre()

    def set_run(self, params):
        snowpack_list = [
            make_snowpack(
                [2000],
                "sticky_hard_spheres",
                density=250,
                temperature=self.temperature_list[i],
                radius=self.radius_list[i],
                stickiness=0.2,
            )
            for i in range(params["n_snowpacks"])
        ]
        self.model.run(self.sensor, snowpack_list, parallel_computation=params['parallel_computation'])


class MultiLayerSnowpack(Benchmark):
    """
    Benchmark time and peak memory for creating a large snowpack and running model

    Inspired from smrt/test/test_coherent_layer.py
    """

    params = [{"solver": "dort", "emmodel": "iba", "n_layers": 200, "parallel_computation": True},
              {"solver": "dort", "emmodel": "iba", "n_layers": 200, "parallel_computation": False}]  # Add values for multiple benchmarks

    def setup(self, params):
        """
        Initiates snowpack, model and sensor for the benchmarks
        """
        self.density = np.linspace(300, 916 / 2, params["n_layers"])
        self.thickness = np.full(params["n_layers"], 200 / params["n_layers"])
        self.temperature = np.linspace(200, 270, params["n_layers"])
        self.corr_length = 200e-6
        self.radiometer = sensor_list.amsre('19')
        # create the EM Model - Equivalent DMRTML
        self.model = make_model(
            params["emmodel"], params["solver"], rtsolver_options={"n_max_stream": 64, "process_coherent_layers": True}
        )
        warnings.simplefilter("ignore", category=SMRTWarning)

    def set_run(self, params):
        sp = make_snowpack(
            self.thickness,
            "exponential",
            density=self.density,
            temperature=self.temperature,
            corr_length=self.corr_length,
        )
        self.model.run(self.radiometer, sp, parallel_computation=params['parallel_computation'])
