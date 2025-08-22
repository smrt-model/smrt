from smrt.inputs.make_medium import make_snowpack
from smrt.core.interface import Substrate
from smrt.core.atmosphere import AtmosphereBase

class Suite:
    def setup_two_snowpacks(self):
        self.sp1 = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
        self.sp2 = make_snowpack([0.5], "exponential", density=400, corr_length=100e-6)

    def time_profile(self):
        sp = make_snowpack([0.1, 0.2, 0.3], "exponential", density=[100, 200, 300], corr_length=200e-6)
        sp.profile('density')

    def peakmem_profile(self):
        sp = make_snowpack([0.1, 0.2, 0.3], "exponential", density=[100, 200, 300], corr_length=200e-6)
        sp.profile('density')

    def time_addition(self):
        self.setup_two_snowpacks()
        self.sp1 + self.sp2

    def peakmem_addition(self):
        self.setup_two_snowpacks()
        self.sp1 + self.sp2

    def time_layer_addition(self):
        self.setup_two_snowpacks()
        self.sp1 + self.sp2.layers[0]

    def peakmem_layer_addition(self):
        self.setup_two_snowpacks()
        self.sp1 + self.sp2.layers[0]

    def time_substrate_addition(self):
        substrate = Substrate()
        sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
        sp += substrate

    def peakmem_substrate_addition(self):
        substrate = Substrate()
        sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
        sp += substrate

    def time_atmosphere_addition(self):
        atmosphere = AtmosphereBase()
        sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
        sp = atmosphere + sp

    def peakmem_atmosphere_addition(self):
        atmosphere = AtmosphereBase()
        sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
        sp = atmosphere + sp

    def time_atmosphere_addition_double_snowpack(self):
        atmosphere = AtmosphereBase()
        sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
        sp2 = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
        sp = (atmosphere + sp) + sp2

    def peakmem_atmosphere_addition_double_snowpack(self):
        atmosphere = AtmosphereBase()
        sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
        sp2 = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
        sp = (atmosphere + sp) + sp2