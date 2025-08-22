import copy

import numpy as np
import xarray as xr

from smrt.core import result

class Suite:
    def setup(self):
        layer_coord = ("layer", [0, 1, 2])

        self.res_example = result.ActiveResult(
            [
                [
                    [[4.01445680e-03, 3.77746658e-03, 0.00000000e00]],
                    [[3.83889082e-03, 3.85904771e-03, 0.00000000e00]],
                    [[2.76453599e-20, -2.73266027e-20, 0.00000000e00]],
                ]
            ],
            coords=[
                ("theta", [35]),
                ("polarization", ["V", "H", "U"]),
                ("theta_inc", [35]),
                ("polarization_inc", ["V", "H", "U"]),
            ],
            channel_map={
                "VV": dict(polarization="V", polarization_inc="V"),
                "VH": dict(polarization="H", polarization_inc="V"),
            },
            other_data={
                "ks": xr.DataArray([1.0, 2.0, 3.0], coords=[layer_coord]),
                "ka": xr.DataArray([3.0, 2.0, 1.0], coords=[layer_coord]),
                "ke": xr.DataArray([4.0, 4.0, 4.0], coords=[layer_coord]),
                "thickness": xr.DataArray([0.1, 0.1, 0.1], coords=[layer_coord]),
            },
        )

        self.res_example2 = result.ActiveResult(
            [
                [
                    [[4e-03, 3e-03, 0], [8e-03, 6e-03, 0]],
                    [[3e-03, 3.85904771e-03, 0], [6e-03, 6.85904771e-03, 0]],
                    [[0, 0, 0], [0, 0, 0]],
                ],
                [
                    [[4e-03, 3e-03, 0], [8e-03, 6e-03, 0]],
                    [[3e-03, 3.85904771e-03, 0], [6e-03, 6.85904771e-03, 0]],
                    [[0, 0, 0], [0, 0, 0]],
                ]
            ],
            coords=[
                ("theta", [45, 50]),
                ("polarization", ["V", "H", "U"]),
                ("theta_inc", [45, 50]),
                ("polarization_inc", ["V", "H", "U"]),
            ],
            channel_map={
                "VV": dict(polarization="V", polarization_inc="V"),
                "VH": dict(polarization="H", polarization_inc="V"),
            },
            other_data={
                "ks": xr.DataArray([2.0, 4.0, 6.0], coords=[layer_coord]),
                "ka": xr.DataArray([3.0, 2.0, 1.0], coords=[layer_coord]),
                "ke": xr.DataArray([5.0, 6.0, 7.0], coords=[layer_coord]),
                "thickness": xr.DataArray([0.1, 0.1, 0.1], coords=[layer_coord]),
            },
        )


    def time_positive_sigmaVV(self):
        self.res_example.sigmaVV()

    def peakmem_positive_sigmaVV(self):
        self.res_example.sigmaVV()

    def time_positive_sigmaVH(self):
        self.res_example.sigmaVH()

    def peakmem_positive_sigmaVH(self):
        self.res_example.sigmaVH()

    def time_positive_sigmaHV(self):
        self.res_example.sigmaHV()

    def peakmem_positive_sigmaHV(self):
        self.res_example.sigmaHV()

    def time_positive_sigmaHH(self):
        self.res_example.sigmaHH()

    def peakmem_positive_sigmaHH(self):
        self.res_example.sigmaHH()

    def time_sigma_dB(self):
        self.res_example.sigmaVV_dB()
        self.res_example.sigmaHH_dB()
        self.res_example.sigmaHV_dB()
        self.res_example.sigmaVH_dB()

    def peakmem_sigma_dB(self):
        self.res_example.sigmaVV_dB()
        self.res_example.sigmaHH_dB()
        self.res_example.sigmaHV_dB()
        self.res_example.sigmaVH_dB()

    def time_sigma_dB_as_dataframe(self):
        self.res_example.sigma_dB_as_dataframe(channel_axis="column")

    def peakmem_sigma_dB_as_dataframe(self):
        self.res_example.sigma_dB_as_dataframe(channel_axis="column")

    def time_to_dataframe_with_channel_axis_on_column(self):
        self.res_example.to_dataframe(channel_axis="column")

    def peakmem_to_dataframe_with_channel_axis_on_column(self):
        self.res_example.to_dataframe(channel_axis="column")

    def time_to_dataframe_without_channel_axis(self):
        self.res_example.to_dataframe(channel_axis=None)

    def peakmem_to_dataframe_without_channel_axis(self):
        self.res_example.to_dataframe(channel_axis=None)

    def time_return_as_series(self):
        self.res_example.to_series()

    def peakmem_return_as_series(self):
        self.res_example.to_series()

    def time_concat_results(self):
        result.concat_results((self.res_example, self.res_example2), coord=("dim0", [0, 1]))

    def peakmem_concat_results(self):
        result.concat_results((self.res_example, self.res_example2), coord=("dim0", [0, 1]))

    def time_concat_results_other_data(self):
        res = copy.deepcopy(self.res_example)
        res2 = copy.deepcopy(self.res_example2)
        result.concat_results((res, res2), coord=("dim0", [0, 1]))

    def peakmem_concat_results_other_data(self):
        res = copy.deepcopy(self.res_example)
        res2 = copy.deepcopy(self.res_example2)
        result.concat_results((res, res2), coord=("dim0", [0, 1]))

    def time_single_scattering_albedo(self):
        self.res_example.single_scattering_albedo()

    def peakmem_single_scattering_albedo(self):
        self.res_example.single_scattering_albedo()

    def time_optical_depth(self):
        self.res_example.optical_depth()

    def peakmem_optical_depth(self):
        self.res_example.optical_depth()