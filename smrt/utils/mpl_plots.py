# coding: utf-8

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, Formatter

from smrt.core.result import make_result
from smrt.core.model import make_model


def plot_snowpack(sp, show_vars=None, show_shade=False, ax=None):

    if ax is None:
        ax = plt.gca()

    depth = np.cumsum(sp.layer_thicknesses)
    xmax = 1.5 * depth[-1]

    ax.plot((0, 100 * xmax), (0, 0), '0.5')
    for lay, z in zip(sp.layers, -depth):
        if show_shade:
            ax.fill_between((0, 100 * xmax), [z] * 2, [z + lay.thickness] * 2, color='#55a9ff', alpha=lay.frac_volume)
        else:
            ax.plot((0, 100 * xmax), (z, z), '0.5')

        if show_vars:
            ax.text(0.8 * xmax, z + lay.thickness / 2, format_vars(lay, show_vars))

    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.set_aspect('equal', 'datalim')
    ax.set_xlim((0, 1))


def plot_streams(sp, emmodel, sensor, ilayer=None, ax=None):

    if ax is None:
        ax = plt.gca()

    depth = np.cumsum(sp.layer_thicknesses)
    xmax = 1.5 * depth[-1]

    if emmodel is None or sensor is None:
        raise RuntimeError("When show_cosine is activated, the 'emmodel' and 'sensor' arguments are compulsary")
    m = make_model(emmodel, CosineComputor)
    sensor.in_layer = ilayer  # a bit tricky to pass an argument...
    cosine = m.run(sensor, sp)

    # draw the stream in the air. Depending on the angle, we've different options
    theta0 = np.arccos(cosine.data.values.flat[0])
    x0 = 0.15 * xmax
    if theta0 > np.radians(45):
        zs = [x0 / np.tan(theta0), 0]
        xs = [0, x0]
        mask = [not np.isfinite(zs[0]), False]
    else:
        zs = [np.median(sp.layer_thicknesses), 0]
        xs = [x0 - np.median(sp.layer_thicknesses) * np.tan(theta0), x0]
        mask = [not np.isfinite(xs[0]), False]

    for lay, mu in zip(sp.layers, cosine.data.values.flat[1:]):
        x_stream = np.tan(np.arccos(mu)) * lay.thickness
        if np.isfinite(x_stream):
            xs.append(xs[-1] + x_stream)
            zs.append(zs[-1] - lay.thickness)
            mask.append(False)
        else:
            # ticky to be sure the next segment will be plot
            xs += [xs[-1] + lay.thickness * np.tan(sensor.theta)] * 2
            zs += [zs[-1] - lay.thickness] * 2
            mask += [True, False]

    xs = np.ma.masked_array(xs, mask)

    ax.plot(xs, zs, label="%g°" % np.degrees(sensor.theta))


def format_vars(lay, show_vars, delimiter=" "):

    # format string and scale
    format_map = dict(density=('%i kgm$^{-3}$', 1),
                      radius=('%i $\mu$m', 1e6),
                      corr_length=('%i $\mu$m', 1e6),
                      temperature=('%g.0 K', 1))
    txt = []
    for v in show_vars:
        x = getattr(lay, v, None)
        if x is None and hasattr(lay, "microstructure"):
            x = getattr(lay.microstructure, v, None)
            if x is None:
                continue

        if v in format_map:
            txt.append(format_map[v][0] % (x * format_map[v][1]))
        else:
            txt.append("%g" % v)
    return delimiter.join(txt)


class CosineComputor(object):

    # this class mimics a RT solver but only do very basic computation and return the cosine angle instead of radiances
    def solve(self, snowpack, emmodel_instances, sensor, atmosphere):

        eps = np.array([emmodel.effective_permittivity() for emmodel in emmodel_instances])
        n = [1] + list(np.real(np.sqrt(eps)))

        sensor_in_layer = getattr(sensor, "in_layer", None)
        if sensor_in_layer:
            n /= n[sensor_in_layer]

        cosine = np.sqrt(1 - (np.sin(sensor.theta) / n)**2)
        return make_result(sensor, cosine, [('layer', np.arange(1 + len(snowpack.layers)))])


#
# register a new reciprocal scale to easily show stickiness plot
# the placement of the tick could be improved
#

class ReciprocalScale(mscale.LinearScale):
    name = 'stickiness_reciprocal'

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(FixedLocator([0.07, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1, 1000]))

        class StickinessFormatter(Formatter):
            def __call__(self, x, pos=None):
                return "%g" % x

        axis.set_major_formatter(StickinessFormatter())
        axis.set_minor_formatter(StickinessFormatter())

    def get_transform(self):
        return self.ReciprocalTransform()

    class ReciprocalTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.reciprocal(np.maximum(a, 0.01))

        def inverted(self):
            return ReciprocalScale.InvertedReciprocalTransform()

    class InvertedReciprocalTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.reciprocal(a)

        def inverted(self):
            return ReciprocalScale.ReciprocalTransform()


# Now that the Scale class has been defined, it must be registered so
# that ``matplotlib`` can find it.
mscale.register_scale(ReciprocalScale)
