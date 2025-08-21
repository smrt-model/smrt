from smrt.core.layer import make_microstructure_model


def time_microstructure_model():
    shs = make_microstructure_model("sticky_hard_spheres", radius=1.0, stickiness=0.5, frac_volume=0.3)

def peakmem_microstructure_model():
    shs = make_microstructure_model("sticky_hard_spheres", radius=1.0, stickiness=0.5, frac_volume=0.3)