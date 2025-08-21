from smrt.core import sensor

def time_iterate():
    freqs = [1e9, 2e9, 3e9]
    s = sensor.active(freqs, 55)
    [sub_s.frequency for sub_s in s.iterate("frequency")]

def peakmem_iterate():
    freqs = [1e9, 2e9, 3e9]
    s = sensor.active(freqs, 55)
    [sub_s.frequency for sub_s in s.iterate("frequency")]

def time_wavelength():
    s = sensor.Sensor(wavelength=0.21, theta_deg=0)
    s.wavelength
    s.frequency

def peakmem_wavelength():
    s = sensor.Sensor(wavelength=0.21, theta_deg=0)
    s.wavelength
    s.frequency

def time_passive_wrong_frequency_units_warning():
    sensor.passive([1e9, 35], theta=55)

def peakmem_passive_wrong_frequency_units_warning():
    sensor.passive([1e9, 35], theta=55)

def time_passive_mode():
    sensor.passive(35e9, 55, polarization="H")

def peakmem_passive_mode():
    sensor.passive(35e9, 55, polarization="H")

def time_active_wrong_frequency_units_warning():
    sensor.active([1e9, 35], 55)

def peakmem_active_wrong_frequency_units_warning():
    sensor.active([1e9, 35], 55)

def time_active_mode():
    sensor.active(35e9, 55)

def peakmem_active_mode():
    sensor.active(35e9, 55)
