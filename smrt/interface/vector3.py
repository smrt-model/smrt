
import numpy as np

class vector3(object):
    __slot__ = ['x', 'y', 'z', 'norm', 'norm2']

    #def __init__(self, x, y, z):
    #   self.x = x 
    #   self.y = y
    #   self.z = z
        #self.norm2 = np.sqrt(self.x**2 +self.y**2 + self.z**2)

    @staticmethod
    def from_angles(norm, mu, phi):
        v = vector3()
        sintheta = np.sqrt(1 - mu**2)
        v.x = norm * np.cos(phi) * sintheta
        v.y = norm * np.sin(phi) * sintheta
        v.z = norm * mu
        v.norm = norm
        v.norm2 = norm**2
        return v