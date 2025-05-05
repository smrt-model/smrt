
import numpy as np

class vector3(object):
    __slot__ = ['x', 'y', 'z', 'norm', 'norm2']

    @staticmethod
    def from_xyz(x, y, z):
        v = vector3()
        v.x = x
        v.y = y
        v.z = z
        v.norm2 = v.x**2 + v.y**2 + v.z**2
        v.norm = np.sqrt(v.norm2)
        return v

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

    def __add__(self, other):
        return vector3.from_xyz(self.x + other.x,
                                self.y + other.y,
                                self.z + other.z)

    def __sub__(self, other):
        return vector3.from_xyz(self.x - other.x,
                                self.y - other.y,
                                self.z - other.z)

    def __mul__(self, other):
        return vector3.from_xyz(self.x * other,
                                self.y * other,
                                self.z * other)
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1. / other)

    def cross(self, other):
        return vector3.from_xyz(self.y * other.z - self.z * other.y,
                                self.z * other.x - self.x * other.z,
                                self.x * other.y - self.y * other.x)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __repr__(self):
        return '{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})'.format(self=self)
