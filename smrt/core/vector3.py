import numpy as np


class vector3(object):
    __slot__ = ["x", "y", "z", "_norm", "_norm2"]

    @staticmethod
    def from_xyz(x, y, z):
        v = vector3()
        v.x = x
        v.y = y
        v.z = z
        v._norm = None
        v._norm2 = None
        return v

    @staticmethod
    def from_array(a):
        v = vector3()
        v.x = a[0]
        v.y = a[1]
        v.z = a[2]
        v._norm = None
        v._norm2 = None
        return v

    @staticmethod
    def from_angles(norm, mu, phi):
        v = vector3()
        sintheta = np.sqrt(1 - mu**2)
        v.x = norm * np.cos(phi) * sintheta
        v.y = norm * np.sin(phi) * sintheta
        v.z = norm * mu
        v._norm = norm
        v._norm2 = norm**2
        return v

    def norm(self):
        if self._norm is None:
            self._norm = np.sqrt(self.norm2())
        return self._norm

    def norm2(self):
        if self._norm2 is None:
            self._norm2 = self.x**2 + self.y**2 + self.z**2
        return self._norm2

    def __neg__(self):
        return vector3.from_xyz(-self.x, -self.y, -self.z)

    def __add__(self, other):
        return vector3.from_xyz(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vector3.from_xyz(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return vector3.from_xyz(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def cross(self, other):
        return vector3.from_xyz(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def normalize(self):
        return self * (1 / self.norm())

    def as_array(self):
        """
        Convert to an nd.array.
        """
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return "{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})".format(self=self)

    @staticmethod
    def matmul(matrix: np.ndarray, v):
        """
        Multiply a matrix (a 3x3 nd array)
        """
        return vector3.from_array(np.matmul(matrix, v.as_array()))
