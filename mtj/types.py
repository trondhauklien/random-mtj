from typing import TypedDict

import numpy.typing as npt
import numpy as np


class MaterialProps(TypedDict):
    """
    Dictionary defining the properties of the material.

    Attributes
    ----------
    K_u: float
        First order crystal anisotropy constant (J/m^3).
    M_s: float
        Saturation magnetization module (A/m).
    u_k: array_like
        Direction of easy axis as a Numpy array of floats with shape (3,).
    p: array_like
        Spin polarization unit vector as a Numpy array of floats with shape (3,).
    a_para: float
        Parallel STT coefficient.
    a_ortho: float
        Orthogonal STT coefficient.
    V: float
        Applied voltage (V).
    H_app: array_like
        Externally applied magnetic field as a Numpy array of floats with shape (3,).
    N: array_like
        Demagnetization tensor as a Numpy array of floats with shape (3,3).
    """

    K_u: float
    """First order crystal anisotropy constant (J/m^3)."""
    M_s: float
    """Saturation magnetization module (A/m)."""
    u_k: npt.NDArray[np.float64 | np.int_]
    """Direction of easy axis as a Numpy array of floats with shape (3,)."""
    p: npt.NDArray[np.float64 | np.int_]
    """Spin polarization unit vector as a Numpy array of floats with shape (3,)."""
    a_para: float
    """Parallel STT coefficient."""
    a_ortho: float
    """Orthogonal STT coefficient."""
    V: float
    """Applied voltage (V)."""
    H_app: npt.NDArray[np.float64 | np.int_]
    """Externally applied magnetic field as a Numpy array of floats with shape (3,)."""
    N: npt.NDArray[np.float64 | np.int_]
    """Demagnetization tensor as a Numpy array of floats with shape (3,3)."""
