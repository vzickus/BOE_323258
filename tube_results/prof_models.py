# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 17:50:54 2016

@author: vytas
"""
import numpy as np
import math
from scipy.optimize import *

# Degrees to radians
d2r = (np.pi / 180.0)


def V_gen(x, y, z, V_max, y_c, z_c, psi, theta, a, b):
    """
    Compute the general equation of the unidirectional flow profile in a tube.

    Coordinates
    -----------
        x, y, z : float

    Parameters
    ----------
        V_max : float
            The peak velocity of the flow.

        y_c, z_c : float
            The centre of the tube in y and z axes.

        R : float
            the radius of the tube.

        psi : float
            The tilt in the x-z plane of the tube in degrees.

        theta : float
            The tilt in the x-y plane of the tube in degrees.

        a, b : float
            The semi-major/minor axes of the elliptical paraboloid.

    Returns
    -------
        V : list of floats
            An array of velocity values at each x, y, z.
    """


    V = V_max * (1.0 - (np.sqrt((y - y_c)**2/a**2 + (z - z_c)**2/b**2))**2)
    V[V < 0.0] = 0.0
    return V


def rotation_matrix(axis, theta):
    """
    Compute the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    Euler-Rodrigues formula.Copy-pasted from:
    http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector

    Parameters
    ----------
        axis : 3d vector
            The (unit) vector along the axis of interest.
        theta : float
            The angle of tilt from the axis in radians.

    Returns
    -------
        3 x 3 Rotation matrix.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def V_xz_tilt_myway(data_coords, V_max, y_c, z_c, psi, theta, a, b):
    """
    Calculate the velocity values in a tilted tube as seen by the camera,
    using the value of height in the tube, that the imaging plane slices
    at a certain distance from the centre along x axis.

    Parameters
    ----------
        Same as V_gen

    Returns
    -------
        The projected (as seen on camera), velocity values on the x-y plane.

    """
    # Calculate the effective z value at a certain distance.
    z_eff = (data_coords[0] - x_c) * np.sin(psi * d2r) - data_coords[2] * (np.tan(psi * d2r)*np.sin(psi * d2r) - np.cos(psi * d2r))
    #By symmetry, effective y will be calculated in a similar way.    
    y_eff = (data_coords[0] - x_c) * np.sin(theta * d2r) - data_coords[1] * (np.tan(theta * d2r)*np.sin(theta * d2r) - np.cos(theta * d2r))
    return np.cos(psi * d2r) * np.cos(theta *d2r) * V_gen(data_coords[0], y_eff, z_eff, V_max, y_c, z_c, psi, theta, a, b)


def V_xz_tilt_jtway(data_coords, V_max, y_c, z_c, psi, a, b):
    """
    Calculate the velocity values in a tilted tube as seen by the camera,
    using the value of height in the tube, that the imaging plane slices
    at a certain distance from the centre along x axis.

    Parameters
    ----------
        Same as V_gen

    Returns
    -------
        The projected (as seen on camera), velocity values on the x-y plane.

    """
    # The axis we want to rotate to.
    axis = (0.0, 1.0, 0.0)
    cam_coords = (data_coords[0] - x_c,
                  data_coords[1] - y_c, data_coords[2] - z_c)
    RotM = rotation_matrix(axis, psi * d2r)
    # Rotate the matrix
    tube_coords = np.dot(RotM, cam_coords)
    xt, yt, zt = tube_coords
    return np.cos(psi * d2r) * V_gen(xt, yt, zt, V_max, y_c, z_c, psi, a, b)
    
