# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:36:01 2016

@author: vytas
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from Tkinter import Tk
from tkFileDialog import askdirectory
import glob
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from numpy import linspace, meshgrid
from matplotlib.mlab import griddata



mpl.rcParams['xtick.labelsize'] = 25
mpl.rcParams['ytick.labelsize'] = 25
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['xtick.labelsize'] = 25
mpl.rcParams['ytick.labelsize'] = 25
mpl.rcParams['lines.linewidth'] = 5
mpl.rc('text', usetex = True)
mpl.rc('font', **{'family':"sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{siunitx}',
    r'\usepackage{sfmath}', r'\sisetup{detect-family = true}',
    r'\usepackage{amsmath}']}
plt.rcParams.update(params)

xyuv_files = sorted(glob.glob(str('/home/vytas/1phd/paper_figures/beads_only_0.5ulmin_50um/2016-11-18 13.48.35 corr_avg 0.5ul_sad_1.05_BF_mask_3_width_12/xyuvms2n' + '/*.txt')))

# ccd pixel size in microns
pixel_size =  6.4500
# magnification
M = 20.0
# effective pixel_size
px = pixel_size/M
# time difference between two pulses or exposures.
dt =  0.3#/1000.0
# z spacing in um, between z slices
spacing = 2#int(input("Enter z spacing (integer microns): "))
#number of z stacks
z = len(xyuv_files)

#Change the threshold of p2p test
p2p_threshold = 1.100001

# data reformating loop
for i in range(z):
    # load the piv_results
    flow_data = np.loadtxt(xyuv_files[i])
    # differentiate the valid vectors from invalid ones
    invalid = flow_data[:,5] < p2p_threshold#.astype('bool')
    valid = ~invalid
    # generate z values to be glued to the piv_results array
    z_vals = np.ones((flow_data[valid].shape[0],1)) * i * spacing
    # stack the piv results and z together
    a = np.hstack((flow_data[valid], z_vals))
    # care must be taken not to stack the initial values twice (i.e. with it self)

    if i == 0:
        b = a


    if i > 0:

        b = np.vstack((b, np.hstack((flow_data[valid], z_vals))))
    
b4 = len(b[:,1])
print "before removal", b4
# Discard y values beyond nominal centre
#both bf and fl
b = b[np.less_equal(np.sqrt((b[:,1] *px - 25)**2 + (b[:,6] -36.0)**2 ), 26) ]

af = len(b[:,1])
print "after removal", af
print "difference = ", b4 - af

#Discard velocity values well above maximum or at zero
before_discard = len(b)
print "len before discard", before_discard
b = b[np.logical_and(b[:,2]*px/dt > 0.0 , b[:,2]*px/dt< 28.0)]
after_discard = len(b)
print "lend after discard", after_discard
print "difference", before_discard - after_discard

# Generate the 3D position coordinates to be fed in the fit.
data_coords = (b[:, 0] * px, b[:, 1] * px, b[:, 6])
psi = 1.0
# The u-component of the velocity.
u_data = np.cos(psi * d2r) * b[:, 2]*px/dt #* 1000 # velocity in mm

print "max value in this region: ", np.amax(u_data)
maxval = np.where(u_data == np.amax(u_data))


def grid(x, y, z, resX=200, resY=200):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp = 'linear')
    X, Y = meshgrid(xi, yi)
    return X, Y, Z


plt.figure(figsize = (8,6))  


XX, YY, ZZ = grid(data_coords[1], data_coords[2], u_data)
plt.contourf(XX, YY, ZZ)
plt.xlabel("y, ${\mu}m$", fontsize=25)
plt.ylabel("z, ${\mu}m$", fontsize=25)
plt.colorbar()
plt.tight_layout()    

