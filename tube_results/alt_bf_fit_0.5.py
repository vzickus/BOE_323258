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
from prof_models import *
import __builtin__
__builtin__.x_c = 511.5000 * 6.4500/20.0000

#print "x_c in microns is:", x_c


mpl.rcParams['xtick.labelsize'] = 25
mpl.rcParams['ytick.labelsize'] = 25
mpl.rcParams['lines.linewidth'] = 2
#mpl.rcParams['lines.markersize'] = 60
mpl.rc('text', usetex = True)
mpl.rc('font', **{'family':"sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{siunitx}',
    r'\usepackage{sfmath}', r'\sisetup{detect-family = true}',
    r'\usepackage{amsmath}']}
plt.rcParams.update(params)

xyuv_files = sorted(glob.glob(str('/home/vytas/1phd/paper_figures/fep_tube_fitting/beads_only_0.5ulmin_50um/2016-11-18 13.48.35 corr_avg 0.5ul_sad_1.05_BF_mask_3_width_12/xyuvms2n' + '/*.txt')))

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
p2p_threshold = 1.10

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


# tube_coords. Obtained from first pass fitting.
print  "u max in mm/s ", np.max(b[:,2] * px/dt)
print  "u mmin in mm/s ", np.min(b[:,2] * px/dt)

y_c =  26.4472072871
z_c =  35
psi =  0.988238648447
theta =  0.196894637307

d2r = (np.pi / 180.0)

z_eff = (b[:,0] * px  - x_c) * np.sin(psi * d2r) - b[:,6]  * (np.tan(psi * d2r)*np.sin(psi * d2r) - np.cos(psi * d2r))
z_eff = np.reshape(z_eff, (len(z_eff),1))

y_eff = (b[:,0] * px - x_c) * np.sin(theta * d2r) - b[:,1]* px  * (np.tan(theta * d2r)*np.sin(theta * d2r) - np.cos(theta * d2r))
y_eff = np.reshape(y_eff, (len(y_eff),1))

u_eff = b[:,2] * px/dt * np.cos(psi * d2r) * np.cos(theta *d2r)
u_eff = np.reshape(u_eff, (len(u_eff),1))

x_same = np.reshape(b[:,0], (len(b[:,0]), 1))

tube_coords_u_eff = np.hstack((x_same * px - x_c, y_eff, z_eff, u_eff))

#assumed tube radius in microns
asmd_tb_r = 25.0

before_discard = np.shape(tube_coords_u_eff)[0] + 0.0

tube_coords_u_eff = tube_coords_u_eff[np.less_equal(np.sqrt((tube_coords_u_eff[:,1] -y_c )**2 + (tube_coords_u_eff[:,2] - z_c)**2 ), asmd_tb_r) ]

after_discard =  np.shape(tube_coords_u_eff)[0] + 0.0

percent_inside = after_discard/before_discard * 100.0
assert(percent_inside < 100.0)
perc_out = np.round(100.0 - percent_inside,3)
print "percent of data point outside of the nominal tube radius", perc_out

tube_coords = tube_coords_u_eff[:,:3]

tube_coords = (tube_coords[:,0], tube_coords[:,1], tube_coords[:,2])

y_eff_min, y_eff_max = np.min(tube_coords[1]), np.max(tube_coords[1])
z_eff_min, z_eff_max = np.min(tube_coords[2]), np.max(tube_coords[2])

assert((y_eff_max - y_eff_min) <= asmd_tb_r*2.0 )
assert((z_eff_max - z_eff_min) <= asmd_tb_r*2.0 )

u_eff_array = tube_coords_u_eff[:,3:4]

u_eff = u_eff_array[:,0]

data_coords = tube_coords

u_data = u_eff

print "max u eff", np.max(u_data)

# Initial guess parameters. Will be close to the initial pass. They DO matter.
V_max_g = 8
y_c_g = 26.0
z_c_g = 30.0
R_g = 26.2
psi_g = 0.0
theta_g = 0.0
a = 25.0
b =  25.0
# Set parameter bounds to something reasonable.
param_bounds=([6, 20, 10, -2, -4, 1, 1],[10, 60, 40, 2, 4, 40, 60])

params, extras = curve_fit(V_xz_tilt_myway, tube_coords, u_eff, bounds = param_bounds, p0=[V_max_g, y_c_g, z_c_g, psi_g, theta_g, a, b])


print "sqrt of diagonals: ", np.sqrt(np.diag(extras))

u_err = 0.25 * px/dt

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

plt.figure(figsize = (8,6))
print "\nBest fit parameters vz: "
print "_" * 25
print "V_max = ", params[0], "\ny_c = ", params[1], "\nz_c = ", params[2],  "\npsi = ", params[3], "\ntheta = ", params[4], "\na = ", params[5], "\nb = ", params[6]
y_c_new = params[1]
z_c_new = params[2]
# The fitted values in 3D space
V_3d_fit  = V_xz_tilt_myway(data_coords,params[0],params[1],params[2],params[3],params[4], params[5], params[6])
print "_" * 25
# calculate residuals
#residuals = u_data - V_3d_fit
## sum of squared residuals
#ssqres = sum(residuals**2)
#print "SSR whole data set:\n", ssqres
#print "_" * 25
# choose a z slice of interest taking the spacing into account.
plt.figure(figsize = (8,6))
print np.unique(z_eff)
z_intr = find_nearest(z_eff, z_c_new)
print "z_intr", z_intr
x_intr =   px * 511.5000
zi = np.where((data_coords[2] == z_intr))# & (data_coords[0] == x_intr))
# axes of interest
X = data_coords[0][zi]
Y = data_coords[1][zi]
Z = data_coords[2][zi]

ylimit = 9
# translate the y points, so we can match xz and xy planes.
y_translate = -y_c_new


plt.errorbar(Y + y_translate, u_data[zi],yerr = u_err, fmt='o')
plt.plot((-asmd_tb_r,-asmd_tb_r),(0,ylimit),'k--')
plt.plot((asmd_tb_r,asmd_tb_r),(0,ylimit),'k--')
plt.axvspan(-(asmd_tb_r + 5.0), -asmd_tb_r, color='gray', alpha=0.5, lw=0)
plt.axvspan(asmd_tb_r, asmd_tb_r + 5.0, color='gray', alpha=0.5, lw=0)

# skip every "skip" value in a list
skip = 20
#For each x position along the tube, plot the fit there.
for X_i in np.unique(X)[::skip]:
    # The coordinates over which the values of the fit will be plotted.
    fit_coords = (np.ones(len(Y)) * X_i, Y , np.ones(len(X)) * z_intr)
    # Calculate the fit values at the above coodrindates
    V_fit = V_xz_tilt_myway(fit_coords, params[0],params[1],params[2],params[3],params[4], params[5], params[6])
    y_model = np.arange(np.amin(data_coords[1] ) - 100,np.amax(data_coords[1]) + 100,0.1,)#
    model_coords  = (np.ones(len(y_model)) * X_i, y_model, np.ones(len(y_model)) * z_intr)
    V_fit = V_xz_tilt_myway(model_coords, params[0],params[1],params[2],params[3],params[4], params[5], params[6])

    # plotting xy plane
    plt.plot(model_coords[1] + y_translate,V_fit,'r-', label = "a = %.3f $\si{\um}$" %params[5]) #, alpha = 0.5)
    plt.legend(loc = 8, fontsize=20)

    plt.xlabel("y, $\si{\um}$",fontsize=25)
    plt.ylabel("u, mms$^{-1}$",fontsize=25)
    plt.ylim(0, ylimit)
    plt.xlim(-(asmd_tb_r + 5.0),asmd_tb_r + 5.0)
    plt.grid(True)
    plt.tight_layout()

y_intr =  find_nearest(y_eff, y_c_new)
print "y_intr",  y_intr
# Look at where in the data we find this y value as well
yi = np.where((data_coords[1] == y_intr))# & (data_coords[0] == x_intr))

Xy = data_coords[0][yi]
Yy = data_coords[1][yi]
Zy = data_coords[2][yi]

print "len Zy", len(Zy)

plt.figure(figsize = (8,6))

z_translate = -z_c_new
plt.errorbar(Zy + z_translate , u_data[yi],yerr = u_err, fmt='o')

plt.plot((-asmd_tb_r,-asmd_tb_r),(0,ylimit),'k--')

plt.plot((asmd_tb_r,asmd_tb_r),(0,ylimit),'k--')
plt.axvspan(-(asmd_tb_r + 5.0), -asmd_tb_r, color='gray', alpha=0.5, lw=0)
plt.axvspan(asmd_tb_r, asmd_tb_r + 5.0, color='gray', alpha=0.5, lw=0)

fit_coords = (np.ones(len(Xy)) * Xy, np.ones(len(Xy)) * Yy, Zy)
V_fit = V_xz_tilt_myway(fit_coords, params[0],params[1],params[2],params[3],params[4], params[5], params[6])

plt.xlabel("z, $\si{\um}$",fontsize=25)
#plt.xticks(rotation=45)
plt.ylabel("u, mms$^{-1}$",fontsize=25)

plt.grid()

for X_i in np.unique(X)[::30]:
    #print X_i
    for Y_i in np.unique(Yy):
        # The coordinates over which the values of the fit will be plotted.
        fit_coords_xz = (np.ones(len(Xy)) * X_i, np.ones(len(Xy)) * Y_i, Zy)
        # Calculate the fit values at the above coodrindates
        V_fit = V_xz_tilt_myway(fit_coords_xz, params[0],params[1],params[2],params[3],params[4], params[5], params[6])

        #populate coords for a smooth fit overlay
        z_model = np.arange(np.amin(data_coords[2]) - 100,np.amax(data_coords[2]) + 100,0.5,)#
        model_coords  = (np.ones(len(z_model)) * X_i, np.ones(len(z_model)) * Y_i, z_model  )
        V_fit = V_xz_tilt_myway(model_coords, params[0],params[1],params[2],params[3],params[4], params[5], params[6])
        plt.plot(model_coords[2] + z_translate, V_fit, 'r-', label = "b = %.3f $\si{\um}$" %params[6])
        plt.legend(loc = 8, fontsize = 20)
        plt.xlim(-(asmd_tb_r + 5.0), asmd_tb_r + 5.0)
        plt.ylim(0, ylimit)
        plt.grid(True)
        plt.tight_layout()



def grid(x, y, z, resX=200, resY=200):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp = 'linear')
    X, Y = meshgrid(xi, yi)
    return X, Y, Z


#plt.figure(figsize = (8,6))
fig, ax = plt.subplots(figsize = (8,6))

shift_centre_x = y_c
shift_centre_y = z_c

# How do I just look at the middle x:
xi_t = np.where(tube_coords[0] == x_intr - x_c)
#print "xi t here: ", xi_t

XX, YY, ZZ = (grid(tube_coords[1][xi_t] - shift_centre_x, tube_coords[2][xi_t] -shift_centre_y, u_eff[xi_t]))
centre = np.where(ZZ == np.max(ZZ))
X_c = 0
Y_c = 0
#print X_c, Y_c

plt.contourf(XX, YY, ZZ)
circle2 = plt.Circle((X_c, Y_c), asmd_tb_r, color='k',lw=3, linestyle = '--', fill=False)
ax.add_artist(circle2)
#plt.title('u-components in yz plane', fontsize=25, y = 1.05)
plt.xlabel("y, $\si{\um}$", fontsize=25)
plt.ylabel("z, $\si{\um}$", fontsize=25)
#plt.xticks(rotation=45)
plt.ylim(-(asmd_tb_r + 5.0), (asmd_tb_r + 5.0))
plt.xlim(-(asmd_tb_r + 5.0),(asmd_tb_r + 5.0))
#plt.zlim(0,7.5)
cbar = plt.colorbar()
cbar.set_label(label='u, mms$^{-1}$', size = 25)
plt.tight_layout()
plt.show()
#plt.savefig('./flow_BF_50um_0.5ulmin_beads_only_contour_%s_percent_outside.eps' %(perc_out),format = 'eps', dpi = 1200)
