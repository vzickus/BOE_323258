# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:21:30 2017

@author: vytas
"""
import glob
from matplotlib.mlab import griddata
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
from collections import deque
from scipy.integrate import simps
import matplotlib as mpl
import scipy

mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['lines.linewidth'] = 1
mpl.rc('text', usetex = True)
mpl.rc('font', **{'family':"sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{siunitx}',
    r'\usepackage{sfmath}', r'\sisetup{detect-family = true}',
    r'\usepackage{amsmath}']}
plt.rcParams.update(params)

def grid(x, y, z, resX=50, resY=50):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    #print "min max x", min(x), max(x)
    dx = xi[1] - xi[0]
    print "dx: ", dx
    yi = np.linspace(min(y), max(y), resY)
    #print "min max y", min(y), max(y)
    dy = yi[1] - yi[0]
    print "dy: ", dy
    Z = griddata(x, y, z, xi, yi, interp = 'linear')
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z, dx, dy


ph_bins = np.round(np.linspace(0.0,2*np.pi,31),3)[:-1]#np.arange(0.000,2*np.pi,0.209)


peak_flow = []
flow_rate = []
flow_rate_pos = []
flow_rate_neg = []

flow_rate_one_two = [-2.1288080687944384, -1.672994892271872, -1.2056210984554423, 1.083197752993259, 2.1663112566668485,2.6914134892102353, 2.4320071957231306, 1.7083900544133925, 1.3754997795562494, 1.0911889972370399, 0.94573237033916357,0.78408922895855526, 0.6700566835142806, 0.56676940665121556, 0.30406499904039458, -0.16571398174268664, -0.38763051191169173,-0.18471031495540963, 0.071827190342187319, 0.203442670277453, 0.36380166050220364, 0.38028855397630779, 0.35996002671583022,0.46920404100071605, 0.63659292851760618, 1.1232396812506686, 1.4241630072649663, 0.43138908424327571, -0.29941111155927991,-0.90704259493898887]

flow_rate_zero_two =  [-1.9690835482344746, -2.8721428954163328, -0.97639303333610339, 0.95798721287750777,2.1705650663230549,2.6494077334896518, 2.4357500147542925, 1.6747400431685151,1.3230569219951847,1.1384134173507334,0.96414785757508725, 0.78235596173805211, 0.69653884641980013, 0.58518580576681112, 0.31266141150452631,-0.19053645882568834, -0.36531389464652531, -0.19871045549043043, 0.061840619103153346, 0.22643646187547151, 0.37547763444616616,0.3664346264065923, 0.36997968501941531, 0.4589567874874626, 0.63364776826600522, 1.1034130331863572, 1.4559685926354435,0.35411442354674244, -0.30767636308962548, -0.89945102744532168]

flow_rate_full = [-2.166637158629968, -2.7029450248669074, -1.0513087161106944, 1.0503871353901391, 2.2132796869401075, 2.6878791683632435, 2.4446087618743966, 1.6629150813230942, 1.3591273190269433, 1.1276894578268146, 0.95293238642899225, 0.78014472954547764, 0.68450968101425635, 0.57772007772872913, 0.3053953659001562, -0.18237864638127632, -0.38733667856909715, -0.1917583152747282, 0.066379169723781578, 0.21677007779644766, 0.36421587849348402, 0.37451753466000148, 0.36670276240040328, 0.46926797536938325, 0.64447311373130534, 1.1122227745862581, 1.4074915409376099, 0.38062102211708843, -0.28517098783262557, -0.9120137193887693]

flow_rate_err =  np.abs(np.asarray(flow_rate_one_two ) - np.asarray(flow_rate_zero_two))

path_to_xyuv_files =  '/home/vytas/1phd/plist parsing/code_for_paper/corr_avg/2018-02-09 13.44.36_single_plane_ipt05_04_0524_24_HW_48_72_ol_hw_12_12_in_bin_0.0_fin_bin_6.28318530718_step_0.209'#'/home/vytas/1phd/plist parsing/corr_avg/2017-08-30 11.47.42_re_split_f1_o0__unnatended_f1_o1__16_16_HW_32_32_ol_hw_8_8sad'#
plt.figure(figsize = (16,8))
for ph_bin in ph_bins:
    xyuv_files= (sorted(glob.glob(str(path_to_xyuv_files + '/xyuvms2n/' +'*bin_%1.3f' + '*.txt')%ph_bin)))
    #print xyuv_files[0]
    bining = 2
    # ccd pixel size in microns
    pixel_size =  6.4500 * bining
    # magnification
    M = 20.0
    # effective pixel_size
    px = pixel_size/M
    # time difference between two pulses or exposures.
    dt =  2.5/1000.0
    # z spacing in um, between z slices
    spacing = 8#int(input("Enter z spacing (integer microns): "))
    #number of z stacks
    z = len(xyuv_files)
    #Change the threshold of p2p test
    p2p_threshold = 1.07
    iw_int_threshold = 5e7

    # data reformating loop
    for i in range(z):
        # load the piv_results
        flow_data = np.loadtxt(xyuv_files[i])
        # differentiate the valid vectors from invalid ones
        invalid = (flow_data[:,5] < p2p_threshold)# & (flow_data[:,7] < iw_int_threshold)  #flow_data[:,5] < p2p_threshold#.astype('bool')
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
#    b = b[np.less(b[:,6],  88) ] 
    #print b.shape
    # Generate the 3D position coordinates to be fed in the fit.
    data_coords = (b[:, 0] * px, b[:, 1] * px, b[:, 6])
    #print np.unique(data_coords[0]/px)
    # The u-component of the velocity.
    u_data = b[:, 2]*px/dt #* 1000 # velocity in mm
    v_data = b[:, 3]*px/dt



    x_intr = 239.5 * px# np.unique(data_coords[0])[np.where(np.unique(data_coords[0]/px) == x_px_of_interest)]
    
#    print "in pixels: ", np.unique(data_coords[0]/px), "\n"

#    x_intr = 162.2175#146.7375#162.2175#138.9975 #154.4775#169.9575
    #print "x intr", x_intr, "in px: ", x_intr/px 
    xi = np.where((data_coords[0] == x_intr))
    # axes of interest
    
    X = data_coords[0][xi]
    Y = data_coords[1][xi] 
    Z = data_coords[2][xi]

    UV_mag= np.sqrt(u_data[xi] ** 2 + v_data[xi] **2)
    U = u_data[xi]
    print "umax, umin:", np.max(U), np.min(U)
    #print u_data[xi]
    #print len(X)
    x_col = np.reshape(X,(len(X),1))
    y_col = np.reshape(Y,(len(X),1))
    z_col = np.reshape(Z,(len(X),1))  
    xyz = np.hstack((x_col, y_col, z_col))


    XX, YY, ZZ, dx, dy = grid(Y, Z, -U) 



    

    
    #calculate flowrate 
    fr = dx*dy*sum(ZZ[~ZZ.mask])/1000000.0
    peak_flow.append(np.mean(ZZ)/1000.0)
    flow_rate.append(dx*dy*sum(ZZ[~ZZ.mask])/1000000.0)
  
    
    if fr >= 0:
        flow_rate_pos.append(fr)
    elif fr < 0:
        flow_rate_neg.append(fr)
#plt.xticks(np.arange(0.0,6.8,0.4))
#plt.title('dx*dx against phase')
#plt.xlabel('phase bin')
#plt.ylabel('dx*dy, microns squared')
#plt.ylim(-0.02,0.14)
#plt.show()
pos_fr = round(sum(flow_rate_pos),2)
neg_fr = round(sum(flow_rate_neg),2)
print "positive", pos_fr
print "negative", neg_fr
#assert  (np.abs(sum(flow_rate) - (sum(flow_rate_pos) + sum(flow_rate_neg))) < 1)
#peak_flow = deque(peak_flow)
print "max flow: ", np.max(peak_flow)
print "max flow rate: ", np.max(flow_rate)
#print "flwo rate errs: ", flow_rate_err
print "flow rate ", flow_rate
flow_rate = deque(flow_rate)
#flow_rate_err = deque(flow_rate_err)
#peak_flow.rotate(6)
#flow_rate.rotate(7)    
#flow_rate_err.rotate(7)
#plt.figure(figsize = (12,8))
#plt.scatter(ph_bins, peak_flow)
#plt.show()



######################################
#
#
#
#dx,dy may not be constant!!!!
#
#
##########################################
dx = 0.016 #in this case, this is the time "or phase 
print "trapz, abs total", np.trapz(np.abs(flow_rate), dx = dx) 
print "trapz, total", np.trapz(flow_rate, dx =dx) 
abs_total_pumped = np.trapz(np.abs(flow_rate), dx =dx)
print "simps, abs total", abs_total_pumped
print "simps, total", scipy.integrate.simps(flow_rate, dx = dx)
print "dx,dy may not be constant!!!!"
regurg = ((np.trapz(np.abs(flow_rate), dx =dx)
 - np.trapz(flow_rate, dx = dx))/2.0)
net_flow = abs_total_pumped - regurg
print "regurgitated: ", regurg
print "net_flow: ", net_flow
#plt.figure()
#plt.plot(ph_bins, np.abs(flow_rate),'o')
#plt.plot([0,6.2],[0,0], 'k--')
#plt.plot(ph_bins, flow_rate, '.')
#plt.show()
print "sum flow_rate: ", sum(np.abs(flow_rate))



ph_offset = np.linspace(0.0,2*np.pi,31)[3]
phases = np.linspace(0.0,2*np.pi,31)[:-1]
#print phases
phases = phases.tolist()
#phases.append(0)
#phases = np.asarray(phases)
phases = (phases + ph_offset)%(2*np.pi)

#phases = (np.linspace(0.0,2*np.pi,31)+ph_offset)%(2*np.pi)
#(np.linspace(0.0,2*np.pi,31)[:-1])+ph_offset)%(2*np.pi)
#phases = np.roll(phases,-3)
#print phases, len(phases)
ticks = np.roll((np.round(np.linspace(0.0,2*np.pi,31)[:-1],3)),2)
#print ticks


#flow_rate = flow_rate.tolist()
#flow_rate.append(flow_rate[0])
#print "b4", flow_rate

#flow_rate = np.roll(flow_rate,-2)
#print np.array(flow_rate)

from scipy.interpolate import spline
xold = np.linspace(0.0,2*np.pi,31)[:-1]
xnew = np.linspace(xold.min(),xold.max(),30)
power_smooth = spline(xold,np.roll(flow_rate,2),xnew)

#,'w-', linewidth = 2, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])





plt.figure(figsize = (12,8))
#plt.xticks(ticks, rotation = 45)#(np.arange(0.0,6.8,0.4))
plt.plot([0.0, 2*np.pi],[0,0], 'k--')
#plt.scatter(ticks, ticks, facecolors='none', edgecolors='none')
plt.xticks(ticks,[i for i in np.roll(ticks,2)], rotation ='45')
#plt.scatter(ph_bins, flow_rate)
plt.errorbar(xnew , np.array(power_smooth),yerr = flow_rate_err, fmt='k', lw = 1.5,capsize=10)
plt.xlabel("phase, rad", fontsize=18)
plt.ylabel("flow rate, nL/s", fontsize=18)
#plt.grid()
plt.plot(xnew,power_smooth,color = 'k')
#plt.plot(phases, flow_rate)
plt.fill_between(xnew, np.array(power_smooth), where=np.array(power_smooth) >= 0,
                 interpolate = True, color='red',alpha=0.5, hatch = '//', label=('Pumped, ' + str(round(net_flow,3)) + " nL"))
plt.fill_between(xnew, np.array(power_smooth), where=np.array(power_smooth) < 0,interpolate = True,
                 color = 'blue',alpha=0.5, hatch = 'x',  label=('Regurgitated, ' + str(round(regurg,3)) + " nL"))
plt.legend(loc='upper right', fontsize = 18)
plt.scatter(phases, np.roll(flow_rate,-1), color = 'k', zorder = 5)
#plt.ylim(-1.5,3.0)
#plt.yticks(np.arange(-1.5,3.5,0.5))
#plt.grid()
plt.xlim(-0.1,6.15)
plt.tight_layout()
#plt.savefig("/home/vytas/1phd/paper_figures/flow_rate" + ".pdf", format = 'pdf', dpi = 300,  bbox_inches='tight')

plt.show()
plt.imshow()

