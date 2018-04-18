import multiprocessing
from multiprocessing import Queue
import numpy as np
import matplotlib.pyplot as pl
from math import log
import scipy.signal
import os, sys
sys.path.insert(0, '/home/vytas/1phd/j_postacquisition_fork')
from j_py_sad_correlation import *
import numpy.lib.stride_tricks
from numpy import ma
from scipy.signal import correlate2d, fftconvolve
from pylab import subplot
import matplotlib.cm as cmps
import errno
import csv
import glob
from tqdm import *

class Multiprocesser():
    def __init__ ( self, data_dir, skip_first = False, first_frame_even = True, frame_format = '*.tif'):
        """A class to handle and process large sets of images.

        This class is responsible of loading image datasets
        and processing them. It has parallelization facilities
        to speed up the computation on multicore machines.
        
        It currently support only image pair obtained from 
        conventional double pulse piv acquisition. Support 
        for continuos time resolved piv acquistion is in the 
        future.
        
        
        Parameters
        ----------
        data_dir : str
            the path where image files are located 
        """
        
        even_frames = []
        odd_frames = []
        all_frames = sorted( glob.glob( os.path.join( os.path.abspath(data_dir), frame_format ) ) )
        #we might want to skip the first frame in a continuous dataset to match the "wrong" frames
        #i.e. prolong the dt
        
        if skip_first:
            all_frames = all_frames[1:]
        
        #We don't know if we every frame has a pair, so discard the last frame in an odd numbered length series
        if len(all_frames) % 2 == 1:
            all_frames = all_frames[:-1]    
        #sort the frames by even and odd depending on the last digit of their filename
        #assumption is that our first image from the series is the first frame in the pair.
        for filename in all_frames:
            basename, ext = os.path.splitext(filename)
            if int(basename[-1]) % 2 == 0:
                even_frames.append(filename)
            else:
                odd_frames.append(filename)
                
        #sanity check 
        if (len(even_frames) + len(odd_frames) ) < len(all_frames):
            raise ValueError('Sum of odd and even frames is less than the total number of frames!') 
        print "total number of frames: ", len(even_frames) + len(odd_frames)
        
        if first_frame_even:
            self.files_a = sorted(even_frames)
            self.files_b = sorted(odd_frames)
        else: 
            self.files_a = sorted(odd_frames)
            self.files_b = sorted(even_frames)
  
        # number of images
        self.n_files = len(self.files_a)
        
        # check if everything was fine
        if not len(self.files_a) == len(self.files_b):
            raise ValueError('Something failed loading the image file. There should be an equal number of "a" and "b" files.')
            
        if not len(self.files_a):
            raise ValueError('Something failed loading the image file. No images were found. Please check directory and image template name.')

    def run( self, func, n_cpus=1 ):
        """Start to process images.
        
        Parameters
        ----------
        
        func : python function which will be executed for each 
            image pair. See tutorial for more details.
        
        n_cpus : int
            the number of processes to launch in parallel.
            For debugging purposes use n_cpus=1
        
        """

        # create a list of tasks to be executed.
        image_pairs = [ (file_a, file_b, i) for file_a, file_b, i in zip( self.files_a, self.files_b, range(self.n_files) ) ]
        
        # for debugging purposes always use n_cpus = 1,
        # since it is difficult to debug multiprocessing stuff.
        if n_cpus > 1:
            pool = multiprocessing.Pool( processes = n_cpus )
            res = pool.map( func, image_pairs )
        else:
            for image_pair in image_pairs:
                func( image_pair )
                

class PivMultiprocesser():
    def __init__ ( self, phase_matched_list):
        """A class to handle and process large sets of images.
        
        Parameters
        ----------
        phase_matched_list : list
            list with phase matched frames 
        """
        frames_a = []
        frames_b = []
        
        for count_phase_matched_frames, frame in enumerate(phase_matched_list[:,:2]):                            
            file_a = frame[0].path
            file_b = frame[1].path
            frames_a.append(file_a)
            frames_b.append(file_b)

        self.files_a = frames_a
        self.files_b = frames_b
        # number of images
        self.n_files = len(self.files_a)
        
        # check if everything was fine
        if not len(self.files_a) == len(self.files_b):
            raise ValueError('Something failed loading the image file. There should be an equal number of "a" and "b" files.')
            
        if not len(self.files_a):
            raise ValueError('Something failed loading the image file. No images were found. Please check directory and image template name.')

    def run( self, func, n_cpus=1 ):
        """Start to process images.
        
        Parameters
        ----------
        
        func : python function which will be executed for each 
            image pair. See tutorial for more details.
        
        n_cpus : int
            the number of processes to launch in parallel.
            For debugging purposes use n_cpus=1
        
        """

        # create a list of tasks to be executed.
        image_pairs = [ (file_a, file_b, i) for file_a, file_b, i in zip( self.files_a, self.files_b, range(self.n_files) ) ]
        
        # for debugging purposes always use n_cpus = 1,
        # since it is difficult to debug multiprocessing stuff.
        #corr_queue = Queue()

        if n_cpus > 1:
            pool = multiprocessing.Pool( processes = n_cpus )
            res = pool.map( func, image_pairs )
        else:
            for image_pair in image_pairs:
                func( image_pair )
                
def pad_array(array, offsets):

    """
    array: Array to be padded

    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros((array.shape[0] + offsets[0],array.shape[1] + offsets[1])) + 0.001
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim]/2, offsets[dim]/2 + array.shape[dim]) for dim in range(result.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result

    
def get_coordinates( image_size, window_height, window_width, overlap_height, overlap_width):

    # get shape of the resulting flow field
    field_shape = get_field_shape( image_size, window_height, window_width, overlap_height, overlap_width )
    # compute grid coordinates of the interrogation window centers
    x = np.arange( field_shape[1] )*(window_width-overlap_width) + (window_width-1)/2.0
    y = np.arange( field_shape[0] )*(window_height-overlap_height) + (window_height-1)/2.0
    return np.meshgrid(x,y[::-1])


def get_field_shape ( image_size, window_height, window_width, overlap_height,overlap_width ):
    return ( (image_size[0] - window_height)//(window_height-overlap_height)+1,
             (image_size[1] - window_width)//(window_width-overlap_width)+1 )

def find_first_peak ( corr ):

    #Will work for inverted SAD matrix, as 0 will be the maximum.

    ind = corr.argmax()

    s = corr.shape[1]


    i = ind // s
    j = ind % s
    assert(isinstance(i,int)==True)
    assert(isinstance(j,int)==True)

    return i, j, corr.max()

def find_second_peak ( corr, i=None, j=None, width=2 ):

    if i is None or j is None:
        #find_first_peak( corr ) gives i, j, corr.max()
        i, j, tmp = find_first_peak( corr )

    # create a masked view
    tmp = corr.view(ma.MaskedArray)

    # set width x width square submatrix around the first correlation peak as masked.
    # Before check if we are not too close to the boundaries, otherwise we have negative indices
    # i initial
    iini = max(0, i-width)
    # i final
    ifin = min(i+width+1, corr.shape[0])
    # j initial
    jini = max(0, j-width)
    #j final
    jfin = min(j+width+1, corr.shape[1])
    # sets the masked array?
    tmp[ iini:ifin, jini:jfin ] = ma.masked

    i, j, corr_max2 = find_first_peak( tmp )


    assert(isinstance(i,int)==True)
    assert(isinstance(j,int)==True)
    return i, j, corr_max2

def find_subpixel_peak_position( corr, subpixel_method, corr_method):

    # initialization
    # i.e. the middle (as in autocorr)
    default_peak_position = ((corr.shape[0]/2) + 0.0 ,(corr.shape[1]/2) + 0.0)

    # the peak locations
    # peak1_i, peak1_j, dummy = i, j, corr.max()
    peak1_i, peak1_j, dummy = find_first_peak( corr )

    if corr_method == 'fft' or corr_method == 'direct':
        try:
            # the peak and its neighbours: left, right, down, up
            c = corr[peak1_i, peak1_j]
            cl = corr[peak1_i-1, peak1_j]
            cr = corr[peak1_i+1, peak1_j]
            cd = corr[peak1_i, peak1_j-1]
            cu = corr[peak1_i, peak1_j+1]
            assert(isinstance(c, np.float64)==True)

            # gaussian fit
            #if any of those coefs are < 0 AND method is gaussian, it will chose centroid method
            if np.any( np.array([c,cl,cr,cd,cu]) < 0 ) and subpixel_method == 'gaussian':
                subpixel_method = 'centroid'
                #print "centroid method"
            try:
                if subpixel_method == 'centroid':
                    # (([shift to left]* left neighbour (function value(at first arg)) + center (i.e. the peak itself) + [shift right]*right neighbour(function value (at first arg) )  / (left value, center, right value) ,
                    # similarly, but up down shif and 2-nd arg func. val.
                    subp_peak_position = ( ((peak1_i-1)*cl+peak1_i*c+(peak1_i+1)*cr)/(cl+c+cr),
                                    ((peak1_j-1)*cd+peak1_j*c+(peak1_j+1)*cu)/(cd+c+cu))

                elif subpixel_method == 'gaussian':
                    subp_peak_position = (peak1_i + ( (log(cl)-log(cr) )/( 2*log(cl) - 4*log(c) + 2*log(cr) )), peak1_j + ( (log(cd)-log(cu) )/( 2*log(cd) - 4*log(c) + 2*log(cu) )))

                elif subpixel_method == 'parabolic':
                    subp_peak_position = (peak1_i + (cl-cr)/(2*cl-4*c+2*cr),
                                        peak1_j + (cd-cu)/(2*cd-4*c+2*cu))
            except:
                subp_peak_position = default_peak_position
        except IndexError:
                subp_peak_position = default_peak_position
        return subp_peak_position[0], subp_peak_position[1]



    elif corr_method == 'sad' or corr_method == 'sad_norm':
        try:
            # the peak and its neighbours: left, right, down, up
            #I am inverting the peak values (since for SAD they are negative) to allow gaussian fitting, but will choose parabolic as a default for SAD...
            c = -corr[peak1_i, peak1_j]
            cl = -corr[peak1_i-1, peak1_j]
            cr = -corr[peak1_i+1, peak1_j]
            cd = -corr[peak1_i, peak1_j-1]
            cu = -corr[peak1_i, peak1_j+1]

            if np.any( np.array([c,cl,cr,cd,cu]) < 0 ) and subpixel_method == 'gaussian':
                subpixel_method = 'centroid'

            try:
                if subpixel_method == 'centroid':
                    # (([shift to left]* left neighbour (function value(at first arg)) + center (i.e. the peak itself) + [shift right]*right neighbour(function value (at first arg) )  / (left value, center, right value) ,
                    # similarly, but up down shif and 2-nd arg func. val.
                    subp_peak_position = ( ((peak1_i-1)*cl+peak1_i*c+(peak1_i+1)*cr)/(cl+c+cr),
                                    ((peak1_j-1)*cd+peak1_j*c+(peak1_j+1)*cu)/(cd+c+cu))

                elif subpixel_method == 'gaussian':
                    subp_peak_position = (peak1_i + ( (log(cl)-log(cr) )/( 2*log(cl) - 4*log(c) + 2*log(cr) )),peak1_j + ( (log(cd)-log(cu) )/( 2*log(cd) - 4*log(c) + 2*log(cu) )))
                elif subpixel_method == 'parabolic':
                    subp_peak_position = (peak1_i + (cl-cr)/(2*cl-4*c+2*cr),
                                        peak1_j + (cd-cu)/(2*cd-4*c+2*cu))
            except:
                subp_peak_position = default_peak_position
        except IndexError:
                subp_peak_position = default_peak_position
        assert(isinstance(subp_peak_position[0], float) == True)
        assert(isinstance(subp_peak_position[1], float) == True)



        return subp_peak_position[0], subp_peak_position[1]

def sig2noise_ratio( corr, sig2noise_method, corr_method, width):

    # compute first peak position
    # i, j, corr.max() = find_first_peak( corr )
    peak1_i, peak1_j, corr_max1 = find_first_peak( corr )
    #remember, we invert the SAD corr matrix by multiplying by -1. And our peak will be the minimum value (as opposed to a maximum)
    corr_min = corr.min()

    if corr_method == 'fft' or corr_method == 'direct':
        # now compute signal to noise ratio
        if sig2noise_method == 'peak2peak':
            # find second peak height
            #
            peak2_i, peak2_j, corr_max2 = find_second_peak( corr , peak1_i, peak1_j, width )

            # if it's an empty interrogation window
            # if the image is lacking particles, totally black it will correlate to very low value, but not zero
            # if the first peak is on the borders, the correlation map is also wrong
            if corr_max1 < 1e-3 or (peak1_i == 0 or peak1_j == corr.shape[0] or peak1_j == 0 or peak1_j == corr.shape[1] or
                                  peak2_i == 0 or peak2_j == corr.shape[0] or peak2_j == 0 or peak2_j == corr.shape[1]):
                # return zero, since we have no signal.
                return 0.0

        elif sig2noise_method == 'peak2mean':
            # find mean of the correlation map
            corr_max2 = corr.mean()

        else:
            raise ValueError('wrong sig2noise_method')

        # avoid dividing by zero
        try:
            sig2noise = corr_max1/corr_max2

            #print "s2n for fft/direct = " + str(sig2noise)
        except ValueError:
            sig2noise = np.inf


        return sig2noise



    elif corr_method == 'sad' or corr_method == 'sad_norm':

        if sig2noise_method == 'peak2peak':
            # find second peak height
            #
            peak2_i, peak2_j, corr_max2 = find_second_peak( corr , peak1_i, peak1_j, width )


            # if the first peak is on the borders, the correlation map is discarded
            if (peak1_i == 0 or peak1_j == corr.shape[0] or peak1_j == 0 or peak1_j == corr.shape[1] or peak2_i == 0 or peak2_j == corr.shape[0] or peak2_j == 0 or peak2_j == corr.shape[1]):

                return 0.0

        elif sig2noise_method == 'peak2mean':
            # find mean of the correlation map
            corr_max2 = corr.mean()

        else:
            raise ValueError('wrong sig2noise_method')

        # avoid dividing by zero
        try:
            sig2noise = (corr_min - corr_max1) / (corr_min - corr_max2)
            #print "s2n sad = " + str(sig2noise)
        except ValueError:
            sig2noise = np.inf

        #print "sig2noise = " + str(sig2noise)
        return sig2noise

def moving_window_array( array, window_height, window_width, overlap_height,overlap_width ):

    sz = array.itemsize
    shape = array.shape
    strides = (sz*shape[1]*(window_height-overlap_height), sz*(window_width-overlap_width), sz*shape[1], sz)
    shape = ( int((shape[0] - window_height)/(window_height-overlap_height))+1, int((shape[1] - window_width)/(window_width-overlap_width))+1 , window_height, window_width)
    final_element = strides[0]*(shape[0]-1) + strides[1]*(shape[1]-1) + strides[2]*(shape[2]-1) + strides[3]*(shape[3]-1)
    assert(final_element < array.size * array.itemsize)

    return numpy.lib.stride_tricks.as_strided( array, strides=strides, shape=shape ).reshape(-1, window_height,window_width)#H,W swapped VZ
    #return numpy.lib.stride_tricks.as_strided( array, strides=strides, shape=shape ).reshape(-1, window_width, window_height)


def moving_sub_window_array( array, window_big_height,window_big_width, overlap_big_height,overlap_big_width, subwindow_inset_height,subwindow_inset_width  ):
    """
This is based on the function moving_window_array, but the idea is to generate
smaller subwindows that are compatible with the larger windows generated by
moving_window_array, but are inset on all boundaries by subwindow_inset.
If we just use exactly the same strides as in moving_window_array, but smaller
dimensions for the array shape, we will get subwindows that line up with the
top left of the larger windows. That is obviously not what we want.
No obvious method springs to mind to achieve that initial offset, so we introduce
a fifth(!) dimension to the array, consisting of two elements with exactly the right
stride offset to take us to where we want our subwindow to start.
Thus we end up with two complete "families" of windows, one starting at the origin
and one starting at the correct place. We then discard the first family and are left
just with what we want
Parameters
----------
subwindow_inset = the size of how far away from the boundary we push the centre of the smaller window (similar to cropping)

overlap = overlap of the BIG IWs

window_size = size of the BIG IW
"""
    
    sz = array.itemsize
    shape = array.shape
    subwindow_height = window_big_height- 2*subwindow_inset_height
    subwindow_width = window_big_width- 2*subwindow_inset_width
    assert(subwindow_height > 0)
    assert(subwindow_width > 0)
    strides = (sz*(shape[1]*subwindow_inset_height+subwindow_inset_width), sz*shape[1]*(window_big_height-overlap_big_height),sz*(window_big_width-overlap_big_width), sz*shape[1], sz)

    shape = (2, int((shape[0] - window_big_height)/(window_big_height-overlap_big_height))+1, int((shape[1] - window_big_width)/(window_big_width-overlap_big_width))+1 , subwindow_height, subwindow_width)


    
    final_element = strides[0]*(shape[0]-1) + strides[1]*(shape[1]-1) + strides[2]*(shape[2]-1) + strides[3]*(shape[3]-1) + strides[4]*(shape[4]-1)
    assert(final_element < array.size * array.itemsize)
    # Consistent with what happens in moving_window_array, this final step flattens the
    # array so we just have a sequence of subwindows - rather than having a 2D array of subwindows
    # Note that we discard the unwanted family of windows that starts at the origin, by taking [1,:,:,:,:]
    return numpy.lib.stride_tricks.as_strided( array, strides=strides, shape=shape )[1,:,:,:,:].reshape(-1, subwindow_height,subwindow_width).astype(np.float64)



def save( x, y, u, v, mask,s2n, filename, fmt='%8.4f', delimiter='\t' ):

    # build output array
    out = np.vstack( [m.ravel() for m in [x, y, u, v, mask, s2n] ] )
    # save data to file.
    np.savetxt( filename, out.T, fmt=fmt, delimiter=delimiter )

def correlate_windows( window_a, window_b, corr_method = 'sad', mode = 'valid'):

    # use "full" mode if same sized windows are used. Make sure to have adequately larger window for different sized windows.
    #if window_a.max() >= coef_a and window_b.max()>=coef_b:

    if corr_method == 'fft':
        return fftconvolve(normalize_intensity(window_b), normalize_intensity(window_a)[::-1,::-1],mode).astype(np.float64)

    elif corr_method == 'direct':
        return correlate2d(normalize_intensity(window_b), normalize_intensity(window_a), mode).astype(np.float64)

    elif corr_method == 'sad':
        return -sad_correlation(window_a, window_b).astype(np.float64)
    else:
        raise ValueError('method is not implemented')
#else:
#    print "brigtest pixel in subwindows is below cut-off"
#    return np.zeroes((exp_corr_size,exp_corr_size))

def piv(frame_a, frame_b, window_big_height,window_big_width, overlap_big_height,overlap_big_width, subwindow_inset_height,subwindow_inset_width, dt, corr_method, subpixel_method, sig2noise_method, width, px_size):

    windows_a = moving_sub_window_array(frame_a, window_big_height,window_big_width, overlap_big_height,overlap_big_width, subwindow_inset_height,subwindow_inset_width).astype(data_type)
    windows_b = moving_window_array( frame_b,window_big_height,window_big_width, overlap_big_height,overlap_big_width).astype(data_type)


    #check correlation size.
    corr_shape = correlate_windows( windows_a[0], windows_b[0], corr_method = corr_method).shape
    print "correlation height, width", corr_shape[0],corr_shape[1]
    corr = np.zeros((windows_a.shape[0],corr_shape[0],corr_shape[1]))

    assert(windows_a.shape[0] == windows_b.shape[0])

    # get shape of the output so that we can preallocate
    # memory for velocity array
    n_rows, n_cols = get_field_shape ( frame_a.shape , window_big_height, window_big_width, overlap_big_height,overlap_big_width )

    print "windows_a.shape[0]", windows_a.shape[0]
    print "n_rows,n_cols",n_rows, n_cols
    print "n_rows*n_cols",n_rows * n_cols

    u = np.zeros(n_rows*n_cols).astype(np.float64)
    v = np.zeros(n_rows*n_cols).astype(np.float64)

    #if we want sig2noise information, allocate memory
    if sig2noise_method == 'peak2peak' or sig2noise_method == 'peak2mean':
        sig2noise = np.zeros(n_rows*n_cols)

    # for each interrogation window in the frame pair

    for i in range(windows_a.shape[0]):

        # get correlation window
        corr[i] = correlate_windows( windows_a[i], windows_b[i], corr_method = corr_method)
        assert(corr.dtype == numpy.float64)


        # get subpixel approximation for peak position row and column index
        row, col = find_subpixel_peak_position( corr[i], subpixel_method=subpixel_method)

        # NOTE! If different sized windows are correlated, this requires the bigger window to be the first argument of correlate/convolve type funtions. This also changes the values (or actually the sign) of the row and colum of the correlation peak. For different sized windows, the below has to have -ve sign in front of the expression for v. This also applies for same sized windows with reversed order in the correlate/convolve function.

        #multiply the pixel shift by the "effective" pixel size (real pixel size on the sensor and maginifaction information is needed)
        u[i], v[i] = px_size*(col - corr_shape[1]/2), -px_size*(row - corr_shape[0]/2)
        if sig2noise_method == 'peak2peak' or sig2noise_method == 'peak2mean':
            sig2noise[i] = sig2noise_ratio( corr[i], sig2noise_method=sig2noise_method, width=width )



    # return output depending if user wanted sig2noise information


    with open("output.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(corr)

    #print corr.shape
    #np.save("corr",corr)

    if sig2noise_method == 'peak2peak' or sig2noise_method == 'peak2mean':
        return u.reshape(n_rows, n_cols)/dt, v.reshape(n_rows, n_cols)/dt, sig2noise.reshape(n_rows, n_cols)
    else:
        return u.reshape(n_rows, n_cols), v.reshape(n_rows, n_cols)


def normalize_intensity( window ):

    return window - window.mean()

def sig2noise_val( u, v, sig2noise, threshold):

    ind = sig2noise < threshold
    u[ind] = np.nan
    v[ind] = np.nan
    mask = np.zeros(u.shape, dtype=bool)
    mask[ind] = True

    return u, v, mask

def save_field( filename, background, new_file, **kw):
    a = np.loadtxt(filename)

    ax = subplot(111)
    bg = background

    #cmap=cmps.gray

    fig = ax.imshow(np.flipud(bg),origin = 'lower',cmap=cmps.gray, interpolation='none')
    pl.hold(True)
    invalid = a[:,4].astype('bool')
    skip = (slice(None,None,3))

    valid = ~invalid


    bi0,bi1,bi2,bi3 = a[invalid,0],a[invalid,1],a[invalid,2],a[invalid,3]
    b0,b1,b2,b3 = a[valid,0],a[valid,1],a[valid,2],a[valid,3]
    pl.quiver(bi0[skip],bi1[skip],bi2[skip],bi3[skip],color='r',**kw)
    q = pl.quiver(b0[skip],b1[skip],b2[skip],b3[skip], color='b',**kw)
#  The actual scaling factor which multiplicatively converts vector component units to physical axis units is width/scale where width is the width of the plot in physical units and scale is the number specified by the scale keyword argument of quiver. That is with scale = 1 a vector representing 1px shift will take up the whole length of the image (for example for a 256x256 it will be 256 px long as they appear on the plot. To get 1px/s vector to be represented on 1px length on the plot, one should choose scale to be 256. It was tested that this works even if the image is not square. The width is taken as the x axis. The quiverkey command produces a legend consisting of a single vector and a label indicating how long it is in physical units. In order, the positi    onal arguments are (1) the variable returned by the quiver command, (2) and (3) the x and y coordinates of the legend, (4) the length of the arrow in physical units, and (5) the legend label. See http://physics.nmt.edu/~raymond/software/python_notes/paper004.html#vectorplot Alternatively, "angles" can be changed with "units = width".
    #p = pl.quiverkey(q,image_size[1]/2,image_size[0] + 5, np.sqrt(vector_length),str(vector_length) + "um/s",coordinates='data',color='r')
    #pl.draw()
    pl.xticks(np.arange(0,image_size[1],window_width))
    pl.yticks(np.arange(0,image_size[0],window_height))
    pl.savefig(new_file, interpolation = "none",cmap=cmps.gray,bbox_inches='tight',pad_inches = 0)
    pl.close()


def query_cont_paired(question, default = "cont"):
	"""Ask if a continuous mode (i.e. frames in sequence) or paired (i.e. frames are paired)"""

	valid = {"cont": True, "c": True, "con": True,
	 "pair": False, "p": False, "prd": False}

	if default is None:
		prompt = " [c/p] "
	elif default == "cont":
		prompt = " [C/p] "
	elif default == "pair":
		prompt = " [c/P] "
	else:
		raise ValueError("invalid default answer: '%s'" % default)

	while True:
		sys.stdout.write(question + prompt)
		choice = raw_input().lower()
		if default is not None and choice == '':
			return valid[default]
		elif choice in valid:
			return valid[choice]
		else:
			sys.stdout.write("Please respond with 'c' or 'p' " "(or 'cont', 'con', 'pair','prd').\n")

def folder_writer(wkdir, folder):
    try:
        os.makedirs(str(wkdir + folder))
        return str(wkdir + folder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise    
            
