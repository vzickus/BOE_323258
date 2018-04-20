import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from math import log, sqrt, sin
import scipy.signal, scipy.ndimage
import os, sys, time, warnings
import plist_wrapper as jPlist
from image_class import *
from periods import *
import tqdm #from tqdm import *
import hashlib

thisModule = sys.modules[__name__]
thisModule.warnedAboutHittingEdge = False

def SortImagesByPhaseOrZstep(images, sortKey='phase', postacq=True, hist=True):
    # Establish an array whose columns represent:
    # - array index
    # - phase
    # - z_step
    sortBy = {'phase':1,'ph':1,'z':2, 'z_step':2, 'zstep':2}
    a = np.zeros((len(images), 3))
    
    a[:,0] = np.arange(len(images))
    a[:,2] = np.vectorize(ImageClass.z_step)(images)
 
    if postacq:
        a[:,1] = np.vectorize(ImageClass.ph_post)(images)
        hist_name = 'postacq. phase'
        #load offline phase info
        offline = np.zeros((len(images),1))
        offline = np.vectorize(ImageClass.ph_off)(images)
    
    else:
        a[:,1] = np.vectorize(ImageClass.ph_off)(images)
        hist_name = 'offline phase'
        
    # Sort by phase-wrapped frame index
    a = a[a[:,sortBy[sortKey]].argsort(kind='mergesort')]     # May as well use a stable sort...
    # I couldn't get "Structured Arrays" to work properly, so make this as int. However, it should
    #be possible to use pandas to store this data better, but perhaps it would take longer to manipulate...
    indices = a[:, 0].astype('int')

    if hist == True:
        #histogram of phases would be useful, Probably easiest to have it for a full dataset.
        #will let us inspect quickly if there are any anomolous parts in the phase assignment.
        plt.hist(a[:,1], bins = np.arange(0.0,6.6,0.2))
        plt.xlim(0,6.4)
        plt.title(hist_name)
        if postacq:
            #offline hist if available    
            plt.figure()
            plt.hist(offline,bins = np.arange(0.0,6.6,0.2) )
            plt.xlim(0,6.4)
            plt.title("offline phase")


        
    sortedImages = [images[i] for i in indices]
    print "len sorted images:", len(sortedImages)
    return sortedImages, a



def LoadImage(imagePath, loadImageData, downsampleFactor, frameIndex, cropRect, cropFollowsZ, timestampKey):
    # Read information from the plist
    plistPath = os.path.splitext(imagePath)[0] + '.plist'
    pl = jPlist.readPlist(plistPath)
    image = ImageClass()
    image.path = imagePath
    image.frameIndex = frameIndex
    image.timestamp = pl[timestampKey]
    image.plist = pl
    image.plistPath = plistPath
    
    if (loadImageData):
        # Note that in July 2016 I noticed a major performance issue with imread on my mac pro (not on my laptop).
        # I tracked it down to PIL.ImageFile.load, which was calling mmap, and that was taking an abnormally long time.
        # I worked around this issue by editing that code for PIL, and setting use_mmap to False in that function.
        im = img.imread(imagePath)
        if (len(im.shape) == 3):
            im = np.sum(im, axis=2)
        # Optional cropping
        # cropRect is ordered (x1, x2, y1, y2)
        if (cropRect is not None):
            cropToUse = cropRect
            if (cropFollowsZ == 'x'):
                assert(0)   # Not yet implemented
            elif (cropFollowsZ == 'y'):
                # Read the z coordinate and correct for z motion
                if ('z_scan' in image.plist):
                    startZ = image.plist['z_scan']['startZ']
                    z = image.plist['stage_positions']['last_known_z']
                    deltaZ = z - startZ
                else:
                    deltaZ = 0
                # For now, I have hard-coded a conversion factor between pixels and z coord
                # TODO: ideally this would be a parameter that could be adjusted.
                # However, in practice it is going to remain unchanged unless we mess with the optics.
                # As a result, I would rather not read the pixel size in the plist, since it's more likely that we won't have
                # updated that when moving the camera between imaging arms!
                binning = 2**(image.plist['binning_code'])
                pixelsPerUm = 0.758 / binning
                correction = int(deltaZ * pixelsPerUm)
                # Limit correction to full size of image
                #print(cropToUse[3], correction, im.shape[0])
                if (cropToUse[3]+correction > im.shape[0]):
                    if (thisModule.warnedAboutHittingEdge == False):
                        print('Hit edge of image during moving-window cropping. Wanted correction', correction, 'for image', imagePath)
                        thisModule.warnedAboutHittingEdge = True
                    correction = im.shape[0] - cropToUse[3]
                cropToUse = (cropToUse[0], cropToUse[1], cropToUse[2] + correction, cropToUse[3] + correction)
            else:
                assert(cropFollowsZ is None)
            
            im = im[cropToUse[2]:cropToUse[3], cropToUse[0]:cropToUse[1]]
        # For now, downsample for speed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = scipy.ndimage.interpolation.zoom(im, zoom=1.0/downsampleFactor, order=1)
    else:
        im = None
    image.image = im

    return image


def MyHash(runningHash, b):
    # Note that in python 3 apparently hash() is for some reason not unique across invocations of python (ouch!)
    # We should be ok if we use hashlib instead
    if isinstance(b, str) == False:
        b = str(b)
    runningHash.update(b.encode('utf-8'))

def LoadOrCacheImagesFromFileList(fileList, loadImageData, downsampleFactor, frameIndexStart, periodRange, plotAllPeriods, cropRect, cropFollowsZ, timestampKey):
    # Loads a sequence of images from fileList
    # For performance reasons, we cache the numpy array so we can load faster if rerunning the same code
    
    # filelist is expected to be sorted, and on Linux this will not be the case by default
    fileList = sorted(fileList)
    # Determine the cache file name appropriate for this fileList
    hashObject = hashlib.md5()
    for f in fileList:
        MyHash(hashObject, f)
    MyHash(hashObject, downsampleFactor)
    MyHash(hashObject, frameIndexStart)
    MyHash(hashObject, cropRect)
    MyHash(hashObject, cropFollowsZ)
    MyHash(hashObject, timestampKey)
    MyHash(hashObject, loadImageData)
    hashResult = hashObject.hexdigest()[-16:]
    cacheName = "%s/%s_%s.npy" % (os.path.dirname(fileList[0]), os.path.basename((os.path.split(fileList[0]))[0]), hashResult)

    try:
        #Try loading an .npy file of the image information.
        images = np.load(cacheName)
        print("Loaded from cache: %s." % cacheName)
    except IOError:
        print("missing npy file: %s. Loading files." % cacheName)
        images = []
        counter = 0
        for imagePath in tqdm(fileList, desc='loading images'):
            images.append(LoadImage(imagePath, loadImageData, downsampleFactor, frameIndexStart + counter, cropRect=cropRect, cropFollowsZ=cropFollowsZ, timestampKey=timestampKey))
            counter = counter + 1
        images = np.array(images)
        try:
            np.save(cacheName, images)
        except IOError:
            print("Harmless warning: unable to save cache (probably a read-only volume?)")

    if (loadImageData and (periodRange is not None)):
        # Estimate approximate period.
        numImagesToUseForPeriodEstimation = min(periodRange[-1] * 12, len(images))
        averagePeriod = EstablishPeriodForImageSequence(images[0:numImagesToUseForPeriodEstimation], periodRange = periodRange, plotAllPeriods = plotAllPeriods)
        print('estimated average(ish) period', averagePeriod, 'from first', numImagesToUseForPeriodEstimation, 'images')
    else:
        averagePeriod = np.nan

    return (images, averagePeriod)


def LoadAllImages(path, loadImageData = True, downsampleFactor = 1, frameIndexStart = 0, earlyTruncation = -1, periodRange = np.arange(20, 50, 0.1), plotAllPeriods=False, cropRect=None, cropFollowsZ=None, timestampKey='time_received'):
	# Load all images found in the directory at 'path'.
	# We also do a rough estimate the average heart period at the start of the dataset (because that is a useful guide for later sync processing)
    fileList = []
    for file in os.listdir(path):
        if (file.endswith(".tif") or file.endswith(".tiff")):
            fileList.append(path+'/'+file)
        if (len(fileList) == earlyTruncation):
            break
    return LoadOrCacheImagesFromFileList(fileList, loadImageData, downsampleFactor, frameIndexStart, periodRange, plotAllPeriods, cropRect, cropFollowsZ, timestampKey)


def LoadImages(path, format, firstImage, numImagesToProcess, loadImageData = True, downsampleFactor = 1, frameIndexStart = 0, periodRange = np.arange(20, 50, 0.1), plotAllPeriods=False, cropRect=None, cropFollowsZ=None, timestampKey='time_processing_started'):
    # Load the source images and metadata for all images found at 'path' that match the supplied filename format with indices given by firstImage and numImagesToProcess.
	# We also do a rough estimate the average heart period at the start of the dataset (because that is a useful guide for later sync processing)
    fileList = []
    for i in range(firstImage, firstImage+numImagesToProcess):
        imagePath = (format % (path, i))
        fileList.append(imagePath)
    return LoadOrCacheImagesFromFileList(fileList, loadImageData, downsampleFactor, frameIndexStart, periodRange, plotAllPeriods, cropRect, cropFollowsZ, timestampKey)
