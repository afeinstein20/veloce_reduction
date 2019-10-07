"""
Created on 13 Apr. 2018

@author: christoph
"""

import astropy.io.fits as pyfits
import numpy as np
from itertools import combinations
import time
import matplotlib.pyplot as plt
from scipy import ndimage

from veloce_reduction.veloce_reduction.helper_functions import correct_orientation, sigma_clip, polyfit2d, polyval2d
from veloce_reduction.veloce_reduction.background import extract_background, fit_background






def make_median_image(imglist, MB=None, correct_OS=True, scalable=False, raw=False):
    """
    Make a median image from a given list of images.

    INPUT:
    'imglist'     : list of files (incl. directories)
    'MB'          : master bias frame - if provided, it will be subtracted from every image before median image is computed
    'correct_OS'  : boolean - do you want to subtract the overscan levels?
    'scalable'    : boolean - do you want to scale this to an exposure time of 1s (AFTER the bias and overscan are subtracted!!!!!)
    'raw'         : boolean - set to TRUE if you want to retain the original size and orientation;
                    otherwise the image will be brought to the 'correct' orientation and the overscan regions will be cropped

    OUTPUT:
    'medimg'   : median image (bias- & overscan-corrected) [still in ADU]
    """

    # from veloce_reduction.calibration import crop_overscan_region

    # prepare array
    allimg = []

    # loop over all files in "dark_list"
    for file in imglist:
        
        # read in dark image
        img = pyfits.getdata(file)
        
        if correct_OS:
            # get the overscan levels so we can subtract them later
#             os_levels = get_bias_and_readnoise_from_overscan_polyfit(img, gain=[1,1,1,1], return_oslevels_only=True)
            os_levels = get_bias_and_readnoise_from_overscan_collapse(img, gain=[1,1,1,1], return_oslevels_only=True)
        if not raw:
            # bring to "correct" orientation
            img = correct_orientation(img)
            # remove the overscan region
            img = crop_overscan_region(img)
        if correct_OS:
            # make (4k x 4k) frame of the offsets
            ny,nx = img.shape
            offmask = np.ones((ny,nx))
            # define four quadrants via masks
            q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
            for q,osl in zip([q1,q2,q3,q4], os_levels):
                offmask[q] = offmask[q] * osl
            # subtract overscan levels
            img = img - offmask           
        if MB is not None:
            # subtract master bias (if provided)
            img = img - MB
        if scalable:
            texp = pyfits.getval(file, 'ELAPSED')
            img /= texp

        # add image to list
        allimg.append(img)

    # get median image
    medimg = np.median(np.array(allimg), axis=0)

    return medimg





def make_quadrant_masks(nx, ny):
    # define four quadrants via masks
    q1 = np.zeros((ny, nx), dtype='bool')
    q1[:(ny / 2), :(nx / 2)] = True
    q2 = np.zeros((ny, nx), dtype='bool')
    q2[:(ny / 2), (nx / 2):] = True
    q3 = np.zeros((ny, nx), dtype='bool')
    q3[(ny / 2):, (nx / 2):] = True
    q4 = np.zeros((ny, nx), dtype='bool')
    q4[(ny / 2):, :(nx / 2)] = True

    return q1, q2, q3, q4





def crop_overscan_region(img, overscan=53):
    """
    As of July 2018, Veloce uses an e2v CCD231-84-1-E74 4kx4k chip.
    Image dimensions are 4096 x 4112 pixels, but the recorded images size including the overscan region is 4202 x 4112 pixels.
    We therefore have an overscan region of size 53 x 4112 at either end. 
    
    raw_img = pyfits.getdata(filename)     -->    raw_img.shape = (4112, 4202)
    img = correct_orientation(raw_img)     -->        img.shape = (4202, 4112)
    """

    #correct orientation if not already done
    if img.shape == (4112, 4202):
        img = correct_orientation(img)

    assert img.shape == (4202, 4112), 'ERROR: wrong image size encountered!!!'

    #crop overscan region
#     good_img = img[53:4149,:]
    good_img = img[overscan : 4096 + overscan, : ]

    return good_img





def extract_overscan_region(img, overscan=53):
    """
    As of July 2018, Veloce uses an e2v CCD231-84-1-E74 4kx4k chip.
    Image dimensions are 4096 x 4112 pixels, but the recorded images size including the overscan region is 4202 x 4112 pixels.
    We therefore have an overscan region of size 53 x 4112 at either end. 
    
    raw_img = pyfits.getdata(filename)     -->    raw_img.shape = (4112, 4202)
    img = correct_orientation(raw_img)     -->        img.shape = (4202, 4112)
    """

    #correct orientation if not already done
    if img.shape == (4112, 4202):
        img = correct_orientation(img)

    assert img.shape == (4202, 4112), 'ERROR: wrong image size encountered!!!'

    ny,nx = img.shape

    #define overscan regions
    os1 = img[:overscan , :nx//2]
    os2 = img[:overscan , nx//2:]
    os3 = img[ny-overscan: , nx//2:]
    os4 = img[ny-overscan: , :nx//2]   
    
    return os1,os2,os3,os4





def get_flux_and_variance_pairs(imglist, MB, MD=None, scalable=True, simu=False, timit=False):
    """
    measure gain from a list of flat-field images as described here:
    https://www.mirametrics.com/tech_note_ccdgain.php

    units = ADUs

    INPUT:
    'imglist'  : list of image filenames (incl. directories)
    'MB'       : the master bias frame
    'simu'     : boolean - are you using simulated spectra?
    'timit'    : boolean - do you want to measure execution run time?

    OUTPUT:
    'f_q1' : median signal for quadrant 1
    'v_q1' : variance in signal for quadrant 1
    (same for other quadrants)
    """

    if timit:
        start_time = time.time()

    # do some formatting things for real observations
    #read dummy image
    img = pyfits.getdata(imglist[0])
    if not simu:
        # bring to "correct" orientation
        img = correct_orientation(img)
        # remove the overscan region, which looks crap for actual bias images
        img = crop_overscan_region(img)

    ny,nx = img.shape

    # define four quadrants via masks
    q1, q2, q3, q4 = make_quadrant_masks(nx, ny)

    #prepare arrays containing flux and variance
    f_q1 = []
    f_q2 = []
    f_q3 = []
    f_q4 = []
    v_q1 = []
    v_q2 = []
    v_q3 = []
    v_q4 = []

    #list of all possible pairs of files
    list_of_combinations = list(combinations(imglist, 2))

    for (name1,name2) in list_of_combinations:
        #read in observations and bring to right format
        img1 = pyfits.getdata(name1)
        img2 = pyfits.getdata(name2)
        if not simu:
            img1 = correct_orientation(img1)
            img1 = crop_overscan_region(img1)
            img2 = correct_orientation(img2)
            img2 = crop_overscan_region(img2)
        #subtract master bias
        img1 = img1 - MB
        if MD is not None:
            if scalable:
                texp1 = pyfits.getval(name1, 'exptime')
                texp2 = pyfits.getval(name1, 'exptime')
                if texp1 == texp2:
                    #subtract (scaled) master dark
                    img1 = img1 - MD*texp1
                    img2 = img2 - MD*texp2
                else:
                    print('ERROR: exposure times for the flat-field pairs do not agree!!!')
                    return
            else:
                #subtract master dark of right exposure time
                img1 = img1 - MD
                img2 = img2 - MD
        
        #take difference and do sigma-clipping
        # diff = img1.astype(long) - img2.astype(long)
        med1_q1 = np.nanmedian(img1[q1])
        med2_q1 = np.nanmedian(img2[q1])
        r_q1 = med1_q1 / med2_q1
        diff_q1 = img1[q1].astype(long) - r_q1 * img2[q1].astype(long)
        var_q1 = (np.nanstd(sigma_clip(diff_q1, 5))/np.sqrt(2))**2
        med1_q2 = np.nanmedian(img1[q2])
        med2_q2 = np.nanmedian(img2[q2])
        r_q2 = med1_q2 / med2_q2
        diff_q2 = img1[q2].astype(long) - r_q2 * img2[q2].astype(long)
        var_q2 = (np.nanstd(sigma_clip(diff_q2, 5)) / np.sqrt(2)) ** 2
        med1_q3 = np.nanmedian(img1[q3])
        med2_q3 = np.nanmedian(img2[q3])
        r_q3 = med1_q3 / med2_q3
        diff_q3 = img1[q3].astype(long) - r_q3 * img2[q3].astype(long)
        var_q3 = (np.nanstd(sigma_clip(diff_q3, 5)) / np.sqrt(2)) ** 2
        med1_q4 = np.nanmedian(img1[q4])
        med2_q4 = np.nanmedian(img2[q4])
        r_q4 = med1_q4 / med2_q4
        diff_q4 = img1[q4].astype(long) - r_q4 * img2[q4].astype(long)
        var_q4 = (np.nanstd(sigma_clip(diff_q4, 5)) / np.sqrt(2)) ** 2

        #fill output arrays
        f_q1.append(med1_q1)
        f_q2.append(med1_q2)
        f_q3.append(med1_q3)
        f_q4.append(med1_q4)
        v_q1.append(var_q1)
        v_q2.append(var_q2)
        v_q3.append(var_q3)
        v_q4.append(var_q4)

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return f_q1,f_q2,f_q3,f_q4,v_q1,v_q2,v_q3,v_q4





def measure_gain_from_slope(signal, variance, debug_level=0):
    """
    In a graph where the y-axis is the variance, and the x-axis is the signal,
    the gain is equal to the inverse of the slope.

    INPUT:
    'signal'       : array of signal values
    'variance'     : array of variance values
    'debug_level'  : for debugging...

    OUTPUT:
    'gain'  : the gain in e-/ADU
    """

    # unweighted linear fit
    p = np.poly1d(np.polyfit(signal, variance, 1))

    #slope is p[1]
    gain = 1./p[1]

    if debug_level >= 1:
        #plot the graph
        plt.figure()
        plt.plot(signal, variance, 'kx')
        plt.xlabel('signal')
        plt.ylabel('variance')
        xr = np.max(signal) - np.min(signal)
        xx = np.linspace(np.min(signal)-0.2*xr,np.max(signal)+0.2*xr)
        plt.plot(xx,p(xx),'g--')
        plt.title('gain = 1/slope = '+str(np.round(gain,2)))
        plt.xlim(np.min(xx),np.max(xx))

    return gain





def measure_gains(filelist, MB, MD=None, scalable=True, timit=False, debug_level=0):
    """
    Measure the gain from a series of flat-field exposures of different exposure times (ie brightness levels).
    
    INPUT:
    'filelist'     : list of filenames for the flat-fields (incl. directories)
    'MB'           : the master bias frame
    'MD'           : the master dark frame
    'scalable'     : boolean - is the master dark frame 'scalable' (ie texp = 1s) ? ignored if MD is None...
    'timit'        : boolean - do you want to measure execution run time?
    'debug_level'  : for debugging...
    
    OUTPUT:
    g1,g2,g3,g4    : the gain values for the four different quadrants [e-/ADU]
    """

    if timit:
        start_time = time.time()

    #we want to sub-group the files in 'filelist' according to their exposure times (ie brightness levels)
    texp = []
    for file in filelist:
        texp.append(pyfits.getval(file, 'exptime'))

    #find unique times
    uniq_times = np.unique(texp)

    #prepare some arrays
    signal_q1 = []
    signal_q2 = []
    signal_q3 = []
    signal_q4 = []
    variance_q1 = []
    variance_q2 = []
    variance_q3 = []
    variance_q4 = []

    #for all brightness levels
    for t in uniq_times:
        sublist = np.array(filelist)[texp == t]
        f_q1, f_q2, f_q3, f_q4, v_q1, v_q2, v_q3, v_q4 = get_flux_and_variance_pairs(sublist, MB, MD=MD, scalable=scalable, timit=timit)
        signal_q1.append(f_q1)
        signal_q2.append(f_q2)
        signal_q3.append(f_q3)
        signal_q4.append(f_q4)
        variance_q1.append(v_q1)
        variance_q2.append(v_q2)
        variance_q3.append(v_q3)
        variance_q4.append(v_q4)

    #have to do this for each quadrant individually
    g1 = measure_gain_from_slope(signal_q1, variance_q1, debug_level=debug_level)
    g2 = measure_gain_from_slope(signal_q2, variance_q2, debug_level=debug_level)
    g3 = measure_gain_from_slope(signal_q3, variance_q3, debug_level=debug_level)
    g4 = measure_gain_from_slope(signal_q4, variance_q4, debug_level=debug_level)

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return g1,g2,g3,g4





def get_bias_and_readnoise_from_overscan_polyfit(img, ramps=[35, 35, 35, 35], gain=None, degpol=5, clip=5, add=0, return_oslevels_only=False, verbose=False, timit=False):
    """
    PURPOSE:
    get an estimate of the bias and the read noise from a selected sub-region of the overscan region for each quadrant
    
    INPUT:
    'img'     - the (raw, ie 4202x4112) image for which to determine the bias and read noise from the overscan region
    'ramps'   - the number of pixels from the edeg to exclude b/c of the weird shape in the overscan region in cross-dispersion direction 
    'gain'    - array of gains for each quadrant (in units of e-/ADU)
    'degpol'  - degree of the polynomial used to fit the medians in dispersion direction
    'clip'    - threshold for sigma clipping
    'add'     - number of ADUs to add (from comparing a number of bias frames with the bias estimate from the overscan regions, there sometimes seems to be a ~1ADU offset)
    'return_oslevels_only'  - boolean - do you want to return the overscan levels only?
    'verbose' - boolean - do you want to print user info to screen?
    'timit'   - boolean - do you want to measure execution run time?
    
    OUTPUT:
    'bias'       - 4096 x 4112 array containing the estimated bias level plus the overscan levls for every "real" pixel [ADU]
    'bias_only'  - 4096 x 4112 array containing the estimated bias level only for every "real" pixel [ADU]
    'offsets'    - the 4 constant offsets per quadrant (ie the overscan levels) [ADU]
    'rons'       - 4-element array containing the read noise for each quadrant [e-]
    """
    
    if timit:
        start_time = time.time()

    if verbose:
        print('Determining offset levels and read-out noise properties from overscan regions for 4 quadrants...')

    # get image dimensions
    ny, nx = crop_overscan_region(correct_orientation(img)).shape

    # extract all four overscan regions
    os1, os2, os3, os4 = extract_overscan_region(img)

    # code defensively...
    assert (ny, nx) == (4096, 4112), 'ERROR: image dimensions not correct!'
    assert os1.shape == (53, 2056), 'ERROR: Overscan region has the wrong shape!'
    assert (os1.shape == os2.shape) and (os1.shape == os3.shape) and (os1.shape == os4.shape), 'ERROR: not all 4 overscan regions have the same dimensions!'

    nxq = os1.shape[1]

    # define good / usable regions within each overscan region (ie where I consider it flat enough)
    good_os1 = os1[ramps[0]:, :]
    # xmeds1 = np.nanmedian(os1, axis=1)
    ymeds1 = np.nanmedian(good_os1, axis=0)
    good_os2 = os2[ramps[1]:, :]
    # xmeds2 = np.nanmedian(os2, axis=1)
    ymeds2 = np.nanmedian(good_os2, axis=0)
    good_os3 = os3[:-ramps[2], :]
#     xmeds3 = np.nanmedian(os3, axis=1)
    ymeds3 = np.nanmedian(good_os3, axis=0)
    good_os4 = os4[:-ramps[3], :]
#     xmeds4 = np.nanmedian(os4, axis=1)
    ymeds4 = np.nanmedian(good_os4, axis=0)

    # now fit polynomial to the medians in the cross-dispersion direction
    # (ie we have a median value for the good part of the overscan region for all 2056 pixel columns per quadrant)
    # note that we do not include the value of the first (Q1 & Q4) / last (Q2 & Q3) pixel column in the fit (replace it with the median value in the adjacent pixel column), 
    # as it is significantly lower in the overscan (but not in the bias frames)
    fit_os1 = np.poly1d(np.polyfit(np.arange(nxq), np.r_[ymeds1[1], ymeds1[1:]], degpol))     
    model_os1_onedim = fit_os1(np.arange(nxq))     
    model_os1 = np.tile(model_os1_onedim, (ny//2, 1))
    fit_os2 = np.poly1d(np.polyfit(np.arange(nxq), np.r_[ymeds2[:-1], ymeds2[-2]], degpol))
    model_os2_onedim = fit_os2(np.arange(nxq)) 
    model_os2 = np.tile(model_os2_onedim, (ny//2, 1))
    fit_os3 = np.poly1d(np.polyfit(np.arange(nxq), np.r_[ymeds3[:-1], ymeds3[-2]], degpol))
    model_os3_onedim = fit_os3(np.arange(nxq)) 
    model_os3 = np.tile(model_os3_onedim, (ny//2, 1))
    fit_os4 = np.poly1d(np.polyfit(np.arange(nxq), np.r_[ymeds4[1], ymeds4[1:]], degpol))
    model_os4_onedim = fit_os4(np.arange(nxq)) 
    model_os4 = np.tile(model_os4_onedim, (ny//2, 1))

    # get the overscan levels, ie the constant offsets (excluding the first/last dodgy pixel column)
    # NOTE: this is usually within +/- 1 ADU of the median of the bias level as measured from bias frames !!! (tested by looking at the OS region of bias frames)
    os_level_1 = np.nanmedian(sigma_clip(good_os1[:, 1:].flatten(), clip))
    os_level_2 = np.nanmedian(sigma_clip(good_os2[:, :-1].flatten(), clip))
    os_level_3 = np.nanmedian(sigma_clip(good_os3[:, :-1].flatten(), clip))
    os_level_4 = np.nanmedian(sigma_clip(good_os4[:, 1:].flatten(), clip))
    offsets = np.array([os_level_1, os_level_2, os_level_3, os_level_4])
    if return_oslevels_only:
        return offsets
    
    # get estimate of the read noise (excluding the first/last dodgy pixel column)
    ron1 = np.nanstd(sigma_clip(good_os1[:, 1:].flatten(), clip))
    ron2 = np.nanstd(sigma_clip(good_os2[:, :-1].flatten(), clip))
    ron3 = np.nanstd(sigma_clip(good_os3[:, :-1].flatten(), clip))
    ron4 = np.nanstd(sigma_clip(good_os4[:, 1:].flatten(), clip))
    rons_adu = np.array([ron1, ron2, ron3, ron4])
    # convert read-out noise (but NOT the bias image!!!) to units of electrons rather than ADUs by multiplying with the gain (which has units of e-/ADU)
    assert gain is not None, 'ERROR: gain is not defined!'
    rons = rons_adu * gain
  
    # create "master bias" (4k x 4k) frame (incl. OS levels) from that (note the order is important, following the definition of the quadrants)
    bias = np.vstack([np.hstack([model_os1, model_os2]), np.hstack([model_os4, model_os3])]) + add
    
    # make (4k x 4k) frame of the offsets
    offmask = np.ones((ny,nx))
    q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
    for q,offset in zip([q1,q2,q3,q4], offsets):
        offmask[q] = offmask[q] * offset
        
    # subtract overscan levels
    bias_only = bias - offmask
    
    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time, 1))+' seconds')
    
    return bias, bias_only, offsets, rons





def get_bias_and_readnoise_from_overscan_collapse(img, gain=None, ramps=[35, 35, 35, 35], clip=5, add=0, return_oslevels_only=False, verbose=False, timit=False):
    """
    PURPOSE:
    get an estimate of the bias and the read noise from a selected sub-region of the overscan region for each quadrant
    
    INPUT:
    'img'     - the (raw, ie 4202x4112) image for which to determine the bias and read noise from the overscan region
    'gain'    - array of gains for each quadrant (in units of e-/ADU)
    'clip'    - threshold for sigma clipping
    'add'     - number of ADUs to add (from comparing a number of bias frames with the bias estimate from the overscan regions, there sometimes seems to be a ~1ADU offset)
    'return_oslevels_only'  - boolean - do you want to return the overscan levels only?
    'verbose' - boolean - do you want to print user info to screen?
    'timit'   - boolean - do you want to measure execution run time?
    
    OUTPUT:
    'bias'       - 4096 x 4112 array containing the estimated bias level plus the overscan levls for every "real" pixel [ADU]
    'rons'       - 4-element array containing the read noise for each quadrant [e-]

    PROCEDURE:
    The procedure that has been developed to overscan subtract Veloce data is as follows.
    Overscan subtraction is performed on a quadrant basis, where each quadrant is mapped (by inverting its X and Y axes),
    so that it looks just like Q2 (i.e. read-out amplifier at the bottom left hand corner).
    (a) Columns 10-51 are averaged (indexed so the first column is column 0) to obtain a shape for the overscan in the X direction,
    which is then subtracted from all rows in the column 10-51 region.
    (b) This "flattened" region is then averaged in the X direction to get a shape for the overscan in the Y direction.
    (d) The Y-profile output from the second pass of step b is then subtracted from the entire quadrant to produce an overscan subtracted quadrant.
    That procedure is then applied to all 4 quadrants, resulting in a single image with mean levels of zero in the overscan region.

    
    TODO:
    step (c) : sigma-clipping and repeating step (a) and (b)
    """
    
    if timit:
        start_time = time.time()

    if verbose:
        print('Determining offset levels and read-out noise properties from overscan regions for 4 quadrants...')

    # get image dimensions
    ny, nx = crop_overscan_region(correct_orientation(img)).shape

    # extract all four overscan regions
    os1, os2, os3, os4 = extract_overscan_region(img)

    # code defensively...
    assert (ny, nx) == (4096, 4112), 'ERROR: image dimensions not correct!'
    assert os1.shape == (53, 2056), 'ERROR: Overscan region has the wrong shape!'
    assert (os1.shape == os2.shape) and (os1.shape == os3.shape) and (os1.shape == os4.shape), 'ERROR: not all 4 overscan regions have the same dimensions!'

    nxq = os1.shape[1]

    # for quadrant 1
    dispdir_avg = np.average(os1[10:52,:], axis=1)
    os1_flat = os1[10:52,:] - np.reshape(np.repeat(dispdir_avg, nxq), (42,nxq))
    dispdir_shape = np.average(os1_flat, axis=0)
    os1_flatflat = os1_flat - np.reshape(np.repeat(dispdir_shape, 42), (nxq,42)).T
    ron1 = np.nanstd(sigma_clip(os1_flatflat.flatten(), clip))
    model_os1 = np.reshape(np.repeat(dispdir_shape, ny//2), (nxq,ny//2)).T
    
    # for quadrant 2
    dispdir_avg = np.average(os2[10:52,:], axis=1)
    os2_flat = os2[10:52,:] - np.reshape(np.repeat(dispdir_avg, nxq), (42,nxq))
    dispdir_shape = np.average(os2_flat, axis=0)
    os2_flatflat = os2_flat - np.reshape(np.repeat(dispdir_shape, 42), (nxq,42)).T
    ron2 = np.nanstd(sigma_clip(os2_flatflat.flatten(), clip))
    model_os2 = np.reshape(np.repeat(dispdir_shape, ny//2), (nxq,ny//2)).T
    
    # for quadrant 3
    dispdir_avg = np.average(os3[-52:-10,:], axis=1)
    os3_flat = os3[-52:-10,:] - np.reshape(np.repeat(dispdir_avg, nxq), (42,nxq))
    dispdir_shape = np.average(os3_flat, axis=0)
    os3_flatflat = os3_flat - np.reshape(np.repeat(dispdir_shape, 42), (nxq,42)).T
    ron3 = np.nanstd(sigma_clip(os3_flatflat.flatten(), clip))
    model_os3 = np.reshape(np.repeat(dispdir_shape, ny//2), (nxq,ny//2)).T
    
    # for quadrant 4
    dispdir_avg = np.average(os4[-52:-10,:], axis=1)
    os4_flat = os4[-52:-10,:] - np.reshape(np.repeat(dispdir_avg, nxq), (42,nxq))
    dispdir_shape = np.average(os4_flat, axis=0)
    os4_flatflat = os4_flat - np.reshape(np.repeat(dispdir_shape, 42), (nxq,42)).T
    ron4 = np.nanstd(sigma_clip(os4_flatflat.flatten(), clip))
    model_os4 = np.reshape(np.repeat(dispdir_shape, ny//2), (nxq,ny//2)).T

    rons_adu = np.array([ron1, ron2, ron3, ron4])
    # convert read-out noise (but NOT the bias image!!!) to units of electrons rather than ADUs by multiplying with the gain (which has units of e-/ADU)
    assert gain is not None, 'ERROR: gain is not defined!'
    rons = rons_adu * gain
  
    # define good / usable regions within each overscan region (ie where I consider it flat enough)
    good_os1 = os1[ramps[0]:, :]
    good_os2 = os2[ramps[1]:, :]
    good_os3 = os3[:-ramps[2], :]
    good_os4 = os4[:-ramps[3], :]
    
    # get the overscan levels, ie the constant offsets (excluding the first/last dodgy pixel column)
    # NOTE: this is usually within +/- 1 ADU of the median of the bias level as measured from bias frames !!! (tested by looking at the OS region of bias frames)
    os_level_1 = np.nanmedian(sigma_clip(good_os1[:, 1:].flatten(), clip))
    os_level_2 = np.nanmedian(sigma_clip(good_os2[:, :-1].flatten(), clip))
    os_level_3 = np.nanmedian(sigma_clip(good_os3[:, :-1].flatten(), clip))
    os_level_4 = np.nanmedian(sigma_clip(good_os4[:, 1:].flatten(), clip))
    offsets = np.array([os_level_1, os_level_2, os_level_3, os_level_4])
    if return_oslevels_only:
        return offsets
 
    # make (4k x 4k) frame of the offsets
    offmask = np.ones((ny,nx))
    q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
    for q,offset in zip([q1,q2,q3,q4], offsets):
        offmask[q] = offmask[q] * offset
  
    # create "master bias" (4k x 4k) frame (incl. OS levels) from that (note the order is important, following the definition of the quadrants)
    bias_only = np.vstack([np.hstack([model_os1, model_os2]), np.hstack([model_os4, model_os3])]) + add
      
    # add overscan levels
    bias = bias_only + offmask
      
    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time, 1))+' seconds')
    
    return bias, bias_only, offsets, rons




def get_bias_and_readnoise_from_bias_frames(bias_list, degpol=5, clip=5, gain=None, save_medimg=True, debug_level=0, timit=False):
    """
    Calculate the median bias frame after subtracting the overscan levels, the remaining offsets in the four different quadrants
    (assuming bias frames are flat within a quadrant), and the read-out noise per quadrant (ie the STDEV of the signal, but from difference images).
    
    INPUT:
    'bias_list'    : list of raw bias image files (incl. directories)
    'degpol'       : order of the polynomial (in each direction) to be used in the 2-dim polynomial surface fits to each quadrant's median bais frame
    'clip'         : number of 'sigmas' used to identify outliers when 'cleaning' each quadrant's median bais frame before the surface fitting
    'gain'         : array of gains for each quadrant (in units of e-/ADU)
    'save_medimg'  : boolean - do you want to save the median image to a FITS file?
    'debug_level'  : for debugging...
    'timit'        : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'medimg'   : the median bias frame [ADU] (after subtracting the overscan levels)
    'coeffs'   : the coefficients that describe the 2-dim polynomial surface fit to the 4 quadrants
    'offsets'  : the remaining 4 constant offsets per quadrant (assuming bias frames are flat within a quadrant) [ADU]
    'rons'     : read-out noise for the 4 quadrants [e-]
    """
    
    if timit:
        start_time = time.time()

    print('Determining offset levels and read-out noise properties from bias frames for 4 quadrants...')
    
    # code defensively...
    assert gain is not None, 'ERROR: gain is not defined!'

    # get image dimensions    
    ny,nx = crop_overscan_region(correct_orientation(pyfits.getdata(bias_list[0]))).shape
    
    # define four quadrants via masks
    q1,q2,q3,q4 = make_quadrant_masks(nx,ny)

    # prepare arrays
    medians_q1 = []
    sigs_q1 = []
    medians_q2 = []
    sigs_q2 = []
    medians_q3 = []
    sigs_q3 = []
    medians_q4 = []
    sigs_q4 = []
    allimg = []

    if debug_level >= 1:
        print('Determining bias levels and read-out noise from '+str(len(bias_list))+' bias frames...')

    # first get mean / median for all bias images (per quadrant)
    for name in bias_list:
        
        if debug_level >= 1:
            print('OK, reading file  "' + name + '"')
        
        img = pyfits.getdata(name)
        
        # get the overscan levels so we can subtract them later
        ### OLD WAY:
        #  offsets = get_bias_and_readnoise_from_overscan(img, gain=gain, return_oslevels_only=True)
#         overscan_img, bias_only, offsets, readnoise = get_bias_and_readnoise_from_overscan_polyfit(img, gain=gain, return_oslevels_only=False)
        overscan_img, bias_only, offsets, readnoise = get_bias_and_readnoise_from_overscan_collapse(img, gain=gain)
        
        # bring to "correct" orientation
        img = correct_orientation(img)
        # remove the overscan region
        img = crop_overscan_region(img)
        
        ### THE OLD WAY (I THINK THAT'S WRONG) ###
        # #  make (4k x 4k) frame of the offsets
        # offmask = np.ones((ny,nx))
        # for q,osl in zip([q1,q2,q3,q4], offsets):
        #     offmask[q] = offmask[q] * osl
        #
        # # subtract overscan levels
        # img = img - offmask
        #####

        # subtract overscan
        img = img - overscan_img

        # append quadrant-medians to lists
        medians_q1.append(np.nanmedian(img[q1]))
        medians_q2.append(np.nanmedian(img[q2]))
        medians_q3.append(np.nanmedian(img[q3]))
        medians_q4.append(np.nanmedian(img[q4]))
        allimg.append(img)

    # get RON from RMS for ALL DIFFERENT COMBINATIONS of length 2 of the images in 'bias_list'
    # by using the difference images we are less susceptible to funny pixels (hot, warm, cosmics, etc.)
    list_of_combinations = list(combinations(bias_list, 2))
    for (name1,name2) in list_of_combinations:
        
        # read in observations and bring to right format
        img1 = pyfits.getdata(name1)
        img2 = pyfits.getdata(name2)

        img1 = correct_orientation(img1)
        img1 = crop_overscan_region(img1)
        img2 = correct_orientation(img2)
        img2 = crop_overscan_region(img2)

        # take difference and do sigma-clipping
        diff = img1.astype(long) - img2.astype(long)
        sigs_q1.append(np.nanstd(sigma_clip(diff[q1], 5))/np.sqrt(2))
        sigs_q2.append(np.nanstd(sigma_clip(diff[q2], 5))/np.sqrt(2))
        sigs_q3.append(np.nanstd(sigma_clip(diff[q3], 5))/np.sqrt(2))
        sigs_q4.append(np.nanstd(sigma_clip(diff[q4], 5))/np.sqrt(2))

    # offset and read-out noise arrays
    offsets = np.array([np.median(medians_q1), np.median(medians_q2), np.median(medians_q3), np.median(medians_q4)])
    rons = np.array([np.median(sigs_q1), np.median(sigs_q2), np.median(sigs_q3), np.median(sigs_q4)])
    
    # get median image as well
    medimg = np.median(np.array(allimg), axis=0)
    # make a copy of that, which we will clean of bad pixels for the surface fits
    clean_medimg = medimg.copy()
    
    ##### now fit a 2D polynomial surface to the median bias image (for each quadrant separately)
        
    # now, because all quadrants are the same size, they have the same "normalized coordinates", so only have to do that once
    xq1 = np.arange(0,(nx/2))
    yq1 = np.arange(0,(ny/2))
    XX_q1,YY_q1 = np.meshgrid(xq1,yq1)
    x_norm = (XX_q1.flatten() / ((len(xq1)-1)/2.)) - 1.
    y_norm = (YY_q1.flatten() / ((len(yq1)-1)/2.)) - 1.
    
    # Quadrant 1
    medimg_q1 = clean_medimg[:(ny/2), :(nx/2)]
    # clean this, otherwise the surface fit will be rubbish
    medimg_q1[np.abs(medimg_q1 - np.median(medians_q1)) > clip * np.median(sigs_q1)] = np.median(medians_q1)
    coeffs_q1 = polyfit2d(x_norm, y_norm, medimg_q1.flatten(), order=degpol)
    
    # Quadrant 2
    medimg_q2 = clean_medimg[:(ny/2), (nx/2):]
    # clean this, otherwise the surface fit will be rubbish
    medimg_q2[np.abs(medimg_q2 - np.median(medians_q2)) > clip * np.median(sigs_q2)] = np.median(medians_q2)
#     xq2 = np.arange((nx/2),nx)
#     yq2 = np.arange(0,(ny/2))
#     XX_q2,YY_q2 = np.meshgrid(xq2,yq2)
#     xq2_norm = (XX_q2.flatten() / ((len(xq2)-1)/2.)) - 3.   #not quite right
#     yq2_norm = (YY_q2.flatten() / ((len(yq2)-1)/2.)) - 1.
    coeffs_q2 = polyfit2d(x_norm, y_norm, medimg_q2.flatten(), order=degpol)
    
    # Quadrant 3
    medimg_q3 = clean_medimg[(ny/2):, (nx/2):]
    # clean this, otherwise the surface fit will be rubbish
    medimg_q3[np.abs(medimg_q3 - np.median(medians_q3)) > clip * np.median(sigs_q3)] = np.median(medians_q3)
    coeffs_q3 = polyfit2d(x_norm, y_norm, medimg_q3.flatten(), order=degpol)
    
    # Quadrant 4
    medimg_q4 = clean_medimg[(ny/2):, :(nx/2)]
    # clean this, otherwise the surface fit will be rubbish
    medimg_q4[np.abs(medimg_q4 - np.median(medians_q4)) > clip * np.median(sigs_q4)] = np.median(medians_q4)
    coeffs_q4 = polyfit2d(x_norm, y_norm, medimg_q4.flatten(), order=degpol)
    
    # return all coefficients as 4-element array
    coeffs = np.array([coeffs_q1, coeffs_q2, coeffs_q3, coeffs_q4])            

    # convert read-out noise (but NOT offsets!!!) to units of electrons rather than ADUs by multiplying with the gain (which has units of e-/ADU)
    rons = rons * gain

    print('Done!!!')
    
    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    if save_medimg:
        # save median bias image
        dum = bias_list[0].split('/')
        path = bias_list[0][0:-len(dum[-1])]
        # write median bias image to file
        pyfits.writeto(path+'median_bias.fits', medimg, clobber=True)
        pyfits.setval(path+'median_bias.fits', 'UNITS', value='ADU')
        pyfits.setval(path+'median_bias.fits', 'HISTORY', value='   median BIAS frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')

    return medimg, coeffs, offsets, rons





def old_get_bias_and_readnoise_from_bias_frames(bias_list, degpol=5, clip=5, gain=None, save_medimg=True, debug_level=0, timit=False):
    """
    Calculate the median bias frame, the offsets in the four different quadrants (assuming bias frames are flat within a quadrant),
    and the read-out noise per quadrant (ie the STDEV of the signal, but from difference images).
    
    INPUT:
    'bias_list'    : list of raw bias image files (incl. directories)
    'degpol'       : order of the polynomial (in each direction) to be used in the 2-dim polynomial surface fits to each quadrant's median bais frame
    'clip'         : number of 'sigmas' used to identify outliers when 'cleaning' each quadrant's median bais frame before the surface fitting
    'gain'         : array of gains for each quadrant (in units of e-/ADU)
    'save_medimg'  : boolean - do you want to save the median image to a FITS file?
    'debug_level'  : for debugging...
    'timit'        : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'medimg'   : the median bias frame [ADU]
    'coeffs'   : the coefficients that describe the 2-dim polynomial surface fit to the 4 quadrants
    'offsets'  : the 4 constant offsets per quadrant (assuming bias frames are flat within a quadrant) [ADU]
    'rons'     : read-out noise for the 4 quadrants [e-]
    """
    
    if timit:
        start_time = time.time()

    print('Determining offset levels and read-out noise properties from bias frames for 4 quadrants...')

    img = pyfits.getdata(bias_list[0])

    # do some formatting things for real observations
    # bring to "correct" orientation
    img = correct_orientation(img)
    # remove the overscan region, which looks crap for actual bias images
    img = crop_overscan_region(img)


    ny,nx = img.shape

    # define four quadrants via masks
    q1,q2,q3,q4 = make_quadrant_masks(nx,ny)

    # co-add all bias frames
    #MB = create_master_img(bias_list, imgtype='bias', with_errors=False, savefiles=False, remove_outliers=True)

    # prepare arrays
    #means_q1 = []
    medians_q1 = []
    sigs_q1 = []
    #means_q2 = []
    medians_q2 = []
    sigs_q2 = []
    #means_q3 = []
    medians_q3 = []
    sigs_q3 = []
    #means_q4 = []
    medians_q4 = []
    sigs_q4 = []
    allimg = []

    if debug_level >= 1:
        print('Determining bias levels and read-out noise from '+str(len(bias_list))+' bias frames...')

    # first get mean / median for all bias images (per quadrant)
    for name in bias_list:
        
        if debug_level >= 1:
            print('OK, reading ',name)
        
        img = pyfits.getdata(name)
        
        # bring to "correct" orientation
        img = correct_orientation(img)
        # remove the overscan region
        img = crop_overscan_region(img)
        
        #means_q1.append(np.nanmean(img[q1]))
        medians_q1.append(np.nanmedian(img[q1]))
        #means_q2.append(np.nanmean(img[q2]))
        medians_q2.append(np.nanmedian(img[q2]))
        #means_q3.append(np.nanmean(img[q3]))
        medians_q3.append(np.nanmedian(img[q3]))
        #means_q4.append(np.nanmean(img[q4]))
        medians_q4.append(np.nanmedian(img[q4]))
        allimg.append(img)

    # now get sigma of RON for ALL DIFFERENT COMBINATIONS of length 2 of the images in 'bias_list'
    # by using the difference images we are less susceptible to funny pixels (hot, warm, cosmics, etc.)
    list_of_combinations = list(combinations(bias_list, 2))
    for (name1,name2) in list_of_combinations:
        
        # read in observations and bring to right format
        img1 = pyfits.getdata(name1)
        img2 = pyfits.getdata(name2)

        img1 = correct_orientation(img1)
        img1 = crop_overscan_region(img1)
        img2 = correct_orientation(img2)
        img2 = crop_overscan_region(img2)

        #take difference and do sigma-clipping
        diff = img1.astype(long) - img2.astype(long)
        sigs_q1.append(np.nanstd(sigma_clip(diff[q1], 5))/np.sqrt(2))
        sigs_q2.append(np.nanstd(sigma_clip(diff[q2], 5))/np.sqrt(2))
        sigs_q3.append(np.nanstd(sigma_clip(diff[q3], 5))/np.sqrt(2))
        sigs_q4.append(np.nanstd(sigma_clip(diff[q4], 5))/np.sqrt(2))

    # now average over all images
    #allmeans = np.array([np.median(medians_q1), np.median(medians_q2), np.median(medians_q3), np.median(medians_q4)])
    offsets = np.array([np.median(medians_q1), np.median(medians_q2), np.median(medians_q3), np.median(medians_q4)])
    rons = np.array([np.median(sigs_q1), np.median(sigs_q2), np.median(sigs_q3), np.median(sigs_q4)])
    
    # get median image as well
    medimg = np.median(np.array(allimg), axis=0)
    # make a copy that we will clean of bad pixels for the surface fits
    clean_medimg = medimg.copy()
    
    ##### now fit a 2D polynomial surface to the median bias image (for each quadrant separately)
        
    # now, because all quadrants are the same size, they have the same "normalized coordinates", so only have to do that once
    xq1 = np.arange(0,(nx/2))
    yq1 = np.arange(0,(ny/2))
    XX_q1,YY_q1 = np.meshgrid(xq1,yq1)
    x_norm = (XX_q1.flatten() / ((len(xq1)-1)/2.)) - 1.
    y_norm = (YY_q1.flatten() / ((len(yq1)-1)/2.)) - 1.
    
    # Quadrant 1
    medimg_q1 = clean_medimg[:(ny/2), :(nx/2)]
    # clean this, otherwise the surface fit will be rubbish
    medimg_q1[np.abs(medimg_q1 - np.median(medians_q1)) > clip * np.median(sigs_q1)] = np.median(medians_q1)
    coeffs_q1 = polyfit2d(x_norm, y_norm, medimg_q1.flatten(), order=degpol)
    
    # Quadrant 2
    medimg_q2 = clean_medimg[:(ny/2), (nx/2):]
    # clean this, otherwise the surface fit will be rubbish
    medimg_q2[np.abs(medimg_q2 - np.median(medians_q2)) > clip * np.median(sigs_q2)] = np.median(medians_q2)
#     xq2 = np.arange((nx/2),nx)
#     yq2 = np.arange(0,(ny/2))
#     XX_q2,YY_q2 = np.meshgrid(xq2,yq2)
#     xq2_norm = (XX_q2.flatten() / ((len(xq2)-1)/2.)) - 3.   #not quite right
#     yq2_norm = (YY_q2.flatten() / ((len(yq2)-1)/2.)) - 1.
    coeffs_q2 = polyfit2d(x_norm, y_norm, medimg_q2.flatten(), order=degpol)
    
    # Quadrant 3
    medimg_q3 = clean_medimg[(ny/2):, (nx/2):]
    # clean this, otherwise the surface fit will be rubbish
    medimg_q3[np.abs(medimg_q3 - np.median(medians_q3)) > clip * np.median(sigs_q3)] = np.median(medians_q3)
    coeffs_q3 = polyfit2d(x_norm, y_norm, medimg_q3.flatten(), order=degpol)
    
    # Quadrant 4
    medimg_q4 = clean_medimg[(ny/2):, :(nx/2)]
    # clean this, otherwise the surface fit will be rubbish
    medimg_q4[np.abs(medimg_q4 - np.median(medians_q4)) > clip * np.median(sigs_q4)] = np.median(medians_q4)
    coeffs_q4 = polyfit2d(x_norm, y_norm, medimg_q4.flatten(), order=degpol)
    
    # return all coefficients as 4-element array
    coeffs = np.array([coeffs_q1, coeffs_q2, coeffs_q3, coeffs_q4])            

    # convert read-out noise (but NOT offsets!!!) to units of electrons rather than ADUs by multiplying with the gain (which has units of e-/ADU)
    if gain is None:
        print('ERROR: gain(s) not set!!!')
        return
    else:
        rons = rons * gain

    if debug_level >= 1:
        # plot stuff
        print('plot the distributions for the four quadrants maybe!?!?!?')

    print('Done!!!')
    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    if save_medimg:
        # save median bias image
        dum = bias_list[0].split('/')
        path = bias_list[0][0:-len(dum[-1])]
        # write median bias image to file
        pyfits.writeto(path+'median_bias.fits', medimg, clobber=True)
        pyfits.setval(path+'median_bias.fits', 'UNITS', value='ADU')
        pyfits.setval(path+'median_bias.fits', 'HISTORY', value='   master BIAS frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')

    return medimg, coeffs, offsets, rons





def make_ronmask(rons, nx, ny, nq=4, gain=None, savefile=False, path=None, timit=False):
    """
    This routine creates a 1-level or 4-level master bias frame of size (ny,nx)
    
    INPUT:
    'rons'       : read-out noise level(s) from "get_offset_and_readnoise_from_bias_frames" [e-]
    'nx'         : image dimensions
    'ny'         : image dimensions
    'nq'         : number of quadrants
    'gain'       : array of gains for each quadrant (in units of e-/ADU) (only needed for writing the header really...)
    'savefile'   : boolean - do you want to save the read-noise mask to a fits file?
    'path'       : path to the output file directory (only needed if savefile is set to TRUE)
    'timit'      : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'ronmask'  : read-out noise mask (or RON-image really...) [e-]
    """

    if timit:
        start_time = time.time()

    if nq == 1:
        ronmask = np.ones((ny,nx)) * rons
    elif nq == 4:
        ronmask = np.ones((ny,nx))
        q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
        for q,RON in zip([q1,q2,q3,q4],rons):
            ronmask[q] = ronmask[q] * RON

    if savefile:
        #check if gain is set
        if gain is None:
            print('ERROR: gain(s) not set!!!')
            return
                
        if path is None:
            print('ERROR: output file directory not provided!!!')
            return
        else:

            # make read-out noise mask and save to fits file
            pyfits.writeto(path+'read_noise_mask.fits', ronmask, clobber=True)
            pyfits.setval(path+'read_noise_mask.fits', 'UNITS', value='ELECTRONS')
            pyfits.setval(path+'read_noise_mask.fits', 'HISTORY', value='   read-noise frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
            if nq == 1:
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN', value=gain, comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE', value=rons, comment='in ELECTRONS')
            elif nq == 4:
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_1', value=rons[0], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_2', value=rons[1], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_3', value=rons[2], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_4', value=rons[3], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_1', value=gain[0], comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_2', value=gain[1], comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_3', value=gain[2], comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_4', value=gain[3], comment='in e-/ADU')

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return ronmask





def make_offmask_and_ronmask(offsets, rons, nx, ny, gain=None, savefiles=False, path=None, timit=False):
    """
    This routine creates a 1-level or 4-level master bias frame of size (ny,nx)
    
    INPUT:
    'offsets'    : offset/bias levels as measured by "get_offset_and_readnoise_from_bias_frames" (either 1-element or 4-element) [ADU]
    'rons'       : read-out noise level(s) from "get_offset_and_readnoise_from_bias_frames" [e-]
    'nx'         : image dimensions
    'ny'         : image dimensions
    'gain'       : array of gains for each quadrant (in units of e-/ADU) (only needed for writing the header really...)
    'savefiles'  : boolean - do you want to save the master bias to a fits file?
    'path'       : path to the output file directory (only needed if savefile is set to TRUE)
    'timit'      : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'offmask'  : master bias image (or offset-image really...) [ADU]
    'ronmask'  : read-out noise mask (or RON-image really...) [e-]
    """

    if timit:
        start_time = time.time()

    nq = len(offsets)

    if nq == 1:
        offmask = np.ones((ny,nx)) * offsets
        ronmask = np.ones((ny,nx)) * rons
    elif nq == 4:
        offmask = np.ones((ny,nx))
        ronmask = np.ones((ny,nx))
        q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
        for q,offset in zip([q1,q2,q3,q4],offsets):
            offmask[q] = offmask[q] * offset
        for q,RON in zip([q1,q2,q3,q4],rons):
            ronmask[q] = ronmask[q] * RON 
    else:
        print('ERROR: "offsets" must either be a scalar (for single-port readout) or a 4-element array/list (for four-port readout)!')
        return
            

    if savefiles:
        #check if gain is set
        if gain is None:
            print('ERROR: gain(s) not set!!!')
            return
                
        if path is None:
            print('ERROR: output file directory not provided!!!')
            return
        else:
            #write offmask to file
            pyfits.writeto(path+'offmask.fits', offmask, clobber=True)
            pyfits.setval(path+'offmask.fits', 'UNITS', value='ADU')
            pyfits.setval(path+'offmask.fits', 'HISTORY', value='   offset mask - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
            if nq == 1:
                pyfits.setval(path+'offmask.fits', 'OFFSET', value=offsets, comment='in ADU')
                pyfits.setval(path+'offmask.fits', 'RNOISE', value=rons, comment='in ELECTRONS')
            elif nq == 4:
                pyfits.setval(path+'offmask.fits', 'OFFSET_1', value=offsets[0], comment='in ADU')
                pyfits.setval(path+'offmask.fits', 'OFFSET_2', value=offsets[1], comment='in ADU')
                pyfits.setval(path+'offmask.fits', 'OFFSET_3', value=offsets[2], comment='in ADU')
                pyfits.setval(path+'offmask.fits', 'OFFSET_4', value=offsets[3], comment='in ADU')
                pyfits.setval(path+'offmask.fits', 'RNOISE_1', value=rons[0], comment='in ELECTRONS')
                pyfits.setval(path+'offmask.fits', 'RNOISE_2', value=rons[1], comment='in ELECTRONS')
                pyfits.setval(path+'offmask.fits', 'RNOISE_3', value=rons[2], comment='in ELECTRONS')
                pyfits.setval(path+'offmask.fits', 'RNOISE_4', value=rons[3], comment='in ELECTRONS')

            #now make read-out noise mask
            pyfits.writeto(path+'read_noise_mask.fits', ronmask, clobber=True)
            pyfits.setval(path+'read_noise_mask.fits', 'UNITS', value='ELECTRONS')
            pyfits.setval(path+'read_noise_mask.fits', 'HISTORY', value='   read-noise frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
            if nq == 1:
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET', value=offsets, comment='in ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN', value=gain, comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE', value=rons, comment='in ELECTRONS')
            elif nq == 4:
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_1', value=offsets[0], comment='in ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_2', value=offsets[1], comment='in ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_3', value=offsets[2], comment='in ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_4', value=offsets[3], comment='in ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_1', value=rons[0], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_2', value=rons[1], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_3', value=rons[2], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_4', value=rons[3], comment='in ELECTRONS')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_1', value=gain[0], comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_2', value=gain[1], comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_3', value=gain[2], comment='in e-/ADU')
                pyfits.setval(path+'read_noise_mask.fits', 'GAIN_4', value=gain[3], comment='in e-/ADU')

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return offmask,ronmask





def make_master_bias_from_coeffs(coeffs, nx, ny, savefile=False, path=None, timit=False):
    """
    Construct the master bais frame from the coefficients for the 2-dim polynomial surface fits to the 4 quadrants of the median bias frame.
    
    INPUT:
    'coeffs'    : coefficients for the 2-dim polynomial surface fits to the 4 quadrants of the median bias frame from "get_bias_and_readnoise_from_bias_frames"
    'nx'        : x-dimension of the full image (dispersion direction)
    'ny'        : y-dimension of the full image (cross-dispersion direction)
    'savefile'  : boolean - do you want to save the master bias frame to a fits file?
    'path'      : path to the output file directory (only needed if savefile is set to TRUE)
    'timit'     : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'master_bias'  : the master bias frame [ADU]
    """
    
    if timit:
        start_time = time.time()
    
    #separate coefficients
    coeffs_q1 = coeffs[0]
    coeffs_q2 = coeffs[1]
    coeffs_q3 = coeffs[2]
    coeffs_q4 = coeffs[3]
    
    #create normalized x- & y-coordinates; size of the quadrants is (nx/2) x (ny/2)
    #the normalized coordinates are the same for all quadrants, of course
    xx_q1 = np.arange(nx/2)    
    yy_q1 = np.arange(ny/2)      
    xxn_q1 = (xx_q1 / (((nx/2)-1)/2.)) - 1. 
    yyn_q1 = (yy_q1 / (((ny/2)-1)/2.)) - 1.
    XX_norm,YY_norm = np.meshgrid(xxn_q1,yyn_q1)
    
    #model the 4 quadrants
    model_q1 = polyval2d(XX_norm, YY_norm, coeffs_q1)
    model_q2 = polyval2d(XX_norm, YY_norm, coeffs_q2)
    model_q3 = polyval2d(XX_norm, YY_norm, coeffs_q3)
    model_q4 = polyval2d(XX_norm, YY_norm, coeffs_q4)
    
    #make master bias frame from 4 quadrant models
    master_bias = np.zeros((ny,nx))
    master_bias[:(ny/2), :(nx/2)] = model_q1
    master_bias[:(ny/2), (nx/2):] = model_q2
    master_bias[(ny/2):, (nx/2):] = model_q3
    master_bias[(ny/2):, :(nx/2)] = model_q4
    
    
    #now save to FITS file
    if savefile:
        if path is None:
            print('ERROR: output file directory not provided!!!')
            return
        else:
            #get header from the read-noise mask file
            h = pyfits.getheader(path+'read_noise_mask.fits')
            #change a few things
            for i in range(1,5):
                del h['offset_'+str(i)]
            h['UNITS'] = 'ADU'
            h['HISTORY'][0] = ('   master BIAS frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
            #write master bias to file
            pyfits.writeto(path+'master_bias.fits', master_bias, h, clobber=True)
    
    
    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')
    
    return master_bias





def make_master_dark(dark_list, MB, gain=None, scalable=False, noneg=False, savefile=True, path=None, debug_level=0, timit=False):
    """
    This routine creates a "MASTER DARK" frame from a given list of dark frames. It also subtracts the MASTER BIAS and the overscan levels 
    from each dark frame before combining them into the master dark frame.
    NOTE: the output is in units of ELECTRONS!!!
    
    INPUT:
    'dark_list'    : list of raw dark image files (incl. directories)
    'MB'           : the master bias frame [ADU]
    'gain'         : the gains for each quadrant [e-/ADU]
    'scalable'     : boolean - do you want to normalize the dark current to an exposure time of 1s? (ie do you want to make it "scalable"?)
    'noneg'        : boolean - do you want to allow negative pixels? (True=no, False=yes) 
    'savefile'     : boolean - do you want to save the master dark frame to a fits file?
    'path'         : path to the output file directory (only needed if savefile is set to TRUE)
    'debug_level'  : for debugging...
    'timit'        : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'MD'  : the master dark frame [e-]
    """

    if timit:
        start_time = time.time()

    if debug_level >= 1:
        print('Creating master dark frame from '+str(len(dark_list))+' dark frames...')      
        
    # code defensively...
    assert gain is not None, 'ERROR: gain is not defined!' 

    # get a list of all the exposure times first
    exp_times = []
    for file in sorted(dark_list):
        exp_times.append(np.round(pyfits.getval(file,'TOTALEXP'),0))

    # list of unique exposure times
    unique_exp_times = np.array(list(sorted(set(exp_times))))
    if debug_level >= 1 and len(unique_exp_times) > 1:
        print('WARNING: not all dark frames have the same exposure times! Found '+str(len(unique_exp_times))+' unique exposure times!!!')

    # create median dark files
    if scalable:
        # get median image (including subtraction of master bias) and scale to texp=1s 
        MD = make_median_image(dark_list, MB=MB, correct_OS=True, scalable=scalable, raw=False)
        ny,nx = MD.shape
        q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
        # convert to units of electrons
        MD[q1] = gain[0] * MD[q1]
        MD[q2] = gain[1] * MD[q2]
        MD[q3] = gain[2] * MD[q3]
        MD[q4] = gain[3] * MD[q4]
        if noneg:
            MD = np.clip(MD, 0, None)
    else:
        # make dark "sublists" for all unique exposure times
        all_dark_lists = []
        for i in range(len(unique_exp_times)):
            all_dark_lists.append( np.array(dark_list)[np.argwhere(exp_times == unique_exp_times[i]).flatten()] )
        # get median image (including subtraction of master bias) for each "sub-list"
        MD = []
        for sublist in all_dark_lists:
            sub_MD = make_median_image(sublist, MB=MB, scalable=scalable, raw=False)
            ny,nx = sub_MD.shape
            q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
            # convert to units of electrons
            sub_MD[q1] = gain[0] * sub_MD[q1]
            sub_MD[q2] = gain[1] * sub_MD[q2]
            sub_MD[q3] = gain[2] * sub_MD[q3]
            sub_MD[q4] = gain[3] * sub_MD[q4]
            if noneg:
                MD = np.clip(MD, 0, None)
            MD.append(sub_MD)

    # save to FITS file
    if savefile:
        if path is None:
            print('WARNING: output file directory not provided!!!')
            print('Using same directory as input file...')
            dum = dark_list[0].split('/')
            path = dark_list[0][0:-len(dum[-1])]
        if scalable:
            outfn = path+'master_dark_scalable.fits'
            #get header from master BIAS frame
            h = pyfits.getheader(path+'master_bias.fits')
            h['HISTORY'][0] = '   MASTER DARK frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
            h['UNITS'] = 'ELECTRONS'
            h['COMMENT'] = 're-normalized to texp=1s to make it scalable'
            h['TOTALEXP'] = (1., 'exposure time [s]')
            pyfits.writeto(outfn, MD, h, clobber=True)
        else:
            for i,submd in enumerate(MD):
                outfn = path+'master_dark_t'+str(int(unique_exp_times[i]))+'.fits'
                #get header from master BIAS frame
                h = pyfits.getheader(path+'master_bias.fits')
                h['HISTORY'][0] = '   MASTER DARK frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
                h['TOTALEXP'] = (unique_exp_times[i], 'exposure time [s]')
                h['UNITS'] = 'ELECTRONS'
                pyfits.writeto(outfn, submd, h, clobber=True)

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return MD





def correct_for_bias_and_dark_from_filename(imgname, MB, MD, gain=None, scalable=False, savefile=False, path=None, simu=False, timit=False):
    """
    This routine subtracts both the MASTER BIAS frame [in ADU], and the MASTER DARK frame [in e-] from a given (single!) image.
    It also corrects the orientation of the image and crops the overscan regions.
    NOTE: the input image has units of ADU, but the output image has units of electrons!!!
    Clone of "correct_for_bias_and_dark", but this one allows us to save output files
    
    INPUT:
    'imgname'   : filename of raw science image (incl. directory)
    'MB'        : the master bias frame (bias only, excluding overscan) [ADU]
    'MD'        : the master dark frame [e-]
    'gain'      : the gains for each quadrant [e-/ADU]
    'scalable'  : boolean - do you want to normalize the dark current to an exposure time of 1s? (ie do you want to make it "scalable"?)
    'savefile'  : boolean - do you want to save the bias- & dark-corrected image (and corresponding error array) to a FITS file?
    'path'      : output file directory
    'simu'      : boolean - are you using Echelle++ simulated observations?
    'timit'     : boolean - do you want to measure the execution run time?
    
    OUTPUT:
    'dc_bc_img'  : the bias- & dark-corrected image [e-] (also has been brought to 'correct' orientation and overscan regions cropped) 
    
    MODHIST:
    # CMB - I removed the 'ronmask' and 'err_MD' INPUTs
    # CMB (12 Jun 2019) - implemented separate overscan and bias removal
    
    """
    if timit:
        start_time = time.time()

    # code defensively...
    assert gain is not None, 'ERROR: gain is not defined!' 

    ### (0) read in raw image [ADU] 
    img = pyfits.getdata(imgname)

    # get the overscan levels so we can subtract them later
    ### OLD WAY:
    #  offsets = get_bias_and_readnoise_from_overscan(img, gain=gain, return_oslevels_only=True)
#     overscan_img, bias_only, offsets, readnoise = get_bias_and_readnoise_from_overscan_polyfit(img, gain=gain, return_oslevels_only=False)
    overscan_img, bias_only, offsets, readnoise = get_bias_and_readnoise_from_overscan_collapse(img, gain=gain)
    
    if not simu:
        # bring to "correct" orientation
        img = correct_orientation(img)
        # remove the overscan region
        img = crop_overscan_region(img)


    ### OLD WAY:
    # # make (4k x 4k) frame of the offsets
    ny,nx = img.shape
    # offmask = np.ones((ny,nx))
    # define four quadrants via masks
    q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
    # for q,osl in zip([q1,q2,q3,q4], offsets):
    #     offmask[q] = offmask[q] * osl
    #
    # # subtract overscan levels
    # img = img - offmask - MB
    #####

    ### (1) BIAS AND OVERSCAN SUBTRACTION [ADU]
    # subtract overscan
    bc_img = img - overscan_img - MB


    ### (2) conversion to ELECTRONS and DARK SUBTRACTION [e-]
    # if the darks have a different exposure time than the image we are trying to correct, we need to re-scale the master dark
    if scalable:
        try:
            texp = pyfits.getval(imgname, 'ELAPSED')
            MD = MD * texp
#             #cannot have an error estimate lower than the read-out noise; this is dodgy but don't know what else to do
#             err_MD = np.maximum(err_MD * texp, ronmask)
        except:
            print('ERROR: "texp" has to be provided when "scalable" is set to TRUE')
            return -1
    # convert image to electrons now    
    bc_img[q1] = gain[0] * bc_img[q1]
    bc_img[q2] = gain[1] * bc_img[q2]
    bc_img[q3] = gain[2] * bc_img[q3]
    bc_img[q4] = gain[3] * bc_img[q4]
    # now subtract master dark frame [e-] to create dark- & bias- & overscan-corrected image [e-]
    dc_bc_img = bc_img - MD


    # if desired, write bias- & dark-corrected image (and error array???) to fits file
    if savefile:
        dum = imgname.split('/')
        dum2 = dum[-1].split('.')
        shortname = dum2[0]
        if path is None:
            print('WARNING: output file directory not provided!!!')
            print('Using same directory as input file...')
            path = imgname[0: -len(dum[-1])]
#         outfn = path+shortname+'_bias_and_dark_corrected.fits'
        outfn = path+shortname+'_BD.fits'
        # get header from the original image FITS file
        h = pyfits.getheader(imgname)
        h['UNITS'] = 'ELECTRONS'
        h['HISTORY'] = '   BIAS- & DARK-corrected image - created ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ' (GMT)'
        pyfits.writeto(outfn, dc_bc_img, h, clobber=True)
#         h_err = h.copy()
#         h_err['HISTORY'] = 'estimated uncertainty in BIAS- & DARK-corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
#         pyfits.append(outfn, err_dc_bc_img, h_err, clobber=True)

    if timit:
        print('Time elapsed: ' + str(np.round(time.time() - start_time,1)) + ' seconds')

#     return dc_bc_img, err_dc_bc_img
    return dc_bc_img





def make_master_calib(file_list, lamptype=None, MB=None, ronmask=None, MD=None, gain=None, chipmask=None, scalable=False, remove_bg=True,
                      savefile=True, saveall=False, path=None, debug_level=0, timit=False):
    """
    Simplified CLONE OF "process_whites".
    This routine processes all calibration lamp images from a given list of files. It corrects the orientation of the image and crops the overscan regions,
    and subtracts both the MASTER BIAS frame [in ADU], and the MASTER DARK frame [in e-] from every image before combining them to create a MASTER frame.
    NOTE: the input image has units of ADU, but the output image has units of electrons!!!
    
    INPUT:
    'file_list'   : list of filenames of raw white images (incl. directories)
    'lamptype'    : type of calibration lamp ['simth' / 'lfc' / 'both'] 
    'MB'          : the master bias frame (bias only, excluding OS levels) [ADU]
    'ronmask'     : the read-noise mask (or frame) [e-]
    'MD'          : the master dark frame [e-]
    'gain'        : the gains for each quadrant [e-/ADU]
    'P_id'        : order tracing dictionary (only needed if remove_bg is set to TRUE)
    'scalable'    : boolean - do you want to normalize the dark current to an exposure time of 1s? (ie do you want to make it "scalable"?)
    'remove_bg'   : boolean - do you want to remove the background from the output master white?
    'savefile'    : boolean - do you want to save the master white frame as a FITS file?
    'saveall'     : boolean - do you want to save all individual bias- & dark-corrected images as well?
    'path'        : path to the output file directory (only needed if savefile is set to TRUE)
    'debug_level' : for debugging...
    'timit'       : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'master'      : the master white image [e-] (also has been brought to 'correct' orientation, overscan regions cropped, and (if desired) bg-corrected) 
    'err_master'  : the corresponding uncertainty array [e-]    
    
    """
    
    if timit:
        start_time = time.time()

    while lamptype.lower() not in ['simth', 'lfc', 'both']:
        lamptype = raw_input("What kind of calibration source is this? ['simth', 'lfc', 'both']: ")
    typestring = lamptype [:]
    if lamptype.lower() == 'both':
        typestring = 'lfc_plus_simth'
    
    if debug_level >= 1:
        print('Creating master ' + typestring.upper() + ' frame from ' + str(len(file_list)) + ' individual exposures...')
    
    # if INPUT arrays are not given, read them from default files
    if path is None:
        print('WARNING: output file directory not provided!!!')
        print('Using same directory as input file...')
        dum = file_list[0].split('/')
        path = file_list[0][0:-len(dum[-1])]
    if MB is None:
        # no need to fix orientation, this is already a processed file [ADU]
#         MB = pyfits.getdata(path+'master_bias.fits')
        MB = pyfits.getdata(path + 'median_bias.fits')
    if ronmask is None:
        # no need to fix orientation, this is already a processed file [e-]
        ronmask = pyfits.getdata(path + 'read_noise_mask.fits')
    if MD is None:
        if scalable:
            # no need to fix orientation, this is already a processed file [e-]
            MD = pyfits.getdata(path + 'master_dark_scalable.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_scalable.fits', 1)
        else:
            # no need to fix orientation, this is already a processed file [e-]
            texp = pyfits.getval(file_list[0])
            MD = pyfits.getdata(path + 'master_dark_t' + str(int(np.round(texp,0))) + '.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_t'+str(int(np.round(texp,0)))+'.fits', 1)

    # prepare arrays
    allimg = []
    allerr = []

    # loop over all files in "file_list"; correct for bias and darks on the fly
    for n,fn in enumerate(sorted(file_list)):
        if debug_level >=1:
            print('Now processing file ' + str(n+1) + '/' + str(len(file_list)) + '   (' + fn + ')')

        # call routine that does all the bias and dark correction stuff and converts from ADU to e-
        if scalable:
            # if the darks have a different exposure time than the whites, then we need to re-scale the master dark
            texp = pyfits.getval(file_list[0], 'ELAPSED')
            img = correct_for_bias_and_dark_from_filename(fn, MB, MD*texp, gain=gain, scalable=scalable, savefile=saveall,
                                                          path=path, timit=timit)     #these are now bias- & dark-corrected images; units are e-
        else:
            img = correct_for_bias_and_dark_from_filename(fn, MB, MD, gain=gain, scalable=scalable, savefile=saveall,
                                                          path=path, timit=timit)     # these are now bias- & dark-corrected images; units are e-

        if debug_level >=2:
            print('min(img) = ' + str(np.min(img)))
        allimg.append(img)
#         err_img = np.sqrt(img + ronmask*ronmask)   # [e-]
        # TEMPFIX: (how should I be doing this properly???)
        err_img = np.sqrt(np.clip(img,0,None) + ronmask*ronmask)   # [e-]
        allerr.append(err_img)

    # list of individual exposure times for all whites (should all be the same, but just in case...)
    texp_list = [pyfits.getval(file, 'ELAPSED') for file in file_list]
    # scale to the median exposure time
    tscale = np.array(texp_list) / np.median(texp_list)


    #########################################################################
    ### now we do essentially what "CREATE_MASTER_IMG" does for whites... ###
    #########################################################################
    # add individual-image errors in quadrature (need it either way, not only for fancy method)
#     err_summed = np.sqrt(np.sum((np.array(allerr)**2), axis=0))
#     # get plain median image
#     medimg = np.median(np.array(allimg), axis=0)
    # take median after scaling to median exposure time 
    medimg = np.median(np.array(allimg) / tscale.reshape(len(allimg), 1, 1), axis=0)
    

    # now set master image equal to median image
    master = medimg.copy()
    nw = len(file_list)     # number of whites 
#         # estimate of the corresponding error array (estimate only!!!)
#         err_master = err_summed / nw     # I don't know WTF I was thinking here...
    # if roughly Gaussian distribution of values: error of median ~= 1.253*error of mean
    err_master = 1.253 * np.std(allimg, axis=0) / np.sqrt(nw-1)     # normally it would be sigma/sqrt(n), but np.std is dividing by sqrt(n), not by sqrt(n-1)
    # err_master = np.sqrt( np.sum( (np.array(allimg) - np.mean(np.array(allimg), axis=0))**2 / (nw*(nw-1)) , axis=0) )   # that is equivalent, but slower
    
    
    # now subtract background (errors remain unchanged)
    if remove_bg:
        if chipmask is None:
            date = file_list[0].split('/')[-2]
            chipmask = np.load('/Users/christoph/OneDrive - UNSW/chipmasks/archive/' + 'chipmask_' + date + '.npy').item()
        
        if lamptype == 'simth':
            lampmask = chipmask['thxe']
        elif lamptype == 'lfc':
            lampmask = chipmask['lfc']
        else:
            lampmask = np.logical_or(chipmask['thxe'], chipmask['lfc'])
            
        # grow this region by 3 pixels either side to avoid contaminating the background
        growkernel = np.ones((7,7))
        extended_lampmask = np.cast['bool'](ndimage.convolve(np.cast['float32'](lampmask), growkernel))    
        # invert mask to identify background regions
        bgmask = np.invert(extended_lampmask)
        # identify and extract background
        bg = extract_background(master, bgmask, timit=timit)
        # fit background
        bg_coeffs, bg_img = fit_background(bg, clip=10, return_full=True, timit=timit)
        # subtract background
        master = master - bg_img


    # now save master frame to file
    if savefile:
        outfn = path + 'master_' + typestring + '.fits'
        pyfits.writeto(outfn, master, clobber=True)
        pyfits.setval(outfn, 'HISTORY', value='   MASTER ' + typestring.upper() + ' frame - created ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ' (GMT)')
        pyfits.setval(outfn, 'MED_TEXP', value=np.median(texp_list), comment='median exposure time [s]')
        pyfits.setval(outfn, 'UNITS', value='ELECTRONS')
        pyfits.setval(outfn, 'METHOD', value='median', comment='method to create master ' + typestring + ' and remove outliers')
        h = pyfits.getheader(outfn)
        h_err = h.copy()
        h_err['HISTORY'] = 'estimated uncertainty in MASTER ' + typestring.upper() + ' frame - created ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ' (GMT)'
        pyfits.append(outfn, err_master, h_err, clobber=True)

    if timit:
        print('Total time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return master, err_master





########################################3
########################################3
########################################3
########################################3
########################################3

#below are old code snippets, currently not in use!!!

def correct_for_bias_and_dark(img, MB, MD, gain=None, scalable=False, texp=None, timit=False):
    """
    This routine subtracts both the MASTER BIAS frame [in ADU], and the MASTER DARK frame [in e-] from a given image.
    Note that the input image has units of ADU, but the output image has units of electrons!!!
    
    NOT UP TO DATE!!!
    """


    #CMB - I removed the 'ronmask' and 'err_MD' INPUTs

    if timit:
        start_time = time.time()

#     #(0) get error estimate for image
#     err_img = np.sqrt(img.astype(float) + ronmask*ronmask)

    #(1) BIAS SUBTRACTION
    #bias-corrected_image [ADU]
    bc_img = img - MB
#     #adjust errors accordingly
#     err_bc_img = np.sqrt(err_img*err_img + ronmask*ronmask)

    #(2) DARK SUBTRACTION
    #if the darks have a different exposure time then the images you are trying to correct, we need to re-scale the master dark
    if scalable:
        if texp is not None:
            MD = MD * texp
#             #cannot have an error estimate lower than the read-out noise; this is dodgy but don't know what else to do
#             err_MD = np.maximum(err_MD * texp, ronmask)
        else:
            print('ERROR: "texp" has to be provided when "scalable" is set to TRUE')
            return
    #dark- & bias-corrected image
    dc_bc_img = bc_img - MD
#     #adjust errors accordingly
#     err_dc_bc_img = np.sqrt(err_bc_img*err_bc_img + err_MD*err_MD)

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

#     return dc_bc_img, err_dc_bc_img
    return dc_bc_img



def bias_subtraction(imglist, MB, noneg=True, savefile=True):
    """
    DUMMY ROUTINE; NOT CURRENTLY USED
    """
    for file in imglist:
        img = pyfits.getdata(file)
        mod_img = img - MB
        if noneg:
            mod_img[mod_img < 0] = 0.
        if savefile:
            dum = file.split('/')
            path = file[:-len(dum[-1])]
            h = pyfits.getheader(file)
            pyfits.writeto(path+'bc_'+dum[-1], mod_img, h, clobber=True)
    return



def dark_subtraction(imglist, MD, noneg=True, savefile=True):
    """
    DUMMY ROUTINE; NOT CURRENTLY USED
    """
    for file in imglist:
        img = pyfits.getdata(file)
        mod_img = img - MD
        if noneg:
            mod_img[mod_img < 0] = 0.
        if savefile:
            dum = file.split('/')
            path = file[:-len(dum[-1])]
            h = pyfits.getheader(file)
            pyfits.writeto(path+'dc_'+dum[-1], mod_img, h, clobber=True)
    return



def read_and_overscan_correct(infile, overscan=53, discard_ramp=17):
    """Read in fits file and overscan correct. Assume that
    the overscan region is independent of binning."""
    dd = pyfits.getdata(infile)
    newshape = (dd.shape[0], dd.shape[1] - 2 * overscan)
    corrected = np.zeros(newshape)

    # Split the y axis in 2
    for y0, y1 in zip([0, dd.shape[0] // 2], [dd.shape[0] // 2, dd.shape[0]]):
        # left overscan region
        loverscan = dd[y0:y1, :overscan]
        overscan_ramp = np.median(loverscan + np.random.random(size=loverscan.shape) - 0.5, axis=0)
        overscan_ramp -= overscan_ramp[-1]
        for i in range(len(loverscan)):
            loverscan[i] = loverscan[i] - overscan_ramp
            corrected[y0 + i, :newshape[1] // 2] = dd[y0 + i, overscan:overscan + newshape[1] // 2] - \
                                                   np.median(loverscan[i] + np.random.random(size=loverscan[i].shape) - 0.5)
        # right overscan region                                           
        roverscan = dd[y0:y1, -overscan:]
        overscan_ramp = np.median(roverscan + np.random.random(size=loverscan.shape) - 0.5, axis=0)
        overscan_ramp -= overscan_ramp[0]
        for i in range(len(roverscan)):
            roverscan[i] = roverscan[i] - overscan_ramp
            corrected[y0 + i, newshape[1] // 2:] = dd[y0 + i, dd.shape[1] // 2:dd.shape[1] - overscan] - \
                                                   np.median(roverscan[i] + np.random.random(size=loverscan[i].shape) - 0.5)
    
    return corrected


# if __name__ == "__main__":
#     corrected = read_and_overscan_correct('bias1.fits')
#     corrected[np.where(np.abs(corrected) > 20)] = 0
#     print(np.std(corrected))

