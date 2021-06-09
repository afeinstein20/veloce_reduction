'''
Created on 25 Oct. 2017

@author: christoph
'''

import time
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.optimize import curve_fit

from .helper_functions import gauss2D, affine_matrix

__all__ =  ['find_laser_peaks_2D', 'find_affine_transformation_matrix_test']


# #some parameters
# RON = 4.                 # my guess for read-out noise
# bg_thresh = 1            # background threshold
# count_thresh = 2000.     # threshold for selection of peaks
# xsigma = 1.1             # sigma of 2D-Gaussian PSF in dispersion direction
# ysigma = 0.7             # sigma of 2D-Gaussian PSF in cross-dispersion direction
# boxwidth = 11            # size of box to use for the 2D Gaussian peak fitting
#
#
# imgname = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_laser_comb.fit'
# img = pyfits.getdata(imgname) + 1
# img1 = img[250:450,250:450]   # contains 19 peaks (1 of which at edge)
# err_img1 = np.sqrt(img1)
# img2 = img[250:450,247:447]   # contains 19 peaks (1 of which at edge)
# err_img2 = np.sqrt(img2)



def find_laser_peaks_2D(img, err_img, boxwidth=11, bg_thresh=1., count_thresh=2000., gauss_filter_size=1., RON=None, xsigma=1.2, ysigma=0.7, smooth=False, timit=False, debug_level=0):

    if timit:
        start_time = time.time()
    
    # define an 8-connected neighbourhood
    neighbourhood = generate_binary_structure(2,2)
    
    ### smooth image slightly for noise reduction
    if smooth:
        filtered_img = ndimage.gaussian_filter(img, gauss_filter_size)   
    else:
        filtered_img = img.copy()
        
#     #WARNING - THIS INTRODUCES ONE EXTRA MAXIMUM at (4095,4095)!!!
#     #add tiny slope in x- and y-directions to make sure neighbouring pixels cannot have identical count values
#     dumxr = np.arange(filtered_img.shape[0]) / (filtered_img.shape[0]*10.)
#     dumyr = np.arange(filtered_img.shape[1]) / (filtered_img.shape[1]*10.)
#     dumxx,dumyy = np.meshgrid(dumxr,dumyr)
#     filtered_img = filtered_img + dumxx + dumyy
    
    #apply the local maximum filter; all pixel of max. value in their neighbourhood are set to 1
    local_max = ndimage.maximum_filter(filtered_img, footprint=neighbourhood)==filtered_img
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.
    
    #we create the mask of the background
    background = (filtered_img <= bg_thresh)
    
    #a little technicality: we must erode the background in order to 
    #successfully subtract it from local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighbourhood, border_value=1)
    
    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask 
    peaks = local_max ^ eroded_background     # XOR operator
    
    labeled, n_peaks = ndimage.label(peaks)
    
    print('Total number of peaks found: '+str(n_peaks))
    
#     if n_peaks != np.sum(peaks):
#         print('ERROR: number of peaks found is inconsistent!')
#         quit()
    
    xy = np.array(ndimage.center_of_mass(filtered_img, labeled, range(1, n_peaks+1)))
        
    #boxwidth = 11     #should probably be an odd number
    padsize = boxwidth / 2
    
    peakshapes = {'number':[], 'x0':[], 'y0':[], 'amp':[], 'x_sigma':[], 'y_sigma':[], 'theta':[], 'chi2red':[]}
                  
    #LOOP OVER ALL LABELLED MAXIMA...
    for i in range(n_peaks):
        
        if debug_level >= 1:
            print(i, xy[i,:])
        
        #initially I had this the wrong way around
#         xr = np.arange(int(round(xy[i,0])) - padsize, int(round(xy[i,0])) + padsize +1)
#         yr = np.arange(int(round(xy[i,1])) - padsize, int(round(xy[i,1])) + padsize +1)
        yr = np.arange(int(round(xy[i,0])) - padsize, int(round(xy[i,0])) + padsize +1)     #this is y
        xr = np.arange(int(round(xy[i,1])) - padsize, int(round(xy[i,1])) + padsize +1)     #this is x
        xx,yy = np.meshgrid(xr,yr)
        z = img[yr[0]:yr[-1]+1, xr[0]:xr[-1]+1] 
        err = err_img[yr[0]:yr[-1]+1, xr[0]:xr[-1]+1] 
        
        #don't use peaks that are too close to the edge of the chip or that have less than a certain number of counts:
        if (z.size == boxwidth**2) and (np.max(z) >= count_thresh):  
            
            # # METHOD 1: ASTROPY (CANNOT GET IT TO WORK...)
            # g2d_init = models.Gaussian2D (amplitude=5.e4, x_mean=28., y_mean=975., x_stddev=2.5, y_stddev=1.5, theta=0, cov_matrix=None)
            # guess = g2d_init.evaluate(xx, yy, 5e4, 28, 975, 2.5, 1.5, 0)
            # fit_p = fitting.LevMarLSQFitter()
            # g2d = fit_p(g2d_init, xx, yy, z)
            
            #METHOD 2: 
            pguess = (np.max(z), xy[i,1], xy[i,0], xsigma, ysigma, 0.)      # (amp, x0, y0, sigma_x, sigma_y, theta)
            guessmodel = gauss2D((xx,yy), *pguess)
            xflat = xx.flatten()
            yflat = yy.flatten()
            zflat = z.flatten()
            errflat = err.flatten()
            lower_bounds = np.array([0., 0., 0., 0., 0., -np.pi/4.])
            upper_bounds = np.array([1e6, img.shape[1]-1., img.shape[0]-1., 4, 3, np.pi/4.])
            popt,pcov = curve_fit(gauss2D, (xflat,yflat), zflat, sigma=errflat, p0=pguess, bounds=(lower_bounds,upper_bounds), maxfev=1000000)
            bestmodel = gauss2D((xx,yy), *popt)
            res = z - bestmodel
            #err = np.sqrt( z + RON**2 )
            chisqarr = (res/err)**2
            chisq = np.sum(chisqarr)
            dof = z.size - len(pguess)
            chi2red = chisq / dof
            
            peakshapes['number'].append(i+1)
            peakshapes['amp'].append(popt[0])
            peakshapes['x0'].append(popt[1])
            peakshapes['y0'].append(popt[2])
            peakshapes['x_sigma'].append(popt[3])
            peakshapes['y_sigma'].append(popt[4])
            peakshapes['theta'].append(180.*popt[5]/np.pi)
            peakshapes['chi2red'].append(chi2red)
    
    
        
    if timit:
        print('Elapsed time for finding peaks: ',time.time() - start_time,' seconds')
    
    return peakshapes





def find_affine_transformation_matrix_test(peakshapes1, peakshapes2):
    
    peaks1_xy = np.array([peakshapes1['x0'],peakshapes1['y0']])
    #go to homogeneous coordinates (ie add a z-component equal to 1, so that we can include translation into the matrix)
    peaks1_xyz = np.vstack((peaks1_xy, np.expand_dims(np.repeat(1,peaks1_xy.shape[1]), axis=0)))
    
    peaks2_xy = np.array([peakshapes2['x0'],peakshapes2['y0']])
    #go to homogeneous coordinates (ie add a z-component equal to 1, so that we can include translation into the matrix)
    peaks2_xyz = np.vstack((peaks2_xy, np.expand_dims(np.repeat(1,peaks2_xy.shape[1]), axis=0)))
    
    #solve matrix equation: M*r1 = r2  (note the transpose, as linalg.lstsq wants row-vectors)
    M, res, rank, s = np.linalg.lstsq(peaks1_xyz.T, peaks2_xyz.T)
    
    #return affine transformation matrix  (note the transpose again, so that you can do :  new_points = np.dot(M.T,points) )
    return M.T





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
