'''
Created on 29 Nov. 2017

@author: christoph
'''

import time
import numpy as np
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy.optimize as op
import lmfit
from lmfit import Model
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel, MoffatModel, Pearson7Model, StudentsTModel, BreitWignerModel, LognormalModel
from lmfit.models import DampedOscillatorModel, DampedHarmonicOscillatorModel, ExponentialGaussianModel, SkewedGaussianModel, DonaichModel
from readcol import readcol 
import datetime
import astropy.io.fits as pyfits
import os
import glob

from veloce_reduction.veloce_reduction.helper_functions import fibmodel_with_amp, CMB_pure_gaussian, multi_fibmodel_with_amp, CMB_multi_gaussian, offset_pseudo_gausslike
from veloce_reduction.veloce_reduction.helper_functions import fit_poly_surface_2D, single_sigma_clip, find_nearest, gaussian_with_offset_and_slope, fibmodel_with_amp_and_offset
from veloce_reduction.utils.linelists import make_gaussmask_from_linelist
from veloce_reduction.veloce_reduction.lfc_peaks import find_affine_transformation_matrix, divide_lfc_peaks_into_orders







def find_suitable_peaks(rawdata, thresh = 5000., bgthresh = 2000., maxthresh = None, gauss_filter_sigma=1., slope=1e-4,
                        clip_edges=True, remove_bg=False, return_masks=False, debug_level=0, timit=False):
    """
    This routine finds (rough, to within 1 pixel) peak locations in 1D data.
    Detection threshold, background threshold, maximum threshold, and smoothing kernel width can be provided as keyword parameters.

    INPUT:
    'rawdata'   : one-dimensional (raw) data array

    KEYWORD PARAMETERS:
    'thresh'              : minimum height of peak in the FILTERED data to be considered a good peak
    'bgthresh'            : minimum height of peak in the filtered data to be considered a real (but not good) peak
    'maxthresh'           : maximum height of peak in the filtered data to be included (ie you can exclude saturated peaks etc)
    'gauss_filter_sigma'  : width of the Gaussian smoothing kernel (set to 0 if you want no smoothing)
    'slope'               : value of the (tiny) slope that's added in order to break possible degeneracies between adjacent pixels
    'clip_edges'          : boolean - do you want to clip the first and last peak, just to be on the safe side and avoid edge effects?
    'remove_bg'           : boolean - do you want to run a crude background removal before identifying peaks?
    'return_masks'        : boolean - do you want to return the masks as well as the data arrays?
    'debug_level'         : for debugging...
    'timit'               : boolean - do you want to clock execution run-time?

    OUTPUT:
    'goodpeaks'   : 1D array containing the "good" peak locations (above thresh and below maxthresh)
    'mostpeaks'   : 1D array containing all "real" peak locations (above bg_thresh and below maxthresh)
    'allpeaks'    : 1D array containing ALL peak locations (ie all local maxima, except for edges)
    'first_mask'  : mask that masks out shallow noise peaks from allpeaks (ONLY IF 'return_masks' is set to TRUE)
    'second_mask' : mask that masks out saturated lines above maxthresh from mostpeaks (ONLY IF 'return_masks' is set to TRUE)
    'third_mask'  : mask that masks out lines below thresh from mostpeaks (ONLY IF 'return_masks' is set to TRUE)
                    ie: goodpeaks = allpeaks[first_mask][second_mask][third_mask]
    
    TODO:
    (1) make a relative threshold criterion
    """

    #this routine is extremely fast, no need to optimise for speed
    if timit:
        start_time = time.time()
    
    xx = np.arange(len(rawdata))
    
    #run a crude background removal first:
    if remove_bg:
        bgfit = ndimage.gaussian_filter(ndimage.minimum_filter(rawdata,size=50.), 10.)
    else:
        bgfit = np.zeros(len(rawdata))
    data = rawdata - bgfit
    
    #smooth data to make sure we are not finding noise peaks, and add tiny slope to make sure peaks are found even when pixel-values are like [...,3,6,18,41,41,21,11,4,...]
    filtered_data = ndimage.gaussian_filter(data.astype(np.float), gauss_filter_sigma) + xx*slope
    
    #find all local maxima in smoothed data (and exclude the leftmost and rightmost maxima to avoid nasty edge effects...)
    allpeaks = signal.argrelextrema(filtered_data, np.greater)[0]
    
    ### this alternative version of finding extrema is completely equivalent (except that the below version accepts first and last as extrema, which I do not want)
    #testix = np.r_[True, filtered_data[1:] > filtered_data[:-1]] & np.r_[filtered_data[:-1] > filtered_data[1:], True]
    #testpeaks = np.arange(len(xx))[testix]
    
    #exclude first and last peaks to avoid nasty edge effects
    if clip_edges:
        allpeaks = allpeaks[1:-1]
    
    #make first mask to determine which peaks from linelist to use in wavelength solution
    first_mask = np.ones(len(allpeaks), dtype='bool')
    
    #remove shallow noise peaks
    first_mask[filtered_data[allpeaks] < bgthresh] = False
    mostpeaks = allpeaks[first_mask]
    
    #make mask which we need later to determine which peaks from linelist to use in wavelength solution
    second_mask = np.ones(len(mostpeaks), dtype='bool')
    
    #remove saturated lines
    if maxthresh is not None:
        second_mask[filtered_data[mostpeaks] > maxthresh] = False
        mostpeaks = mostpeaks[second_mask]
     
    #make mask which we need later to determine which peaks from linelist to use in wavelength solution    
    third_mask = np.ones(len(mostpeaks), dtype='bool')
    
    #only select good peaks higher than a certain threshold
    third_mask[filtered_data[mostpeaks] < thresh] = False
    goodpeaks = mostpeaks[third_mask]           #ie goodpeaks = allpeaks[first_mask][second_mask][third_mask]
    
    #for testing and debugging...
    if debug_level >= 1:
        print('Total number of peaks found: '+str(len(allpeaks)))
        print('Number of peaks found that are higher than '+str(int(thresh))+' counts: '+str(len(goodpeaks)))
        plt.figure()
        plt.plot(data)
        plt.plot(filtered_data)
        plt.scatter(goodpeaks, data[goodpeaks], marker='x', color='r', s=40)
        plt.plot((0,len(data)),(bgthresh,bgthresh),'r--')
        plt.plot((0,len(data)),(thresh,thresh),'g--')
        plt.xlabel('pixel')
        plt.ylabel('counts')
        #plt.vlines(thar_pos_guess, 0, np.max(data))
        plt.show()
    
    if timit:
        delta_t = time.time() - start_time
        print('Time elapsed: '+str(round(delta_t,5))+' seconds')
    
    if return_masks:
        return goodpeaks, mostpeaks, allpeaks, first_mask, second_mask, third_mask
    else:
        return goodpeaks, mostpeaks, allpeaks





def fit_emission_lines(data, fitwidth=4, thresh = 5000., bgthresh = 2000., maxthresh = None, slope=1e-4, laser=False,
                       varbeta=True, offset=False, minsigma=0.2, maxsigma=np.inf, sigma_0=1., minamp=0., maxamp=np.inf,
                       minbeta=1., maxbeta=4., beta_0=2., return_all_pars=False, return_qualflag=False, verbose=False, timit=False):
    """
    This routine identifies and fits emission lines in a 1-dim spectrum (ie generally speaking it finds and fits peaks in a 1dim array), using "scipy.optimize.curve_fit".
    Detection threshold, background threshold, and maximum threshold can be provided as keyword parameters. Different models for the peak-like function can be selected.

    INPUT:
    'data'   : one-dimensional data array

    KEYWORD PARAMETERS:
    'fitwidth'         : range around the identified peaks to be used for the peak-fitting 
    'thresh'           : minimum height of peak in the FILTERED data to be considered a good peak
    'bgthresh'         : minimum height of peak in the FILTERED data to be considered a real (but not good) peak
    'maxthresh'        : maximum height of peak in the FILTERED data to be included (ie you can exclude saturated peaks etc)
    'slope'            : value of the (tiny) slope that's added in order to break possible degeneracies between adjacent pixels
    'laser'            : boolean - is this a LFC spectrum? (if set to true, there is no check for blended lines)
    'varbeta'          : boolean - if set to TRUE, use a Gauss-like function for fitting, if set to FALSE use a plain Gaussian
    'offset'           : boolean - do you want to fit an offset to each peak as well?
    'minsigma'         : lower threshold for allowed sigma values
    'maxsigma'         : upper threshold for allowed sigma values
    'sigma_0'          : initial guess for sigma
    'minamp'           : lower threshold for allowed amplitude values
    'maxamp'           : upper threshold for allowed amplitude values
    'minbeta'          : lower threshold for allowed beta values
    'maxbeta'          : upper threshold for allowed beta values
    'beta_0'           : initial guess for beta
    'return_all_pars'  : boolean - do you want to return all fit parameters?
    'return_qualflag'  : boolean - do you want to return a quality flag for each line fit?
    'verbose'          : boolean - for debugging...
    'timit'            : boolean - do you want to clock execution run-time?    
    
    OUTPUT:
    'line_pos_fitted'    : fitted line positions
    'line_sigma_fitted'  : fitted line sigmas
    'line_amp_fitted'    : fitted line amplitudes
    'line_beta_fitted'   : fitted line betas (exponent in Gauss-like functions, which is equal to 2 for pure Gaussians)
    'qualflag'           : quality flag (1=good, 0=bad) for each line (only if 'return_qualflag' is set to TRUE)
    
    TODO:
    (1) make a relative threshold criterion
    """
    
    print('ATTENTION: You should consider using the more up-to-date and more robust function "fit_emission_lines_lmfit" instead!')
#     choice = None
#     while choice is None:
#         choice = raw_input("Do you want to continue? [y/n]: ")
#         if choice.lower() not in ('y','n'):
#             print('Invalid input! Please try again...')
#             choice = None
#    
#     #stop here if you'd rather use "fit_emission_lines_lmfit"
#     if choice.lower() == 'n':
#         return
    
    #OK, continue, you have been warned...
    if timit:
        start_time = time.time()
    
    xx = np.arange(len(data))
    
    #find rough peak locations
    goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data, thresh=thresh, bgthresh=bgthresh, maxthresh=maxthresh, slope=slope)
    
    if verbose:
        print('Fitting '+str(len(goodpeaks))+' emission lines...')
    
    line_pos_fitted = []
    if return_all_pars:
        line_amp_fitted = []
        line_sigma_fitted = []
        if varbeta:
            line_beta_fitted = []
    if return_qualflag:
        qualflag = []
        
    for xguess in goodpeaks:
        if verbose:
            print('xguess = ',xguess)
        ################################################################################################################################################################################
        #METHOD 1 (using curve_fit; slightly faster than method 2, but IDK how to make sure the fit converged (as with .ier below))

        if not laser:
            #check if there are any other peaks in the vicinity of the peak in question (exclude the peak itself)
            checkrange = np.r_[xx[np.max([0,xguess - 2*fitwidth]) : xguess], xx[xguess+1 : np.min([xguess + 2*fitwidth+1, len(data)-1])]]
            peaks = np.r_[xguess]
            #while len((set(checkrange) & set(allpeaks))) > 0:    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
            while len((set(checkrange) & set(mostpeaks))) > 0:
                #where are the other peaks?
                #other_peaks = np.intersect1d(checkrange, allpeaks)    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
                other_peaks = np.intersect1d(checkrange, mostpeaks)
                peaks = np.sort(np.r_[peaks, other_peaks])
                #define new checkrange
                checkrange = xx[np.max([0,peaks[0] - 2*fitwidth]) : np.min([peaks[-1] + 2*fitwidth + 1,len(data)-1])]
                dum = np.in1d(checkrange, peaks)
                checkrange = checkrange[~dum]
        else:
            peaks = np.r_[xguess]

        npeaks = len(peaks)
        xrange = xx[np.max([0,peaks[0] - fitwidth]) : np.min([peaks[-1] + fitwidth + 1,len(data)-1])]      # this should satisfy: len(xrange) == len(checkrange) - 2*fitwidth + len(peaks)

        if npeaks == 1:
            if offset:
                if varbeta:
                    guess = np.array([xguess, sigma_0, data[xguess], beta_0, np.min(data[xrange])])
                    popt, pcov = op.curve_fit(fibmodel_with_amp_and_offset, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp,minbeta,-np.inf],[xguess+2,maxsigma,maxamp,maxbeta,np.inf]))
                else:
                    guess = np.array([xguess, sigma_0, data[xguess], np.min(data[xrange])])
                    popt, pcov = op.curve_fit(gaussian_with_offset, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp,-np.inf],[xguess+2,maxsigma,maxamp,np.inf]))
            else:
                if varbeta:
                    guess = np.array([xguess, sigma_0, data[xguess], beta_0])
                    popt, pcov = op.curve_fit(fibmodel_with_amp, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp,minbeta],[xguess+2,maxsigma,maxamp,maxbeta]))
                else:
                    guess = np.array([xguess, sigma_0, data[xguess]])
                    popt, pcov = op.curve_fit(CMB_pure_gaussian, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp],[xguess+2,maxsigma,maxamp]))
            fitted_pos = popt[0]
            if return_all_pars:
                fitted_sigma = popt[1]
                fitted_amp = popt[2]
                if varbeta:
                    fitted_beta = popt[3]
        else:
            guess = []
            lower_bounds = []
            upper_bounds = []
            for i in range(npeaks):
                if varbeta:
                    guess.append(np.array([peaks[i], sigma_0, data[peaks[i]], beta_0]))
                    lower_bounds.append([peaks[i]-2,minsigma,minamp,minbeta])
                    upper_bounds.append([peaks[i]+2,maxsigma,maxamp,maxbeta])
                else:
                    guess.append(np.array([peaks[i], sigma_0, data[peaks[i]]]))
                    lower_bounds.append([peaks[i]-2,minsigma,minamp])
                    upper_bounds.append([peaks[i]+2,maxsigma,maxamp])
            guess = np.array(guess).flatten()
            lower_bounds = np.array(lower_bounds).flatten()
            upper_bounds = np.array(upper_bounds).flatten()
            if varbeta:
                popt, pcov = op.curve_fit(multi_fibmodel_with_amp, xrange, data[xrange], p0=guess, bounds=(lower_bounds,upper_bounds))
            else:
                popt, pcov = op.curve_fit(CMB_multi_gaussian, xrange, data[xrange], p0=guess, bounds=(lower_bounds,upper_bounds))

            #now figure out which peak is the one we wanted originally
            q = np.argwhere(peaks==xguess)[0]
            if varbeta:
                fitted_pos = popt[q*4]
                if return_all_pars:
                    fitted_sigma = popt[q*4+1]
                    fitted_amp = popt[q*4+2]
                    fitted_beta = popt[q*4+3]
            else:
                fitted_pos = popt[q*3]
                if return_all_pars:
                    fitted_sigma = popt[q*3+1]
                    fitted_amp = popt[q*3+2]


        #make sure we actually found a good peak
        if abs(fitted_pos - xguess) >= 2.:
            line_pos_fitted.append(xguess)
            if return_qualflag:
                qualflag.append(0)
            if return_all_pars:
                line_sigma_fitted.append(fitted_sigma)
                line_amp_fitted.append(fitted_amp)
                if varbeta:
                    line_beta_fitted.append(fitted_beta)
        else:
            line_pos_fitted.append(fitted_pos)
            if return_qualflag:
                qualflag.append(1)
            if return_all_pars:
                line_sigma_fitted.append(fitted_sigma)
                line_amp_fitted.append(fitted_amp)
                if varbeta:
                    line_beta_fitted.append(fitted_beta)
        ################################################################################################################################################################################
        
#         ################################################################################################################################################################################
#         #METHOD 2 (using lmfit) (NOTE THAT THE TWO METHODS HAVE different amplitudes for the Gaussian b/c of different normalization, but we are only interested in the position)
#         #xguess = int(xguess)
#         gm = GaussianModel()
#         gm_pars = gm.guess(data[xguess - fitwidth:xguess + fitwidth], xx[xguess - fitwidth:xguess + fitwidth])
#         gm_fit_result = gm.fit(data[xguess - fitwidth:xguess + fitwidth], gm_pars, x=xx[xguess - fitwidth:xguess + fitwidth])
#         
#         #make sure we actually found the correct peak
#         if gm_fit_result.ier not in (1,2,3,4):     #if this is any other value it means the fit did not converge
#         #if gm_fit_result.ier > 4:   
#             # gm_fit_result.plot()
#             # plt.show()
#             thar_pos_fitted.append(xguess)
#         elif abs(gm_fit_result.best_values['center'] - xguess) > 2.:
#             thar_pos_fitted.append(xguess)
#         else:
#             thar_pos_fitted.append(gm_fit_result.best_values['center'])
#         ################################################################################################################################################################################
    
    if verbose:    
        plt.figure()
        plt.plot(xx,data)
        #plt.vlines(thar_pos_guess, 0, np.max(data))
        plt.vlines(line_pos_fitted, 0, np.max(data) * 1.2, color='g', linestyles='dotted')
        plt.show()
    
    if timit:
        print('Time taken for fitting emission lines: '+str(time.time() - start_time)+' seconds...')
    
    if return_all_pars:
        if varbeta:
            if return_qualflag:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(line_beta_fitted), np.array(qualflag)
            else:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(line_beta_fitted)
        else:
            if return_qualflag:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(qualflag)
            else:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted)
    else:
        if return_qualflag:
            return np.array(line_pos_fitted), np.array(qualflag)
        else:
            return np.array(line_pos_fitted)





def fit_emission_lines_lmfit(data, fitwidth=None, thresh=5000., bgthresh=2000., maxthresh=None, laser=False, model='gauss', no_weights=False,
                             offset=False, return_all_pars=False, return_qualflag=False, return_stats=False, return_full=False, timit=False, debug_level=0):
    """
    This routine identifies and fits emission lines in a 1-dim spectrum (ie generally speaking it finds and fits peaks in a 1dim array), using the LMFIT package.
    Detection threshold, background threshold, and maximum threshold can be provided as keyword parameters. Different models for the peak-like function can be selected.

    INPUT:
    'data'   : one-dimensional data array

    KEYWORD PARAMETERS:
    'fitwidth'         : range around the identified peaks to be used for the peak-fitting 
                         (if None, it will automatically be determined by the minimum spacing between peaks - this option should only be used for LFC spectra, won't work properly for ThAr or ThXe)
    'thresh'           : minimum height of peak in the FILTERED data to be considered a good peak
    'bgthresh'         : minimum height of peak in the FILTERED data to be considered a real (but not good) peak
    'maxthresh'        : maximum height of peak in the FILTERED data to be included (ie you can exclude saturated peaks etc)
    'laser'            : boolean - is this a LFC spectrum? (if set to true, there is no check for blended lines)
    'model'            : peak-like function model to be used for the fitting; valid options are (check code below for exact names):
                         [gaussian, gauss-like, double gaussian, lorentzian, voigt, pseudo-voigt, offset pseudo-voigt, offset "pseudo-gausslike", 
                         moffat, pearson7, students, breit-wigner, lognormal, damped oscillator, damped harmonic oscillator, exp gauss, skew gauss, donaich] 
    'no_weights'       : boolean - set to TRUE if you DO NOT WANT to use weights (not recommended)                        
    'offset'           : boolean - do you want to estimate and subtract an offset before fitting?                        
    'return_all_pars'  : boolean - do you want to return all fit parameters?
    'return_qualflag'  : boolean - do you want to return a quality flag for each line fit?
    'return_stats'     : boolean - do you want to return goodness of fit statistics (ie AIC, BIC, CHISQ and REDCHISQ)?
    'return_full'      : boolean - do you want to return the full "fit_result" class for each line? (WARNING: this ONLY returns the line positions and the full 'fit_result' classes,
                         ie it overwrites the 'return_all_pars' / 'return_qualflag' / 'return_stats' keywords)
    'debug_level'      : boolean - for debugging...
    'timit'            : boolean - do you want to clock execution run-time?    
    
    OUTPUT:
    'full_results'     : the full results, ie the instance of the LMFIT 'ModelResult' class (if 'return_full' is set to TRUE)
    -------------------
    'line_pos_fitted'  : fitted line positions
    'allpars'          : (only if 'return_all_pars' is set to TRUE)
    'qualflag'         : quality flag (1=good, 0=bad) for each line (only if 'return_qualflag' is set to TRUE)
    'stats'            : (only if 'return_stats' is set to TRUE)
    
    TODO:
    (1) make a relative threshold criterion
    """

    if timit:
        start_time = time.time()

    xx = np.arange(len(data))

    if model.lower() == 'gausslike':
        mod = Model(fibmodel_with_amp)
    if model.lower() in ('double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
        mod1 = GaussianModel(prefix='first_')
        mod2 = GaussianModel(prefix='second_')
        mod = mod1 + mod2
    if model.lower() in ('gauss', 'gaussian'):
        mod = GaussianModel()
    if model.lower() in ('lorentz', 'lorentzian'):
        mod = LorentzianModel()
    if model.lower() == 'voigt':
        mod = VoigtModel()
    if model.lower() in ('pseudo', 'pseudovoigt'):
        mod = PseudoVoigtModel()
    if model.lower() == 'offset_pseudo':
        gmod = GaussianModel(prefix='G_')
        lmod = LorentzianModel(prefix='L_')
        mod = gmod + lmod
    if model.lower() == 'offset_pseudo_gausslike':
        mod = Model(offset_pseudo_gausslike)
    if model.lower() == 'moffat':
        mod = MoffatModel()
    if model.lower() in ('pearson', 'pearson7'):
        mod = Pearson7Model(nan_policy='omit')
    if model.lower() in ('student', 'students', 'studentst'):
        mod = StudentsTModel()
    if model.lower() == 'breitwigner':
        mod = BreitWignerModel()
    if model.lower() == 'lognormal':
        mod = LognormalModel()
    if model.lower() == 'dampedosc':
        mod = DampedOscillatorModel()
    if model.lower() == 'dampedharmosc':
        mod = DampedHarmonicOscillatorModel()
    if model.lower() == 'expgauss':
        mod = ExponentialGaussianModel(nan_policy='omit')
    if model.lower() == 'skewgauss':
        mod = SkewedGaussianModel()
    if model.lower() == 'donaich':
        mod = DonaichModel()


    # find rough peak locations
    goodpeaks, mostpeaks, allpeaks = find_suitable_peaks(data, thresh=thresh, bgthresh=bgthresh, maxthresh=maxthresh, remove_bg=True)

    #determine fit-window if not explicitly given
    if fitwidth is None:
        fitwidth = np.min(np.diff(goodpeaks))//2
    
    print('Fitting ' + str(len(goodpeaks)) + ' emission lines...')

    #prepare output variables
    line_pos_fitted = []
    if return_full:
        full_results = []
    else:
        if return_qualflag:
            qualflag = []
        if return_all_pars:
            allpars = []
        if return_stats:
            stats = []
    
    
    #loop over all lines
    for xguess in goodpeaks:
        
        if debug_level >= 2:
            print('xguess = ',xguess)

        if not laser:
            # check if there are any other peaks in the vicinity of the peak in question (exclude the peak itself)
            checkrange = np.r_[xx[np.max([0, xguess - 2 * fitwidth]): xguess], xx[xguess + 1: np.min([xguess + 2 * fitwidth + 1, len(data) - 1])]]
            peaks = np.r_[xguess]
            # while len((set(checkrange) & set(allpeaks))) > 0:    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
            while len((set(checkrange) & set(mostpeaks))) > 0:
                # where are the other peaks?
                # other_peaks = np.intersect1d(checkrange, allpeaks)    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
                other_peaks = np.intersect1d(checkrange, mostpeaks)
                peaks = np.sort(np.r_[peaks, other_peaks])
                # define new checkrange
                checkrange = xx[np.max([0, peaks[0] - 2 * fitwidth]): np.min([peaks[-1] + 2 * fitwidth + 1, len(data) - 1])]
                dum = np.in1d(checkrange, peaks)
                checkrange = checkrange[~dum]
        else:
            peaks = np.r_[xguess]

        npeaks = len(peaks)
        xrange = xx[np.max([0, peaks[0] - fitwidth]): np.min([peaks[-1] + fitwidth + 1, len(data) - 1])]   # this should satisfy: len(xrange) == len(checkrange) - 2*fitwidth + len(peaks)


        ################################################################################################################################################################################
        #using LMFIT package:

        if npeaks == 1:

            #estimate and subtract an offset if desired
            if offset:
                off = np.mean([data[xrange][0],data[xrange][1],data[xrange][-1],data[xrange][-2]])
            else:
                off = 0.
            fitdata = data[xrange] - off

            # create instance of Parameters-class needed for fitting with LMFIT
            if model.lower() in ('gausslike', 'offset_pseudo', 'offset_pseudo_gausslike', 'double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                parms = lmfit.Parameters()
            else:
                parms = mod.guess(fitdata, x=xrange)

            # fill/tweak initial Parameters() instance
            if model.lower() in ('gauss', 'gaussian'):
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
            if model.lower() in ('double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                parms = mod1.guess(fitdata, x=xrange)
                parms.update(mod2.guess(fitdata, x=xrange))
                parms['second_amplitude'].set(0., vary=True)
                parms['first_center'].set(min=xguess - 3, max=xguess + 3)
                parms['second_center'].set(min=xguess - 3, max=xguess + 3)
            if model.lower() == 'gausslike':
                gmod = GaussianModel()
                gparms = gmod.guess(fitdata,xrange)
                parms.add('mu', xguess, min=xguess - 3, max=xguess + 3)
                parms.add('sigma', gparms['sigma'].value, min=0.2)
                parms.add('amp', data[xguess], min=0.)
                parms.add('beta', 2., min=1., max=4.)
            if model.lower() == 'moffat':
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
                parms['beta'].set(min=0.)
            if model.lower() in ('pseudo', 'pseudovoigt'):
                # parms['fraction'].set(0.5,min=0.,max=1.)
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
            if model.lower() == 'lognormal':
                #parms['sigma'].set(value=1e-4, vary=True, expr='')
                parms['center'].set(value=np.log(xguess), vary=True, expr='', min=0., max=8.5)
                parms['amplitude'].set(data[xguess], vary=True, min=0., expr='')
            # if model.lower() == 'dampedosc':
            #     parms['sigma'].set(1e-4, vary=True, expr='')
            #     parms['amplitude'].set(1e-4, vary=True, min=0., expr='')
            if model.lower() == 'offset_pseudo':
                parms = gmod.guess(fitdata, x=xrange)
                parms.update(lmod.guess(fitdata, x=xrange))
                parms['G_amplitude'].set(parms['G_amplitude'] / 2., min=0., vary=True)
                parms['L_amplitude'].set(parms['L_amplitude'] / 2., min=0., vary=True)
                parms['G_center'].set(min=xguess - 3, max=xguess + 3)
                parms['L_center'].set(min=xguess - 3, max=xguess + 3)
            if model.lower() == 'skewgauss':
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
            if model.lower() == 'offset_pseudo_gausslike':
                gmod = GaussianModel(prefix='G_')
                lmod = LorentzianModel(prefix='L_')
                parms = gmod.guess(fitdata, x=xrange)
                parms.update(lmod.guess(fitdata, x=xrange))
                parms['G_amplitude'].set(parms['G_amplitude'] / 2., min=0., vary=True)
                parms['L_amplitude'].set(parms['L_amplitude'] / 2., min=0., vary=True)
                parms['G_center'].set(min=xguess - 3, max=xguess + 3)
                parms['L_center'].set(min=xguess - 3, max=xguess + 3)
                parms.add('beta', 2., min=1., max=4.)

            
            #perform the actual fit (get weights from array before offset subtraction)
            if not no_weights:
                fit_result = mod.fit(fitdata, parms, x=xrange, weights=np.sqrt(data[xrange]))
            else:
                fit_result = mod.fit(fitdata, parms, x=xrange)

            #goodness of fit statistics
            fitstats = {}
            fitstats['aic'] = fit_result.aic
            fitstats['bic'] = fit_result.bic
            fitstats['chi2'] = fit_result.chisqr
            fitstats['chi2red'] = fit_result.redchi 

            #fill bestpos variable depending on model used
            if model.lower() not in ('offset_pseudo', 'offset_pseudo_gausslike', 'double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                try:
                    bestpos = fit_result.best_values['center']
                except:
                    bestpos = fit_result.best_values['mu']
            elif model.lower() in ('offset_pseudo', 'offset_pseudo_gausslike'):
                bestpos = np.array((fit_result.best_values['G_center'], fit_result.best_values['L_center']))
            elif model.lower() in ('double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                bestpos = np.array((fit_result.best_values['first_center'], fit_result.best_values['second_center']))
            else:
                print('ERROR!!! No valid model defined!!!')
                return

            #make sure we actually found a good / the correct peak
            if fit_result.ier not in (1,2,3,4):     #if this is any other value it means the fit did not converge
            #if fit_result.ier > 4:
                fit_result.plot()
                plt.show()
                print('WARNING: Bad fit encountered!')
                choice = raw_input('Do you wish to continue? [y/n]')
                #choice = 'y'
                if choice.lower() == 'y':
                    line_pos_fitted.append(xguess)
                    if return_full:
                        full_results.append(fit_result)
                    else:
                        if return_qualflag:
                            qualflag.append(0)
                        if return_all_pars:
                            allpars.append(fit_result.params)
                        if return_stats:
                            stats.append(fitstats)    
                else:
                    return
            elif np.all(abs(bestpos - xguess) >= 2.):
                fit_result.plot()
                plt.show()
                print('WARNING: Bad fit encountered!')
                choice = raw_input('Do you wish to continue? [y/n]')
                #choice = y
                if choice.lower() == 'y':
                    line_pos_fitted.append(xguess)
                    if return_full:
                        full_results.append(fit_result)
                    else:
                        if return_qualflag:
                            qualflag.append(0)
                        if return_all_pars:
                            allpars.append(fit_result.params)
                        if return_stats:
                            stats.append(fitstats)
                else:
                    return
            else:
                line_pos_fitted.append(bestpos)
                if return_full:
                    full_results.append(fit_result)
                else:
                    if return_qualflag:
                        qualflag.append(0)
                    if return_all_pars:
                        allpars.append(fit_result.params)
                    if return_stats:
                        stats.append(fitstats)

# #             # # #this way you can quickly plot the indidvidual fits for debugging
#             plot_osf = 10
#             plot_os_grid = np.linspace(xrange[0], xrange[-1], plot_osf * (len(xrange) - 1) + 1)
#             guessmodel = mod.eval(fit_result.init_params, x=plot_os_grid)
#             bestmodel = mod.eval(fit_result.params, x=plot_os_grid)
#             plt.figure()
#             plt.title('model = ' + model.title())
#             plt.xlabel('pixel number (dispersion direction)')
#             plt.plot(xrange, fitdata, 'bo')
#             plt.plot(plot_os_grid, guessmodel, 'k--', label='initial guess')
#             plt.plot(plot_os_grid, bestmodel, 'r-', label='best-fit model')
#             plt.legend()


        else:
            print('ERROR: multi-peak fitting with LMFIT has not been implemented yet')
            return
        ################################################################################################################################################################################

    if debug_level >= 1:
        plt.figure()
        plt.plot(xx, data)
        # plt.vlines(thar_pos_guess, 0, np.max(data))
        plt.vlines(line_pos_fitted, 0, np.max(data) * 1.2, color='g', linestyles='dotted')
        plt.show()

    if timit:
        print('Time taken for fitting emission lines: ' + str(time.time() - start_time) + ' seconds...')

    #return results
    if return_full:
        return full_results 
    else:
        if return_stats:
            if return_all_pars:
                if return_qualflag:
                    return np.array(line_pos_fitted), allpars, np.array(qualflag), stats
                else:
                    return np.array(line_pos_fitted), allpars, stats
            else:
                if return_qualflag:
                    return np.array(line_pos_fitted), np.array(qualflag), stats
                else:
                    return np.array(line_pos_fitted), stats
        else:
            if return_all_pars:
                if return_qualflag:
                    return np.array(line_pos_fitted), allpars, np.array(qualflag)
                else:
                    return np.array(line_pos_fitted), allpars
            else:
                if return_qualflag:
                    return np.array(line_pos_fitted), np.array(qualflag)
                else:
                    return np.array(line_pos_fitted)





def get_wavelength_solution_from_thorium(thflux, poly_deg=5, polytype='chebyshev', savetable=False, return_full=True, timit=False):
    """ 
    INPUT:
    'thflux'           : extracted 1-dim thorium / laser-only image (initial tests used this one : '/Users/christoph/OneDrive - UNSW/veloce_spectra/reduced/tests/...')
    'poly_deg'         : the order of the polynomials to use in the fit (for both dimensions)
    'polytype'         : either 'polynomial', 'legendre', or 'chebyshev' (default) are accepted 
    'return_full'      : boolean - if TRUE, then the wavelength solution for each pixel for each order is returned; otherwise just the set of coefficients that describe it
    'saveplots'        : boolean - do you want to create plots for each order? 
    'savetable'        : boolean - if TRUE, then an output file is created, containing a summary of all lines used in the fit, and details about the fit
    'timit'            : time it...
    
    OUTPUT:
    'p'      : functional form of the coefficients that describe the wavelength solution
    'p_wl'   : wavelength solution for each pixel for each order (n_ord x n_pix numpy-array) 
    
    TODO:
    include order 40 (m=65) as well (just 2 lines!?!?!?)
    figure out how to properly use weights here (ie what should we use as weights?)
    """
    
    if timit:
        start_time = time.time()
    
#     #read in pre-defined thresholds (needed for line identification)
#     thresholds = np.load('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/thresholds.npy').item()
    
    #prepare arrays for fitting
    x = []
    order = []
    m_order = []
    wl = []

    for ord in sorted(thflux.keys()):    #make sure there are enough lines in every order
    #for ord in sorted(thflux.keys())[:-1]:    #don't have enough emission lines in order 40
        ordnum = ord[-2:]
        #m = 105 - int(ordnum)   #original orientation of the spectra
        m = 64 + int(ordnum)    #'correct' orientation of the spectra
        print('OK, fitting '+ord+'   (m = '+str(m)+')')
        
        data = thflux[ord]        
        xx = np.arange(len(data))
        
        #adjust thresholds here
        fitted_line_pos = fit_emission_lines(data,return_all_pars=False,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
        #fitted_line_pos = fit_emission_lines_lmfit(data, fitwidth=4, model='gausslike', return_all_pars=False,thresh=thresholds['thresh'][ord],
        #                                          bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord]) 
        goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])    
        
        #########################################################################################################################################
        #THIS IS THE STEP THAT NEEDS TO BE AUTOMATED
        line_number, refwlord = readcol('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/ThAr_linelist_order_'+ordnum+'.dat',fsep=';',twod=False)        
        #this tells us which peaks we have known wavelengths for
        mask_order = np.load('/Users/christoph/OneDrive - UNSW/linelists/posmasks/mask_order'+ordnum+'.npy')
        xord = fitted_line_pos[mask_order]
        
#         #ie we want to do sth like this:
#         guess_wls = get_wavelengths_for_peaks(fitted_line_pos)   # ie get a rough guess for thew wavelengths of the lines we found
#         refwlord = identify_lines(guess_wls, linelist)              # ie find the corresponding "true" wavelengths
        #########################################################################################################################################
        
        #fill arrays for fitting
        x.append(xord)
        x = np.array(x)
        order.append(np.repeat(int(ordnum), len(xord)))
        order = np.array(order)
        #m_order.append(np.repeat(105 - int(ordnum), len(xord)))
        m_order.append(np.repeat(64 + int(ordnum), len(xord)))
        m_order = np.array(m_order)
        wl.append(refwlord)
        wl = np.array(wl)
    
    
    #now re-normalize arrays to [-1,+1]
    x_norm = (x / ((len(data)-1)/2.)) - 1.
    order_norm = ((order-1) / ((len(thflux)-1)/2.)) - 1.       
           
    #call the fitting routine
    p = fit_poly_surface_2D(x_norm, order_norm, wl, weights=None, polytype=polytype, poly_deg=poly_deg, debug_level=0)    
    
#     if return_all_pars:
#         return dispsol,fitshapes
#     else:
#         return dispsol

    if savetable:
        now = datetime.datetime.now()
        model_wl = p(x_norm, order_norm)
        resid = wl - model_wl
        outfn = '/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/lines_used_in_fit_as_of_'+str(now)[:10]+'.dat'
        outfn = open(outfn, 'w')
        outfn.write('line number   order_number   physical_order_number     pixel      reference_wl[A]   model_wl[A]    residuals[A]\n')
        outfn.write('=====================================================================================================================\n')
        for i in range(len(x)):
                outfn.write("   %3d             %2d                 %3d           %11.6f     %11.6f     %11.6f     %9.6f\n" %(i+1, order[i], m_order[i], x[i], wl[i], model_wl[i], resid[i]))
        outfn.close()
              

    if return_full:
        #xx = np.arange(len(data))            # already done above
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,len(thflux))
        oon = ((oo-1) / ((len(thflux)-1)/2.)) - 1.   
        X,O = np.meshgrid(xxn,oon)
        p_wl = p(X,O)

    
    if timit:
        print('Time elapsed: ',time.time() - start_time,' seconds')
        

    if return_full:
        return p,p_wl
    else:
        return p





def get_wl(p,xx,yy):
    """
    Get the wavelength for "full" 2-dim grid of x-pixel number and order number, or x-pixel number and y-pixel number, 
    depending on how 'p' was created, ie either lambda(x,y) or lambda (x,ord).
    
    Example: for the full Veloce Rosso chip (nx,ny) = (4112,4096), call it like so:
             wl = get_wl(p, np.arange(4112), np.arange(4096))
             OR for (nx,n_ord) = (4112,40)
             wl = get_wl(p, np.arange(4112), np.arange(1,41))   #note that orders range from 1...40 (not from 0...39)
    
    INPUT:
    'p'     : the coefficients describing the 2-dim polynomial surface fit from "fit_poly_surface_2D"   
    'xx'    : x-coordinate(s) at which you want to evaluate the wavelength solution
    'yy'    : y-coordinate(s) at which you want to evaluate the wavelength solution
    
    OUTPUT:
    p_wl'  : wavelength solution for each pair of coordinates
    """
    
    #re-normalize arrays to [-1,+1]
    print('WRONG!!!!!')
    xxn = np.linspace(-1, 1, len(xx))
    yyn = np.linspace(-1, 1, len(yy))
    #make 2D grids of coordinates
    X,Y = np.meshgrid(xxn,yyn)
    #actually calculate the wavelengths from the polynomial coefficients
    p_wl = p(X,Y)
    
    return p_wl





def get_wavelength_solution_labtests(thflux, thflux2, poly_deg=5, polytype='chebyshev', savetable=False, return_full=True, saveplots=False, timit=False):
    """ 
    INPUT:
    'thflux'           : extracted 1-dim thorium / laser-only image (initial tests used this one : '/Users/christoph/OneDrive - UNSW/veloce_spectra/reduced/tests/...')
    'poly_deg'         : the order of the polynomials to use in the fit (for both dimensions)
    'polytype'         : either 'polynomial', 'legendre', or 'chebyshev' (default) are accepted 
    'return_full'      : boolean - if TRUE, then the wavelength solution for each pixel for each order is returned; otherwise just the set of coefficients that describe it
    'saveplots'        : boolean - do you want to create plots for each order? 
    'savetable'        : boolean - if TRUE, then an output file is created, containing a summary of all lines used in the fit, and details about the fit
    'timit'            : time it...
    
    OUTPUT:
    EITHER
    'p'      : functional form of the coefficients that describe the wavelength solution
    OR 
    'p_wl'   : wavelength solution for each pixel for each order (n_ord x n_pix numpy-array) 
    (selection between outputs is controlled by the 'return_full' keyword)
    
    TODO:
    clean-up the 2-version thing about which extracted spectrum to use (thflux / thflux2)
    include order 40 (m=65) as well (just 2 lines!?!?!?)
    figure out how to properly use weights here (ie what should we use as weights?)
    """
    
    if timit:
        start_time = time.time()
    
    #read in pre-defined thresholds (needed for line identification)
    thresholds = np.load('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/thresholds.npy').item()
    
    #wavelength solution from Zemax as a reference
    if saveplots:
        zemax_dispsol = np.load('/Users/christoph/OneDrive - UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()
    
    #prepare arrays for fitting
    x = []
    order = []
    m_order = []
    wl = []

    for ord in sorted(thflux.keys())[:-1]:    #don't have enough emission lines in order 40
        ordnum = ord[-2:]
        m = 105 - int(ordnum)   #original orientation of the spectra
        #m = 64 + int(ordnum)    #'correct' orientation of the spectra
        print('OK, fitting '+ord+'   (m = '+str(m)+')')
        coll = thresholds['collapsed'][ord]
        if coll:
            data = thflux2[ord]
        else:
            data = thflux[ord]
        
        xx = np.arange(len(data))
        
#         if return_all_pars:
#             fitted_line_pos,fitted_line_sigma,fitted_line_amp = fit_emission_lines(data,return_all_pars=return_all_pars,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
#         else:
#             fitted_line_pos = fit_emission_lines(data,return_all_pars=return_all_pars,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
        fitted_line_pos = fit_emission_lines(data,return_all_pars=False,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
        #fitted_line_pos = fit_emission_lines_lmfit(data, fitwidth=4, model='gausslike', return_all_pars=False,thresh=thresholds['thresh'][ord],
        #                                          bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord]) 
        goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])    
        
        #########################################################################################################################################
        #THIS IS THE STEP THAT NEEDS TO BE AUTOMATED
        line_number, refwlord = readcol('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/ThAr_linelist_order_'+ordnum+'.dat',fsep=';',twod=False)
        #lam = refwlord.copy()  
        #wl_ref[ord] = lam
        
        #this tells us which peaks we have known wavelengths for
        mask_order = np.load('/Users/christoph/OneDrive - UNSW/linelists/posmasks/mask_order'+ordnum+'.npy')
        xord = fitted_line_pos[mask_order]
        #########################################################################################################################################
        
        #stupid python!?!?!?
        if ordnum == '30':
            xord = np.array([xord[0][0],xord[1][0],xord[2],xord[3]])
        
        if saveplots:
            zemax_wl = 10. * zemax_dispsol['order'+str(m)]['model'](xx[::-1])
        
        #fill arrays for fitting
        x.append(xord)
        x = np.array(x)
        order.append(np.repeat(int(ordnum), len(xord)))
        order = np.array(order)
        m_order.append(np.repeat(105 - int(ordnum), len(xord)))
        m_order = np.array(m_order)
        wl.append(refwlord)
        wl = np.array(wl)
        
        
#         #perform the fit
#         fitdegpol = degpol
#         while fitdegpol > len(x)/2:
#             fitdegpol -= 1
#         if fitdegpol < 2:
#             fitdegpol = 2
#         thar_fit = np.poly1d(np.polyfit(x, lam, fitdegpol))
#         dispsol[ord] = thar_fit
#         if return_all_pars:
#             fitshapes[ord] = {}
#             fitshapes[ord]['x'] = x
#             fitshapes[ord]['y'] = P_id[ord](x)
#             fitshapes[ord]['FWHM'] = 2.*np.sqrt(2.*np.log(2.)) * fitted_line_sigma[mask_order]
#         
#         #calculate RMS of residuals in terms of RV
#         resid = thar_fit(x) - lam
#         rv_resid = 3e8 * resid / lam
#         rms = np.std(rv_resid)
#         
#         if saveplots:
#             #first figure: lambda vs x with fit and zemax dispsol
#             fig1 = plt.figure()
#             plt.plot(x,lam,'bo')
#             plt.plot(xx,thar_fit(xx),'g',label='fitted')
#             plt.plot(xx,zemax_wl,'r--',label='Zemax')
#             plt.title('Order '+str(m))
#             plt.xlabel('pixel number')
#             plt.ylabel(ur'wavelength [\u00c5]')
#             plt.text(3000,thar_fit(500),'n_lines = '+str(len(x)))
#             plt.text(3000,thar_fit(350),'deg_pol = '+str(fitdegpol))
#             plt.text(3000,thar_fit(100),'RMS = '+str(round(rms, 1))+' m/s')
#             plt.legend()
#             plt.savefig('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/fit_to_order_'+ordnum+'.pdf')
#             plt.close(fig1)
#             
#             #second figure: spectrum vs fitted dispsol
#             fig2 = plt.figure()
#             plt.plot(thar_fit(xx),data)
#             #plt.scatter(thar_fit(x), data[x.astype(int)], marker='o', color='r', s=40)
#             plt.scatter(thar_fit(goodpeaks), data[goodpeaks], marker='o', color='r', s=30)
#             plt.title('Order '+str(m))
#             plt.xlabel(ur'wavelength [\u00c5]')
#             plt.ylabel('counts')
#             plt.savefig('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/ThAr_order_'+ordnum+'.pdf')
#             plt.close(fig2)
    
    
    #re-normalize arrays to [-1,+1]
    x_norm = (x / ((len(data)-1)/2.)) - 1.
    order_norm = ((order-1) / (38./2.)) - 1.       #TEMP, TODO, FUGANDA, PLEASE FIX ME!!!!!
    #order_norm = ((m-1) / ((len(P_id)-1)/2.)) - 1.
           
    #call the fitting routine
    p = fit_poly_surface_2D(x_norm, order_norm, wl, weights=None, polytype=polytype, poly_deg=poly_deg, debug_level=0)    
    
#     if return_all_pars:
#         return dispsol,fitshapes
#     else:
#         return dispsol

    if savetable:
        now = datetime.datetime.now()
        model_wl = p(x_norm, order_norm)
        resid = wl - model_wl
        outfn = '/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/lines_used_in_fit_as_of_'+str(now)[:10]+'.dat'
        outfn = open(outfn, 'w')
        outfn.write('line number   order_number   physical_order_number     pixel      reference_wl[A]   model_wl[A]    residuals[A]\n')
        outfn.write('=====================================================================================================================\n')
        for i in range(len(x)):
                outfn.write("   %3d             %2d                 %3d           %11.6f     %11.6f     %11.6f     %9.6f\n" %(i+1, order[i], m_order[i], x[i], wl[i], model_wl[i], resid[i]))
        outfn.close()
              

    if return_full:
        #xx = np.arange(4112)            # already done above
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,len(thflux))
        oon = ((oo-1) / (38./2.)) - 1.        #TEMP, TODO, FUGANDA, PLEASE FIX ME!!!!!
        #oon = ((oo-1) / ((len(thflux)-1)/2.)) - 1.   
        X,O = np.meshgrid(xxn,oon)
        p_wl = p(X,O)

    
    if timit:
        print('Time elapsed: ',time.time() - start_time,' seconds')
        

    if return_full:
        return p,p_wl
    else:
        return p





def get_simu_dispsol(fibre=None, path='/Users/christoph/OneDrive - UNSW/dispsol/', npix=4096):
    """
    Get wavelength solution from Veloce Zemax file for a given fibre (1...28)
    WARNING: If no fibre is given, a mean wavelength solution across all orders is calculated!!!
    """
    #read dispersion solution from file
    if fibre is None:
        dispsol = np.load(path + 'mean_dispsol_by_orders_from_zemax.npy').item()
        orders = dispsol.keys()
    else:
        dbf = np.load(path + 'dispsol_by_fibres_from_zemax.npy').item()
        orders = dbf['fiber_1'].keys()
     
    #read extracted spectrum from files (obviously this needs to be improved)
    xx = np.arange(npix)
     
    #this is so as to match the order number with the physical order number (66 <= m <= 108)
    # order01 corresponds to m=66
    # order43 corresponds to m=108
    wl = {}
    for o in orders:
        m = o[5:]
        ordnum = str(int(m)-65).zfill(2)
        if fibre is None:
            wl['order_'+ordnum] = dispsol['order'+m]['model'](xx)
        else:
            wl['order_'+ordnum] = dbf['fiber_'+str(fibre)][o]['fitparms'](xx)
 
    return wl





def quick_bg_fix(raw_data, npix=4112):
    left_xx = np.arange(npix/2)
    right_xx = np.arange(npix/2, npix)
    left_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[:npix/2],3), size=100)
    right_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[npix/2:],3), size=100)
    data = raw_data.copy()
    data[left_xx] = raw_data[left_xx] - left_bg
    data[right_xx] = raw_data[right_xx] - right_bg
    return data





def get_arc_dispsol(thflux, satmask=None, polytype='chebyshev', lamptype=None, deg_spectral=5, deg_spatial=3,
                    return_full=True, savetables=True, debug_level=0, timit=False):

    if lamptype.lower() not in ['thar', 'thxe']:
        lamptype = raw_input('Please enter valid lamp type ["thar" / "thxe"]: ')

    # read the sacred Michael Murphy ThAr line list 
    wn, wlair, relint, species1, species2, reference = readcol('/Users/christoph/OneDrive - UNSW/linelists/thar_mm_new.txt', twod=False, skipline=1)
    wlvac = 1.e8 / wn
    
    if debug_level >= 1:
        # mm_grid, mm_spec_air = make_gaussmask_from_linelist(wlair, relint=10.**relint)
        mm_grid, mm_spec_unit_air = make_gaussmask_from_linelist(wlair, relint=None)
        # mm_grid, mm_spec_vac = make_gaussmask_from_linelist(wlvac, relint=10.**relint)
        # mm_grid, mm_spec_unit_vac = make_gaussmask_from_linelist(wlvac, relint=None)
    
    # read best existing wavelength solution
    # p = np.load('/Users/christoph/OneDrive - UNSW/dispsol/veloce_thar_dispsol_coeffs_20180921.npy').item()
    # thar_dispsol = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/veloce_thar_dispsol_20180916.fits')
    p = np.load('/Users/christoph/OneDrive - UNSW/dispsol/veloce_thar_dispsol_coeffs_air_as_of_20180923.npy').item()
    thar_dispsol = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/veloce_thar_dispsol_17sep30080_7x7_air.fits')
    ref_linenum, ref_ordnum, ref_m, ref_pix, ref_wl, _, _, _, _, _ = readcol('/Users/christoph/OneDrive - UNSW/linelists/thar_lines_used_in_7x7_fit_as_of_2018-10-19.dat', twod=False, skipline=2)
    
    # prepare output file
    if savetables:
        now = datetime.datetime.now()
        #model_wl = p(x_norm, order_norm)
        #resid = wl - model_wl
        outfn = '/Users/christoph/OneDrive - UNSW/linelists/'+lamptype+'_lines_found_as_of_'+str(now)[:10]+'.dat'
        outfile = open(outfn, 'w')
        outfile.write('line number   order_number   physical_order_number     pixel       air_ref_wl[A]   vac_ref_wl[A] \n')
        outfile.write('=================================================================================================\n')
        outfile.close()
    
    # prepare global output arrays
    x = []
    nlines = []
    order = []
    ref_wl_air = []
    ref_wl_vac = []
    
    # if not passed into the routine, define a mask that masks out saturated arc lines and some other crap
    if satmask is None:
        n_ord = thflux.shape[0]
        satmask = define_arc_mask(lamptype, n_ord)
    
    # search region size in Angstroms
    # search_region_size = 0.7
    search_region_size = 0.5
        
    #######################################################################################################################################
    #loop over orders
    for ord in range(thflux.shape[0]):
    
        # order numbers for output file
        ordnum = str(ord+1).zfill(2)
        mord = (ord+1) + 64
    
        # prepare output arrays for this order
        ord_x = []
        ord_ref_wl_air = []
        ord_ref_wl_vac = []
        
        # normalized co-ordinates
        ord_norm = (ord / ((40-1)/2.)) - 1.
        
        #define raw 1-dim data
        raw_data = thflux[ord,:]
        
        # clean up data
        data = quick_bg_fix(raw_data)
        
        # mask out saturated lines
        ordmask = satmask[ord,:]
        data[ordmask] = 0.
        
        # find peaks
        ####### TODO: automate this!!!
        thresh = 1500
        bgthresh = 1000
        fitted_line_pos = fit_emission_lines(data, return_all_pars=False, varbeta=False, timit=False, verbose=False, thresh=thresh, bgthresh=bgthresh)
        
        if debug_level >= 1:
            print(str(len(fitted_line_pos)) + ' lines found in order ' + ordnum)
            
        # select appropriate region in master line list
        testrange = np.logical_and(mm_grid > np.min(thar_dispsol[ord,:]) - 20, mm_grid < np.max(thar_dispsol[ord,:]) + 20)
        
        # PLOT
        if debug_level >= 1:
            xx = np.arange(len(data))
            plt.plot(mm_grid[testrange], mm_spec_unit_air[testrange])
            plt.plot(thar_dispsol[ord,:],data/np.max(data),'.-')
            plt.title('Order '+str(ord+1).zfill(2)) 
        
        # loop over all lines found - search for possible matches in the ThAr master line list
        for i,checkval in enumerate(fitted_line_pos):
            
            # normalized co-ordinates
            x_norm = (checkval / ((len(data)-1)/2.)) - 1.
            
            if debug_level >= 1:
                print('Trying to identify line in order ' + ordnum + ', at pixel ' + str(checkval) + '  -  rough wl estimate: ' + str(p(x_norm, ord_norm)) + ' A')
            
            if debug_level >= 3:
                # print(i,checkval)
                # print(i, checkval, thar_dispsol[order, np.round(checkval,0).astype(int)])
                # print(i, checkval, thar_dispsol[order, np.round(checkval,0).astype(int)], p(x_norm, ord_norm))
                print(i, checkval, p(x_norm, ord_norm))
            
            # select match(es)
            # q = np.logical_and(ord_pix > checkval + wshift - search_region_size, ord_pix < checkval + wshift + search_region_size)
            # q = np.logical_and(wlair > thar_dispsol[order, np.round(checkval,0).astype(int)] - search_region_size, wlair < thar_dispsol[order, np.round(checkval,0).astype(int)] + search_region_size)
            q = np.logical_and(wlair > p(x_norm, ord_norm) - search_region_size, wlair < p(x_norm, ord_norm) + search_region_size)
            
            found = np.sum(q)
            if found > 0:
                if debug_level >= 1:
                    print(str(found) + ' possible match(es) found!')
                if found == 1:
                    ord_x.append(checkval)
                    ord_ref_wl_air.append(wlair[q])
                    ord_ref_wl_vac.append(wlvac[q])
                else:
                    if debug_level >= 1:
                        print('Selecting closest match...')
                    qq = find_nearest(wlair, p(x_norm, ord_norm), return_index=True)
                    ord_x.append(checkval)
                    ord_ref_wl_air.append(wlair[qq])
                    ord_ref_wl_vac.append(wlvac[qq])
            else:
                if debug_level >= 1:
                    print('NO LINE FOUND in MASTERTABLE!!!')
        
        # convert to numpy arrays
        ord_x = np.array(ord_x)
        ord_ref_wl_air = np.array(ord_ref_wl_air)
        ord_ref_wl_vac = np.array(ord_ref_wl_vac)
        
        if debug_level >= 1:
            print(str(len(ord_x)) + ' lines could initially be matched to known lines from MM line list')
        
        
        # perform sanity-check 1D fit
        order_fit = np.poly1d(np.polyfit(ord_x, ord_ref_wl_air, 3))
        if debug_level >= 2:
            plt.figure()
            plt.title('Order '+str(ord+1).zfill(2))
            plt.scatter(ord_x, ord_ref_wl_air, color='r')
            plt.plot(xx, order_fit(xx))
            plt.xlim(0,4111)
        # remove obvious outliers
        fitres = ord_ref_wl_air - order_fit(ord_x)
        # do sigma clipping
        goodres,goodix,badix = single_sigma_clip(fitres, 2, return_indices=True)
        if debug_level >= 2:
            plt.figure()
            plt.title('Order '+str(ord+1).zfill(2)+'  -  residuals')
            plt.plot(ord_x, fitres,'x-')
            plt.scatter(ord_x[badix], fitres[badix], marker='o', color='red')
            plt.xlim(0,4111)
        # this does not work properly yet, do it manually for now
#         while len(badix) > 0:
#             ord_x = ord_x[goodix]
#             ord_ref_wl_air = ord_ref_wl_air[goodix]
#             ord_ref_wl_vac = ord_ref_wl_vac[goodix]
#             order_fit = np.poly1d(np.polyfit(ord_x, ord_ref_wl_air, 3))
#             fitres = ord_ref_wl_air - order_fit(ord_x)
#             goodres,goodix,badix = single_sigma_clip(fitres, 2, return_indices=True)
#             plt.clf()
#             plt.plot(ord_x, fitres,'x')
#             plt.scatter(ord_x[badix], fitres[badix], marker='o', color='red')
        
        ######################################################################
        allix = np.arange(len(ord_x))
        #badix = [2,4,8,9]
        goodix = [ix for ix in allix if ix not in badix]
        
        ord_x = ord_x[goodix]
        ord_ref_wl_air = ord_ref_wl_air[goodix]
        ord_ref_wl_vac = ord_ref_wl_vac[goodix]
        order_fit = np.poly1d(np.polyfit(ord_x, ord_ref_wl_air, 3))
        fitres = ord_ref_wl_air - order_fit(ord_x)
        plt.plot(ord_x, fitres,'x-')
        ######################################################################
        
        # now we can fill the global arrays and save info to table 
        if debug_level >= 1:
            print(str(len(ord_x)) + ' good lines found in order ' + ordnum)
        nlines.append(len(ord_x))
        order.append(np.repeat(ord+1, len(ord_x)))
        x.append(ord_x)
        ref_wl_air.append(ord_ref_wl_air)
        ref_wl_vac.append(ord_ref_wl_vac)
        if savetables:
            outfile = open(outfn, 'a+')
            for i in range(len(ord_x)):
                outfile.write("   %3d             %2d                 %3d           %11.6f     %11.6f     %11.6f\n" %(i+1, int(ordnum), mord, ord_x[i], ord_ref_wl_air[i], ord_ref_wl_vac[i]))
            outfile.close()                                                                                                                                                                        
    #######################################################################################################################################
    # end of loop over all orders
    
    # convert to numpy arrays
    x = np.array(x)
    nlines = np.array(nlines)
    order = np.array(order)
    ref_wl_air = np.array(ref_wl_air)
    ref_wl_vac = np.array(ref_wl_vac)
        
    # go to normalized co-ordinates
    x_norm = (x / ((len(data)-1)/2.)) - 1.
    order_norm = ((order-1) / ((40-1)/2.)) - 1.

    # call the 2D fitting routine
    p_air = fit_poly_surface_2D(x_norm, order_norm, ref_wl_air, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)    
    p_vac = fit_poly_surface_2D(x_norm, order_norm, ref_wl_vac, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)    
    
    if savetables:
        # get residuals from fit
        model_wl_air = p_air(x_norm, order_norm)
        model_wl_vac = p_vac(x_norm, order_norm)
        resid_air = ref_wl_air - model_wl_air
        resid_vac = ref_wl_vac - model_wl_vac
        now = datetime.datetime.now()
        outfn2 = '/Users/christoph/OneDrive - UNSW/linelists/'+lamptype+'_lines_used_in_' + str(deg_spectral) + 'x' + str(deg_spatial) + '_fit_as_of_'+str(now)[:10]+'.dat'
        outfile2 = open(outfn2, 'w')
        outfile2.write('line number   order_number   physical_order_number     pixel       air_ref_wl[A]   vac_ref_wl[A]   air_model_wl[A]   vac_model_wl[A]   air_resid[A]   vac_resid[A] \n')
        outfile2.write('===================================================================================================================================================================\n')
        for i in range(len(x)):
            outfile2.write("   %3d             %2d                 %3d           %11.6f     %11.6f     %11.6f     %11.6f       %11.6f    %11.6f    %11.6f\n" 
                         %(i+1, order[i], order[i]+64, x[i], ref_wl_air[i], ref_wl_vac[i], model_wl_air[i], model_wl_vac[i], resid_air[i], resid_vac[i]))
        outfile2.close()                                                                                                                                                                     

    # evaluate for every pixel along each order
    if return_full:
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,len(thflux)+1)
        oon = ((oo-1) / ((len(thflux)-1)/2.)) - 1.   
        X,O = np.meshgrid(xxn,oon)
        p_air_wl = p_air(X,O)
        p_vac_wl = p_vac(X,O)
        return p_air_wl, p_vac_wl
    else:
        return p_air, p_vac
    




def define_arc_mask(lamptype, n_ord):

    mask = np.zeros((n_ord,4112), dtype='bool')

    if lamptype.lower() == 'thar':
        mask[1, 3060:3140] = True
        mask[2, 2340:2590] = True
        mask[6, 1100:1230] = True
        mask[7, 1650:1870] = True
        mask[7, 3550:3670] = True
        mask[7, 3850:3945] = True
        mask[8, 1180:1450] = True
        mask[8, 1560:1830] = True
        mask[9, 2260:2420] = True
        mask[10, 2920:3500] = True
        mask[11, 500:1150] = True
        mask[11, 2930:3260] = True
        mask[12, 475:850] = True
        mask[12, 2120:2320] = True
        mask[14, 2640:2820] = True
        mask[15, 2380:2685] = True
        mask[16, 0:50] = True
        mask[16, 3060:3160] = True
        mask[16, 3265:3435] = True
        mask[17, 785:960] = True
        mask[17, 1100:1275] = True
        mask[18, 1850:2050] = True
        mask[22, 1326:1420] = True
        mask[23, 1850:2050] = True
        mask[24, 1390:1765] = True
    elif lamptype.lower() == 'thxe':
        mask[2, 1520:1800] = True
        mask[3, 1240:1360] = True
        mask[3, 3060:3155] = True
        mask[4, 240:390] = True
        mask[4, 850:920] = True
        mask[4, 1390:1445] = True
        mask[4, 2340:2430] = True
        mask[4, 3020:3212] = True
        mask[5, 300:540] = True
        mask[5, 2260:2350] = True
        mask[8, 1630:1750] = True
        mask[8, 2935:3000] = True
        mask[9, 1880:2105] = True
        mask[9, 2900:3100] = True
        mask[10, 350:550] = True
        mask[11, 2080:2180] = True
        mask[15, 2335:2445] = True
        mask[16, 1450:1570] = True
        mask[21, 2120:2180] = True
    else:
        print('ERROR: lamp type must either be "ThAr" or "ThXe"!!!')
        return -1
    
    return mask



     

def xcorr_thflux(thflux, thflux2, scale=300., masking=True, satmask=None, lamptype='thar', fitwidth=7):
    
    #########################
    # preparation steps
    #########################

#     # trying to figure out the offsets between each fibre in dispersion direction
#     path = '/Users/christoph/OneDrive - UNSW/dispsol_tests/20180917/'
#     indfib_thflux = pyfits.getdata(path + 'ARC_ThAr_17sep30080_optimal3a_extracted.fits')
#     indfib_thflux_so = pyfits.getdata(path + 'ARC_ThAr_17sep30080_optimal3a_extracted_with_slope_and_offset.fits')
#     nfib = indfib_thflux.shape[1]
#     thflux2 = pyfits.getdata(path +'ARC_ThAr_17sep30080_quick_extracted_39_orders.fits')
#     n_ord = thflux2.shape[0]
#     


# evaluation steps afterwards:

#     fibshift = np.zeros((nfib,n_ord))
#     w_fibshift = np.zeros((nfib,n_ord))
#     # fibshift2 = np.zeros((nfib,n_ord))
#     # w_fibshift2 = np.zeros((nfib,n_ord))
#     # fibshift3 = np.zeros((nfib,n_ord))
#     # w_fibshift3 = np.zeros((nfib,n_ord))
#     
#     #loop over all fibres
#     for i in np.arange(nfib):
#         print('Fibre '+str(i+1).zfill(2))
#         thflux = indfib_thflux[:,i,:]
#         thflux2 = indfib_thflux[:,0,:]
#         # fibshift[i,:], w_fibshift[i,:], fibshift2[i,:], w_fibshift2[i,:], fibshift3[i,:], w_fibshift3[i,:] = xcorr_thflux(thflux, thflux2)
#         fibshift[i,:], w_fibshift[i,:] = xcorr_thflux(thflux, thflux2, masking=False)
    
    
    if thflux.shape[0] == thflux2.shape[0]:
    
        shift = []
#         shift2 = []
#         shift3 = []
        wshift = []
#         wshift2 = []
#         wshift3 = []
    
        # if not passed into the routine, define a mask that masks out saturated arc lines and some other crap
        if masking:
            if satmask is None:
                n_ord = thflux.shape[0]
                satmask = define_arc_mask(lamptype, n_ord)
    
        for o in np.arange(thflux.shape[0]):
            
            raw_data = thflux[o, :]
            raw_data2 = thflux2[o, :]
        
            data = quick_bg_fix(raw_data)
            data2 = quick_bg_fix(raw_data2)
        
            # NAN fix
            data[np.isnan(data)] = 0.
            data2[np.isnan(data2)] = 0.
            
            # mask out saturated lines
            if masking:
                ordmask = satmask[o, :]
                data[ordmask] = 0. 
        
            # scale = 300
            wdata = np.arcsinh(data / scale)
            wdata2 = np.arcsinh(data2 / scale)
        
            xc = np.correlate(data, data2, mode='same')
            wxc = np.correlate(wdata, wdata2, mode='same')
        
            # xrange = np.arange(np.argmax(xc) - fitwidth, np.argmax(xc) + fitwidth + 1, 1)
            xrange = np.arange(2056 - fitwidth, 2056 + fitwidth + 1, 1)
        
            # fit Gauss-like function to central part of xcorr function
            mod = Model(fibmodel_with_amp_and_offset)
            # plain (naive) cross-correlation function
            parms = lmfit.Parameters()
            gmod = GaussianModel()
            gparms = gmod.guess(xc[xrange],x=xrange)
            parms.add('mu', gparms['center'], min=gparms['center'] - 2, max=gparms['center'] + 2)
            parms.add('sigma', gparms['sigma'].value, min=1.)
            parms.add('amp', gparms['height'], min=0.)
            parms.add('beta', 2., min=0.1, max=10.)
            parms.add('offset', np.min(xc[xrange]), min=0., max=np.max(xc[xrange]))
            result = mod.fit(xc[xrange],parms,x=xrange,weights=None)
            # "arcsinh-scaled"
            wparms = lmfit.Parameters()
            gwmod = GaussianModel()
            gwparms = gwmod.guess(wxc[xrange],x=xrange)
            wparms.add('mu', gwparms['center'], min=gwparms['center'] - 2, max=gwparms['center'] + 2)
            wparms.add('sigma', gwparms['sigma'].value, min=1.)
            wparms.add('amp', gwparms['height'], min=0.)
            wparms.add('beta', 2., min=0.1, max=10.)
            wparms.add('offset', np.min(wxc[xrange]), min=0., max=np.max(wxc[xrange]))
            wresult = mod.fit(wxc[xrange],wparms,x=xrange,weights=None)
#             # fit Gaussian and Lorentzian to central part of CCF
#             mod2 = GaussianModel()
#             mod3 = LorentzianModel()
#             parms2 = mod2.guess(xc[xrange],x=xrange)
#             parms3 = mod3.guess(xc[xrange],x=xrange)
#             result2 = mod2.fit(xc[xrange],parms2,x=xrange,weights=None)
#             result3 = mod3.fit(xc[xrange],parms3,x=xrange,weights=None)
#             wparms2 = mod2.guess(wxc[xrange], x=xrange)
#             wparms3 = mod3.guess(wxc[xrange], x=xrange)
#             wresult2 = mod2.fit(wxc[xrange], wparms2, x=xrange, weights=None)
#             wresult3 = mod3.fit(wxc[xrange], wparms3, x=xrange, weights=None)
        
        
            # #plot it?
            # plot_osf = 10
            # plot_os_grid = np.linspace(xrange[0],xrange[-1],plot_osf * (len(xrange)-1) + 1)
            # guessmodel = mod.eval(result.init_params,x=plot_os_grid)
            # bestmodel = mod.eval(result.params,x=plot_os_grid)
            # wguessmodel = mod.eval(wresult.init_params,x=plot_os_grid)
            # wbestmodel = mod.eval(wresult.params,x=plot_os_grid)
            # plt.plot(wxc, 'b.-')
            # plt.plot(xrange, wxc[xrange], 'y.-')
            # plt.xlim(np.argmax(xc) - 2*fitwidth, np.argmax(xc) + 2*fitwidth + 1)
            # plt.plot(plot_os_grid, wguessmodel, 'r', label='initial guess')
            # plt.plot(plot_os_grid, wbestmodel, 'g', label='best fit')
            # plt.legend()
            
            mu = result.best_values['mu']
            wmu = wresult.best_values['mu']
#             mu2 = result2.best_values['center']
#             wmu2 = wresult2.best_values['center']
#             mu3 = result3.best_values['center']
#             wmu3 = wresult3.best_values['center']
            
            shift.append(mu - (len(xc)//2))
            wshift.append(wmu - (len(wxc)//2))
#             shift2.append(mu2 - (len(xc)//2))
#             wshift2.append(wmu2 - (len(wxc) // 2))
#             shift3.append(mu3 - (len(xc)//2))
#             wshift3.append(wmu3 - (len(wxc) // 2))

    else:
        print('ERROR: dimensions of "thflux" and "thflux2" do not agree!!!')
        return -1
    
    # return np.median(shift), np.median(wshift), np.median(shift2), np.median(wshift2), np.median(shift3), np.median(wshift3)
    # return shift, wshift, shift2, wshift2, shift3, wshift3
    return shift, wshift





def get_dispsol_from_known_lines(thflux, fibre=None, date=None, fitwidth=4, search_width=None, satmask=None, lamptype='thxe', minsigma=0.4, maxsigma=2.,
                                 sigma_0=0.85, minamp=0., maxamp=np.inf, return_all_pars=False, deg_spectral=7, deg_spatial=7,
                                 polytype='chebyshev', return_full=True, savetable=True, outpath=None, debug_level=0, timit=False):

    '''
    thflux   -   shape (n_ord, n_pix), ie for each fibre individually, or quick-extracted format
    '''

    if timit:
        start_time = time.time()

    #read master table
    linenum, order, m, pix, wlref, vac_wlref, _, _, _, _ = readcol('/Users/christoph/OneDrive - UNSW/linelists/thar_lines_used_in_7x7_fit_as_of_2018-10-19.dat', twod=False, skipline=2)
    del _
   
    xx = np.arange(thflux.shape[1])
   
    all_pix = []
    all_wl = []
    all_vac_wl = []
    all_order = []
    if return_all_pars:
        allsigma = []
        allamp = []
        alloffset = []
        allslope = []
   
    if satmask is None:
        n_ord = thflux.shape[0]
        satmask = define_arc_mask(lamptype.lower(), n_ord)
   
    if search_width is None:
        search_width = fitwidth 
   
    # loop over all orders
    for ord in range(thflux.shape[0]):
        
        # order numbers for output file
        ordnum = str(ord+1).zfill(2)
        mord = (ord+1) + 64
    
        if debug_level >= 1:
            print('Processing Order_' + ordnum)
        
        #define raw 1-dim data
        raw_data = thflux[ord,:]
        
        # clean up data
        data = quick_bg_fix(raw_data)
        
        # mask out saturated lines
        ordmask = satmask[ord,:]
        data[ordmask] = 0.
        
        # select individual orders from master table
        ix = np.argwhere(m == mord).flatten()
        ord_pix = pix[ix]
        ord_wl = wlref[ix]
        ord_vac_wl = vac_wlref[ix]
        
        # some housekeeping...
        new_ord_pix = []
        new_ord_wl = []
        new_ord_vac_wl = []
        new_ord_order = []
        if return_all_pars:
            new_ord_sigma = []
            new_ord_amp = []
            new_ord_offset = []
            new_ord_slope = []
            
        # loop over all good lines in that order
        for xguess, wl, vac_wl in zip(ord_pix, ord_wl, ord_vac_wl):
            if debug_level >= 2:
                print(xguess)
            peaks = np.r_[int(np.round(xguess, 0))] 
            search_range = xx[np.max([0,peaks[0] - fitwidth]) : np.min([peaks[-1] + fitwidth + 1,len(data)-1])] 
#             guess = np.array([xguess, sigma_0, data[int(np.round(xguess, 0))], np.min(data[xrange]), 0.])
#             popt, pcov = op.curve_fit(gaussian_with_offset_and_slope, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp,-np.inf,-np.inf],[xguess+2,maxsigma,maxamp,np.inf,np.inf]))
            
            peak_found = search_range[np.argmax(data[search_range])]
            xrange = xx[np.max([0,peak_found - fitwidth]) : np.min([peak_found + fitwidth + 1,len(data)-1])] 
            # shift the peak to roughly zero, so that offset has sensible values 
            guess = np.array([0, sigma_0, np.max(data[xrange]), np.min(data[xrange]), 0.])
            try:
                popt, pcov = op.curve_fit(gaussian_with_offset_and_slope, xrange - peak_found, data[xrange], p0=guess, 
                                          bounds=([-2,minsigma,minamp,-np.inf,-np.inf], [2,maxsigma,maxamp,np.inf,np.inf]))
                
                # remove rubbish fits (very low signal)
                if popt[2] > 50 and np.max(data[xrange]) > 30:
                    new_ord_pix.append(popt[0] + peak_found)
                    new_ord_wl.append(wl)
                    new_ord_vac_wl.append(vac_wl)
                    new_ord_order.append(ord)
                    if return_all_pars:
                        new_ord_sigma.append(popt[1])
                        new_ord_amp.append(popt[2])
                        new_ord_offset.append(popt[3])
                        new_ord_slope.append(popt[4])
                else:
                    if debug_level >= 1:
                        print('WARNING: not enough signal to fit a good peak at location', xguess)
                    
#             except RuntimeError:
            except:
                if debug_level >= 1:
                    print('WARNING: failed to fit a peak at location', xguess)
                
#             # plot a single fit for debugging
#             plot_osf = 10
#             plot_os_grid = np.linspace((xrange-peak_found)[0], (xrange-peak_found)[-1], plot_osf * (len(xrange)-1) + 1)
#             plt.plot(xx-peak_found, data,'b.-')
#             plt.plot(xrange-peak_found, data[xrange],'y.-')
#             plt.xlim(-10,10)
#             plt.ylim(np.min(data[xrange]) - 0.25 * (np.max(data[xrange] - np.min(data[xrange]))), np.max(data[xrange]) + 0.25 * (np.max(data[xrange] - np.min(data[xrange])))) 
#             plt.plot(plot_os_grid, gaussian_with_offset_and_slope(plot_os_grid, *guess),'r--', label='initial guess')
#             plt.plot(plot_os_grid, gaussian_with_offset_and_slope(plot_os_grid, *popt),'g-', label='best fit')
#             plt.axvline(popt[0], color='g', linestyle=':')
#             plt.legend()
#             plt.xlabel('pixel')
#             plt.ylabel('counts')
#             plt.text(5, np.max(data[xrange]) - 0.1 * (np.max(data[xrange] - np.min(data[xrange]))), 'mu = '+str(np.round(popt[0],4)))
    
        all_pix.append(new_ord_pix)
        all_wl.append(new_ord_wl)
        all_vac_wl.append(new_ord_vac_wl)
        all_order.append(new_ord_order)
        if return_all_pars:
            allsigma.append(new_ord_sigma)
            allamp.append(new_ord_amp)
            alloffset.append(new_ord_offset)
            allslope.append(new_ord_slope)
            
#     if return_all_pars:
#         return all_pix, all_wl, all_vac_wl, allsigma, allamp, alloffset, allslope
#     else:
#         return all_pix, all_wl, all_vac_wl
    
    
    # now make everything a simple 1-dim array
    x = np.array([item for sublist in all_pix for item in sublist])
    lam = np.array([item for sublist in all_wl for item in sublist])
    vac_lam = np.array([item for sublist in all_vac_wl for item in sublist])
    all_order_correct_format = np.array([item for sublist in all_order for item in sublist])
    
    # go to normalized co-ordinates
    x_norm = np.squeeze( (x / ((len(data)-1)/2.)) - 1. )
    order_norm = np.squeeze( ((all_order_correct_format) / ((40-1)/2.)) - 1. )
    
    # call the 2D fitting routine
    p_air = fit_poly_surface_2D(x_norm, order_norm, lam, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)    
    p_vac = fit_poly_surface_2D(x_norm, order_norm, vac_lam, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)  
    
    if savetable:
        # get residuals from fit
#         model_wl_air = p_air(x_norm, order_norm)
#         model_wl_vac = p_vac(x_norm, order_norm)
#         resid_air = lam - model_wl_air
#         resid_vac = vac_lam - model_wl_vac
        if date is None:
            now = datetime.datetime.now()
            if outpath is None:
                outpath = '/Users/christoph/OneDrive - UNSW/linelists/'
            if fibre is None:
                outfn = outpath + lamptype + '_lines_as_of_'+str(now)[:10]+'.dat'
            else:
                outfn = outpath + lamptype + '_lines_fibre_' + fibre + '_as_of_'+str(now)[:10]+'.dat'
        else:
            if outpath is None:
                outpath = '/Users/christoph/OneDrive - UNSW/dispsol/fibth_line_tables/'
            if fibre is None:
                outfn = outpath + lamptype + '_lines_quick_for_' + date + '.dat'
            else:
                outfn = outpath + lamptype + '_lines_fibre_' + fibre + '_for_' + date + '.dat'
        outfile = open(outfn, 'w')
        outfile.write('line number   order_number   physical_order_number     pixel       air_ref_wl[A]   vac_ref_wl[A] \n') #  air_model_wl[A]   vac_model_wl[A]   air_resid[A]   vac_resid[A] \n')
        outfile.write('=================================================================================================\n')
        for i in range(len(x)):
            outfile.write("   %3d             %2d                 %3d           %11.6f     %11.6f     %11.6f \n" #     %11.6f       %11.6f    %11.6f    %11.6f\n" 
                         %(i+1, all_order_correct_format[i]+1, all_order_correct_format[i]+65, x[i], lam[i], vac_lam[i])) #, model_wl_air[i], model_wl_vac[i], resid_air[i], resid_vac[i]))
        outfile.close()                                                                                                                                                                     

    if timit:
        print('Time elapsed for getting dispsol from known lines: ',time.time() - start_time,' seconds')

    # evaluate for every pixel along each order
    if return_full:
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,len(thflux)+1+1)     # note the double-adding of 1 is intentional in order to make a wl-solution for 40 orders!!!
        oon = ((oo-1) / ((40-1)/2.)) - 1.   
        X,O = np.meshgrid(xxn,oon)
        p_air_wl = p_air(X,O)
        p_vac_wl = p_vac(X,O)
        return p_air_wl, p_vac_wl
    else:
        return p_air, p_vac
    
    
    
    

def define_pixel_offsets_between_fibres(date, relto='S1', savedict=False, saveplots=False, extra_lfc_offset=True,
                                        return_all=False, debug_level=0, timit=False):
    if timit:
        start_time = time.time()

    # difference in pixel space (ie peak locations)
    refpath = '/Users/christoph/OneDrive - UNSW/dispsol/fibth_line_tables/'
    path = '/Users/christoph/OneDrive - UNSW/dispsol/pixfit_coeffs/'
    xx = np.arange(4112)
    fibslot = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25]
    fibname = ['S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09',
               '03', '08', '19', '02', 'S4', 'S3', 'S1']

    pixfit_coeffs = {}
    zeroth_coeffs = np.zeros((39, 24))
    first_coeffs = np.zeros((39, 24))

    if date == '20180917':
        lamptype = 'thar'
    else:
        lamptype = 'thxe'

    for o in range(39):

        ord = 'order_' + str(o + 1).zfill(2)

        ref_filename = refpath + lamptype + '_lines_fibre_' + relto + '_for_' + date + '.dat'
        linenum, order, m, pix, wlref, vac_wlref = readcol(ref_filename, twod=False, skipline=2)
        # linenum, order, m, pix, wlref, vac_wlref, _, _, _, _ = readcol('/Users/christoph/OneDrive - UNSW/linelists/thar_lines_used_in_7x7_fit_as_of_2018-10-19.dat', twod=False, skipline=2)
        ix = np.argwhere(order == o + 1).flatten()
        ref_x = pix[ix]
        ref_wlref = wlref[ix]
        #     order_fit = np.poly1d(np.polyfit(ref_x, ref_wlref, 5))
        #     ref_fit = order_fit(xx)

        pixfit_coeffs[ord] = {}
        ord_zeroth_coeffs = []
        ord_first_coeffs = []

        for fib in fibname:
            filename = refpath + lamptype + '_lines_fibre_' + fib + '_for_' + date + '.dat'
            linenum, order, m, pix, wlref, vac_wlref = readcol(filename, twod=False, skipline=2)
            ix = np.argwhere(order == o + 1).flatten()
            x = pix[ix]
            lam = wlref[ix]
            matched_ref_x = ref_x[np.in1d(ref_wlref, lam)]
            matched_x = x[np.in1d(lam, ref_wlref)]
            delta_x = matched_x - matched_ref_x
            #     plt.plot(matched_ref_x, x - matched_ref_x, 'x-', label='fibre '+fib)
            # perform sanity-check 1D fit (remove only very obvious outliers, as there is quite a bit of scatter)
            test_fit = np.poly1d(np.polyfit(matched_ref_x, delta_x, 1))
            if debug_level >= 2:
                plt.figure()
                plt.plot(matched_ref_x, delta_x, 'x-')
                plt.title('fibre ' + fib)
                plt.plot(xx, test_fit(xx), 'r--')
                plt.xlim(0, 4111)
            # remove obvious outliers
            fitres = delta_x - test_fit(matched_ref_x)
            # do single sigma clipping
            goodres, goodix, badix = single_sigma_clip(fitres, 2, return_indices=True)
            # fit again
            pix_fit = np.poly1d(np.polyfit(matched_ref_x[goodix], delta_x[goodix], 1))
            if debug_level >= 2:
                plt.scatter(matched_ref_x[badix], delta_x[badix], color='r', marker='o')
                plt.plot(xx, pix_fit(xx), 'g-')

            pixfit_coeffs[ord]['fibre_' + fib] = pix_fit
            ord_zeroth_coeffs.append(pix_fit[0])
            ord_first_coeffs.append(pix_fit[1])
        #         #now fit new dispsol to the model-shifted lines
        #         order_fit = np.poly1d(np.polyfit(ref_x + pix_fit(ref_x), ref_wlref, 5))
        #         plt.plot(xx, order_fit(xx) - ref_fit, label='fibre '+fib)

        # now fill array of fit coefficients
        zeroth_coeffs[o, :] = ord_zeroth_coeffs
        first_coeffs[o, :] = ord_first_coeffs

        # now estimate the extra offset between fibre 'S1' and the LFC fibre by extrapolating to the next fibre slot (ie slot 26)
        if extra_lfc_offset:
            ex_fit_0 = np.poly1d(np.polyfit(fibslot, zeroth_coeffs[o, :], 2))
            ex_fit_1 = np.poly1d(np.polyfit(fibslot, first_coeffs[o, :], 2))
            zeroth_coeffs[o, :] -= ex_fit_0(26)
            first_coeffs[o, :] -= ex_fit_1(26)
            for fib in pixfit_coeffs[ord].keys():
                pixfit_coeffs[ord][fib][0] -= ex_fit_0(26)
                pixfit_coeffs[ord][fib][1] -= ex_fit_1(26)

        # now save the dictionary containing the fit coefficients that describe shift and slope in pixel space to file
        if savedict:
            # now = datetime.datetime.now()
            # if extra_lfc_offset:
            #     np.save(path + 'pixfit_coeffs_relto_LFC_as_of_' + str(now)[:10] + '.npy', pixfit_coeffs)
            # else:
            #     np.save(path + 'pixfit_coeffs_relto_' + relto + '_as_of_' + str(now)[:10] + '.npy', pixfit_coeffs)
            if extra_lfc_offset:
                np.save(path + 'pixfit_coeffs_relto_LFC_for_' + date + '.npy', pixfit_coeffs)
            else:
                np.save(path + 'pixfit_coeffs_relto_' + relto + '_for_' + date + '.npy', pixfit_coeffs)

        # now save a plot of both 0th and 1st order terms
        if saveplots:
            ordstring = 'Order_' + str(o + 1).zfill(2)

            #     plt.figure()
            plt.plot(fibslot, zeroth_coeffs, 'bo-')
            plt.title('0th order coefficients  -  ' + ordstring)
            plt.xlim(-2, 27)
            plt.xlabel('fibre')
            plt.ylabel('offset [pix]')
            plt.axvline(2, color='gray', linestyle=':')
            plt.axvline(22, color='gray', linestyle=':')
            plt.savefig(path + '0th_order_coeffs_pixel_shits_' + ordstring + '.eps')
            plt.clf()

            #     plt.figure()
            plt.plot(fibslot, first_coeffs, 'bo')
            plt.title('1st order coefficients  -  ' + ordstring)
            plt.xlim(-2, 27)
            plt.xlabel('fibre')
            plt.ylabel('slope')
            plt.axvline(2, color='gray', linestyle=':')
            plt.axvline(22, color='gray', linestyle=':')
            slope_fit = np.poly1d(np.polyfit(fibslot, first_coeffs, 2))
            plot_x = np.arange(-1, 26, 0.1)
            plt.plot(plot_x, slope_fit(plot_x), 'r')
            plt.savefig(path + '1st_order_coeffs_pixel_shits_' + ordstring + '.eps')
            plt.clf()

    if timit:
        delta_t = time.time() - start_time
        print('Time taken for defining pixel offsets: ' + str(delta_t) + ' seconds')

    if return_all:
        return pixfit_coeffs, zeroth_coeffs, first_coeffs
    else:
        return pixfit_coeffs





def old_get_dispsol_for_all_fibs(obsname, date=None, relto='LFC', twod=False, degpol=7, deg_spectral=7, deg_spatial=7, fibs='stellar', nx=4112,
                             polytype='chebyshev', refit=False, debug_level=0, timit=False, fudge=1., signflip_shift=True,
                             signflip_slope=True, signflip_secord=True, nightly_coeffs=True):

    '''using Duncan's xcorrs to measure LFC shifts'''

    if timit:
        start_time = time.time()

    # if no date is provided, get it from the obsname
    if date is None:
        day = obsname[:2]
        monstr = obsname[2:5]
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        mon = str(1 + np.argwhere(np.array(months) == monstr)[0][0]).zfill(2)
        if mon in ['01', '02']:
            yr = '2019'
        else:
            yr = '2018'
        date = yr + mon + day

    # need to do this b/c not all files have a pixfit coeffs file
    if nightly_coeffs:
        pc_fn = '/Users/christoph/OneDrive - UNSW/dispsol/pixfit_coeffs/pixfit_coeffs_relto_' + relto + '_for_' + date + '.npy'
        while not os.path.isfile(pc_fn):
            date = str(int(date) - 1)
            pc_fn = '/Users/christoph/OneDrive - UNSW/dispsol/pixfit_coeffs/pixfit_coeffs_relto_' + relto + '_for_' + date + '.npy'
    else:
        # load default coefficients
        pc_fn = '/Users/christoph/OneDrive - UNSW/dispsol_tests/20180917/pixfit_coeffs_relto_' + relto + '_as_of_2018-11-14.npy'
    pixfit_coeffs = np.load(pc_fn).item()

    # fibslot = [0,1,   3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,   23,24,25]
    fibname = ['S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09', '03', '08', '19', '02', 'S4', 'S3', 'S1']

    nfib = len(fibname)

    # read master LFC linelist
    lfc_ord, lfc_pix, lfc_wl = readcol('/Users/christoph/OneDrive - UNSW/dispsol/laser_dispsol_20181015/PeakPos.txt', twod=False)
    lfc_pix -= 1.   # b/c DW is using Matlab, which starts indexing at 1 not at 0!!!!!!!

    # read file containing slope and offset as measured from LFC peak positions
    # read file and see how many coefficients are provided
    fn = '/Users/christoph/OneDrive - UNSW/dispsol/laser_offsets/relto_21sep30019/' + obsname + '_Slope_and_Offset.txt'
    dum = readcol(fn)
    n_coeffs = dum.shape[1]
    assert n_coeffs in [2,3], "ERROR: DW's LFC drift coefficient files do not have exactly two or three colums!!!"
    if n_coeffs == 2:
        lfc_slope, lfc_shift = readcol(fn, twod=False)
        lfc_secord = np.zeros(lfc_slope.shape)
    if n_coeffs == 3:
        lfc_secord, lfc_slope, lfc_shift = readcol(fn, twod=False)
        # TESTING ONLY!!!
        # lfc_shift = np.zeros(lfc_slope.shape)
        # lfc_secord = np.zeros(lfc_slope.shape)
        # lfc_slope = np.zeros(lfc_slope.shape)

    if signflip_shift:
        lfc_shift *= -1.
    if signflip_slope:
        lfc_slope *= -1.
    if signflip_secord:
        lfc_secord *= -1.

    # some housekeeping...
    n_ord = len(np.unique(lfc_ord))
    xx = np.arange(nx)

    # prepare outputs
    wldict = {}
    # wl = np.zeros((n_ord, nfib, nx))
    wl = np.zeros((40, nfib, nx))

    if not twod:
        if debug_level >= 1:
            print('OK, getting a 1-D wavelength solution...')
        # loop over all orders
        for o in np.unique(lfc_ord)[:-1]:
            ord = 'order_'+str(o+1).zfill(2)     # DW's order numbers are offset by one to CB's
            wldict[ord] = {}
            ix = np.argwhere(lfc_ord == o).flatten()
            x0 = lfc_pix[ix]
            lam = lfc_wl[ix]
            master_lfc_fit = np.poly1d(np.polyfit(x0, lam, degpol))

            # either perform a new fit using the new x-values
            if refit:
                # FFFFUUUUU, broke my brain turning that around, but don't need to, b/c DW's LFC spectra are flipped (left-right) wrt to CBs
                # # now apply the shift to the pixel positions
                # diff = - lfc_shift[o-1] + lfc_slope[o-1] * x0     # signs are confusing here, but comparison to DW's results suggests this is correct
                # x = x0 + diff - np.max(diff) + np.min(diff)       # signs are confusing here, but comparison to DW's results suggests this is correctly
                diff = lfc_shift[o-1] + lfc_slope[o-1] * x0 + lfc_secord[o-1] * x0**2        # need the o-1 here because Duncan's indices start with 1
                x = x0 - fudge * diff
                lfc_fit = np.poly1d(np.polyfit(x, lam, degpol))
                newly_fit_dispsol = lfc_fit(xx)
                wldict[ord]['laser'] = newly_fit_dispsol[::-1]     # need to turn around, b/c DW's LFC spectra are flipped (left-right) wrt to CBs
                # need to flip x now, because the pixfit_coeffs are using CMB layout
                x_flipped = (nx - 1) - x
                # loop over all fibres
                for i,fib in enumerate(fibname):
                    xfib = x_flipped + pixfit_coeffs[ord]['fibre_'+fib](x_flipped)
                    fib_fit = np.poly1d(np.polyfit(xfib, lam, degpol))
                    wldict[ord][fib] = fib_fit(xx)
                    wl[o,i,:] = fib_fit(xx)
            # or re-evaluate the master fit at new x-values (better b/c less margin for fitting variations and hence more constancy)
            else:
                diff = lfc_shift[o-1] + lfc_slope[o-1] * x0 + lfc_secord[o-1] * x0**2
                xx_prime = xx + fudge * diff
                eval_xprime_dispsol = master_lfc_fit(xx_prime)
                wldict[ord]['laser'] = eval_xprime_dispsol[::-1]   # need to turn around, b/c DW's LFC spectra are flipped (left-right) wrt to CBs
                # need to flip x now, because the pixfit_coeffs are using CMB layout
                x0_flipped = (nx - 1) - x0
                # loop over all fibres
                for i,fib in enumerate(fibname):
                    xfib_flipped = x0_flipped + pixfit_coeffs[ord]['fibre_'+fib](x0_flipped)
                    xfib = (nx - 1) - xfib_flipped   # and flip it back --> can't think right now, can I combine those two flips???
                    master_lfc_fit_fib = np.poly1d(np.polyfit(xfib, lam, degpol))
                    diff_xx = lfc_shift[o-1] + lfc_slope[o-1] * xx + lfc_secord[o-1] * xx**2
                    xx_prime_fib = xx + fudge * diff_xx
                    eval_xprime_dispsol_fib = master_lfc_fit_fib(xx_prime_fib)
                    wldict[ord][fib] = eval_xprime_dispsol_fib[::-1]
                    wl[o,i,:] = eval_xprime_dispsol_fib[::-1]

    else:

        print('THIS IS STILL WRONG (needs to be flipped) - USE THE 1-DIM VERSION FOR NOW!!!')

        if debug_level >= 1:
            print('OK, getting a 2-D wavelength solution...')

        order = lfc_ord.copy() + 1     # DW's order numbers are offset by one to CB's
        pixel = np.tile(lfc_pix.copy(), (nfib,1))
        lam = lfc_wl.copy()

        # temporarily remove the blue-most order until I have incorporated that into the rest of the reduction
        dumix = np.argwhere(lfc_ord == 39).flatten()
        order = np.delete(order, dumix)
        pixel = np.delete(pixel,dumix,1)
        lam = np.delete(lam,dumix)

        # fit the pure LFC dispsol
        x_norm = np.squeeze( (pixel[0,:] / ((nx-1)/2.)) - 1. )
        order_norm = np.squeeze( ((order-1) / ((40-1)/2.)) - 1. )
        p_wl_lfc = fit_poly_surface_2D(x_norm, order_norm, lam, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)
        # calculate wavelengths from model for full (n_ord x n_pix)-array
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,41)
        oon = ((oo-1) / ((40-1)/2.)) - 1.
        X,O = np.meshgrid(xxn,oon)
        wl_lfc = p_wl_lfc(X,O)

        # loop over all orders
        for o in np.unique(lfc_ord)[:-1]:
            ord = 'order_'+str(o+1).zfill(2)     # DW's order numbers are offset by one to CB's
            wldict[ord] = {}
            ix = np.argwhere(lfc_ord == o).flatten()
            x0 = lfc_pix[ix]
            # now apply the shift to the pixel positions
            diff = - lfc_shift[o-1] + lfc_slope[o-1] * x0   # signs are confusing here, but comparison to DW's results suggests this is correct
            x = x0 + diff - np.max(diff) + np.min(diff)     # signs are confusing here, but comparison to DW's results suggests this is correct
            # loop over all fibres
            for i,fib in enumerate(fibname):
                pixel[i,ix] = x + pixfit_coeffs[ord]['fibre_'+fib](x)

        # now fit a 2-D wavelength solution for each fibre individually
        # loop over all fibres
        for i,fib in enumerate(fibname):
            if debug_level >= 1:
                print(fib)
            # print(fib)
            # go to normalized co-ordinates
            x_norm = np.squeeze( (pixel[i,:] / ((nx-1)/2.)) - 1. )
            order_norm = np.squeeze( ((order-1) / ((40-1)/2.)) - 1. )

            # call the 2D fitting routine
            p_wl = fit_poly_surface_2D(x_norm, order_norm, lam, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)

            # calculate wavelengths from model for full (n_ord x n_pix)-array
            xxn = (xx / ((len(xx)-1)/2.)) - 1.
            oo = np.arange(1,41)
            oon = ((oo-1) / ((40-1)/2.)) - 1.
            X,O = np.meshgrid(xxn,oon)
            wl[:,i,:] = p_wl(X,O)

        for i,o in enumerate(np.unique(lfc_ord)[:-1]):
            ord = 'order_'+str(o+1).zfill(2)
            for j,fib in enumerate(fibname):
                wldict[ord]['fibre_'+fib] = wl[o,j,:].copy()
            wldict[ord]['laser'] = wl_lfc[o,:].copy()

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    if fibs not in ['all', 'stellar']:
        print('ERROR: "fibs" not valid!!!')
        return
    elif fibs == 'stellar':
        wl = wl[:,2:21,:]

    return wldict,wl





def get_dispsol_for_all_fibs(obsname, date=None, relto='LFC', twod=False, degpol=7, deg_spectral=7, deg_spatial=7, fibs='stellar', nx=4112,
                             polytype='chebyshev', refit=False, debug_level=0, timit=False, fudge=1., fibtofib=True, nightly_coeffs=True):

    '''using Duncan's xcorrs to measure LFC shifts'''

#     print('haehaehae!!!!!')

    if timit:
        start_time = time.time()

    # if no date is provided, get it from the obsname
    if date is None:
        day = obsname[:2]
        monstr = obsname[2:5]
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        mon = str(1 + np.argwhere(np.array(months) == monstr)[0][0]).zfill(2)
        if mon in ['01', '02']:
            yr = '2019'
        else:
            yr = '2018'
        date = yr + mon + day

    if fibtofib:
        if nightly_coeffs:
            # need to do this b/c not all nights have a pixfit coeffs file
            pc_fn = '/Users/christoph/OneDrive - UNSW/dispsol/pixfit_coeffs/pixfit_coeffs_relto_' + relto + '_for_' + date + '.npy'
            while not os.path.isfile(pc_fn):
                date = str(int(date) - 1)
                pc_fn = '/Users/christoph/OneDrive - UNSW/dispsol/pixfit_coeffs/pixfit_coeffs_relto_' + relto + '_for_' + date + '.npy'
        else:
            # load default coefficients
            pc_fn = '/Users/christoph/OneDrive - UNSW/dispsol_tests/20180917/pixfit_coeffs_relto_' + relto + '_as_of_2018-11-14.npy'
        pixfit_coeffs = np.load(pc_fn).item()

    # fibslot = [0,1,   3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,   23,24,25]
    fibname = ['S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09', '03', '08', '19', '02', 'S4', 'S3', 'S1']

    nfib = len(fibname)

    # read master LFC linelist
    lfc_ord, lfc_pix, lfc_wl = readcol('/Users/christoph/OneDrive - UNSW/dispsol/laser_dispsol_20181015/PeakPos.txt', twod=False)
    lfc_pix = 4112 - lfc_pix   # b/c DW is using Matlab, which starts indexing at 1 not at 0, and also his pixels increase with wl, mine decrease

    # read file containing slope and offset as measured from LFC peak positions
    # read file and see how many coefficients are provided
    fn = '/Users/christoph/OneDrive - UNSW/dispsol/laser_offsets/relto_21sep30019/' + obsname + '_Slope_and_Offset.txt'
    dum = readcol(fn)
    n_coeffs = dum.shape[1]
    assert n_coeffs in [2,3], "ERROR: DW's LFC drift coefficient files do not have exactly two or three colums!!!"
    if n_coeffs == 2:
        lfc_slope, lfc_shift = readcol(fn, twod=False)
        lfc_secord = np.zeros(lfc_slope.shape)
    if n_coeffs == 3:
        lfc_secord, lfc_slope, lfc_shift = readcol(fn, twod=False)
        # # TESTING ONLY!!!
        # lfc_shift = np.zeros(lfc_slope.shape)
        # lfc_secord = np.zeros(lfc_slope.shape)
        # lfc_slope = np.zeros(lfc_slope.shape)


    # some housekeeping...
    n_ord = len(np.unique(lfc_ord))
    xx = np.arange(nx)

    # prepare outputs
    wldict = {}
    # wl = np.zeros((n_ord, nfib, nx))
    wl = np.zeros((40, nfib, nx))

    if not twod:
        if debug_level >= 1:
            print('OK, getting a 1-D wavelength solution...')
        # loop over all orders
        for o in np.unique(lfc_ord)[:-1]:
            ord = 'order_'+str(o+1).zfill(2)     # DW's order numbers are offset by one to CB's
            wldict[ord] = {}
            ix = np.argwhere(lfc_ord == o).flatten()
            x0 = lfc_pix[ix]
            lam = lfc_wl[ix]
            master_lfc_fit = np.poly1d(np.polyfit(x0, lam, degpol))

            # either perform a new fit using the new x-values
            if refit:
                # no more x-flipping needed b/c already done above, but need to add one b/c Python vs Matlab indexing
                diff = lfc_shift[o-1] + lfc_slope[o-1] * (x0+1) + lfc_secord[o-1] * (x0+1)**2     # also, the o-1 is needed b/c Duncan starts at order_02
                x = x0 - fudge * diff
                lfc_fit = np.poly1d(np.polyfit(x, lam, degpol))
                newly_fit_dispsol = lfc_fit(xx)
                wldict[ord]['laser'] = newly_fit_dispsol
                # loop over all fibres
                for i,fib in enumerate(fibname):
                    if fibtofib:
                        xfib = x + pixfit_coeffs[ord]['fibre_'+fib](x)
                    else:
                        xfib = x.copy()
                    fib_fit = np.poly1d(np.polyfit(xfib, lam, degpol))
                    wldict[ord][fib] = fib_fit(xx)
                    wl[o,i,:] = fib_fit(xx)
            # or re-evaluate the master fit at new x-values (better b/c less margin for fitting variations and hence more constancy)
            else:
                diff_xx = lfc_shift[o-1] + lfc_slope[o-1] * (xx+1) + lfc_secord[o-1] * (xx+1)**2  # also, the o-1 is needed b/c Duncan starts at order_02
                xx_prime = xx + fudge * diff_xx
                eval_xprime_dispsol = master_lfc_fit(xx_prime)
                wldict[ord]['laser'] = eval_xprime_dispsol
                # loop over all fibres
                for i,fib in enumerate(fibname):
                    if fibtofib:
                        xfib = x0 + pixfit_coeffs[ord]['fibre_'+fib](x0)
                    else:
                        xfib = x0.copy()
                    # xfib = (nx - 1) - xfib_flipped   # and flip it back --> can't think right now, can I combine those two flips???
                    master_lfc_fit_fib = np.poly1d(np.polyfit(xfib, lam, degpol))
                    # diff_xx = lfc_shift[o-1] + lfc_slope[o-1] * xx + lfc_secord[o-1] * xx**2
                    xx_prime_fib = xx + fudge * diff_xx   # same as xx_prime, but OK
                    eval_xprime_dispsol_fib = master_lfc_fit_fib(xx_prime_fib)
                    wldict[ord][fib] = eval_xprime_dispsol_fib
                    wl[o,i,:] = eval_xprime_dispsol_fib

    else:

        print('THIS IS STILL WRONG (needs to be flipped) - USE THE 1-DIM VERSION FOR NOW!!!')

        if debug_level >= 1:
            print('OK, getting a 2-D wavelength solution...')

        order = lfc_ord.copy() + 1     # DW's order numbers are offset by one to CB's
        pixel = np.tile(lfc_pix.copy(), (nfib,1))
        lam = lfc_wl.copy()

        # temporarily remove the blue-most order until I have incorporated that into the rest of the reduction
        dumix = np.argwhere(lfc_ord == 39).flatten()
        order = np.delete(order, dumix)
        pixel = np.delete(pixel,dumix,1)
        lam = np.delete(lam,dumix)

        # fit the pure LFC dispsol
        x_norm = np.squeeze( (pixel[0,:] / ((nx-1)/2.)) - 1. )
        order_norm = np.squeeze( ((order-1) / ((40-1)/2.)) - 1. )
        p_wl_lfc = fit_poly_surface_2D(x_norm, order_norm, lam, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)
        # calculate wavelengths from model for full (n_ord x n_pix)-array
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,41)
        oon = ((oo-1) / ((40-1)/2.)) - 1.
        X,O = np.meshgrid(xxn,oon)
        wl_lfc = p_wl_lfc(X,O)

        # loop over all orders
        for o in np.unique(lfc_ord)[:-1]:
            ord = 'order_'+str(o+1).zfill(2)     # DW's order numbers are offset by one to CB's
            wldict[ord] = {}
            ix = np.argwhere(lfc_ord == o).flatten()
            x0 = lfc_pix[ix]
            # now apply the shift to the pixel positions
            diff = - lfc_shift[o-1] + lfc_slope[o-1] * x0   # signs are confusing here, but comparison to DW's results suggests this is correct
            x = x0 + diff - np.max(diff) + np.min(diff)     # signs are confusing here, but comparison to DW's results suggests this is correct
            # loop over all fibres
            for i,fib in enumerate(fibname):
                pixel[i,ix] = x + pixfit_coeffs[ord]['fibre_'+fib](x)

        # now fit a 2-D wavelength solution for each fibre individually
        # loop over all fibres
        for i,fib in enumerate(fibname):
            if debug_level >= 1:
                print(fib)
            # print(fib)
            # go to normalized co-ordinates
            x_norm = np.squeeze( (pixel[i,:] / ((nx-1)/2.)) - 1. )
            order_norm = np.squeeze( ((order-1) / ((40-1)/2.)) - 1. )

            # call the 2D fitting routine
            p_wl = fit_poly_surface_2D(x_norm, order_norm, lam, weights=None, polytype=polytype, poly_deg_x=deg_spectral, poly_deg_y=deg_spatial, debug_level=0)

            # calculate wavelengths from model for full (n_ord x n_pix)-array
            xxn = (xx / ((len(xx)-1)/2.)) - 1.
            oo = np.arange(1,41)
            oon = ((oo-1) / ((40-1)/2.)) - 1.
            X,O = np.meshgrid(xxn,oon)
            wl[:,i,:] = p_wl(X,O)

        for i,o in enumerate(np.unique(lfc_ord)[:-1]):
            ord = 'order_'+str(o+1).zfill(2)
            for j,fib in enumerate(fibname):
                wldict[ord]['fibre_'+fib] = wl[o,j,:].copy()
            wldict[ord]['laser'] = wl_lfc[o,:].copy()

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    if fibs not in ['all', 'stellar']:
        print('ERROR: "fibs" not valid!!!')
        return
    elif fibs == 'stellar':
        wl = wl[:,2:21,:]

    return wldict,wl





def get_dispsol_for_all_fibs_2(obsname, date=None, relto='LFC', degpol=7, fibs='stellar', nx=4112, eps=0.5, fudge=1., refit=False, debug_level=0, timit=False):
    '''using CGT's DAOPHOT results to measure LFC shifts'''

    # TODO: instead of reading P_id for the LFC, do a divide into orders and fit new x-y-traces
    # TODO: use meansep - get_mean_fibre_separation instead of 1.97

    lfc_path = '/Users/christoph/OneDrive - UNSW/lfc_peaks/'

    if timit:
        start_time = time.time()

    # make sure a date is provided
    assert date is not None, 'ERROR: "date" not provided'
    
    if fibtofib:
        if nightly_coeffs:
            # need to do this b/c not all nights have a pixfit coeffs file
            pc_fn = '/Users/christoph/OneDrive - UNSW/dispsol/pixfit_coeffs/pixfit_coeffs_relto_' + relto + '_for_' + date + '.npy'
            while not os.path.isfile(pc_fn):
                date = str(int(date) - 1)
                pc_fn = '/Users/christoph/OneDrive - UNSW/dispsol/pixfit_coeffs/pixfit_coeffs_relto_' + relto + '_for_' + date + '.npy'
        else:
            # load default coefficients
            pc_fn = '/Users/christoph/OneDrive - UNSW/dispsol_tests/20180917/pixfit_coeffs_relto_' + relto + '_as_of_2018-11-14.npy'
        pixfit_coeffs = np.load(pc_fn).item()

    # load default coefficients
    pixfit_coeffs = np.load('/Users/christoph/OneDrive - UNSW/dispsol_tests/20180917/pixfit_coeffs_relto_' + relto + '_as_of_2018-11-14.npy').item()

    # fibslot = [0,1,   3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,   23,24,25]
    fibname = ['S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09',
               '03', '08', '19', '02', 'S4', 'S3', 'S1']
    offsets = 1.97 * np.array(range(3) + range(4,23) + range(24,26))[::-1]   # 1.97 pix is the median fib-to-fib separation
    if not refit:
        print('WARNING: median fibre to fibre separation not correct!!!')

    nfib = len(fibname)

    # read master LFC linelist
    lfc_ord, lfc_pix, lfc_wl = readcol('/Users/christoph/OneDrive - UNSW/dispsol/laser_dispsol_20181015/PeakPos.txt',
                                       twod=False)
    lfc_pix -= 1.  # b/c DW is using Matlab, which starts indexing at 1 not at 0!!!!!!!

    # read file containing LFC peak positions of observation
    _, yref, xref, _, _, _, _, _, _, _, _ = readcol(lfc_path + '21sep30019olc.nst', twod=False, skipline=2)
    try:
        _, y, x, _, _, _, _, _, _, _, _ = readcol(lfc_path + 'all/' + obsname + 'olc.nst', twod=False, skipline=2)
    except:
        _, y, x, _, _, _, _, _, _ = readcol(lfc_path + 'all/' + obsname + 'olc.nst', twod=False, skipline=2)
    del _
    xref = nx - xref
    x = nx - x
    yref = yref - 54.   # or 53??? but does not matter for getting the transformation matrix
    y = y - 54.         # or 53??? but does not matter for getting the transformation matrix
    # ref_peaks = divide_lfc_peaks_into_orders(xref, yref)
    # peaks = divide_lfc_peaks_into_orders(x, y)

    # some housekeeping...
    xx = np.arange(nx)

    if refit:
        # divide LFC peak positions into orders
        peaks = divide_lfc_peaks_into_orders(x, y)
    else:
        # get affine transformation matrix
        #     if   x' = x * M
        # # then   x  = x' * M_inv
        Minv = find_affine_transformation_matrix(xref, yref, x, y, timit=True, eps=2.)
        # also get the rough trace of the LFC fibre (needed below)
        pid = np.load(lfc_path + 'lfc_P_id.npy').item()

    # prepare outputs
    wldict = {}
    # wl = np.zeros((n_ord, nfib, nx))
    wl = np.zeros((40, nfib, nx))

    # loop over all orders
    for o in np.unique(lfc_ord)[:-1]:
        ord = 'order_' + str(o + 1).zfill(2)  # DW's order numbers are offset by one to CB's
        wldict[ord] = {}
        ix = np.argwhere(lfc_ord == o).flatten()
        x0 = lfc_pix[ix]
        lam = lfc_wl[ix]
        master_lfc_fit = np.poly1d(np.polyfit(x0, lam, degpol))

        if refit:
            x = np.array(peaks[ord])[:,0]
            # check which peaks match peaks from the master solution above
            # also need to turn around x0 here
            x0 = (nx - 1) - x0
            good_x = []
            good_wls = []
            for xpos in x:
                distance = np.abs(x0 - xpos)
                if np.sum(distance < eps) > 0:
                    if np.sum(distance < eps) > 1:
                        print('FUGANDA: @', xpos)
                    # get indices of those peaks and hence their wavelengths
                    good_x.append(xpos)
                    good_wls.append(lam[distance < eps])

            # re-fit
            lfc_fit = np.poly1d(np.polyfit(good_x, np.squeeze(good_wls), degpol))
            newly_fit_dispsol = lfc_fit(xx)
            # no need to turn around anything anymore here
            wldict[ord]['laser'] = newly_fit_dispsol
            # loop over all fibres
            for i, fib in enumerate(fibname):
                xfib = good_x + pixfit_coeffs[ord]['fibre_' + fib](good_x)
                fib_fit = np.poly1d(np.polyfit(xfib, np.squeeze(good_wls), degpol))
                wldict[ord][fib] = fib_fit(xx)
                wl[o, i, :] = fib_fit(xx)

        else:
            # now get shifted coordinates
            coords = np.array([xx, pid[ord](xx), np.ones(nx)]).T
            shifted_coords = np.dot(coords, Minv)
            xx_prime = shifted_coords[:,0]

            # now re-evaluate the master fit at new x-values
            eval_xprime_dispsol = master_lfc_fit(xx_prime)

            wldict[ord]['laser'] = eval_xprime_dispsol[::-1]  # need to turn around, b/c DW's LFC spectra are flipped (left-right) wrt to CBs
            # need to flip x now, because the pixfit_coeffs are using CMB layout
            x0_flipped = (nx - 1) - x0

            # loop over all fibres
            for i, fib in enumerate(fibname):
                xfib_flipped = x0_flipped + pixfit_coeffs[ord]['fibre_' + fib](x0_flipped)
                xfib = (nx - 1) - xfib_flipped  # and flip it back --> can't think right now, can I combine those two flips???
                master_lfc_fit_fib = np.poly1d(np.polyfit(xfib, lam, degpol))

                # TODO :
                # coords_fib = np.array([xx, fibre_traces[fib](xx), np.ones(nx)]).T
                coords_fib = np.array([xx, pid[ord](xx) - offsets[i], np.ones(nx)]).T
                shifted_coords_fib = np.dot(coords_fib, Minv)
                xx_prime_fib = shifted_coords_fib[:,0]

                eval_xprime_dispsol_fib = master_lfc_fit_fib(xx_prime_fib)
                wldict[ord][fib] = eval_xprime_dispsol_fib[::-1]
                wl[o, i, :] = eval_xprime_dispsol_fib[::-1]

    if timit:
        print('Time elapsed: ' + str(np.round(time.time() - start_time, 1)) + ' seconds')

    if fibs not in ['all', 'stellar']:
        print('ERROR: "fibs" not valid!!!')
        return
    elif fibs == 'stellar':
        wl = wl[:, 2:21, :]

    return wldict, wl





def get_dispsol_for_all_fibs_3(obsname, date=None, relto='LFC', degpol=7, nx=4112, refit=True, fibtofib=True, return_auxdata=False, 
                               norm_coords=False, laptop=False, debug_level=0, timit=False):
    '''using CGT's new DAOPHOT results to measure LFC shifts; anchor using the wl-solution from the master ARC that night'''

    # TODO: IDEA: instead of reading P_id for the LFC, do a divide into orders and fit new x-y-traces
    # TODO: use meansep - get_mean_fibre_separation instead of 1.97

    # make sure a date is provided
    assert date is not None, 'ERROR: "date" not provided'
    year = str(date)[:4]
    
    if laptop:
        raw_path = '/Users/christoph/data/raw_goodonly/' + date + '/'
        red_path = '/Users/christoph/data/reduced/' + date + '/'
        lfc_path = '/Users/christoph/data/lfc_peaks/'
    else:
        raw_path = '/Volumes/BERGRAID/data/veloce/raw_goodonly/' + date + '/'
        red_path = '/Volumes/BERGRAID/data/veloce/reduced/' + date + '/'
        lfc_path = '/Volumes/BERGRAID/data/veloce/lfc_peaks/'
        
    # read in LFC vac wls (f0 = 9.56 GHz   &   f_rep = 25 GHz)    
    lfc_vac_wls = np.squeeze(np.array(readcol('/Users/christoph/OneDrive - UNSW/dispsol/lfc_vac_wls_nm.txt', twod=False))) * 10.    # in Angstroms
    
    if timit:
        start_time = time.time()

    # load default coefficients
    pixfit_coeffs = np.load('/Users/christoph/OneDrive - UNSW/dispsol_tests/20180917/pixfit_coeffs_relto_' + relto + '_as_of_2018-11-14.npy').item()

    # fibslot = [(0),1,2,   3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,   22,23,24,(25)]
    fibname = ['S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09', '03', '08', '19', '02', 'S4', 'S3', 'S1']
    
    if not refit:
        offsets = 1.97 * np.array(range(3) + range(4,23) + range(24,26))[::-1]   # 1.97 pix is the median fib-to-fib separation
        print('WARNING: median fibre to fibre separation not correct!!!')

#     nfib = len(fibname)
    nfib = len(fibname) + 2   # we want to include the simcalib fibres, although they're not used in the pixfit_coeffs (LFC wl-sol is saved, simThXe is not)

    # what kind of arc lamp is it?
    if date == '20180917':
        lamptype = 'thar'
    else:
        lamptype = 'thxe'
        
    # read master ARC dispsol for that night
    assert (os.path.isfile(red_path + lamptype + '_dispsol_' + date + '.fits')) or (os.path.isfile(raw_path + lamptype + '_dispsol_' + date + '.fits')), 'ERROR: master ARC wl-solution does not exist for ' + str(date) + ' !!!'
    try:
        # air_wl = pyfits.getdata(red_path + lamptype + '_dispsol_' + date + '.fits', 0)
        vac_wl = pyfits.getdata(red_path + lamptype + '_dispsol_' + date + '.fits', 1)
    except:
        # air_wl = pyfits.getdata(raw_path + lamptype + '_dispsol_' + date + '.fits', 0)
        vac_wl = pyfits.getdata(raw_path + lamptype + '_dispsol_' + date + '.fits', 1)
#         vac_wl = pyfits.getdata('/Volumes/BERGRAID/data/veloce/reduced/20190621/thxe_dispsol_20190621.fits', 1)
            
    # read file containing LFC peak positions of observation
#     _, yref, xref, _, _, _, _, _, _, _, _ = readcol(lfc_path + '21sep30019olc.nst', twod=False, skipline=2)
    try:
        _, y, x, _, _, _, _, _, _, _, _ = readcol(lfc_path + 'all/' + year + '/' + obsname + 'olc.nst', twod=False, skipline=2)
    except:
        _, y, x, _, _, _, _, _, _ = readcol(lfc_path + 'all/' + year + '/' + obsname + 'olc.nst', twod=False, skipline=2)
    del _
#     xref = nx - xref
    x = nx - x
#     yref = yref - 54.   # cut off 53 overscan pixels that DAOPHOT includes + 1 pixel b/c it starts indexing from 1
    y = y - 54.         # cut off 53 overscan pixels that DAOPHOT includes + 1 pixel b/c it starts indexing from 1
    
    # some housekeeping...
    xx = np.arange(nx)

    if refit:
        # divide LFC peak positions into orders
        peaks = divide_lfc_peaks_into_orders(x, y)
    else:
        print('FEHLER: muss ich erst noch machen')
#         # get affine transformation matrix
#         #     if   x' = x * M
#         # # then   x  = x' * M_inv
#         Minv = find_affine_transformation_matrix(xref, yref, x, y, timit=True, eps=2.)
#         # also get the rough trace of the LFC fibre (needed below)
#         pid = np.load(lfc_path + 'lfc_P_id.npy').item()

    # prepare outputs
    wldict = {}
    auxdata = {}
    # wl = np.zeros((n_ord, nfib, nx))
    wl = np.zeros((40, nfib, nx))
     
    # loop over all orders
    for o in range(vac_wl.shape[0]):
        
        # some housekeeping...
        ord = 'order_'+str(o+1).zfill(2)
        if debug_level > 0:
            print(ord)
        wldict[ord] = {}
        auxdata[ord] = {}
        
        # divide DAOPHOT LFC peaks into orders
#         peaks = divide_lfc_peaks_into_orders(x, y)
        ord_x = np.array(peaks[ord])[:,0]
        ord_y = np.array(peaks[ord])[:,1]
        ord_y = ord_y[ord_x.argsort()]
        ord_x = ord_x[ord_x.argsort()]
        
        # this just recovers the 7th order polynomial used in the 2-dim ThXe dispsol for this order (almost perfect)
        ord_wlfit = np.poly1d(np.polyfit(xx, vac_wl[o,-2,:], degpol))
        peak_thxe_wls = ord_wlfit(ord_x)
        
        # find the nearest entry in the array of theoretical wavelengths for each peak based in the ThXe dispsol
        theo_wls = np.array([find_nearest(lfc_vac_wls, ord_wlfit(peak_x)) for peak_x in ord_x])
        
        if norm_coords:
            # go to normalized co-ordinates (constrain the x-values to [0,1])
            ord_x_norm = (ord_x / (nx-1))
#             ord_xx_norm = (xx.astype(float) / (nx-1)) 
            ord_lam_norm = (theo_wls - np.min(theo_wls)) / (np.max((theo_wls - np.min(theo_wls)))) 
            new_fit = np.poly1d(np.polyfit(ord_x_norm, theo_wls, degpol))
            # residuals in Angstromgs
            wlres = theo_wls - new_fit(ord_x_norm)
            # fit the inverse problem as well, so that we can estimate the RMS in pixels (note that I subtract min(theo_wls) so as to "normalize" wls for the fit (otherwise it goes haywire numreically)
            inv_new_fit = np.poly1d(np.polyfit(ord_lam_norm, ord_x, degpol))
            # residuals in pixels
            pixres = ord_x - inv_new_fit(ord_lam_norm)
        else:   
            # make new wl-vs-x-fit and reject outliers
            new_fit = np.poly1d(np.polyfit(ord_x, theo_wls, degpol))
            # residuals in Angstromgs
            wlres = theo_wls - new_fit(ord_x)
            # fit the inverse problem as well, so that we can estimate the RMS in pixels (note that I subtract min(theo_wls) so as to "normalize" wls for the fit (otherwise it goes haywire numreically)
            inv_new_fit = np.poly1d(np.polyfit(theo_wls - np.min(theo_wls), ord_x, degpol))
            # residuals in pixels
            pixres = ord_x - inv_new_fit(theo_wls - np.min(theo_wls))
        
        # remove outliers and refit
        clipped, goodboolix, goodix, badix = single_sigma_clip(wlres, 3, return_indices=True)
        new_outies = len(badix)
        while new_outies > 0:
            ord_x = ord_x[goodboolix]
            ord_y = ord_y[goodboolix]
            theo_wls = theo_wls[goodboolix]
            if norm_coords:
                ord_x_norm = (ord_x / (nx-1))
                ord_lam_norm = (theo_wls - np.min(theo_wls)) / (np.max((theo_wls - np.min(theo_wls)))) 
                new_fit = np.poly1d(np.polyfit(ord_x_norm, theo_wls, degpol))
                inv_new_fit = np.poly1d(np.polyfit(ord_lam_norm, ord_x, degpol))
                wlres = theo_wls - new_fit(ord_x_norm)
                pixres = ord_x - inv_new_fit(ord_lam_norm)
            else:
                new_fit = np.poly1d(np.polyfit(ord_x, theo_wls, degpol))
                inv_new_fit = np.poly1d(np.polyfit(theo_wls - np.min(theo_wls), ord_x, degpol))
                wlres = theo_wls - new_fit(ord_x)
                pixres = ord_x - inv_new_fit(theo_wls - np.min(theo_wls))
            clipped, goodboolix, goodix, badix = single_sigma_clip(wlres, 3, return_indices=True)
            new_outies = len(badix)
        
        # if lam = f(x), and x = g(lam), then it should hold true that x = g(f(x)), or equivalently, lam = f(g(lam))
        
        # save data on LFC fit to output variables
        if return_auxdata:
            auxdata[ord]['wlres'] = wlres
            auxdata[ord]['pixres'] = pixres
            auxdata[ord]['x'] = ord_x
            auxdata[ord]['y'] = ord_y
            auxdata[ord]['peak_wl'] = theo_wls
            auxdata[ord]['rms_wl'] = np.std(wlres)
            auxdata[ord]['rms_pix'] = np.std(pixres)
            auxdata[ord]['n_peaks'] = len(ord_x)
        newly_fit_dispsol = new_fit(xx)
        wldict[ord]['laser'] = newly_fit_dispsol  
        wl[o,-1,:] = newly_fit_dispsol

        # now apply the fibre-to-fibre shifts
        # loop over all fibres
        for i,fib in enumerate(fibname):
            if fibtofib:
                xfib = ord_x + pixfit_coeffs[ord]['fibre_'+fib](ord_x)
            else:
                xfib = ord_x.copy()
            fib_fit = np.poly1d(np.polyfit(xfib, theo_wls, degpol))
            wldict[ord][fib] = fib_fit(xx)
            wl[o,i+1,:] = fib_fit(xx)       # note the i+1 because wl includes the two simcalib fibres, but we don't have pixfit_coeffs for these

    if timit:
        print('Time elapsed: ' + str(np.round(time.time() - start_time, 1)) + ' seconds')

    # for ord in sorted(auxdata.keys()):
    #     plt.hlines(int(ord[-2:]), xmin=0, xmax=4111, color='gray', linestyle='--')
    #     plt.plot(auxdata[ord]['x'], int(ord[-2:]) + auxdata[ord]['pixres'], 'b.')
    #     plt.xlabel('pixel (dispersion direction)')
    #     plt.ylabel('Order number  |  residuals in dispersion direction (pix)')
    #     #         plt.text(4112, int(ord[-2:]), 'RMS = '+str(np.round(auxdata[ord]['rms_pix'],3))+' pix', verticalalignment='center')
    #     plt.text(4112, int(ord[-2:]), 'RMS = {0:.3f} pix'.format(auxdata[ord]['rms_pix']), verticalalignment='center')
    #     plt.xlim(-150, 4700)

    if return_auxdata:
        return wldict, wl, auxdata
    else:
        return wldict, wl





def make_master_fibth(date=None, savefile=True, overwrite=False, laptop=False):
    
    # make sure we have an existing date
    if date is None:
        ok=0
        while ok == 0:
            date = raw_input("Please enter date of observations (YYYYMMDD): ")
            # check if file directory exists
            if os.path.isdir("/Volumes/BERGRAID/data/veloce/reduced/" + date + "/"):
                ok = 1
    
    if laptop:
        path = "/Users/christoph/data/reduced/" + date + "/"
    else:
        path = "/Volumes/BERGRAID/data/veloce/reduced/" + date + "/"
                
    # list of all Fibre Thoriums (ThAr for 20180917; ThXe for all later nights)
    arc_list = glob.glob(path + 'ARC*optimal*')
    if len(arc_list) == 0:
        print('WARNING: no Fibre Thorium (ARC) exposures found for '+date+' !!!')
        return (-1,-1)
    
    # prepare arrays
    allspec = []
    allerr = []
    # get lists of the reduced spectra and respective error arrays
    for i,fn in enumerate(arc_list):
        allspec.append(pyfits.getdata(fn,0))
        allerr.append(pyfits.getdata(fn,1))
            
    # list of individual exposure times for all whites (should all be the same, but just in case...)
    texp_list = [pyfits.getval(fn, 'ELAPSED') for fn in arc_list]
    # scale to the median exposure time
    tscale = np.array(texp_list) / np.median(texp_list)
                
    # median spectrum                
    medspec = np.nanmedian(np.array(allspec) / tscale.reshape(len(allspec), 1, 1, 1), axis=0)
    n_arc = len(arc_list)
    err_medspec = 1.253 * np.std(allspec, axis=0) / np.sqrt(n_arc-1)     # normally it would be sigma/sqrt(n), but np.std is dividing by sqrt(n), not by sqrt(n-1)
    
    if savefile:
        pyfits.writeto(path + 'master_ARC_' + date + '.fits', medspec, clobber=overwrite)
        pyfits.append(path + 'master_ARC_' + date + '.fits', err_medspec, clobber=overwrite)
    
    return medspec, err_medspec





def make_arc_dispsols(date, deg_spectral=7, deg_spatial=7, polytype='chebyshev', savetable=False, savefits=True, overwrite=False, 
                      save_individual=False, laptop=False, debug_level=0, timit=False):

    """
    for making the one per night fibre ThAr/ThXe dispsols
    """

    if timit:
        start_time = time.time()

    # fibslot = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25]
    fibname = ['simth', 'S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09',
               '03', '08', '19', '02', 'S4', 'S3', 'S1', 'LFC']

    print('Creating ARC wavelength solution for ' + date + ' ...')

    if laptop:
        path = "/Users/christoph/data/reduced/" + date + "/"
    else:
        path = "/Volumes/BERGRAID/data/veloce/reduced/" + date + "/"
    
    # check if wl-solution file already exists
    if os.path.isfile(path + 'master_ARC_' + date + '.fits'):
        master_arc = pyfits.getdata((path + 'master_ARC_' + date + '.fits'))
    else:
        # make master arc for that night
        master_arc, err_master_arc = make_master_fibth(date=date, savefile=True, laptop=laptop)
#     # assert master_arc != [], 'ERROR: No FibThs found for '+date
#     if master_arc == []:
#         print('ERROR: No FibThs found for ' + date)
#         return -1

    # prepare output array
    all_air_wl = np.zeros(master_arc.shape)
    all_vac_wl = np.zeros(master_arc.shape)

    # loop over all fibres (excluding the simTh & LFC fibre)
    for fib in range(1,master_arc.shape[1]-1):     

        print('Processing fibre ' + fibname[fib])

        # spectrum for that particular fibre
        thflux = master_arc[:, fib, :]

        # what kind of arc lamp is it?
        if date == '20180917':
            lamptype = 'thar'
        else:
            lamptype = 'thxe'

        # calculate wl-solution
        air_wl, vac_wl = get_dispsol_from_known_lines(thflux, fibre=fibname[fib], lamptype=lamptype, deg_spectral=deg_spectral,
                                                      deg_spatial=deg_spatial, polytype=polytype, return_full=True,
                                                      savetable=savetable, debug_level=debug_level, timit=timit)

        # save to individual-fibre fits file(s)
        if savefits:
            # save to all-fibres-combined array
            all_air_wl[:, fib, :] = air_wl[:-1, :].copy()  # do NOT include the blue-most order
            all_vac_wl[:, fib, :] = vac_wl[:-1, :].copy()
            if save_individual:
                pyfits.writeto(path + lamptype + '_dispsol_' + date + '_fibre_' + fibname[fib] + '.fits', air_wl, clobber=overwrite)
                pyfits.append(path + lamptype + '_dispsol_' + date + '_fibre_' + fibname[fib] + '.fits', vac_wl)

    # save to all-fibres-combined fits file
    if savefits:
        pyfits.writeto(path + lamptype + '_dispsol_' + date + '.fits', all_air_wl, clobber=overwrite)
        pyfits.append(path + lamptype + '_dispsol_' + date + '.fits', all_vac_wl)

    if timit:
        delta_t = time.time() - start_time
        print('Time taken for creating ARC dispsol: ' + str(delta_t) + ' seconds')

    return 1





def make_arc_dispsols_for_all_nights(outpath='/Users/christoph/OneDrive - UNSW/dispsol/arc_dispsols/', deg_spectral=7, deg_spatial=7,
                                     polytype='chebyshev', savetable=False, savefits=True, overwrite=False, save_individual=False):
    
    # get list of all nights
    datedir_list = glob.glob('/Volumes/BERGRAID/data/veloce/reduced/2019*')
    datedir_list.sort()

#     fibslot = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25]
    fibname = ['S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09', '03', '08', '19', '02', 'S4', 'S3', 'S1']

    # loop over all nights
    for datedir in datedir_list:
        datedir += '/'
        date = datedir[-9:-1]
        print('Creating ARC wavelength solution for ' + date + ' ...')
        
        # make master arc for that night
        master_arc = make_master_fibth(date=date)
        if master_arc == []:
            continue
        
        # prepare output array
        all_air_wl = np.zeros(master_arc.shape)
        
        # loop over all fibres
        for fib in range(master_arc.shape[1]):
            
            print('Processing fibre ' + fibname[fib])
            
            # spectrum for that particular fibre
            thflux = master_arc[:,fib,:]
            
            # what kind of arc lamp is it?
            if date == '20180917':
                lamptype = 'thar'
            else:
                lamptype = 'thxe'
            
            # calculate wl-solution    
            air_wl, vac_wl = get_dispsol_from_known_lines(thflux, lamptype=lamptype, deg_spectral=deg_spectral, deg_spatial=deg_spatial,
                                                          polytype=polytype, return_full=True, savetable=savetable, debug_level=0, timit=False)
            
            # save to all-fibres-combined array
            all_air_wl[:,fib,:] = air_wl[:-1,:].copy()     # do NOT include the blue-most order
            
            # save to individual-fibre fits file(s)
            if savefits:
                if save_individual:
                    pyfits.writeto(outpath + lamptype + '_dispsol_' + date + '_fibre_' + fibname[fib] + '.fits', air_wl, clobber=overwrite)
                
        # save to all-fibres-combined fits file
        if savefits:
            pyfits.writeto(outpath + lamptype + '_dispsol_' + date + '.fits', all_air_wl, clobber=overwrite)
        
    return





def get_dispsol_for_all_fibs_from_fibth(fn, date=None, path=None, deg_spectral=7, deg_spatial=7, polytype='chebyshev', savefits=False, debug_level=0, timit=False):
        
    if timit:
        start_time = time.time()    
    
#     assert date is not None, 'ERROR: date not provided!!!'
#     if savefits:
#         assert path is not None, 'ERROR: cannot save to file: output directory ("path") not provided!!!'

    if date is None:
        date = fn.split('/')[-2]
    if path is None:
        path = fn.split(date)[0] + date + '/'
            
    spec = pyfits.getdata(fn)
    
    # only want the 24 fibres for which we have the fibre ThXe
    if spec.shape[1] == 24:
        userange = np.arange(24)
    elif spec.shape[1] == 26:
        userange = np.arange(1,25)
        
    # fibslot = [0,1,  3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,  24,25]
    fibname = ['ThXe', 'S5', 'S2', '07', '18', '17', '06', '16', '15', '05', '14', '13', '01', '12', '11', '04', '10', '09',
               '03', '08', '19', '02', 'S4', 'S3', 'S1', 'LC']

    # prepare output arrays
    all_air_wl = np.zeros(spec.shape)
    all_vac_wl = np.zeros(spec.shape)

    # loop over all fibres
    for fib in userange:
        
        if debug_level > 0:
            print('Processing fibre ' + fibname[fib])

        # spectrum for that particular fibre
        thflux = spec[:, fib, :]

        # what kind of arc lamp is it?
        if date == '20180917':
            lamptype = 'ThAr'
        else:
            lamptype = 'ThXe'

        # calculate wl-solution
        air_wl, vac_wl = get_dispsol_from_known_lines(thflux, fibre=fibname[fib], lamptype=lamptype, deg_spectral=deg_spectral,
                                                      deg_spatial=deg_spatial, polytype=polytype, return_full=True,
                                                      savetable=False, timit=timit)

        # save to all-fibres-combined array
        all_air_wl[:, fib, :] = air_wl[:-1, :].copy()  # do NOT include the blue-most order
        all_vac_wl[:, fib, :] = vac_wl[:-1, :].copy()  # do NOT include the blue-most order
        
#         if save_individual:
#             pyfits.writeto(outpath + lamptype + '_dispsol_' + date + '_fibre_' + fibname[fib] + '.fits', air_wl, clobber=overwrite)
    

    # save to all-fibres-combined fits file
    if savefits:
        shortname = fn.split('/')[-1]
        obsname = shortname.split('_')[-3]
        obj = 'ARC_'
        pyfits.writeto(path + obj + lamptype + '_' + obsname + '_air_dispsol_' + date + '.fits', all_air_wl, clobber=True)
        pyfits.writeto(path + obj + lamptype + '_' + obsname + '_vac_dispsol_' + date + '.fits', all_vac_wl, clobber=True)
    
    if timit:
        delta_t = time.time() - start_time
        print('Time taken for creating ARC dispsol for all fibres: ' + str(np.round(delta_t,1)) + ' seconds')
    
    return all_air_wl, all_vac_wl





def interpolate_dispsols(wl1, wl2, t1, t2, tobs):
    
    assert wl1.shape == wl2.shape, 'ERROR: wl1 and wl2 do not have the same dimensions!!!'
    assert t1 <= t2, 'ERROR: t2 is not greater than or equal to t1'
    assert (tobs >= t1) and (tobs <= t2), 'ERROR: tobs does not lie between t1 and t2'
    if t1 == t2:
        assert wl1 == wl2, 'ERROR: t1 is equal to t2, but the wl-solutions are not the same!!!'
        return wl1
    
    #prepare output array
    interp_wl = np.zeros(wl1.shape)
    
    # difference array
    delta_wl = wl2 - wl1
    
    # total time between the two reference wl-solutions
    delta_t = t2 - t1
    
    # fractional time ([0...1]) that is the target for interpolation
    frac = (tobs - t1) / delta_t
    
    # now linearly interpolate between wl1 and wl2 to tobs
    interp_wl = wl1 + frac * delta_wl
    
    return interp_wl







def WIP():
    # OLDER CODE SNIPPETS
    #
    # THE following routine is now called fit_poly_surface_2D and lives in "helper_functions"
    # def fit_dispsol_2D(x_norm, ord_norm, WL, weights=None, polytype = 'chebyshev', poly_deg=5, debug_level=0):
    #     """
    #     Calculate 2D polynomial wavelength fit to normalized x and order values.
    #     Wrapper function for using the astropy fitting library.
    #
    #     'x_norm'   : x-values (pixels) of all the lines, re-normalized to [-1,+1]
    #     'ord_norm' : order numbers of all the lines, re-normalized to [-1,+1]
    #     'WL'       :
    #     'polytype' : either '(p)olynomial' (default), '(l)egendre', or '(c)hebyshev' are accepted
    #     """
    #
    #     if polytype in ['Polynomial','polynomial','p','P']:
    #         p_init = models.Polynomial2D(poly_deg)
    #         if debug_level > 0:
    #             print('OK, using standard polynomials...')
    #     elif polytype in ['Chebyshev','chebyshev','c','C']:
    #         p_init = models.Chebyshev2D(poly_deg,poly_deg)
    #         if debug_level > 0:
    #             print('OK, using Chebyshev polynomials...')
    #     elif polytype in ['Legendre','legendre','l','L']:
    #         p_init = models.Legendre2D(poly_deg,poly_deg)
    #         if debug_level > 0:
    #             print('OK, using Legendre polynomials...')
    #     else:
    #         print("ERROR: polytype not recognised ['(P)olynomial' / '(C)hebyshev' / '(L)egendre']")
    #
    #     fit_p = fitting.LevMarLSQFitter()
    #
    #     with warnings.catch_warnings():
    #         # Ignore model linearity warning from the fitter
    #         warnings.simplefilter('ignore')
    #         p = fit_p(p_init, x_norm, ord_norm, WL, weights=weights)
    #
    #
    # #     if debug_level > 0:
    # #         plt.figure()
    # #         index_include = np.array(weights, dtype=bool)
    # #         plt.scatter(x_norm[index_include], WL[index_include], c=order_norm[index_include])
    # #         plt.scatter(x_norm[np.logical_not(index_include)], WL[np.logical_not(index_include)], facecolors='none',
    # #                     edgecolors='r')
    # #
    # #         for x, o, oo, wl in zip(x_norm[index_include], order_norm[index_include], orders[index_include],
    # #                                 WL[index_include]):
    # #             plt.arrow(x, wl, 0, (p(x, o) / oo - wl) * 1000., head_width=0.00005, head_length=0.0001, width=0.00005)
    # #
    # #         xi = np.linspace(min(x_norm[index_include]), max(x_norm[index_include]), 101)
    # #         yi = np.linspace(min(order_norm[index_include]), max(order_norm[index_include]), 101)
    # #         zi = griddata((x_norm[index_include], order_norm[index_include]),
    # #                       ((WL[index_include] - p(x_norm[index_include], order_norm[index_include]) / orders[
    # #                           index_include]) / np.mean(WL[index_include])) * 3e8,
    # #                       (xi[None, :], yi[:, None]), method='linear')
    # #         fig, ax = plt.subplots()
    # #         ax.set_xlim((np.min(xi), np.max(xi)))
    # #         ax.set_ylim((np.min(yi), np.max(yi)))
    # #         ax.set_xlabel('Detector x normalized')
    # #         ax.set_ylabel('order normalized')
    # #         plt.title('Legendre Polynomial Degree: ' + str(poly_deg) + "\n" + "#pars: " + str(len(p.parameters)))
    # #
    # #         im = ax.imshow(zi, interpolation='nearest', extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)])
    # #
    # #         divider = make_axes_locatable(ax)
    # #         cax = divider.append_axes("right", size="5%", pad=0.05)
    # #
    # #         cb = plt.colorbar(im, cax=cax)
    # #         cb.set_label('RV deviation [m/s]')
    # #
    # #         plt.tight_layout()
    #
    #     return p

    #
    #
    #
    # thardata = np.load('/Users/christoph/OneDrive - UNSW/rvtest/thardata.npy').item()
    # laserdata = np.load('/Users/christoph/OneDrive - UNSW/rvtest/laserdata.npy').item()
    # thdata = thardata['flux']['order_01']
    # ldata = laserdata['flux']['order_01']
    #
    #
    # dispsol = np.load('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/thar_dispsol.npy').item()
    # OR
    # dispsol = np.load('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/thar_dispsol_2D.npy').item()
    # fitshapes = np.load('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/fitshapes.npy').item()
    # wavelengths = np.load('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/wavelengths.npy').item()
    # wl_ref = np.load('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/wl_ref.npy').item()
    # x = np.array([])
    # m = np.array([])
    # wl = np.array([])
    # wl_ref_arr = np.array([])
    # for ord in sorted(dispsol.keys()):
    #     ordnum = int(ord[-2:])
    #     x = np.append(x,fitshapes[ord]['x'])
    #     order = np.append(order, np.repeat(ordnum,len(fitshapes[ord]['x'])))
    #     wl = np.append(wl, dispsol[ord](fitshapes[ord]['x']))
    #     wl_ref_arr = np.append(wl_ref_arr, wl_ref[ord])
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # # ###########################################
    # # thar_refwlord01, thar_relintord01, flag = readcol('/Users/christoph/OneDrive - UNSW/linelists/test_thar_list_order_01.dat',fsep=';',twod=False)
    # # thar_refwlord01 *= 1e3
    # # refdata = {}
    # # refdata['order_01'] = {}
    # # refdata['order_01']['wl'] = thar_refwlord01[np.argwhere(flag == ' resolved')][::-1]          #note the array is turned around to match other arrays
    # # refdata['order_01']['relint'] = thar_relintord01[np.argwhere(flag == ' resolved')][::-1]     #note the array is turned around to match other arrays
    # # ###########################################
    #
    #
    #
    #
    #
    # def get_dispsol_from_thar(thardata, refdata, deg_polynomial=5, timit=False, verbose=False):
    #     """
    #     NOT CURRENTLY USED
    #     """
    #     if timit:
    #         start_time = time.time()
    #
    #     thar_dispsol = {}
    #
    #     #loop over all orders
    #     #for ord in sorted(thardata['flux'].iterkeys()):
    #     for ord in ['order_01']:
    #
    #         if verbose:
    #             print('Finding wavelength solution for '+str(ord))
    #
    #         #find fitted x-positions of ThAr peaks
    #         fitted_thar_pos, thar_qualflag = fit_emission_lines(thardata['flux'][ord], return_all_pars=False, return_qualflag=True, varbeta=False)
    #         x = fitted_thar_pos.copy()
    #
    #         #these are the theoretical wavelengths from the NIST linelists
    #         lam = (refdata[ord]['wl']).flatten()
    #
    #         #exclude some peaks as they are a blend of multiple lines: TODO: clean up
    #         filt = np.ones(len(fitted_thar_pos),dtype='bool')
    #         filt[[33,40,42,58,60]] = False
    #         x = x[filt]
    #
    #         #fit polynomial to lambda as a function of x
    #         thar_fit = np.poly1d(np.polyfit(x, lam, deg_polynomial))
    #         #save to output dictionary
    #         thar_dispsol[ord] = thar_fit
    #
    #     if timit:
    #         print('Time taken for finding ThAr wavelength solution: '+str(time.time() - start_time)+' seconds...')
    #
    #     return thar_dispsol
    #
    #
    # # '''xxx'''
    # # #################################################################
    # # # the following is needed as input for "get_dispsol_from_laser" #
    # # #################################################################
    # # laser_ref_wl,laser_relint = readcol('/Users/christoph/OneDrive - UNSW/linelists/laser_linelist_25GHz.dat',fsep=';',twod=False)
    # # laser_ref_wl *= 1e3
    # #
    # # #wavelength solution from HDF file
    # # #read dispersion solution from file
    # # dispsol = np.load('/Users/christoph/OneDrive - UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()
    # # #read extracted spectrum from files (obviously this needs to be improved)
    # # xx = np.arange(4096)
    # # #this is so as to match the order number with the physical order number (66 <= m <= 108)
    # # # order01 corresponds to m=66
    # # # order43 corresponds to m=108
    # # wl = {}
    # # for ord in dispsol.keys():
    # #     m = ord[5:]
    # #     ordnum = str(int(m)-65).zfill(2)
    # #     wl['order_'+ordnum] = dispsol['order'+m]['model'](xx)
    #
    #
    #
    #
    #
    # def get_dispsol_from_laser(laserdata, laser_ref_wl, deg_polynomial=5, timit=False, verbose=False, return_stats=False, varbeta=False):
    #     """
    #     NOT CURRENTLY USED
    #     """
    #
    #     if timit:
    #         start_time = time.time()
    #
    #     if return_stats:
    #         stats = {}
    #
    #     #read in mask for fibre_01 (ie the Laser-comb fibre) from order_tracing as a first step in excluding low-flux regions
    #     mask_01 = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/masks/mask_01.npy').item()
    #
    #     laser_dispsol = {}
    #
    #     #loop over all orders
    #     #order 43 does not work properly, as some laser peaks are missing!!!
    #     for ord in sorted(laserdata['flux'].iterkeys())[:-1]:
    #
    #         if verbose:
    #             print('Finding wavelength solution for '+str(ord))
    #
    #         #find fitted x-positions of ThAr peaks
    #         data = laserdata['flux'][ord] * mask_01[ord]
    #         goodpeaks,mostpeaks,allpeaks,first_mask,second_mask,third_mask = find_suitable_peaks(data,return_masks=True)    #from this we just want the masks this time (should be very fast)
    #         #fitted_laser_pos, laser_qualflag = fit_emission_lines(data, laser=True, return_all_pars=False, return_qualflag=True, varbeta=varbeta)
    #         if varbeta:
    #             fitted_laser_pos, fitted_laser_sigma, fitted_laser_amp, fitted_laser_beta = fit_emission_lines(data, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta, timit=timit, verbose=verbose)
    #         else:
    #             fitted_laser_pos, fitted_laser_sigma, fitted_laser_amp = fit_emission_lines(data, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta, timit=timit, verbose=verbose)
    #         x = fitted_laser_pos.copy()
    #         #exclude the leftmost and rightmost peaks (nasty edge effects...)
    # #         blue_cutoff = int(np.round((x[-1]+x[-2])/2.,0))
    # #         red_cutoff = int(np.round((x[0]+x[1])/2.,0))
    #         blue_cutoff = int(np.round(allpeaks[-1]+((allpeaks[-1] - allpeaks[-2])/2),0))
    #         red_cutoff = int(np.round(allpeaks[0]-((allpeaks[1] - allpeaks[0])/2),0))
    #         cond1 = (laser_ref_wl >= wl[ord][blue_cutoff])
    #         cond2 = (laser_ref_wl <= wl[ord][red_cutoff])
    #         #these are the theoretical wavelengths from the NIST linelists
    #         lam = laser_ref_wl[np.logical_and(cond1,cond2)][::-1]
    #         lam = lam[first_mask][second_mask][third_mask]
    #
    #         #check if the number of lines found equals the number of lines from the line list
    # #         if verbose:
    # #             print(len(x),len(lam))
    #         if len(x) != len(lam):
    #             print('fuganda')
    #             return 'fuganda'
    #
    #         #fit polynomial to lambda as a function of x
    #         laser_fit = np.poly1d(np.polyfit(x, lam, deg_polynomial))
    #
    #         if return_stats:
    #             stats[ord] = {}
    #             resid = laser_fit(x) - lam
    #             stats[ord]['resids'] = resid
    #             #mean error in RV for a single line = c * (stddev(resid) / mean(lambda))
    #             stats[ord]['single_rverr'] = 3e8 * (np.std(resid) / np.mean(lam))
    #             stats[ord]['rverr'] = 3e8 * (np.std(resid) / np.mean(lam)) / np.sqrt(len(lam))
    #             stats[ord]['n_lines'] = len(lam)
    #
    #         #save to output dictionary
    #         laser_dispsol[ord] = laser_fit
    #
    #
    #     #let's do order 43 differently because it has the stupid gap in the middle
    #     #find fitted x-positions of ThAr peaks
    #     ord = 'order_43'
    #     if verbose:
    #             print('Finding wavelength solution for '+str(ord))
    #     data = laserdata['flux'][ord] * mask_01[ord]
    #     data1 = data[:2500]
    #     data2 = data[2500:]
    #     goodpeaks1,mostpeaks1,allpeaks1,first_mask1,second_mask1,third_mask1 = find_suitable_peaks(data1,return_masks=True)    #from this we just want use_mask this time (should be very fast)
    #     goodpeaks2,mostpeaks2,allpeaks2,first_mask2,second_mask2,third_mask2 = find_suitable_peaks(data2,return_masks=True)    #from this we just want use_mask this time (should be very fast)
    #     #fitted_laser_pos1, laser_qualflag1 = fit_emission_lines(data1, laser=True, return_all_pars=False, return_qualflag=True, varbeta=varbeta)
    #     #fitted_laser_pos2, laser_qualflag2 = fit_emission_lines(data2, laser=True, return_all_pars=False, return_qualflag=True, varbeta=varbeta)
    #     if varbeta:
    #         fitted_laser_pos1, fitted_laser_sigma1, fitted_laser_amp1, fitted_laser_beta1 = fit_emission_lines(data1, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
    #         fitted_laser_pos2, fitted_laser_sigma2, fitted_laser_amp2, fitted_laser_beta2 = fit_emission_lines(data2, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
    #     else:
    #         fitted_laser_pos1, fitted_laser_sigma1, fitted_laser_amp1 = fit_emission_lines(data1, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
    #         fitted_laser_pos2, fitted_laser_sigma2, fitted_laser_amp2 = fit_emission_lines(data2, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
    #     x1 = fitted_laser_pos1.copy()
    #     x2 = fitted_laser_pos2.copy() + 2500
    #     #exclude the leftmost and rightmost peaks (nasty edge effects...)
    # #         blue_cutoff = int(np.round((x[-1]+x[-2])/2.,0))
    # #         red_cutoff = int(np.round((x[0]+x[1])/2.,0))
    #     blue_cutoff1 = int(np.round(allpeaks1[-1]+((allpeaks1[-1] - allpeaks1[-2])/2),0))
    #     blue_cutoff2 = int(np.round(allpeaks2[-1]+((allpeaks2[-1] - allpeaks2[-2])/2)+2500,0))
    #     red_cutoff1 = int(np.round(allpeaks1[0]-((allpeaks1[1] - allpeaks1[0])/2),0))
    #     red_cutoff2 = int(np.round(allpeaks2[0]-((allpeaks2[1] - allpeaks2[0])/2)+2500,0))
    #     cond1_1 = (laser_ref_wl >= wl[ord][blue_cutoff1])
    #     cond1_2 = (laser_ref_wl >= wl[ord][blue_cutoff2])
    #     cond2_1 = (laser_ref_wl <= wl[ord][red_cutoff1])
    #     cond2_2 = (laser_ref_wl <= wl[ord][red_cutoff2])
    #     #these are the theoretical wavelengths from the NIST linelists
    #     lam1 = laser_ref_wl[np.logical_and(cond1_1,cond2_1)][::-1]
    #     lam2 = laser_ref_wl[np.logical_and(cond1_2,cond2_2)][::-1]
    #     lam1 = lam1[first_mask1][second_mask1][third_mask1]
    #     lam2 = lam2[first_mask2][second_mask2][third_mask2]
    #
    #     x = np.r_[x1,x2]
    #     lam = np.r_[lam1,lam2]
    #
    #     #check if the number of lines found equals the number of lines from the line list
    #     if verbose:
    #         print(len(x),len(lam))
    #     if len(x) != len(lam):
    #         print('fuganda')
    #         return 'fuganda'
    #
    #     #fit polynomial to lambda as a function of x
    #     laser_fit = np.poly1d(np.polyfit(x, lam, deg_polynomial))
    #
    #     if return_stats:
    #         stats[ord] = {}
    #         resid = laser_fit(x) - lam
    #         stats[ord]['resids'] = resid
    #         #mean error in RV for a single line = c * (stddev(resid) / mean(lambda))
    #         stats[ord]['single_rverr'] = 3e8 * (np.std(resid) / np.mean(lam))
    #         stats[ord]['rverr'] = 3e8 * (np.std(resid) / np.mean(lam)) / np.sqrt(len(lam))
    #         stats[ord]['n_lines'] = len(lam)
    #
    #     #save to output dictionary
    #     laser_dispsol[ord] = laser_fit
    #
    #     if timit:
    #         print('Time taken for finding Laser-comb wavelength solution: '+str(time.time() - start_time)+' seconds...')
    #
    #     if return_stats:
    #         return laser_dispsol, stats
    #     else:
    #         return laser_dispsol
    #
    #
    #
    #
    #
    # ###########################################################################################################
    # # laser_dispsol2,stats2 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=2)
    # # laser_dispsol3,stats3 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=3)
    # # laser_dispsol5,stats5 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=5)
    # # laser_dispsol11,stats11 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=11)
    # ###########################################################################################################





    #     if new_ordnum == 2:
    #         mask = np.zeros(len(data), dtype='bool')
    #         mask[1,:3060:3140] = True
    #         data[mask] = 0
    #     if new_ordnum == 3:
    #         mask = np.zeros(len(data),dtype='bool')
    #         mask[2340:2590] = True
    #         data[mask] = 0
    #     if new_ordnum == 8:
    #         mask = np.zeros(len(data),dtype='bool')
    #         mask[1650:1870] = True
    #         mask[3550:3670] = True
    #         data[mask] = 0
    #     if new_ordnum == 9:
    #         mask = np.zeros(len(data),dtype='bool')
    #         mask[1180:1450] = True
    #         mask[1560:1830] = True
    #         data[mask] = 0
    #     if new_ordnum == 10:
    #         mask = np.zeros(len(data),dtype='bool')
    #         mask[2260:2420] = True
    #         data[mask] = 0
    #     if new_ordnum == 11:
    #         mask = np.zeros(len(data),dtype='bool')
    #         mask[2920:3500] = True
    #         data[mask] = 0
    #     if new_ordnum == 12:
    #         mask = np.zeros(len(data),dtype='bool')
    #         mask[500:1150] = True
    #         mask[2930:3260] = True
    #         data[mask] = 0
    #     if new_ordnum == 13:
    #         mask = np.zeros(len(data),dtype='bool')
    #         mask[475:850] = True
    #         mask[2120:2320] = True
    #         data[mask] = 0
    #     if new_ordnum == 15:
    #         mask = np.zeros(len(data), dtype='bool')
    #         mask[2640:2820] = True
    #         data[mask] = 0
    #     if new_ordnum == 16:
    #         mask = np.zeros(len(data), dtype='bool')
    #         mask[2380:2685] = True
    #         data[mask] = 0
    #     if new_ordnum == 17:
    #         mask = np.zeros(len(data),dtype='bool')
    #         mask[0:50] = True
    #         mask[3060:3160] = True
    #         mask[3265:3435] = True
    #         data[mask] = 0
    #     if new_ordnum == 18:
    #         mask = np.zeros(len(data),dtype='bool')
    #         mask[785:960] = True
    #         mask[1100:1275] = True
    #         data[mask] = 0
    #     if new_ordnum == 19:
    #         mask = np.zeros(len(data), dtype='bool')
    #         mask[1850:2050] = True
    #         data[mask] = 0
    #     if new_ordnum == 23:
    #         mask = np.zeros(len(data), dtype='bool')
    #         mask[1326:1420] = True
    #         data[mask] = 0
    #     if new_ordnum == 24:
    #         mask = np.zeros(len(data), dtype='bool')
    #         mask[1850:2050] = True
    #         data[mask] = 0
    #     if new_ordnum == 25:
    #         mask = np.zeros(len(data), dtype='bool')
    #         mask[1390:1765] = True
    #         data[mask] = 0
    
    # nlines = np.array([12, 25, 28, 30, 27, 34, 25, 17, 23, 22, 23, 21, 25, 28, 27, 33, 33, 30, 36, 30, 33, 36, 36, 25, 29, 33, 40, 35, 30, 23, 29, 29, 35, 28, 25, 41, 36, 40, 31, 21])
    
    return










