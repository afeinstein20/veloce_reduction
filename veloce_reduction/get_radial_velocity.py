'''
Created on 9 Nov. 2017

@author: christoph
'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as op
import time
from scipy import ndimage
from readcol import readcol

from veloce_reduction.veloce_reduction.helper_functions import xcorr, gausslike_with_amp_and_offset, gausslike_with_amp_and_offset_and_slope, central_parts_of_mask, spectres, cmb_scrunch, multi_fibmodel_with_amp_and_offset
from veloce_reduction.veloce_reduction.flat_fielding import deblaze_orders, onedim_pixtopix_variations



# #speed of light in m/s
# c = 2.99792458e8
# 
# #read dispersion solution from file
# dispsol = np.load('/Users/christoph/OneDrive - UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()
# 
# #read extracted spectrum from files (obviously this needs to be improved)
# xx = np.arange(4096)
# 
# #for simulated data only
# wl = get_simu_dispsol()
#     
# 
# #re-bin into logarithmic wavelength space for cross-correlation, b/c delta_log_wl = c * delta_v
# refdata = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/high_SNR_solar_template.txt', names=('pixnum','wl','flux','err'))
# # flux1 = data1['flux']
# # err1 = data1['err']
# #data2 = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/solar_red100ms.txt', names=('pixnum','wl','flux','err'))
# # flux2 = data2['flux']
# # err2 = data2['err']
# # obsdata = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/solar_0ms.txt', names=('pixnum','wl','flux','err'))
# # obsdata = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/solar_red100ms.txt', names=('pixnum','wl','flux','err'))
# obsdata = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/solar_red1000ms.txt', names=('pixnum','wl','flux','err'))
# # flux3 = data3['flux']
# # err3 = data3['err']
# flatdata = ascii.read('/Users/christoph/OneDrive - UNSW/rvtest/tramline_extracted_flat.txt', names=('pixnum','wl','flux','err'))
# # flatflux = flatdata['flux']
# # flaterr = flatdata['err']
# #wl1 = data1['wl']
# #wl2 = data2['wl']
# # wl = data1['wl']
# # pixnum = data1['pixnum']





# # f0 = template
# f0 = {}
# err0 = {}
# # f = observation
# f = {}
# err = {}
# # f_flat = master_white
# f_flat = {}
# err_flat = {}
# 
# ref_wl = {}
# obs_wl = {}
# flat_wl = {}
# for m in range(len(wl)):
#     ord = 'order_'+str(m+1).zfill(2)
#     f0[ord] = np.array(refdata['flux'][m*4096:(m+1)*4096])[::-1]              # need to turn around if using Zemax dispsol, otherwise np.interp fails
#     f[ord] = np.array(obsdata['flux'][m*4096:(m+1)*4096])[::-1]
#     f_flat[ord] = np.array(flatdata['flux'][m*4096:(m+1)*4096])[::-1]
#     err0[ord] = np.array(refdata['err'][m*4096:(m+1)*4096])[::-1]
#     err[ord] = np.array(obsdata['err'][m*4096:(m+1)*4096])[::-1]
#     err_flat[ord] = np.array(flatdata['err'][m*4096:(m+1)*4096])[::-1]
#     ref_wl[ord] = np.array(refdata['wl'][m*4096:(m+1)*4096])[::-1]            # units do not matter, as log(x) - log(y) = log(ax) - log(ay) = log(a) + log(x) - log(a) - log(y)
#     obs_wl[ord] = np.array(obsdata['wl'][m*4096:(m+1)*4096])[::-1]
#     flat_wl[ord] = np.array(flatdata['wl'][m*4096:(m+1)*4096])[::-1]
    
    

def get_rvs_from_xcorr(extracted_spectra, obsnames, mask, smoothed_flat, debug_level=0):
    """
    This is a wrapper for the actual RV routine "get_RV_from_xcorr", which is called for all observations within 'obsnames'.
    
    INPUT:
    'extracted_spectra'  : dictionary containing keys 'pix', 'wl', 'flux', and 'err' for every observation (each containing keys=orders), plus the template and the master white
    'obsnames'           : list containing the names of the observations
    'mask'               : dictionary of masks from 'find_stripes' (with the keys being the orders)
    'smoothed_flat'      : dictionary containing the smoothed master white for each order (with the keys being the orders)
    'debug_level'        : boolean - for debugging...
    
    OUTPUT:
    'rv'     : RVs (dictionary with 'obsnames' --> 'orders' as keys)
    'rverr'  : RV errors ((dictionary with 'obsnames' --> 'orders' as keys)
    """
    
    rv = {}
    rverr = {}
    
    f0 = extracted_spectra['template']['flux'].copy()
    wl0 = extracted_spectra['template']['wl'].copy()
    
    for obs in sorted(obsnames):
        #prepare arrays
        f = extracted_spectra[obs]['flux'].copy()
        err = extracted_spectra[obs]['err'].copy()
        wl = extracted_spectra[obs]['wl'].copy()
        
        #if using cross-correlation, we need to de-blaze the spectra first
        f_dblz, err_dblz = deblaze_orders(f, wl, smoothed_flat, mask, err=err)
        f0_dblz = deblaze_orders(f0, wl0, smoothed_flat, mask, err=None)
        
        #we also only want to use the central TRUE parts of the masks, ie want ONE consecutive stretch per order
        cenmask = central_parts_of_mask(mask)
        
        #call RV routine
        if debug_level >= 1:
            print('Calculating RV for observation: '+obs)
        rv[obs],rverr[obs] = get_RV_from_xcorr(f_dblz, err_dblz, wl, f0_dblz, wl0, mask=cenmask, filter_width=25, debug_level=0)

    return rv,rverr
  
    
    
def get_RV_from_xcorr(f, err, wl, f0, wl0, mask=None, smoothed_flat=None, osf=2, delta_log_wl=1e-6, relgrid=False,
                      filter_width=25, bad_threshold=0.05, simu=False, debug_level=0, timit=False):
    """
    This routine calculates the radial velocity of an observed spectrum relative to a template using cross-correlation. 
    Note that input spectra should be de-blazed already!!!
    If the mask from "find_stripes" has gaps, do the filtering for each segment independently. If no mask is provided, create a simple one on the fly.
    
    INPUT:
    'f'             : dictionary containing the observed flux (keys = orders)
    'err'           : dictionary containing the uncertainties in the observed flux (keys = orders)
    'wl'            : dictionary containing the wavelengths of the observed spectrum (keys = orders)
    'f0'            : dictionary containing the template (keys = orders) (errors assumed to be negligible)
    'wl0'           : dictionary containing the wavelengths of the template spectrum (keys = orders)
    'mask'          : mask-dictionary from "find_stripes" (keys = orders)
    'smoothed_flat' : if no mask is provided, need to provide the smoothed_flat, so that a mask can be created on the fly
    'osf'           : oversampling factor for the logarithmic wavelength rebinning (only used if 'relgrid' is TRUE)
    'delta_log_wl'  : stepsize of the log-wl grid (only used if 'relgrid' is FALSE)
    'relgrid'       : boolean - do you want to use an absolute stepsize of the log-wl grid, or relative using 'osf'?
    'filter_width'  : width of smoothing filter in pixels; needed b/c of edge effects of the smoothing; number of pixels to disregard should be >~ 2 * width of smoothing kernel  
    'bad_threshold' : if no mask is provided, create a mask that requires the flux in the extracted white to be larger than this fraction of the maximum flux in that order
    'simu'          : boolean - are you using ES simulated spectra? (only used if mask is not provided)
    'debug_level'   : boolean - for debugging...
    'timit'         : boolean - for timing the execution run time...
    
    OUTPUT:
    'rv'         : dictionary with the measured RVs for each order
    'rverr'      : dictionary with the uncertainties in the measured RVs for each order
    
    MODHIST:
    Dec 2017 - CMB create
    04/06/2018 - CMB fixed bug when turning around arrays (need to use new variable)
    28/06/2018 - CMB fixed bug with interpolation of log wls
    """
    
    if timit:
        start_time = time.time()
    
    # speed of light in m/s
    c = 2.99792458e8
    
    rv = {}
    rverr = {}
    
    # loop over orders
    for ord in sorted(f.iterkeys()):
        
        if debug_level >= 1:
            print(ord)
        
        # only use pixels that have enough signal
        if mask is None:
            normflat = smoothed_flat[ord]/np.max(smoothed_flat[ord])
            ordmask = np.ones(len(normflat), dtype = bool)
            if np.min(normflat) < bad_threshold:
                ordmask[normflat < bad_threshold] = False
                #once the blaze function falls below a certain value, exclude what's outside of that pixel column, even if it's above the threshold again, ie we want to only use a single consecutive region
                leftmask = ordmask[: len(ordmask)//2]
                leftkill_index = [i for i,x in enumerate(leftmask) if not x]
                try:
                    ordmask[: leftkill_index[0]] = False
                except:
                    pass
                rightmask = ordmask[len(ordmask)//2 :]
                rightkill_index = [i for i,x in enumerate(rightmask) if not x]
                if ord == 'order_01' and simu:
                    try:
                        #if not flipped then the line below must go in the leftkillindex thingy
                        #ordmask[: leftkill_index[-1] + 100] = False
                        ordmask[len(ordmask)//2 + rightkill_index[0] - 100 :] = False
                    except:
                        pass
                else:
                    try:
                        ordmask[len(mask)//2 + rightkill_index[-1] + 1 :] = False
                    except:
                        pass
        else:
            #ordmask  = mask[ord][::-1]
            ordmask  = mask[ord]
        
        # either way, disregard #(edge_cut) pixels at either end; this is slightly dodgy, but the gaussian filtering above introduces edge effects due to mode='reflect'
        ordmask[:2*int(filter_width)] = False
        ordmask[-2*int(filter_width):] = False
    
#         f0_unblazed = f0[ord] / np.max(f0[ord]) / normflat   #see COMMENT above
#         f_unblazed = f[ord] / np.max(f[ord]) / normflat
        
        #create logarithmic wavelength grid
        logwl = np.log(wl[ord])
        logwl0 = np.log(wl0[ord])
        if relgrid:
            logwlgrid = np.linspace(np.min(logwl[ordmask]), np.max(logwl[ordmask]), osf*np.sum(ordmask))
            delta_log_wl = logwlgrid[1] - logwlgrid[0]
            if debug_level >= 2:
                print(ord, ' :  delta_log_wl = ',delta_log_wl)
        else:
            logwlgrid = np.arange(np.min(logwl[ordmask]), np.max(logwl[ordmask]), delta_log_wl)
        
        #wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(logwl) < 0).any():
            logwl_sorted = logwl[::-1].copy()
            logwl0_sorted = logwl0[::-1].copy()
            ordmask_sorted = ordmask[::-1].copy()
            ord_f0_sorted = f0[ord][::-1].copy()
            ord_f_sorted = f[ord][::-1].copy()
        else:
            logwl_sorted = logwl.copy()
            logwl0_sorted = logwl0.copy()
            ordmask_sorted = ordmask.copy()
            ord_f0_sorted = f0[ord].copy()
            ord_f_sorted = f[ord].copy()
        
        # rebin spectra onto logarithmic wavelength grid
#         rebinned_f0 = np.interp(logwlgrid,logwl[mask],f0_unblazed[mask])
#         rebinned_f = np.interp(logwlgrid,logwl[mask],f_unblazed[mask])
        spl_ref_f0 = interp.InterpolatedUnivariateSpline(logwl0_sorted[ordmask_sorted], ord_f0_sorted[ordmask_sorted], k=3)    #slightly slower than linear, but best performance for cubic spline
        rebinned_f0 = spl_ref_f0(logwlgrid)
        spl_ref_f = interp.InterpolatedUnivariateSpline(logwl_sorted[ordmask_sorted], ord_f_sorted[ordmask_sorted], k=3)    #slightly slower than linear, but best performance for cubic spline
        rebinned_f = spl_ref_f(logwlgrid)
    
        # do we want to cross-correlate the entire order???
        #xc = np.correlate(rebinned_f0 - np.median(rebinned_f0), rebinned_f - np.median(rebinned_f), mode='same')
    #     #now this is slightly dodgy, but cutting off the edges works better because the division by the normflat introduces artefacts there
    #     if ord == 'order_01':
    #         xcorr_region = np.arange(2500,16000,1)
    #     else:
    #         xcorr_region = np.arange(2500,17500,1)
        
        xc = np.correlate(rebinned_f0 - np.median(rebinned_f0), rebinned_f - np.median(rebinned_f), mode='same')
        #now fit Gaussian to central section of CCF
        if relgrid:
            fitrangesize = osf*6    #this factor was simply eye-balled
        else:
            #fitrangesize = 30
            fitrangesize = int(np.round(0.0036 * len(xc) / 2. - 1,0))     #this factor was simply eye-balled
            
        xrange = np.arange(np.argmax(xc)-fitrangesize, np.argmax(xc)+fitrangesize+1, 1)
        #parameters: mu, sigma, amp, beta, offset, slope
        guess = np.array((np.argmax(xc), 0.0006 * len(xc), (xc[np.argmax(xc)]-xc[np.argmax(xc)-fitrangesize]), 2., xc[np.argmax(xc)-fitrangesize], 0.))
        #guess = np.array((np.argmax(xc), 5., (xc[np.argmax(xc)]-xc[np.argmax(xc)-fitrangesize]), 2., xc[np.argmax(xc)-fitrangesize], 0.))
        #guess = np.array((np.argmax(xc), 10., (xc[np.argmax(xc)]-xc[np.argmax(xc)-fitrangesize])/np.max(xc), 2., xc[np.argmax(xc)-fitrangesize]/np.max(xc), 0.))
        #popt, pcov = op.curve_fit(gaussian_with_offset_and_slope, xrange, xc[np.argmax(xc)-fitrangesize : np.argmax(xc)+fitrangesize+1]/np.max(xc[xrange]), p0=guess)
        #popt, pcov = op.curve_fit(gausslike_with_amp_and_offset_and_slope, xrange, xc[xrange]/np.max(xc[xrange]), p0=guess)
        popt, pcov = op.curve_fit(gausslike_with_amp_and_offset_and_slope, xrange, xc[xrange], p0=guess)
        mu = popt[0]
#         print(ord, f[ord][3000:3003])
#         print(ord, f[ord][::-1][3000:3003])
        mu_err = np.sqrt(pcov[0,0])
        #convert to RV in m/s
        rv[ord] = c * (mu - (len(xc)//2)) * delta_log_wl
        rverr[ord] = c * mu_err * delta_log_wl
    
    if timit:
        delta_t = time.time() - start_time
        print('Time taken for calculating RV: '+str(np.round(delta_t,2))+' seconds')
    
    return rv,rverr



def get_RV_from_xcorr_2(f, wl, f0, wl0, bc=0, bc0=0, mask=None, smoothed_flat=None, delta_log_wl=1e-6, addrange=150, fitrange=35, npeaks=1,
                        taper=True, taper_width=0.05, flipped=False, deg_interp=3, individual_fibres=True, individual_orders=True,
                        scrunch=False, fit_slope=False, norm_cont=False, synthetic_template=False, old_ccf=False, debug_level=0, timit=False):
    """
    This routine calculates the radial velocity of an observed spectrum relative to a template using cross-correlation.
    Note that input spectra should be de-blazed already!!!
    If the mask from "find_stripes" has gaps, do the filtering for each segment independently. If no mask is provided, create a simple one on the fly.

    INPUT:
    'f'             : numpy array containing the observed flux (n_ord, n_fib, n_pix)
    'wl'            : numpy array containing the wavelengths of the observed spectrum (n_ord, n_fib, n_pix)
    'f0'            : numpy array containing the flux of the template spectrum (n_ord, n_fib, n_pix)
    'wl0'           : numpy array containing the wavelengths of the template spectrum (n_ord, n_fib, n_pix)
    'bc'            : the barycentric velocity correction for the observed spectrum
    'bc0'           : the barycentric velocity correction for the template spectrum
    'mask'          : mask-dictionary from "find_stripes" (keys = orders)
    'smoothed_flat' : if no mask is provided, need to provide the smoothed_flat, so that a mask can be created on the fly
    'delta_log_wl'  : stepsize of the log-wl grid
    'addrange'      : the central (2*addrange + 1) pixels of the CCFs will be added
    'fitrange'      : a Gauss-like function will be fitted to the central (2*fitrange + 1) pixels
    'npeaks'        : the number of peaks to be fitted in the central region of the CCF (essentially that's the number of stars, but allows room for experiments)
    'taper'         : boolean - do you want to taper off the edges for the inputs to the xcorr-routine?
    'taper_width'   : fractional width/length of the input arrays to the xcorr-routine that will be tapered to zero at both ends
    'flipped'       : boolean - reverse order of inputs to xcorr routine?
    'deg_interp'         : the degree of the interpolation (1 = linear, 3 = cubic)
    'individual_fibres'  : boolean - do you want to return the RVs for individual fibres? (if FALSE, then the RV is calculated from the sum of the ind. fib. CCFs)
    'individual_orders'  : boolean - do you want to return the RVs for individual orders? (if FALSE, then the RV is calculated from the sum of the ind. ord. CCFs)
    'scrunch'            : boolean - do you want to scrunch the spectra or simply interpolate them when rebinning?
    'norm_cont'          : boolean - do you want to normalise the continuum?
    'synthetic_template' : boolean - are you using a synthetic template?
    'debug_level'   : boolean - for debugging...
    'timit'         : boolean - for timing the execution run time...

    OUTPUT:
    'rv'         : dictionary with the measured RVs for each order
    'rverr'      : dictionary with the uncertainties in the measured RVs for each order

    MODHIST:
    Dec 2017 - CMB create
    04/06/2018 - CMB fixed bug when turning around arrays (need to use new variable)
    28/06/2018 - CMB fixed bug with interpolation of log wls
    """

    assert delta_log_wl > 0, 'ERROR: "delta_log_wl" must ge greater than zero!!!'
    assert fitrange > 0 and fitrange.__class__ == int, 'ERROR: "fitrange" must be a positive integer'
    assert addrange > 0 and addrange.__class__ == int and addrange >= fitrange, 'ERROR: "addrange" must be a positive integer, and must be larger than "fitrange"'
    assert npeaks > 0 and npeaks.__class__ == int, 'ERROR: "npeaks" must be a positive integer'

    if timit:
        start_time = time.time()

    # speed of light in m/s
    c = 2.99792458e8

    # the dummy wavelengths for orders 'order_01' and 'order_40' cannot be zero as we're taking a log!!!
    if wl.shape[0] == 40:
        wl[0, :, :] = 1.
        wl[-1, :, :] = 1.
        if not synthetic_template:
            wl0[0, :, :] = 1.
            wl0[-1, :, :] = 1.
    elif wl.shape[0] == 39:
        wl[0, :, :] = 1.
        if not synthetic_template:
            wl0[0, :, :] = 1.

    # make cross-correlation functions (list of length n_orders used)
    if not old_ccf:
        xcs = make_ccfs(f, wl, f0, wl0, bc=bc, bc0=bc0, smoothed_flat=smoothed_flat, delta_log_wl=delta_log_wl, deg_interp=deg_interp, flipped=flipped,
                        individual_fibres=individual_fibres, scrunch=scrunch, synthetic_template=synthetic_template, norm_cont=norm_cont, debug_level=debug_level, timit=timit)
    else:
        xcs = old_make_ccfs(f, wl, f0, wl0, bc=bc, bc0=bc0, mask=mask, smoothed_flat=smoothed_flat, delta_log_wl=delta_log_wl, relgrid=False,
                            flipped=flipped, individual_fibres=individual_fibres, synthetic_template=synthetic_template, debug_level=debug_level, timit=timit)


    if individual_fibres:

        # make array only containing the central parts of the CCFs (which can have different total lengths) for fitting
        xcarr = np.zeros((len(xcs), len(xcs[0]), 2 * addrange + 1))
        for i in range(xcarr.shape[0]):
            for j in range(xcarr.shape[1]):
                dum = np.array(xcs[i][j])
                xcarr[i, j, :] = dum[len(dum) // 2 - addrange: len(dum) // 2 + addrange + 1]

        if individual_orders:
            if debug_level >= 1:
                print('Calculating independent RVs for each order and for each fibre...')
            xcsum = np.sum(np.sum(xcarr, axis=0), axis=0)
        else:
            if debug_level >= 1:
                print('Calculating one RV for each fibre (summing up CCFs over individual orders)...')
            # sum up the CCFs for all orders
            xcarr = np.sum(xcarr, axis=0)
            xcarr = xcarr[np.newaxis, :]  # need that extra dimension for the for-loop below
            xcsum = np.sum(xcarr, axis=0)

        # format is (n_ord, n_fib, npeaks)
        rv = np.zeros((xcarr.shape[0], xcarr.shape[1], npeaks))
        rverr = np.zeros((xcarr.shape[0], xcarr.shape[1], npeaks))
        for o in range(xcarr.shape[0]):
            for f in range(xcarr.shape[1]):
                if debug_level >= 3:
                    print('order = ', o, ' ; fibre = ', f)
                xc = xcarr[o, f, :]

                # find peaks (the highest of which we assume is the real one we want) in case the delta-rvabs is non-zero
                peaks = np.r_[True, xc[1:] > xc[:-1]] & np.r_[xc[:-1] > xc[1:], True]
                # filter out maxima too close to the edges to avoid problems
                peaks[:5] = False
                peaks[-5:] = False
                if npeaks == 1:
                    guessloc = np.argmax(xc*peaks)
                    if guessloc >= len(xc)//2:
                        xrange = np.arange(np.minimum(len(xc) - 2*fitrange-1, guessloc - fitrange), np.minimum(guessloc + fitrange + 1, len(xc)), 1)
                    else:
                        xrange = np.arange(np.maximum(0, guessloc - fitrange), np.maximum(guessloc + fitrange + 1, 2*fitrange+1), 1)
                else:
                    # find the npeaks highest peaks in the CCF
                    guesslocs = np.argsort(xc * peaks)[-npeaks:]
                    xrange = np.arange(np.maximum(0, np.min(guesslocs) - fitrange), np.minimum(np.max(guesslocs) + fitrange + 1, len(xc)), 1)
#                 xrange = np.arange(guessloc - fitrange, guessloc + fitrange + 1, 1)
#                 xrange = np.arange(np.argmax(xc) - fitrange, np.argmax(xc) + fitrange + 1, 1)

                # make sure we have a dynamic range
                xc -= np.min(xc[xrange])
                # "normalize" it
                xc /= np.max(xc)
                xc *= 0.9
                xc += 0.1

                if fit_slope:
                    print('latest version OKOK')
                    assert npeaks == 1, 'ERROR: multi-peak fitting together with slope fitting has not been implemented yet!!!'
                    # parameters: mu, sigma, amp, beta, offset, slope
                    guess = np.array([guessloc, fitrange//3, 0.9, 2., np.min(xc[xrange]), 0.])

                    try:
                        # subtract the minimum of the fitrange so as to have a "dynamic range"
                        popt, pcov = op.curve_fit(gausslike_with_amp_and_offset_and_slope, xrange, xc[xrange], p0=guess, maxfev=1000000)
                        mu = popt[0]
                        mu_err = np.sqrt(pcov[0, 0])
                        if debug_level >= 1:
                            print('Fit successful...')
                    except:
                        popt, pcov = (np.nan, np.nan)
                        mu = np.nan
                        mu_err = np.nan
                else:
                    print('latest version OKOK')
                    
#                 # OLD VERSION: can only accommodate one peak, but same as below
#                 # parameters: mu, sigma, amp, beta, offset
#                 guess = np.array([guessloc, fitrange//3, 0.9, 2., np.min(xc[xrange])])
# 
#                 try:
#                     # subtract the minimum of the fitrange so as to have a "dynamic range"
#                     popt, pcov = op.curve_fit(gausslike_with_amp_and_offset, xrange, xc[xrange], p0=guess, maxfev=1000000)
#                     mu = popt[0]
#                     mu_err = np.sqrt(pcov[0, 0])
#                     if debug_level >= 1:
#                         print('Fit successful...')
#                 except:
#                     popt, pcov = (np.nan, np.nan)
#                     mu = np.nan
#                     mu_err = np.nan

                    # NEW: THIS WAY WE CAN ACCOMMODATE ANY (user-defined) NUMBER OF PEAKS
                    guess = []
                    lower_bounds = []
                    upper_bounds = []
                    for n in range(npeaks):
                        # mu, sigma, amp, beta
                        guess.append(np.array([guesslocs[n], fitrange//3, xc[guesslocs[n]], 2.]))
                        lower_bounds.append([xrange[0], 0, 0, 1])
                        upper_bounds.append([xrange[-1], np.inf, np.inf, 4])
                    # reformat arrays
                    guess = np.array(guess).flatten()
                    lower_bounds = np.array(lower_bounds).flatten()
                    upper_bounds = np.array(upper_bounds).flatten()
                    # append offset
                    guess = np.append(guess, np.min(xc[xrange]))
                    lower_bounds = np.append(lower_bounds, -1)
                    upper_bounds = np.append(upper_bounds, 1)
                    # fit multiple peaks
                    try:
                        popt, pcov = op.curve_fit(multi_fibmodel_with_amp_and_offset, xrange, xc[xrange], p0=guess, bounds=(lower_bounds, upper_bounds), maxfev=100000)
                    except:
                        popt, pcov = (np.ones(4*npeaks+1)*np.nan, np.ones(4*npeaks+1)*np.nan)
                    popt_arr = np.reshape(popt[:-1], (npeaks, 4))
                    pcov_arr = np.reshape(pcov[:-1], (npeaks, 4))
    #                 offset = popt[-1]
                    mu = popt_arr[:,0]
                    mu_err = popt_arr[:,0]
                    ixs = np.arange(0,npeaks*4,4)
                    pcov_dum = 
                    mu_err = np.sqrt(np.array([pcov[ix,ix] for ix in ixs]))    
                    
                # convert to RV in m/s
                rv[o, f, :] = c * (mu - (len(xc) // 2)) * delta_log_wl
                rverr[o, f, :] = c * mu_err * delta_log_wl

    else:

        # make array only containing the central parts of the CCFs (which can have different total lengths) for fitting
        xcarr = np.zeros((len(xcs), 2 * addrange + 1))
        for i in range(xcarr.shape[0]):
            dum = np.array(xcs[i])
            xcarr[i, :] = dum[len(dum) // 2 - addrange: len(dum) // 2 + addrange + 1]

        if individual_orders:
            if debug_level >= 1:
                print('Calculating one RV for each order (summing up CCFs over individual fibres)...')
        else:
            if debug_level >= 1:
                print('Calculating one RV (summing up CCFs over individual fibres and over individual orders)')
            # sum up the CCFs for all orders
            xcarr = np.sum(xcarr, axis=0)
            xcarr = xcarr[np.newaxis, :]  # need that extra dimension for the for-loop below

        xcsum = np.sum(xcarr, axis=0)
        # format is (n_ord, npeaks)
        rv = np.zeros((xcarr.shape[0], npeaks))
        rverr = np.zeros((xcarr.shape[0], npeaks))
        for o in range(xcarr.shape[0]):
            if debug_level >= 2:
                print('order ' + str(o+1))
            xc = xcarr[o, :]
            # want to fit a symmetric region around the peak, not around the "centre" of the xc

            # find peaks (the highest of which we assume is the real one we want) in case the delta-rvabs is non-zero
            peaks = np.r_[True, xc[1:] > xc[:-1]] & np.r_[xc[:-1] > xc[1:], True]
            # filter out maxima too close to the edges to avoid problems
            peaks[:5] = False
            peaks[-5:] = False
            if npeaks == 1:
                guessloc = np.argmax(xc*peaks)
                if guessloc >= len(xc)//2:
                    xrange = np.arange(np.minimum(len(xc) - 2*fitrange-1, guessloc - fitrange), np.minimum(guessloc + fitrange + 1, len(xc)), 1)
                else:
                    xrange = np.arange(np.maximum(0, guessloc - fitrange), np.maximum(guessloc + fitrange + 1, 2*fitrange+1), 1)
            else:
                # find the npeaks highest peaks in the CCF
                guesslocs = np.argsort(xc * peaks)[-npeaks:]
                xrange = np.arange(np.maximum(0, np.min(guesslocs) - fitrange), np.minimum(np.max(guesslocs) + fitrange + 1, len(xc)), 1)
#             xrange = np.arange(guessloc - fitrange, guessloc + fitrange + 1, 1)
#           xrange = np.arange(np.argmax(xc) - fitrange, np.argmax(xc) + fitrange + 1, 1)

            # make sure we have a dynamic range
            if debug_level >= 3:
                print(xrange)
            xc -= np.min(xc[xrange])
            # "normalize" it
            xc /= np.max(xc)
            xc *= 0.9
            xc += 0.1

            if fit_slope:
                print('latest version OKOK')
                assert npeaks == 1, 'ERROR: multi-peak fitting together with slope fitting has not been implemented yet!!!'
                # parameters: mu, sigma, amp, beta, offset, slope
                guess = np.array([guessloc, fitrange//3, 0.9, 2., np.min(xc[xrange]), 0.])

                try:
                    # subtract the minimum of the fitrange so as to have a "dynamic range"
                    popt, pcov = op.curve_fit(gausslike_with_amp_and_offset_and_slope, xrange, xc[xrange], p0=guess, maxfev=1000000)
                    mu = popt[0]
                    mu_err = np.sqrt(pcov[0, 0])
                    if debug_level >= 1:
                        print('Fit successful...')
                except:
                    popt, pcov = (np.nan, np.nan)
                    mu = np.nan
                    mu_err = np.nan
            else:
                print('latest version OKOK')
                
#                 # OLD VERSION: can only accommodate one peak, but same as below
#                 # parameters: mu, sigma, amp, beta, offset
#                 guess = np.array([guessloc, fitrange//3, 0.9, 2., np.min(xc[xrange])])
# 
#                 try:
#                     # subtract the minimum of the fitrange so as to have a "dynamic range"
#                     popt, pcov = op.curve_fit(gausslike_with_amp_and_offset, xrange, xc[xrange], p0=guess, maxfev=1000000)
#                     mu = popt[0]
#                     mu_err = np.sqrt(pcov[0, 0])
#                     if debug_level >= 1:
#                         print('Fit successful...')
#                 except:
#                     popt, pcov = (np.nan, np.nan)
#                     mu = np.nan
#                     mu_err = np.nan

                # NEW: THIS WAY WE CAN ACCOMMODATE ANY (user-defined) NUMBER OF PEAKS
                guess = []
                lower_bounds = []
                upper_bounds = []
                for n in range(npeaks):
                    # mu, sigma, amp, beta
                    guess.append(np.array([guesslocs[n], fitrange//3, xc[guesslocs[n]], 2.]))
                    lower_bounds.append([xrange[0], 0, 0, 1])
                    upper_bounds.append([xrange[-1], np.inf, np.inf, 4])
                # reformat arrays
                guess = np.array(guess).flatten()
                lower_bounds = np.array(lower_bounds).flatten()
                upper_bounds = np.array(upper_bounds).flatten()
                # append offset
                guess = np.append(guess, np.min(xc[xrange]))
                lower_bounds = np.append(lower_bounds, -1)
                upper_bounds = np.append(upper_bounds, 1)
                # fit multiple peaks
                try:
                    popt, pcov = op.curve_fit(multi_fibmodel_with_amp_and_offset, xrange, xc[xrange], p0=guess, bounds=(lower_bounds, upper_bounds), maxfev=100000)
                except:
                    popt, pcov = (np.ones(4*npeaks+1)*np.nan, np.ones(4*npeaks+1)*np.nan)
                popt_arr = np.reshape(popt[:-1], (npeaks, 4))
                pcov_arr = np.reshape(pcov[:-1], (npeaks, 4))
#                 offset = popt[-1]
                mu = popt_arr[:,0]
                mu_err = popt_arr[:,0]
                ixs = np.arange(0,npeaks*4,4)
                pcov_dum = 
                mu_err = np.sqrt(np.array([pcov[ix,ix] for ix in ixs]))
                    
            # convert to RV in m/s
            rv[o,:] = c * (mu - (len(xc) // 2)) * delta_log_wl
            rverr[o,:] = c * mu_err * delta_log_wl
#             # # plot a single fit for debugging
#             plot_osf = 10
#             plot_os_grid = np.linspace(xrange[0], xrange[-1], plot_osf * (len(xrange)-1) + 1)
#             plt.plot(c * (xrange - (len(xc) // 2)) * delta_log_wl, xc[xrange], 'bx', label='data')
# #             plt.plot(c * (plot_os_grid - (len(xc) // 2)) * delta_log_wl, gausslike_with_amp_and_offset(plot_os_grid, *guess),'r--', label='initial guess')
# #             plt.plot(c * (plot_os_grid - (len(xc) // 2)) * delta_log_wl, gausslike_with_amp_and_offset(plot_os_grid, *popt),'g-', label='best fit')
#             plt.plot(c * (plot_os_grid - (len(xc) // 2)) * delta_log_wl, gausslike_with_amp_and_offset_and_slope(plot_os_grid, *popt), 'g-', label='best fit')
#             plt.plot(c * (plot_os_grid - (len(xc) // 2)) * delta_log_wl, gausslike_with_amp_and_offset_and_slope(plot_os_grid, *guess),'r--', label='initial guess')
#             plt.axvline(c * (mu - (len(xc) // 2)) * delta_log_wl, color='g', linestyle=':')
#             plt.legend()
#             plt.xlabel('delta RV [m/s]')
#             plt.ylabel('power')
#             plt.title('CCF')

    if timit:
        delta_t = time.time() - start_time
        print('Time taken for calculating RV: ' + str(np.round(delta_t, 2)) + ' seconds')

    if individual_fibres:
        if individual_orders:
            return rv, rverr, np.array(xcsum)
        else:
            return rv, rverr, np.array(xcsum)
    else:
        if individual_orders:
            return rv, rverr, np.array(xcsum)
        else:
            return rv, rverr, np.array(xcsum)





def make_ccfs(f, wl, f0, wl0, bc=0., bc0=0., smoothed_flat=None, delta_log_wl=1e-6, deg_interp=1, flipped=False, individual_fibres=True, scrunch=False,
              norm_cont=True, taper=True, taper_width=0.05, use_orders=None, synthetic_template=False, n_stellar_fibs=19, debug_level=0, timit=False):
    """
    This routine calculates the CCFs of an observed spectrum and a template spectrum for each order.
    Note that input spectra should be de-blazed for the cross-correlation, so can do that either externally, or internally
    by setting norm_cont to TRUE !!!

    INPUT:
    'f'                  : numpy array containing the observed flux (n_ord, n_fib, n_pix)
    'wl'                 : numpy array containing the wavelengths of the observed spectrum (n_ord, n_fib, n_pix)
    'f0'                 : numpy array containing the flux of the template spectrum (n_ord, n_fib, n_pix)
    'wl0'                : numpy array containing the wavelengths of the template spectrum (n_ord, n_fib, n_pix)
    'bc'                 : the barycentric velocity correction for the observed spectrum
    'bc0'                : the barycentric velocity correction for the template spectrum
    'smoothed_flat'      : used as the blaze function for continuum normalisation and CCF weighting
    'delta_log_wl'       : step size of the log-wl grid (only used if 'relgrid' is FALSE)
    'deg_interp'         : the degree of the interpolation (1 = linear, 3 = cubic)
    'flipped'            : boolean - reverse order of inputs to xcorr routine?
    'individual_fibres'  : boolean - do you want to return the CCFs for individual fibres? (if FALSE, then the sum of the ind. fib. CCFs is returned)
    'scrunch'            : boolean - do you want to scrunch the spectra or simply interpolate them when rebinning?
    'norm_cont'          : boolean - do you want to normalise the continuum?
    'taper'              : boolean - do you want to taper off the edges for the inputs to the xcorr-routine?
    'taper_width'        : fractional width/length of the input arrays to the xcorr-routine that will be tapered to zero at both ends
    'use_orders'         : which orders do you want to use for the xcorr?
    'synthetic_template' : boolean - are you using a synthetic template?
    'n_stellar_fibs'     : number of stellar fibres - sholud always be 19, except for testing purposes
    'debug_level'        : for debugging...
    'timit'              : boolean - do you want to measure execution run time?

    OUTPUT:
    'xcs'   : list containing the ind. fib. CCFs / sum of ind. fib. CCFs for each order (len = n_ord)

    MODHIST:
    July 2019 - CMB create (major changes from "old_make_ccfs")
    """

    if timit:
        start_time = time.time()

    # speed of light in m/s
    c = 2.99792458e8
    
    
    # make sure that f and f0 have the same shape
    assert f.shape == f0.shape, 'ERROR: observation and template do not have the same shape!!!'
    assert wl.shape == wl0.shape, 'ERROR: observation and template wavelength-arrays do not have the same shape!!!'
    
    if smoothed_flat is None:
        smoothed_flat = np.ones(f.shape)
        blaze_provided = False
    else:
        blaze_provided = True
    
    n_ord, n_fib, n_pix = f.shape
    # only proceed with stellar fibres from here on
    assert n_fib in [19,24,26], 'ERROR: spectrum dimensions not recognised: wrong number of fibres encountered (nfib = ' + str(n_fib) + ')'
    # trim everything down to stellar fibres only
    if n_fib == 26:
        f = f[:,3:22,:]
        f0 = f0[:,3:22,:]
        wl = wl[:,3:22,:]
        wl0 = wl0[:,3:22,:]
        smoothed_flat = smoothed_flat[:,3:22,:]
    elif n_fib == 24:
        f = f[:,2:21,:]
        f0 = f0[:,2:21,:]
        wl = wl[:,2:21,:]
        wl0 = wl0[:,2:21,:]
        smoothed_flat = smoothed_flat[:,2:21,:]
    
    # the dummy wavelengths for orders 'order_01' and 'order_40' cannot be zero as we're taking a log!!!
    if wl.shape[0] == 40:
        if (wl[0,:,:] == 0).all():
            wl[0,:,:] = 1.
        if (wl[-1,:,:] == 0).all():
            wl[-1,:,:] = 1.
        if not synthetic_template:
            if (wl0[0,:,:] == 0).all():
                wl0[0,:,:] = 1.
            if (wl0[-1,:,:] == 0).all():
                wl0[-1,:,:] = 1.
    if wl.shape[0] == 39:
        if (wl[0,:,:] == 0).all():
            wl[0,:,:] = 1.
        if not synthetic_template:
            if (wl0[0,:,:] == 0).all():
                wl0[0,:,:] = 1.


    # read min and max of the wl ranges for the logwlgrid for the xcorr
    # HARD-CODED...UGLY!!!
    dumord, min_wl_arr, max_wl_arr = readcol('/Users/christoph/OneDrive - UNSW/dispsol/veloce_xcorr_wlrange.txt', twod=False, verbose=False)
    # index -723 is the last non-NaN one for order_38 (WTF???), which is ~5956.2 A, so allow enough room
    min_wl_arr[37] = 5958.5

    # TESTING ONLY
    min_wl_arr[5] = 8650.
    max_wl_arr[5] = 8810.
    min_wl_arr[6] = 8530.
    max_wl_arr[6] = 8705.
    min_wl_arr[17] = 7390.
    max_wl_arr[17] = 7530.
    min_wl_arr[25] = 6740.
    max_wl_arr[25] = 6864.
    min_wl_arr[26] = 6665.
    max_wl_arr[26] = 6785.
    min_wl_arr[27] = 6600.
    max_wl_arr[27] = 6715.
    min_wl_arr[31] = 6330.
    max_wl_arr[31] = 6430.
    min_wl_arr[34] = 6138.
    max_wl_arr[34] = 6240.
    min_wl_arr[35] = 6080.
    max_wl_arr[35] = 6175.
    min_wl_arr[36] = 6020.
    max_wl_arr[36] = 6110.
    
    # prepare output variable
    xcs = []

    #####  LOOP OVER ORDERS  #####
#     for o in range(n_ord):
#     for o in range(1,n_ord):
    # from Brendan's plots/table:
#     for o in [5, 6, 7, 17, 26, 27, 34, 35, 36, 37]:
    # Duncan's suggestion
#     for o in [4,5,6,25,26,33,34,35]:
#     for o in [5, 6, 17, 25, 26, 27, 31, 34, 35, 36, 37]:
#     if use_orders == 'all':
#         use_orders = np.arange(1,39)
#     if use_orders is None:
#     use_orders = [5, 6, 17, 25, 26, 27, 31, 34, 35, 36]         # at the moment 17 and 34 give the lowest scatter
#     use_orders = [5, 17, 25, 26, 27, 31, 34, 35, 36]
    use_orders = [5, 17, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36]    # for TOI192
#     use_orders = [5, 6, 17, 25, 27, 31, 36]
#     use_orders = [17]
#     use_orders = [35]
#     use_orders = np.arange(1,39)
#     use_orders = np.arange(1,37)
#     use_orders = np.arange(n_ord)

    print('Using ' + str(len(use_orders)) + ' orders for CCF...')
    print("haehaehae_20191206")

    for o in use_orders:         # at the moment 17 and 34 give the lowest scatter

        if debug_level >= 2:
            print('Order ' + str(o + 1).zfill(2))

        ord = 'order_' + str(o + 1).zfill(2)

        # define parts of the spectrum to use (in wl space)
        min_wl = min_wl_arr[o]
        max_wl = max_wl_arr[o]

        # now apply barycentric correction to wl and wl0 so that we can ALWAYS USE THE SAME PARTS OF THE SPECTRUM for X-CORR!!!!!
#         wl_bcc = (1 + bc / c) * wl   # NO! That would be for frequencies!
#         wl_bcc = wl / (1 - bc / c)
        wl_bcc = wl * np.sqrt((1 + bc / c) / (1 - bc / c))
        # create logarithmic wavelength grid
        logwl = np.log(wl_bcc[o, :, :])
        if not synthetic_template:
#             wl0_bcc = (1 + bc0 / c) * wl0   # NO! That would be for frequencies!
#             wl0_bcc = wl0 / (1 - bc0 / c) 
            wl0_bcc = wl0 * np.sqrt((1 + bc0 / c) / (1 - bc0 / c))
            logwl0 = np.log(wl0_bcc[o, :, :])
        else:
            # create logarithmic wavelength grid for the synthetic template (first trim, then go to log space)
            ordix_0 = (wl0 >= np.min(wl[o, 9, :])) & (wl0 <= np.max(wl[o, 9, :]))
            wl0_ord = wl0[ordix_0]
            logwl0 = np.log(wl0_ord)
            f0_ord = f0[ordix_0]

        # use range as defined above, and SAME range for all observations!!!
        # logwlgrid = np.arange(np.log(min_wl), np.log(max_wl), delta_log_wl)
        logwlgrid = np.arange(np.log(min_wl) + delta_log_wl, np.log(max_wl) - delta_log_wl, delta_log_wl)
#         logwlgrid = np.arange(np.log(min_wl) + 5*delta_log_wl, np.log(max_wl) - 5*delta_log_wl, delta_log_wl)

        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(logwl) < 0).any():
            logwl_sorted = logwl[:,::-1].copy()
            ord_f_sorted = f[o,:,::-1].copy()
            ord_blaze_sorted = smoothed_flat[o,:,::-1].copy()
            if not synthetic_template:
                logwl0_sorted = logwl0[:,::-1].copy()
                ord_f0_sorted = f0[o,:,::-1].copy()
            else:
                logwl0_sorted = np.array([logwl0,] * n_fib)
                ord_f0_sorted = np.array([f0_ord,] * n_fib)
        else:
            logwl_sorted = logwl.copy()
            ord_f_sorted = f[o,:,:].copy()
            ord_blaze_sorted = smoothed_flat[o,:,:].copy()
            if not synthetic_template:
                logwl0_sorted = logwl0.copy()
                ord_f0_sorted = f0[o,:,:].copy()
            else:
                logwl0_sorted = np.array([logwl0,]*n_fib)
                ord_f0_sorted = np.array([f0_ord,]*n_fib)

        # rebin spectra onto logarithmic wavelength grid
        rebinned_f0 = np.zeros((n_stellar_fibs, len(logwlgrid)))
        rebinned_f = np.zeros((n_stellar_fibs, len(logwlgrid)))
        rebinned_blaze = np.ones((n_stellar_fibs, len(logwlgrid)))
        # make sure we don't have any NaNs (which throw off the interpolation routine)
        ord_f0_sorted[np.isnan(ord_f0_sorted)] = 0.
        ord_f_sorted[np.isnan(ord_f_sorted)] = 0.
        ord_blaze_sorted[np.isnan(ord_blaze_sorted)] = 0.

        # only use stellar fibres now
        for i in range(n_stellar_fibs):   
            # (i) using SpectRes
#             rebinned_f0[i, :] = spectres(logwlgrid, logwl0_sorted[i, :], ord_f0_sorted[i, :])
#             rebinned_f[i, :] = spectres(logwlgrid, logwl_sorted[i, :], ord_f_sorted[i, :])
            if scrunch:
                # (ii) using cmb_scrunch
                rebinned_f0[i, :] = cmb_scrunch(logwlgrid, logwl0_sorted[i, :], ord_f0_sorted[i, :])
                rebinned_f[i, :] = cmb_scrunch(logwlgrid, logwl_sorted[i, :], ord_f_sorted[i, :])
            else:
                # (iii) using interpolation (using pixel centres)
                spl_ref_f0 = interp.InterpolatedUnivariateSpline(logwl0_sorted[i, :], ord_f0_sorted[i, :], k=deg_interp)
                # if k=1, then that is equivalent to spl_ref_f0 = interp.interp1d(logwl0_sorted[i, :], ord_f0_sorted[i, :])
                # if k=3, then that is equivalent to spl_ref_f0 = interp.interp1d(logwl0_sorted[i, :], ord_f0_sorted[i, :], kind='cubic')
                rebinned_f0[i, :] = spl_ref_f0(logwlgrid)
                spl_ref_f = interp.InterpolatedUnivariateSpline(logwl_sorted[i, :], ord_f_sorted[i, :], k=deg_interp)  
                rebinned_f[i, :] = spl_ref_f(logwlgrid)
            if blaze_provided:
                # (i) using SpectRes
#                 rebinned_blaze[i, :] = spectres(logwlgrid, logwl_sorted[i, :], ord_blaze_sorted[i, :])
                if scrunch:
                    # (ii) using cmb_scrunch
                    rebinned_blaze[i, :] = cmb_scrunch(logwlgrid, logwl_sorted[i, :], ord_blaze_sorted[i, :])
                else:
                    # (iii) using interpolation (using pixel centres)
                    spl_ref_blaze = interp.InterpolatedUnivariateSpline(logwl_sorted[i, :], ord_blaze_sorted[i, :], k=deg_interp)  
                    rebinned_blaze[i, :] = spl_ref_blaze(logwlgrid)
         
        
        # actually perform the cross-correlation        
        if individual_fibres:
            ord_xcs = []
            
            if not blaze_provided:
                rebinned_blaze = np.ones(rebinned_f.shape)
            
            # only use stellar fibres now
            for fib in range(n_stellar_fibs): 
                
                # further continuum-normalize spectra
                if norm_cont:
                    rebinned_f0[fib,:] = rebinned_f0[fib,:] / (rebinned_blaze[fib,:] / np.nanmax(rebinned_blaze[fib,:]))
                    rebinned_f[fib,:] = rebinned_f[fib,:] / (rebinned_blaze[fib,:] / np.nanmax(rebinned_blaze[fib,:]))
                    maxfiltersize = np.round(0.016*len(logwlgrid),0).astype(int)
                    gaussfiltersize = np.round(0.011*len(logwlgrid),0).astype(int)
                    cont_f0 = ndimage.gaussian_filter(ndimage.maximum_filter(rebinned_f0[fib,:], size=maxfiltersize), gaussfiltersize)
                    cont_f = ndimage.gaussian_filter(ndimage.maximum_filter(rebinned_f[fib,:], size=maxfiltersize), gaussfiltersize)
                    rebinned_f0[fib,:] /= cont_f0
                    rebinned_f[fib,:] /= cont_f
                
                # re-scale spectra
                rebinned_f0[fib,:] /= np.nanmax(rebinned_f0[fib,:])
                rebinned_f[fib,:] /= np.nanmax(rebinned_f[fib,:])
                if blaze_provided:
                    rebinned_blaze[fib,:] /= np.nanmax(rebinned_blaze[fib,:])       
                
                if taper:
                    assert (taper_width >= 0.001) & (taper_width <=0.5), 'ERROR: taper_width must be within [0.001...0.5]'
                    taper_range = int(np.ceil(len(rebinned_f0[fib,:]) * taper_width))
                    taper_start = np.linspace(-np.pi, 0, taper_range)
                    taper_end = np.linspace(0, np.pi, taper_range)
                    taper_func = np.ones(len(rebinned_f0[fib,:]))
                    taper_func[:taper_range] = (np.cos(taper_start) / 2.) + 0.5
                    taper_func[-taper_range:] = (np.cos(taper_end) / 2.) + 0.5
                
                if not flipped:
#                     xc = xcorr((rebinned_f0[fib,:] - 1.)*rebinned_blaze[fib,:], rebinned_f[fib,:] - 1., scale='unbiased')
#                     xc = xcorr((rebinned_f0[fib,:] - 1.)*taper_func, (rebinned_f[fib,:] - 1.)*taper_func, scale='unbiased')      # xc4b for tests on 22/10/2019
#                     xc = xcorr((rebinned_f0[fib,:] - 1.)*rebinned_blaze[fib,:]*taper_func, (rebinned_f[fib,:] - 1.)*rebinned_blaze[fib,:]*taper_func, scale='unbiased')      # xc8 for tests on 22/10/2019
#                     xc = xcorr((rebinned_f0[fib,:] - 1.)*rebinned_blaze[fib,:]*taper_func, (rebinned_f[fib,:] - 1.)*taper_func, scale='unbiased')      # xc7 for tests on 22/10/2019
#                     xc = xcorr(((rebinned_f0[fib,:]/np.max(rebinned_f0[fib,:])) - 1.)*(rebinned_blaze[fib,:]/np.max(rebinned_blaze[fib,:])), (rebinned_f[fib,:]/np.max(rebinned_f[fib,:])) - 1., scale='unbiased')
#                     xc = np.correlate(rebinned_f0[fib, :], rebinned_f[fib, :], mode='full')
                    print('xc4b')
                    f0_in = (rebinned_f0[fib,:] - 1.)*taper_func
                    f_in = (rebinned_f[fib,:] - 1.)*taper_func
                    f0_in -= np.mean(f0_in)
                    f_in -= np.mean(f_in)
                    xc = xcorr(f0_in, f_in, scale='unbiased')
                else:
                    xc = xcorr(rebinned_f[fib,:] - 1., (rebinned_f0[fib,:] - 1.)*rebinned_blaze[fib,:], scale='unbiased')
#                     xc = xcorr((rebinned_f[fib,:]/np.max(rebinned_f[fib,:])) - 1., ((rebinned_f0[fib,:]/np.max(rebinned_f0[fib,:])) - 1.)*(rebinned_blaze[fib,:]/np.max(rebinned_blaze[fib,:])), scale='unbiased')
#                     xc = np.correlate(rebinned_f[fib, :], rebinned_f0[fib, :], mode='full')
                ord_xcs.append(xc)
            xcs.append(ord_xcs)
            
        else:
            
            # combine fibres
            rebinned_f = np.sum(rebinned_f, axis=0)
            rebinned_f0 = np.sum(rebinned_f0, axis=0)
            if blaze_provided:
                rebinned_blaze = np.sum(rebinned_blaze, axis=0)
            else:
                rebinned_blaze = np.ones(rebinned_f.shape)
                
            # further continuum-normalize spectra
            if norm_cont:
                rebinned_f0 = rebinned_f0 / (rebinned_blaze / np.nanmax(rebinned_blaze))
                rebinned_f = rebinned_f / (rebinned_blaze / np.nanmax(rebinned_blaze))
                maxfiltersize = np.round(0.016*len(logwlgrid),0).astype(int)
                gaussfiltersize = np.round(0.011*len(logwlgrid),0).astype(int)
                cont_f0 = ndimage.gaussian_filter(ndimage.maximum_filter(rebinned_f0, size=maxfiltersize), gaussfiltersize)
                cont_f = ndimage.gaussian_filter(ndimage.maximum_filter(rebinned_f, size=maxfiltersize), gaussfiltersize)
                rebinned_f0 /= cont_f0
                rebinned_f /= cont_f
                
            # re-scale spectra
            rebinned_f0 /= np.nanmax(rebinned_f0)
            rebinned_f /= np.nanmax(rebinned_f)
            if blaze_provided:
                rebinned_blaze /= np.nanmax(rebinned_blaze)           
            
            if taper:
                assert (taper_width >= 0.001) & (taper_width <=0.5), 'ERROR: taper_width must be within [0.001...0.5]' 
                taper_range = int(np.ceil(len(rebinned_f0) * taper_width))
                taper_start = np.linspace(-np.pi, 0, taper_range)
                taper_end = np.linspace(0, np.pi, taper_range)
                taper_func = np.ones(len(rebinned_f0))
                taper_func[:taper_range] = (np.cos(taper_start) / 2.) + 0.5
                taper_func[-taper_range:] = (np.cos(taper_end) / 2.) + 0.5
                
            if not flipped:
#                 xc = xcorr((rebinned_f0 - 1.)*rebinned_blaze, rebinned_f - 1., scale='unbiased')                                # xc0 for tests on 22/10/2019
#                 xc = xcorr((rebinned_f0 - 1.), rebinned_f - 1., scale='unbiased')                                             # xc1 for tests on 22/10/2019
#                 xc = xcorr(rebinned_f0, rebinned_f, scale='unbiased')                                                       # xc2 for tests on 22/10/2019
#                 xc = xcorr((rebinned_f0 - 1.)*rebinned_blaze, (rebinned_f - 1.)*rebinned_blaze, scale='unbiased')             # xc3 for tests on 22/10/2019
#                 xc = xcorr((rebinned_f0 - 1.)*taper_func+1., (rebinned_f - 1.)*taper_func+1., scale='unbiased')                   # xc4a for tests on 22/10/2019
#                 xc = xcorr((rebinned_f0 - 1.)*taper_func, (rebinned_f - 1.)*taper_func, scale='unbiased')                         # xc4b for tests on 22/10/2019
#                 xc = xcorr((rebinned_f0 - 1.)*rebinned_blaze + 1, rebinned_f, scale='unbiased')                               # xc5 for tests on 22/10/2019
#                 xc = xcorr((rebinned_f0 - 1.)*rebinned_blaze + 1, (rebinned_f - 1.)*rebinned_blaze + 1, scale='unbiased')     # xc6 for tests on 22/10/2019
#                 xc = xcorr((rebinned_f0 - 1.)*rebinned_blaze*taper_func, (rebinned_f - 1.)*taper_func, scale='unbiased')      # xc7 for tests on 22/10/2019
#                 xc = xcorr((rebinned_f0 - 1.)*rebinned_blaze*taper_func, (rebinned_f - 1.)*rebinned_blaze*taper_func, scale='unbiased')      # xc8 for tests on 22/10/2019
#                 xc = xcorr((rebinned_f0 - 1.)*rebinned_blaze*taper_func, (rebinned_f - 1.)*rebinned_blaze*taper_func, scale='unbiased')      # xc9 for tests on 22/10/2019
                print('xc4b')
                f0_in = (rebinned_f0 - 1.)*taper_func
                f_in = (rebinned_f - 1.)*taper_func
                f0_in -= np.mean(f0_in)
                f_in -= np.mean(f_in)
                xc = xcorr(f0_in, f_in, scale='unbiased')
            else:
                xc = xcorr(rebinned_f - 1., (rebinned_f0 - 1.)*rebinned_blaze, scale='unbiased')
#                 xc = np.correlate(rebinned_f, rebinned_f0, mode='full')
            xcs.append(xc)

    if timit:
        delta_t = time.time() - start_time
        print('Time taken for creating CCFs: ' + str(np.round(delta_t, 2)) + ' seconds')

    return xcs





def old_make_ccfs(f, wl, f0, wl0, bc=0., bc0=0., mask=None, smoothed_flat=None, delta_log_wl=1e-6, relgrid=False, osf=5,
              filter_width=25, bad_threshold=0.05, flipped=False, individual_fibres=True, synthetic_template=False,
              debug_level=0, timit=False):
    """
    This routine calculates the CCFs of an observed spectrum and a template spectrum for each order.
    Note that input spectra should be de-blazed already!!!
    If the mask from "find_stripes" has gaps, do the filtering for each segment independently. If no mask is provided, create a simple one on the fly.

    INPUT:
    'f'                  : numpy array containing the observed flux (n_ord, n_fib, n_pix)
    'wl'                 : numpy array containing the wavelengths of the observed spectrum (n_ord, n_fib, n_pix)
    'f0'                 : numpy array containing the flux of the template spectrum (n_ord, n_fib, n_pix)
    'wl0'                : numpy array containing the wavelengths of the template spectrum (n_ord, n_fib, n_pix)
    'bc'                 : the barycentric velocity correction for the observed spectrum
    'bc0'                : the barycentric velocity correction for the template spectrum
    'mask'               : mask-dictionary from "find_stripes" (keys = orders)
    'smoothed_flat'      : if no mask is provided, need to provide the smoothed_flat, so that a mask can be created on the fly
    'delta_log_wl'       : stepsize of the log-wl grid (only used if 'relgrid' is FALSE)
    'relgrid'            : boolean - do you want to use an absolute stepsize of the log-wl grid (DEFAULT), or relative using 'osf'?
    'osf'                : oversampling factor for the logarithmic wavelength rebinning (only used if 'relgrid' is TRUE)
    'filter_width'       : width of smoothing filter in pixels; needed b/c of edge effects of the smoothing; number of pixels to disregard should be >~ 2 * width of smoothing kernel
    'bad_threshold'      : if no mask is provided, create a mask that requires the flux in the extracted white to be larger than this fraction of the maximum flux in that order
    'flipped'            : boolean - reverse order of inputs to xcorr routine?
    'individual_fibres'  : boolean - do you want to return the CCFs for individual fibres? (if FALSE, then the sum of the ind. fib. CCFs is returned)
    'debug_level'        : for debugging...
    'timit'              : boolean - do you want to measure execution run time?

    OUTPUT:
    'xcs'   : list containing the ind. fib. CCFs / sum of ind. fib. CCFs for each order (len = n_ord)

    MODHIST:
    Nov 2018 - CMB create
    """

    if timit:
        start_time = time.time()

    # speed of light in m/s
    c = 2.99792458e8
    
    # make sure that f and f0 have the same shape
    assert f.shape == f0.shape, 'ERROR: observation and template do not have the same shape!!!'
    

    # the dummy wavelengths for orders 'order_01' and 'order_40' cannot be zero as we're taking a log!!!
    if wl.shape[0] == 40:
        if (wl[0,:,:] == 0).all():
            wl[0,:,:] = 1.
        if (wl[-1,:,:] == 0).all():
            wl[-1,:,:] = 1.
        if not synthetic_template:
            if (wl0[0,:,:] == 0).all():
                wl0[0,:,:] = 1.
            if (wl0[-1,:,:] == 0).all():
                wl0[-1,:,:] = 1.
    if wl.shape[0] == 39:
        if (wl[0,:,:] == 0).all():
            wl[0,:,:] = 1.
        if not synthetic_template:
            if (wl0[0,:,:] == 0).all():
                wl0[0,:,:] = 1.


    # read min and max of the wl ranges for the logwlgrid for the xcorr
    # HARD-CODED...UGLY!!!
    dumord, min_wl_arr, max_wl_arr = readcol('/Users/christoph/OneDrive - UNSW/dispsol/veloce_xcorr_wlrange.txt', twod=False)

    # prepare output variable
    xcs = []

    # loop over orders
    # for ord in sorted(f.iterkeys()):
    # for o in range(wl.shape[0]):
    # from Brendan's plots/table:
    # for o in [5, 6, 7, 17, 26, 27, 34, 35, 36, 37]:
    # Duncan's suggestion
    # for o in [4,5,6,25,26,33,34,35]:
    for o in [5, 6, 17, 25, 26, 27, 31, 34, 35, 36, 37]:
    # for o in [5]:

        if debug_level >= 2:
            print('Order ' + str(o + 1).zfill(2))

        ord = 'order_' + str(o + 1).zfill(2)

        # # only use pixels that have enough signal
        # if mask is None:
        #     normflat = smoothed_flat[ord] / np.max(smoothed_flat[ord])
        #     ordmask = np.ones(len(normflat), dtype=bool)
        #     if np.min(normflat) < bad_threshold:
        #         ordmask[normflat < bad_threshold] = False
        #         # once the blaze function falls below a certain value, exclude what's outside of that pixel column, even if it's above the threshold again, ie we want to only use a single consecutive region
        #         leftmask = ordmask[: len(ordmask) // 2]
        #         leftkill_index = [i for i, x in enumerate(leftmask) if not x]
        #         try:
        #             ordmask[: leftkill_index[0]] = False
        #         except:
        #             pass
        #         rightmask = ordmask[len(ordmask) // 2:]
        #         rightkill_index = [i for i, x in enumerate(rightmask) if not x]
        #         if ord == 'order_01' and simu:
        #             try:
        #                 # if not flipped then the line below must go in the leftkillindex thingy
        #                 # ordmask[: leftkill_index[-1] + 100] = False
        #                 ordmask[len(ordmask) // 2 + rightkill_index[0] - 100:] = False
        #             except:
        #                 pass
        #         else:
        #             try:
        #                 ordmask[len(mask) // 2 + rightkill_index[-1] + 1:] = False
        #             except:
        #                 pass
        # else:
        #     # ordmask  = mask[ord][::-1]
        #     ordmask = mask[ord]
        #
        # ordmask = np.ones(4112, dtype=bool)
        # ordmask[:200] = False
        # ordmask[4000:] = False
        #
        #
        # # either way, disregard #(edge_cut) pixels at either end; this is slightly dodgy, but the gaussian filtering above introduces edge effects due to mode='reflect'
        # ordmask[:2 * int(filter_width)] = False
        # ordmask[-2 * int(filter_width):] = False
        #
        # f0_unblazed = f0[ord] / np.max(f0[ord]) / normflat   #see COMMENT above
        # f_unblazed = f[ord] / np.max(f[ord]) / normflat

        # define parts of the spectrum to use (based on the (unshifted) wl-solution of the central fibre in the template observation)
        # # TODO: try and optimize that for each order and even for each target
        # min_wl = np.min(wl0[o, 9, :]) + 20.
        # max_wl = np.max(wl0[o, 9, :]) - 12.
        min_wl = min_wl_arr[o]
        max_wl = max_wl_arr[o]

        # now apply barycentric correction to wl and wl0 so that we can always use the SAME PARTS OF THE SPECTRUM for X-CORR
        wl_bcc = (1 + bc / c) * wl
        # create logarithmic wavelength grid
        logwl = np.log(wl_bcc[o, :, :])
        if not synthetic_template:
            wl0_bcc = (1 + bc0 / c) * wl0
            logwl0 = np.log(wl0_bcc[o, :, :])
        else:
            # create logarithmic wavelength grid for the synthetic template (first trim, then go to log space)
            ordix_0 = (wl0 >= np.min(wl[o, 9, :])) & (wl0 <= np.max(wl[o, 9, :]))
            wl0_ord = wl0[ordix_0]
            logwl0 = np.log(wl0_ord)
            f0_ord = f0[ordix_0]

        if relgrid:
            logwlgrid = np.linspace(np.min(logwl[ordmask]), np.max(logwl[ordmask]), osf * np.sum(ordmask))
            delta_log_wl = logwlgrid[1] - logwlgrid[0]
            if debug_level >= 2:
                print(ord, ' :  delta_log_wl = ', delta_log_wl)
        else:
            # logwlgrid = np.arange(np.min(logwl[-1,ordmask]), np.max(logwl[-1,ordmask]), delta_log_wl)
            # # use range from the maximum of the indfib-minima to the minimum of the indfib-maxima (also across wl and wl0) - use list comprehensions...
            # logwlgrid = np.arange(np.max([np.min(logwl[fib,ordmask]) for fib in range(logwl.shape[0])] + [np.min(logwl0[fib,ordmask]) for fib in range(logwl0.shape[0])]),
            #                       np.min([np.max(logwl[fib,ordmask]) for fib in range(logwl.shape[0])] + [np.max(logwl0[fib,ordmask]) for fib in range(logwl0.shape[0])]),
            #                       delta_log_wl)
            # NO!!! use range as defined above, but same for all observations!!!
            logwlgrid = np.arange(np.log(min_wl), np.log(max_wl), delta_log_wl)

        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(logwl) < 0).any():
            logwl_sorted = logwl[:,::-1].copy()
            ord_f_sorted = f[o,:,::-1].copy()
            # ordmask_sorted = ordmask[::-1].copy()
            if not synthetic_template:
                logwl0_sorted = logwl0[:,::-1].copy()
                ord_f0_sorted = f0[o,:,::-1].copy()
            else:
                logwl0_sorted = np.array([logwl0,]*nfib)
                ord_f0_sorted = np.array([f0_ord,]*nfib)
        else:
            logwl_sorted = logwl.copy()
            ord_f_sorted = f[ord].copy()
            # ordmask_sorted = ordmask.copy()
            if not synthetic_template:
                logwl0_sorted = logwl0.copy()
                ord_f0_sorted = f0[ord].copy()
            else:
                logwl0_sorted = np.array([logwl0,]*19)
                ord_f0_sorted = np.array([f0_ord,]*nfib)

        # rebin spectra onto logarithmic wavelength grid
        # rebinned_f0 = np.interp(logwlgrid,logwl[mask],f0_unblazed[mask])
        # rebinned_f = np.interp(logwlgrid,logwl[mask],f_unblazed[mask])
        nfib = ord_f_sorted.shape[0]   # (already done above now)
        rebinned_f0 = np.zeros((nfib, len(logwlgrid)))
        rebinned_f = np.zeros((nfib, len(logwlgrid)))
        for i in range(nfib):
            # spl_ref_f0 = interp.InterpolatedUnivariateSpline(logwl0_sorted[i,ordmask_sorted], ord_f0_sorted[i,ordmask_sorted], k=3)  # slightly slower than linear, but best performance for cubic spline
            # rebinned_f0[i,:] = spl_ref_f0(logwlgrid)
            # spl_ref_f = interp.InterpolatedUnivariateSpline(logwl_sorted[i,ordmask_sorted], ord_f_sorted[i,ordmask_sorted], k=3)  # slightly slower than linear, but best performance for cubic spline
            # rebinned_f[i,:] = spl_ref_f(logwlgrid)
            spl_ref_f0 = interp.InterpolatedUnivariateSpline(logwl0_sorted[i, :], ord_f0_sorted[i, :],
                                                             k=3)  # slightly slower than linear, but best performance for cubic spline
            rebinned_f0[i, :] = spl_ref_f0(logwlgrid)
            spl_ref_f = interp.InterpolatedUnivariateSpline(logwl_sorted[i, :], ord_f_sorted[i, :],
                                                            k=3)  # slightly slower than linear, but best performance for cubic spline
            rebinned_f[i, :] = spl_ref_f(logwlgrid)

        if individual_fibres:
            ord_xcs = []
            for fib in range(nfib):
                if not flipped:
                    xc = np.correlate(rebinned_f0[fib, :], rebinned_f[fib, :], mode='full')
                else:
                    xc = np.correlate(rebinned_f[fib, :], rebinned_f0[fib, :], mode='full')
                ord_xcs.append(xc)
            xcs.append(ord_xcs)
        else:
            rebinned_f = np.sum(rebinned_f, axis=0)
            rebinned_f0 = np.sum(rebinned_f0, axis=0)
            # xc = np.correlate(rebinned_f0 - np.median(rebinned_f0), rebinned_f - np.median(rebinned_f), mode='full')
            if not flipped:
                xc = np.correlate(rebinned_f0, rebinned_f, mode='full')
            else:
                xc = np.correlate(rebinned_f, rebinned_f0, mode='full')
            xcs.append(xc)

    if timit:
        delta_t = time.time() - start_time
        print('Time taken for creating CCFs: ' + str(np.round(delta_t, 2)) + ' seconds')

    return xcs




def make_ccfs_quick(f, wl, f0, wl0, smoothed_flat, bc=0., bc0=0., rvabs=0., rvabs0=0., delta_log_wl=1e-6, synthetic_template=False, debug_level=0, timit=False):
    """
    This routine calculates the CCFs of a quick-extracted observed spectrum and a quick-extracted template spectrum for each order.
    Note that input spectra should be de-blazed and cosmic-cleaned already!!!

    INPUT:
    'f'                  : numpy array containing the observed flux (n_ord, n_pix)
    'wl'                 : numpy array containing the wavelengths of the observed spectrum (n_ord, n_pix)
    'f0'                 : numpy array containing the flux of the template spectrum (n_ord, n_pix)
    'wl0'                : numpy array containing the wavelengths of the template spectrum (n_ord, n_pix)
    'smoothed_flat'      : numpy array containing the flux of the template spectrum (n_ord, n_pix)
    'bc'                 : the barycentric velocity correction for the observed spectrum
    'bc0'                : the barycentric velocity correction for the template spectrum
    'rvabs'              : absolute RV of observed target [in km/s]
    'rvabs0'             : absolute RV of template star [in km/s]
    'delta_log_wl'       : stepsize of the log-wl grid (only used if 'relgrid' is FALSE)
    'synthetic_template' : boolean - are you using a synthetic template?
    'debug_level'        : for debugging...
    'timit'              : boolean - do you want to measure execution run time?

    OUTPUT:
    'xcs'   : list containing the CCFs for each order (len = n_ord)

    MODHIST:
    June 2019 - CMB create(clone of "make_ccfs")
    """

    if timit:
        start_time = time.time()

    # speed of light in m/s
    c = 2.99792458e8
    
    # make sure that f and f0 have the same shape
    assert f.shape == f0.shape, 'ERROR: observation and template do not have the same shape!!!'
    
    # the dummy wavelengths for orders 'order_01' and 'order_40' cannot be zero as we're taking a log!!!
    if wl.shape[0] == 40:
        if (wl[0,:] == 0).all():
            wl[0,:] = 1.
        if (wl[-1,:] == 0).all():
            wl[-1,:] = 1.
        if not synthetic_template:
            if (wl0[0,:] == 0).all():
                wl0[0,:] = 1.
            if (wl0[-1,:] == 0).all():
                wl0[-1,:] = 1.
    if wl.shape[0] == 39:
        if (wl[0,:] == 0).all():
            wl[0, :, :] = 1.
        if not synthetic_template:
            if (wl0[0,:] == 0).all():
                wl0[0,:] = 1.


    # read min and max of the wl ranges for the logwlgrid for the xcorr
    # HARD-CODED...UGLY!!!
#     min_wl_arr, max_wl_arr = np.loadtxt('C:\\veloce_data\\veloce_xcorr_wlrange.txt', usecols = (1,2), unpack=True)
    dumord, min_wl_arr, max_wl_arr = readcol('/Users/christoph/OneDrive - UNSW/dispsol/veloce_xcorr_wlrange.txt', twod=False)
   
    # prepare output variable
    xcs = []

    # loop over orders
    # for ord in sorted(f.iterkeys()):
    # for o in range(wl.shape[0]):
    # from Brendan's plots/table:
    # for o in [5, 6, 7, 17, 26, 27, 34, 35, 36, 37]:
    # Duncan's suggestion
    # for o in [4,5,6,25,26,33,34,35]:
    # for o in [5, 6, 17, 25, 26, 27, 31, 34, 35, 36, 37]:
    # don't use order 5 if some obs have ThXe (strongly affected)
    for o in [6, 17, 25, 26, 27, 31, 34, 35, 36, 37]:
    # for o in [5]:

        if debug_level >= 2:
            print('Order ' + str(o + 1).zfill(2))

        ord = 'order_' + str(o + 1).zfill(2)

        min_wl = min_wl_arr[o]
        max_wl = max_wl_arr[o]

        # now apply barycentric correction to wl and wl0 so that we can always use the SAME PARTS OF THE SPECTRUM for X-CORR
        wl_bcc = (1 + (bc-rvabs*1e3) / c) * wl
        # create logarithmic wavelength grid
        logwl = np.log(wl_bcc[o,:])
        if not synthetic_template:
            wl0_bcc = (1 + (bc0-rvabs0*1e3) / c) * wl0
            logwl0 = np.log(wl0_bcc[o,:])
        else:
            # create logarithmic wavelength grid for the synthetic template (first trim, then go to log space)
            ordix_0 = (wl0 >= np.min(wl[o,:])) & (wl0 <= np.max(wl[o,:]))
            wl0_ord = wl0[ordix_0]
            logwl0 = np.log(wl0_ord)
            f0_ord = f0[ordix_0]

        logwlgrid = np.arange(np.log(min_wl), np.log(max_wl), delta_log_wl)

        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(logwl) < 0).any():
            logwl_sorted = logwl[::-1].copy()
            ord_f_sorted = f[o,::-1].copy()
            ord_blaze_sorted = smoothed_flat[o,::-1].copy()
            # ordmask_sorted = ordmask[::-1].copy()
            if not synthetic_template:
                logwl0_sorted = logwl0[::-1].copy()
                ord_f0_sorted = f0[o,::-1].copy()
            else:
                logwl0_sorted = np.array([logwl0,])
                ord_f0_sorted = np.array([f0_ord,])
        else:
            logwl_sorted = logwl[o,:].copy()
            ord_f_sorted = f[o,:].copy()
            ord_blaze_sorted = smoothed_flat[o,:].copy()
            # ordmask_sorted = ordmask.copy()
            if not synthetic_template:
                logwl0_sorted = logwl0[o,:].copy()
                ord_f0_sorted = f0[o,:].copy()
            else:
                logwl0_sorted = np.array([logwl0,])
                ord_f0_sorted = np.array([f0_ord,])

        # rebin spectra onto logarithmic wavelength grid
        # rebinned_f0 = np.interp(logwlgrid,logwl[mask],f0_unblazed[mask])
        # rebinned_f = np.interp(logwlgrid,logwl[mask],f_unblazed[mask])
        # nfib = ord_f_sorted.shape[0] (already done above now)
        rebinned_f0 = np.zeros(len(logwlgrid))
        rebinned_f = np.zeros(len(logwlgrid))
        spl_ref_f0 = interp.InterpolatedUnivariateSpline(logwl0_sorted, ord_f0_sorted, k=3)  # slightly slower than linear, but best performance for cubic spline
        rebinned_f0 = spl_ref_f0(logwlgrid)
        spl_ref_f = interp.InterpolatedUnivariateSpline(logwl_sorted, ord_f_sorted, k=3)  # slightly slower than linear, but best performance for cubic spline
        rebinned_f = spl_ref_f(logwlgrid)
        rebinned_blaze = np.zeros(len(logwlgrid))
        spl_ref_blaze = interp.InterpolatedUnivariateSpline(logwl_sorted, ord_blaze_sorted, k=3)
        rebinned_blaze = spl_ref_blaze(logwlgrid)
            
        # normalize spectra
        rebinned_f0 /= np.max(rebinned_f0)
        rebinned_f /= np.max(rebinned_f)
        rebinned_blaze /= np.max(rebinned_blaze)
        
        # actually do the cross-correlation
        
#         xc = np.correlate(rebinned_f0, rebinned_f, mode='full')
        
#         # we already know that the two arrays we are xcorr'ing have the same length!
#         # xc = np.correlate(rebinned_f0, rebinned_f, mode='full')
#         # lags = np.arange(-(rebinned_f.size - 1), rebinned_f.size)
#         # xc /= (rebinned_f.size - np.abs(lags))
#         # OR, equivalently:
        xc = xcorr((rebinned_f0 - 1.)*rebinned_blaze, rebinned_f - 1., scale='unbiased')

        xcs.append(xc)

    if timit:
        delta_t = time.time() - start_time
        print('Time taken for creating CCFs: ' + str(np.round(delta_t, 2)) + ' seconds')

    return xcs





def make_self_indfib_ccfs(f, wl, relto=9, mask=None, smoothed_flat=None, delta_log_wl=1e-6, filter_width=25, bad_threshold=0.05, debug_level=0, timit=False):
    """
    This routine calculates the CCFs of all fibres with respect to one user-specified (default = central) fibre for a given observation.
    If the mask from "find_stripes" has gaps, do the filtering for each segment independently. If no mask is provided, create a simple one on the fly.

    INPUT:
    'f'                  : numpy array containing the observed flux (n_ord, n_fib, n_pix)
    'wl'                 : numpy array containing the wavelengths of the observed spectrum (n_ord, n_fib, n_pix)
    'relto'              : which fibre do you want to use as the reference fibre [0, 1, ... , 18]
    'mask'               : mask-dictionary from "find_stripes" (keys = orders)
    'smoothed_flat'      : if no mask is provided, need to provide the smoothed_flat, so that a mask can be created on the fly
    'delta_log_wl'       : stepsize of the log-wl grid (only used if 'relgrid' is FALSE)
    'filter_width'       : width of smoothing filter in pixels; needed b/c of edge effects of the smoothing; number of pixels to disregard should be >~ 2 * width of smoothing kernel
    'bad_threshold'      : if no mask is provided, create a mask that requires the flux in the extracted white to be larger than this fraction of the maximum flux in that order
    'debug_level'        : for debugging...
    'timit'              : boolean - do you want to measure execution run time?

    OUTPUT:
    'xcs'   : list containing the ind. fib. CCFs / sum of ind. fib. CCFs for each order (len = n_ord)

    MODHIST:
    Dec 2018 - CMB create (clone of "make_ccfs") 
    """

    assert relto in np.arange(19), 'reference fibre not recognized (must be in [0, 1, ... , 18])'

    if timit:
        start_time = time.time()

    # the dummy wavelengths for orders 'order_01' and 'order_40' cannot be zero as we're taking a log!!!
    if wl.shape[0] == 40:
        wl[0, :, :] = 1.
        wl[-1, :, :] = 1.
    if wl.shape[0] == 39:
        wl[0, :, :] = 1.

    xcs = []

    # loop over orders
    # for ord in sorted(f.iterkeys()):
    # for o in range(wl.shape[0]):
    # for o in [4,5,6,25,26,33,34,35]:
    for o in [5, 6, 17, 25, 26, 27, 31, 34, 35, 36, 37]:

        if debug_level >= 2:
            print('Order ' + str(o+1).zfill(2))

        # # only use pixels that have enough signal
        # if mask is None:
        #     normflat = smoothed_flat[ord] / np.max(smoothed_flat[ord])
        #     ordmask = np.ones(len(normflat), dtype=bool)
        #     if np.min(normflat) < bad_threshold:
        #         ordmask[normflat < bad_threshold] = False
        #         # once the blaze function falls below a certain value, exclude what's outside of that pixel column, even if it's above the threshold again, ie we want to only use a single consecutive region
        #         leftmask = ordmask[: len(ordmask) // 2]
        #         leftkill_index = [i for i, x in enumerate(leftmask) if not x]
        #         try:
        #             ordmask[: leftkill_index[0]] = False
        #         except:
        #             pass
        #         rightmask = ordmask[len(ordmask) // 2:]
        #         rightkill_index = [i for i, x in enumerate(rightmask) if not x]
        #         if ord == 'order_01' and simu:
        #             try:
        #                 # if not flipped then the line below must go in the leftkillindex thingy
        #                 # ordmask[: leftkill_index[-1] + 100] = False
        #                 ordmask[len(ordmask) // 2 + rightkill_index[0] - 100:] = False
        #             except:
        #                 pass
        #         else:
        #             try:
        #                 ordmask[len(mask) // 2 + rightkill_index[-1] + 1:] = False
        #             except:
        #                 pass
        # else:
        #     # ordmask  = mask[ord][::-1]
        #     ordmask = mask[ord]

        ordmask = np.ones(4112, dtype=bool)
        ordmask[:200] = False
        ordmask[4000:] = False


        # either way, disregard #(edge_cut) pixels at either end; this is slightly dodgy, but the gaussian filtering above introduces edge effects due to mode='reflect'
        ordmask[:2 * int(filter_width)] = False
        ordmask[-2 * int(filter_width):] = False

        # f0_unblazed = f0[ord] / np.max(f0[ord]) / normflat   #see COMMENT above
        # f_unblazed = f[ord] / np.max(f[ord]) / normflat

        # create logarithmic wavelength grid
        logwl = np.log(wl[o,:,:])
        # use range from the maximum of the indfib-minima to the minimum of the indfib-maxima - use list comprehensions...
        logwlgrid = np.arange(np.max([np.min(logwl[fib,ordmask]) for fib in range(logwl.shape[0])]), 
                              np.min([np.max(logwl[fib,ordmask]) for fib in range(logwl.shape[0])]), 
                              delta_log_wl)

        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(logwl) < 0).any():
            logwl_sorted = logwl[:, ::-1].copy()
            ordmask_sorted = ordmask[::-1].copy()
            ord_f_sorted = f[o,:,::-1].copy()
        else:
            logwl_sorted = logwl.copy()
            ordmask_sorted = ordmask.copy()
            ord_f_sorted = f[ord].copy()

        # rebin spectra onto logarithmic wavelength grid
        nfib = ord_f_sorted.shape[0]
        rebinned_f = np.zeros((nfib, len(logwlgrid)))
        for i in range(nfib):
            spl_ref_f = interp.InterpolatedUnivariateSpline(logwl_sorted[i,ordmask_sorted], ord_f_sorted[i,ordmask_sorted], k=3)  # slightly slower than linear, but best performance for cubic spline
            rebinned_f[i,:] = spl_ref_f(logwlgrid)

        
        ord_xcs = []
        for fib in range(nfib):
            xc = np.correlate(rebinned_f[fib,:], rebinned_f[relto,:], mode='full')
            ord_xcs.append(xc)
        xcs.append(ord_xcs)
        

    if timit:
        delta_t = time.time() - start_time
        print('Time taken for creating CCFs: ' + str(np.round(delta_t, 2)) + ' seconds')

    return xcs




#############################################################################################################################    
#############################################################################################################################    
#############################################################################################################################    
#############################################################################################################################    
#############################################################################################################################    

    
    
    
    
    
    
    
    
# testrv = np.array([value for (key, value) in sorted(rv.items())])
# testerr = np.array([value for (key, value) in sorted(rverr.items())])  
# testw = 1./(testerr**2)
# print(np.average(testrv, weights=testw))
#     
#     
#  
# #############################################################################################################################    
#     
#     
#     
# # or do we maybe want to cut it up into chunks, and determine a RV for every chunk???
# dum1 = (rebinned_flux1 - np.median(rebinned_flux1)).reshape((osf*16,256))
# dum3 = (rebinned_flux3 - np.median(rebinned_flux3)).reshape((osf*16,256))
# dumwl = logwlgrid.reshape((osf*16,256))   
# rv = [] 
# for i in range(len(dum1)):
#     ref = dum1[i,:]
#     flux = dum3[i,:]
#     xc = np.correlate(ref, flux, mode='same')
#     #now fit Gaussian to central section of CCF
#     fitrangesize = 9
#     xrange = np.arange(np.argmax(xc)-fitrangesize, np.argmax(xc)+fitrangesize+1, 1)
#     guess = np.array((np.argmax(xc), 3., (xc[np.argmax(xc)]-xc[np.argmax(xc)-fitrangesize])/np.max(xc), xc[np.argmax(xc)-fitrangesize]/np.max(xc), 0.))
#     #maybe use a different model, ie include a varying beta-parameter???
#     popt, pcov = op.curve_fit(gaussian_with_offset_and_slope, xrange, xc[np.argmax(xc)-fitrangesize : np.argmax(xc)+fitrangesize+1]/np.max(xc), p0=guess)
#     shift = popt[0]
#     rv.append(c * (shift - (len(xc)//2)) * delta_log_wl)
    
    
# start_time = time.time()
# for i in range(1000):
#     spl_ref1 = interp.InterpolatedUnivariateSpline(logwl, f1_unblazed, k=1)
#     rebinned_f1_xxx = spl_ref1(logwlgrid)
#     #rebinned_flux1 = np.interp(logwlgrid,logwl,f1_unblazed)
# print(str(time.time() - start_time), 'seconds')


####################################################################################################################################################3


##########################################################################################
### CMB - 15/11/2017                                                                   ###
### The following functions are based on the RV parts from Mike Ireland's "pymfe"      ###
### I also made them stand-alone routines rather than part of an object-oriented class ###
##########################################################################################    

def rv_shift_resid(parms, wl, f, err, spline_ref, return_spect=False):
    """Find the residuals to a fit of a (subsampled) reference spectrum to an 
    observed spectrum. 
    
    The function for parameters p[0] through p[3] is:
    
    .. math::
        y(x) = Ref[ wave(x) * (1 - p[0]/c) ] * exp(p[1] * x^2 + p[2] * x + p[3])
    
    Here "Ref" is a function f(wave)
    
    Parameters
    ----------        
    params: array-like
    wl: float array
        Wavelengths for the observed spectrum.        
    f: float array
       The observed spectra     
    err: float array
        standard deviation of the input spectra 
    spline_ref: InterpolatedUnivariateSpline instance
        For interpolating the reference spectrum
    return_spect: boolean
        Whether to return the fitted spectrum or the residuals.
    wave_ref: float array
        The wavelengths of the reference spectrum
    ref: float array
        The reference spectrum
    
    Returns
    -------
    resid: float array
        The fit residuals
    """
    
    # speed of light in m/s
    c = 2.99792458e8
    
    ny = len(f)
    
    # CMB change: necessary to make xx go smoothly from -0.5 to 0.5, rather than a step function (step at ny//2) from -1.0 to 0.0      
    #xx = (np.arange(ny)-ny//2)/ny
    xx = (np.arange(ny) - ny//2) / float(ny)
    norm = np.exp(parms[1]*xx*xx + parms[2]*xx + parms[3])     
    
    # Lets get this sign correct. A redshift (positive radial velocity) means that a given wavelength for the reference corresponds to a longer  
    # wavelength for the target, which in turn means that the target wavelength has to be interpolated onto shorter wavelengths for the reference.
#     fitted_spect = spline_ref(wl * (1.0 - parms[0]/c)) * norm
#     fitted_spect = spline_ref(wl * np.sqrt((1 + parms[0] / c) / (1 - parms[0] / c))) * norm   # NOPE! that's the wrong way around...
    fitted_spect = spline_ref(wl * np.sqrt((1 - parms[0] / c) / (1 + parms[0] / c))) * norm
    
    if return_spect:
        return fitted_spect
    else:
        return (fitted_spect - f)/err



def rv_shift_chi2(parms, wl, f, err, spline_ref):
    """Find the chi-squared for an RV fit. Just a wrapper for rv_shift_resid,
    so the docstring is cut and paste!
    
    The function for parameters p[0] through p[3] is:
    
    .. math::
        y(x) = Ref[ wave(x) * (1 - p[0]/c) ] * exp(p[1] * x^2 + p[2] * x + p[3])
    
    Here "Ref" is a function f(wave)
     
    Parameters
    ----------
    
    params: 
        ...
    wl: float array
        Wavelengths for the observed spectrum.
    f: float array
        The observed spectrum
    err: 
        ...
    spline_ref: 
        ...
        
    Returns
    -------
    chi2:
        The fit chi-squared
    """
    return np.sum(rv_shift_resid(parms, wl, f, err, spline_ref)**2)



def rv_shift_jac(parms, wl, f, err, spline_ref):
    """Explicit Jacobian function for rv_shift_resid. 
    
    This is not a completely analytic solution, but without it there seems to be 
    numerical instability.
    
    The key equations are:
    
    .. math:: f(x) = R( \lambda(x)  (1 - p_0/c) ) \times \exp(p_1 x^2 + p_2 x + p_3)
    
       g(x) = (f(x) - d(x))/\sigma(x)
    
       \frac{dg}{dp_0}(x) \approx  [f(x + 1 m/s) -f(x) ]/\sigma(x)
    
       \frac{dg}{dp_1}(x) = x^2 f(x) / \sigma(x)
    
       \frac{dg}{dp_2}(x) = x f(x) / \sigma(x)
    
       \frac{dg}{dp_3}(x) = f(x) / \sigma(x)
    
    Parameters
    ----------
    
    params: float array
    wl: float array
        Wavelengths for the observed spectrum.
    f: float array
        The observed spectrum
    err: float array
        Error in the observed spectrum
    spline_ref: 
        ...
        
    Returns
    -------
    jac: 
        The Jacobian.
    """
    # speed of light in m/s
    c = 2.99792458e8
    
    ny = len(f)
    
    # CMB change: necessary to make xx go smoothly from -0.5 to 0.5, rather than a step function (step at ny//2) from -1.0 to 0.0      
    #xx = (np.arange(ny)-ny//2)/ny
    xx = (np.arange(ny) - ny//2) / float(ny)
    norm = np.exp(parms[1]*xx*xx + parms[2]*xx + parms[3])     
    
    # Lets get this sign correct. A redshift (positive radial velocity) means that a given wavelength for the reference corresponds to a longer  
    # wavelength for the target, which in turn means that the target wavelength has to be interpolated onto shorter wavelengths for the reference.
    fitted_spect = spline_ref(wl * (1.0 - parms[0]/c)) * norm
#     fitted_spect = spline_ref(wave * np.sqrt((1 + parms[0] / c) / (1 - parms[0] / c))) * norm
#     fitted_spect = spline_ref(wave * np.sqrt((1 - parms[0] / c) / (1 + parms[0] / c))) * norm
    
    #The Jacobian is the derivative of fitted_spect/err with respect to p[0] through p[3]
    jac = np.empty((ny,4))
    jac[:,3] = fitted_spect / err
    jac[:,2] = fitted_spect*xx / err
    jac[:,1] = fitted_spect*xx*xx / err     
#     jac[:,0] = (spline_ref(wave*(1.0 - (parms[0] + 1.0)/const.c.si.value))*norm - fitted_spect)/spect_sdev
    jac[:,0] = ((spline_ref(wl * (1.0 - (parms[0] + 1.0)/c)) * norm) - fitted_spect) / err
    
    return jac



def calculate_rv_shift(f0, wl0, f, wl, err, bc=0, bc0=0, return_fitted_spects=False, rvguess=None, edge=20, bad_threshold=10):
    """Calculates the Radial Velocity of each spectrum
    
    The radial velocity shift of the reference spectrum required
    to match the flux in each order for one input spectrum is calculated.
    
    The input flux to this method should be flat-fielded data, which are then fitted with 
    a barycentrically corrected reference spectrum :math:`R(\lambda)`, according to 
    the following equation:

    .. math::
        f(x) = R( \lambda(x)  (1 - p_0/c) ) \\times \exp(p_1 x^2 + p_2 x + p_3)

    The first term in this equation is simply the velocity corrected spectrum, based on a 
    the arc-lamp derived reference wavelength scale :math:`\lambda(x)` for pixels coordinates x.
    The second term in the equation is a continuum normalisation - a shifted Gaussian was 
    chosen as a function that is non-zero everywhere. The scipy.optimize.leastsq function is used
    to find the best fitting set fof parameters :math:`p_0` through to :math`p_3`. 

    The reference spectrum function :math:`R(\lambda)` is created using a wavelength grid 
    which is over-sampled with respect to the data by a factor of 2. Individual fitted 
    wavelengths are then found by cubic spline interpolation on this :math:`R_j(\lambda_j)` 
    discrete grid.
    
    Parameters
    ----------
    wl0: 3D np.array(float)
        Wavelength coordinate map of form (Order, Wavelength/pixel*2+2), 
        where the wavelength scale has been interpolated.
    f0: 3D np.array(float)
        Reference spectrum of form (Order, Flux/pixel*2+2), 
        where the flux scale has been interpolated.
    f: 3D np.array(float)
        Fluxes of form (Order, Fibre, Flux/pixel)
    wl: 3D np.array(float)
        Wavelength coordinate map of form (Order, Fibre, Wavelength/pixel)   
    err: 3D np.array(float)
        Variance of form (Order, Fibre, Error/pixel)    
     

    Returns
    -------
    rvs: 2D np.array(float)
        Radial velocities of format (Observation, Order)
    rv_sigs: 2D np.array(float)
        Radial velocity sigmas of format (Observation, Order)

    TODO:
    oversample the template spectrum before regridding onto bc-shifted wl-grid of observation
    """
    
#     nm = f.shape[0]
#     nf = f.shape[1]
#     npix = f.shape[2]
    (nm, nf, npix) = f.shape
    
    # speed of light in m/s
    c = 2.99792458e8
    
    # initialise output arrays
    rvs = np.zeros( (nm,nf) )
    rv_sigs = np.zeros( (nm,nf) )
    redchi2_arr = np.zeros( (nm,nf) )
#     thefit_100 = np.zeros( (nm,nf) )

    # initial fit parameters
    initp = np.zeros(4)
#     initp[3] = 0.5
    if rvguess is not None:
        initp[0] = rvguess

    if return_fitted_spects:
        fitted_spects = np.empty(f.shape)
    
    # shift wavelength arrays by the respective barycentric corrections
#     wl_bcc = (1 + bc / c) * wave
    wl_bcc = wl * np.sqrt((1 + bc / c) / (1 - bc / c))
#     wl0_bcc = (1 + bc0 / c) * wave_ref
    wl0_bcc = wl0 * np.sqrt((1 + bc0 / c) / (1 - bc0 / c))
    
    print("Order ")
    
    # loop over all orders (skipping first order, as wl-solution is crap!)
#     for o in range(1,nm):
    for o in range(nm):

        print(str(o+1)),
        
        ord_wl = wl_bcc[o,:,:]
        ord_f = f[o,:,:]
        ord_err = err[o,:,:]
        ord_wl0 = wl0_bcc[o,:,:]
        ord_f0 = f0[o,:,:]
        
        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(ord_wl) < 0).any():
            ord_wl_sorted = ord_wl[:,::-1].copy()
            ord_f_sorted = ord_f[:,::-1].copy()
            ord_err_sorted = ord_err[:,::-1].copy()
            ord_wl0_sorted = ord_wl0[:,::-1].copy()
            ord_f0_sorted = ord_f0[:,::-1].copy()
        else:
            ord_wl_sorted = ord_wl.copy()
            ord_f_sorted = ord_f.copy()
            ord_err_sorted = ord_err.copy()
            ord_wl0_sorted = ord_wl0.copy()
            ord_f0_sorted = ord_f0.copy()
        
        nbad = 0
        
        # loop over all fibres
        for fib in range(nf):
            
            # deal with NaNs (exclude any pixels that are NaNs in any of the five arrays)
            nanpix = np.isnan(ord_wl0_sorted[fib,:]) | np.isnan(ord_wl_sorted[fib,:]) | np.isnan(ord_f0_sorted[fib,:]) | np.isnan(ord_f_sorted[fib,:]) | np.isnan(ord_err_sorted[fib,:])
            
            spl_ref = interp.InterpolatedUnivariateSpline(ord_wl0_sorted[fib,~nanpix], ord_f0_sorted[fib,~nanpix], k=3)
            args = (ord_wl_sorted[fib,~nanpix], ord_f_sorted[fib,~nanpix], ord_err_sorted[fib,~nanpix], spl_ref)
            
#             # need to turn arrays around as my wl-solution is decreasing within the order
#             spl_ref = interp.InterpolatedUnivariateSpline(wl0_bcc[o,fib,::-1], f0[o,fib,::-1], k=3)
#             args = (wl_bcc[o,fib,:], f[o,fib,:], err[o,fib,:], spl_ref)
            
            # Remove edge effects in a slightly dodgy way by making their error bars infinity and hence giving them zero weights (20 pixels is about 30km/s) 
            args[2][:edge] = np.inf
            args[2][-edge:] = np.inf
            
            # calculate the model and residuals starting with initial parameters
#             the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True, xtol=1e-6)
            the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=None, full_output=True, epsfcn=1e-3)   
            # the_fit[0] are the best-fit parms
            # the_fit[1] is the covariance matrix
            # the_fit[2] is auxiliary information on the fit (in form of a dictionary)
            # the_fit[3] is a string message giving information about the cause of failure.
            # the_fit[4] is an integer flag 
            # (if it is equal to 1, 2, 3 or 4, the solution was found. Otherwise, the solution was not found -- see op.leastsq documentation for more info)
            model_before = rv_shift_resid(the_fit[0], *args, return_spect=True)
            resid_before = rv_shift_resid(the_fit[0], *args)
            wbad = np.where(np.abs(resid_before) > bad_threshold)[0]   # I should probably return the abs residuals and then divide them by the error outside "rv_shift_resid"
            nbad += len(wbad)
            
            chi2 = rv_shift_chi2(the_fit[0], *args)
            redchi2 = chi2 / (npix - len(initp))
            
            try:
#                 errorSigma = np.sqrt(chi2 * the_fit[1][0,0])
##                 normalisedValsBefore = normalised(residBefore, errorSigma)
                
#                 fitted_spec = rv_shift_resid(the_fit[0], *args, return_spect=True)
                
                # make the errors for the "bad" pixels infinity (so as to make the weights zero)
                args[2][np.where(np.abs(resid_before) > bad_threshold)] = np.inf

#                 the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True, xtol=1e-6)
                the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=None, full_output=True, epsfcn=1e-3)   
#                 residAfter = rv_shift_resid(the_fit[0], *args)
                chi2 = rv_shift_chi2(the_fit[0], *args)
                redchi2 = chi2 / (npix - len(initp))
                
#                 errorSigma = np.sqrt(chi2 * the_fit[1][0,0])
##                 normalisedValsAfter = normalised(residAfter, errorSigma)

                redchi2_arr[o,fib] = redchi2

            except:
                pass

            
            # Some outputs for testing
            if return_fitted_spects:
                fitted_spects[o,fib,:] = rv_shift_resid(the_fit[0], *args, return_spect=True)
            
            # Save the fit and the uncertainty (the_fit[0][0] is the RV shift)
            rvs[o,fib] = the_fit[0][0]

            try:
#                 rv_sigs[o,fib] = np.sqrt(redchi2 * the_fit[1][0,0])
                rv_sigs[o,fib] = np.sqrt(the_fit[1][0,0])
            except:
                rv_sigs[o,fib] = np.nan
                
#             # the_fit[1] is the covariance matrix
#             try:
#                 thefit_100[o,fib] = the_fit[1][0,0]
#             except:
#                 thefit_100[o,fib] = np.NaN

    if return_fitted_spects:
        return rvs, rv_sigs, redchi2_arr, fitted_spects
    else:
        return rvs, rv_sigs, redchi2_arr
    
    
    
def normalised(residVals, errorSigma):
    return residVals/errorSigma



def old_calculate_rv_shift(wave_ref, ref_spect, fluxes, vars, bcors, wave, return_fitted_spects=False, bad_threshold=10):
    """
    Calculates the Radial Velocity of each spectrum!
    
    The radial velocity shift of the reference spectrum required
    to match the flux in each order in each input spectrum is calculated.
    
    The input fluxes to this method are flat-fielded data, which are then fitted with 
    a barycentrically corrected reference spectrum :math:`R(\lambda)`, according to 
    the following equation:

    .. math::
        f(x) = R( \lambda(x)  (1 - p_0/c) ) \\times \exp(p_1 x^2 + p_2 x + p_3)

    The first term in this equation is simply the velocity corrected spectrum, based on a 
    the arc-lamp derived reference wavelength scale :math:`\lambda(x)` for pixels coordinates x.
    The second term in the equation is a continuum normalisation - a shifted Gaussian was 
    chosen as a function that is non-zero everywhere. The scipy.optimize.leastsq function is used
    to find the best fitting set of parameters :math:`p_0` through to :math`p_3`. 

    The reference spectrum function :math:`R(\lambda)` is created using a wavelength grid 
    which is over-sampled with respect to the data by a factor of 2. Individual fitted 
    wavelengths are then found by cubic spline interpolation on this :math:`R_j(\lambda_j)` 
    discrete grid.
    
    Parameters
    ----------
    wave_ref: 2D np.array(float)
        Wavelength coordinate map of form (Order, Wavelength/pixel*2+2), 
        where the wavelength scale has been interpolated.
    ref_spect: 2D np.array(float)
        Reference spectrum of form (Order, Flux/pixel*2+2), 
        where the flux scale has been interpolated.
    fluxes: 3D np.array(float)
        Fluxes of form (Observation, Order, Flux/pixel)
    vars: 3D np.array(float)
        Variance of form (Observation, Order, Variance/pixel)    
    bcors: 1D np.array(float)
        Barycentric correction for each observation.
    wave: 2D np.array(float)
        Wavelength coordinate map of form (Order, Wavelength/pixel)

    Returns
    -------
    rvs: 2D np.array(float)
        Radial velocities of format (Observation, Order)
    rv_sigs: 2D np.array(float)
        Radial velocity sigmas of format (Observation, Order)
    """
    nm = fluxes.shape[1]
    ny = fluxes.shape[2]
    nf = fluxes.shape[0]
    
    # initialise output arrays
    rvs = np.zeros( (nf,nm) )
    rv_sigs = np.zeros( (nf,nm) )
    initp = np.zeros(4)
    initp[3]=0.5
    initp[0]=0.0
    spect_sdev = np.sqrt(vars)
    fitted_spects = np.empty(fluxes.shape)
    
    # loop over all fibres(?)
    for i in range(nf):
        # Start with initial guess of no intrinsic RV for the target.
        initp[0] = -bcors[i] #!!! New Change 
        nbad = 0
        #loop over all orders(?)
        for j in range(nm):
            # This is the *only* non-linear interpolation function that 
            # doesn't take forever
            spl_ref = interp.InterpolatedUnivariateSpline(wave_ref[j,::-1], ref_spect[j,::-1])
            args = (wave[j,:], fluxes[i,j,:], spect_sdev[i,j,:], spl_ref)
            
            # Remove edge effects in a slightly dodgy way. 
            # 20 pixels is about 30km/s. 
            args[2][:20] = np.inf
            args[2][-20:] = np.inf
            the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True)
            #the_fit = op.leastsq(self.rv_shift_resid, initp, args=args,diag=[1e3,1e-6,1e-3,1], full_output=True,epsfcn=1e-9)
            
            #The following line also doesn't work "out of the box".
            #the_fit = op.minimize(self.rv_shift_chi2,initp,args=args)
            #pdb.set_trace()
            #Remove bad points...
            resid = rv_shift_resid( the_fit[0], *args)
            wbad = np.where( np.abs(resid) > bad_threshold)[0]
            nbad += len(wbad)
            #15 bad pixels in a single order is *crazy*
            if len(wbad)>20:
                fitted_spect = rv_shift_resid(the_fit[0], *args, return_spect=True)
                plt.clf()
                plt.plot(args[0], args[1])
                plt.plot(args[0][wbad], args[1][wbad],'o')
                plt.plot(args[0], fitted_spect)
                plt.xlabel("Wavelength")
                plt.ylabel("Flux")
                #print("Lots of 'bad' pixels. Type c to continue if not a problem")
                #pdb.set_trace()
            
            # make the errors for the "bad" pixels infinity (so as to make the weights zero)
            args[2][wbad] = np.inf
            the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True)
            #the_fit = op.leastsq(self.rv_shift_resid, initp,args=args, diag=[1e3,1e-6,1e-3,1], full_output=True, epsfcn=1e-9)
            
            # Some outputs for testing
            fitted_spects[i,j] = rv_shift_resid(the_fit[0], *args, return_spect=True)
            # the_fit[0][0] is the RV shift
            if ( np.abs(the_fit[0][0] - bcors[i]) < 1e-4 ):
                #pdb.set_trace() #This shouldn't happen, and indicates a problem with the fit.
                pass
            #Save the fit and the uncertainty.
            rvs[i,j] = the_fit[0][0]
            try:
                rv_sigs[i,j] = np.sqrt(the_fit[1][0,0])
            except:
                rv_sigs[i,j] = np.NaN
        print("Done file {0:d}. Bad spectral pixels: {1:d}".format(i,nbad))
        
    if return_fitted_spects:
        return rvs, rv_sigs, fitted_spects
    else:
        return rvs, rv_sigs
 
#########################################################################################
#########################################################################################
#########################################################################################











