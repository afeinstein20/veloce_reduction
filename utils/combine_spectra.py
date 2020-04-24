'''
Created on 22 Aug. 2019

@author: christoph
'''

import numpy as np
import glob
import time
import astropy.io.fits as pyfits
import scipy.interpolate as interp
from scipy import ndimage
import barycorrpy

from veloce_reduction.veloce_reduction.barycentric_correction import get_bc_from_gaia
from veloce_reduction.veloce_reduction.wavelength_solution import interpolate_dispsols
from veloce_reduction.veloce_reduction.cosmic_ray_removal import onedim_medfilt_cosmic_ray_removal



def combine_fibres(f, err, wl, osf=5, fibs='stellar', ref=12, timit=False):
    
    if timit:
        start_time = time.time()  
    
    assert f.shape == (39,26,4112), 'ERROR: unknown format encountered (f does NOT have dimensions of (39,26,4112) !!!'
    assert f.shape == err.shape, 'ERROR: dimensions of flux and error arrays do not agree!!!'
    assert f.shape == wl.shape, 'ERROR: dimensions of flux and wavelength arrays do not agree!!!'
    
    #     if fibs.lower() == 'stellar':
    if fibs in ['stellar', 'Stellar', 'STELLAR']:
        fib_userange = np.arange(3,22)
        # let's use the wl-solution for the central fibre (index 12) as our reference wl solution for the stellar fibres
        ref_wl = wl[:,ref,:]
    elif fibs in ['sky', 'Sky', 'SKY']:
        fib_userange = [1, 2, 22, 23, 24]
        # let's use the wl-solution for the central fibre (index 12) as our reference wl solution for the stellar fibres
        ref_wl = wl[:,ref,:]
    else:
        fib_userange = fibs[:]
        # let's use the wl-solution for the central fibre (index 12) as our reference wl solution for the stellar fibres
        ref_wl = wl[:,ref,:]

    # prepare some arrays
    os_f = np.zeros((f.shape[0], f.shape[1], osf*f.shape[2]))
    os_err = np.zeros((f.shape[0], f.shape[1], osf*f.shape[2]))
    comb_f = np.zeros((f.shape[0], f.shape[2]))
    comb_err = np.zeros((f.shape[0], f.shape[2]))
    
    # loop over orders
    for ord in range(f.shape[0]):
        
        # make oversampled wl-grid for this order
        ord_ref_wl = ref_wl[ord,:]
        os_wlgrid_sorted = np.linspace(np.min(ord_ref_wl), np.max(ord_ref_wl), osf*len(ord_ref_wl))
    
        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(ord_ref_wl) < 0).any():
            ord_wl_sorted = wl[ord,:,::-1]
            ord_f_sorted = f[ord,:,::-1]
            ord_err_sorted = err[ord,:,::-1]
        else:
            ord_wl_sorted = wl[ord,:,:].copy()
            ord_f_sorted = f[ord,:,:].copy()
            ord_err_sorted = err[ord,:,:].copy()
        
        # loop over fibres
        for fib in fib_userange:
            # rebin spectra in individual fibres onto oversampled wavelength grid
            spl_ref_f = interp.InterpolatedUnivariateSpline(ord_wl_sorted[fib,:], ord_f_sorted[fib,:], k=3)    # slightly slower than linear, but best performance for cubic spline
            os_f[ord,fib,:] = spl_ref_f(os_wlgrid_sorted)
            spl_ref_err = interp.InterpolatedUnivariateSpline(ord_wl_sorted[fib,:], ord_err_sorted[fib,:], k=3)    # slightly slower than linear, but best performance for cubic spline
            os_err[ord,fib,:] = spl_ref_err(os_wlgrid_sorted)
    
    # now that everything is on the same wl-grid, we can add the flux-/error-arrays of the individual fibres up
    os_f_sum = np.sum(os_f, axis=1)   # strictly speaking we only want to sum over the used fibres, but the rest is zero, so that's OK
    os_err_sum = np.sqrt(np.sum(os_err**2, axis=1))
    
    # downsample to original wl grid again and turn around once more
    for ord in range(f.shape[0]):
        
        # make oversampled wl-grid for this order
        ord_ref_wl = ref_wl[ord,:]
        os_wlgrid_sorted = np.linspace(np.min(ord_ref_wl), np.max(ord_ref_wl), osf*len(ord_ref_wl))
        
        spl_ref_f = interp.InterpolatedUnivariateSpline(os_wlgrid_sorted, os_f_sum[ord,:], k=3)    # slightly slower than linear, but best performance for cubic spline
        comb_f[ord,:] = spl_ref_f(ref_wl[ord,::-1])[::-1]
        spl_ref_err = interp.InterpolatedUnivariateSpline(os_wlgrid_sorted, os_err_sum[ord,:], k=3)    # slightly slower than linear, but best performance for cubic spline
        comb_err[ord,:] = spl_ref_err(ref_wl[ord,::-1])[::-1]
        
    if timit:
        delta_t = time.time() - start_time
        print('Time elapsed: ' + str(np.round(delta_t,1)) + ' seconds')
        
    return comb_f, comb_err, ref_wl



def subtract_sky(f, err, wl, f_sky, err_sky, wl_sky, osf=5, fibs='stellar', ref='stellar', timit=False):
    '''subtract one sky spectrum (39x4112) from one stellar spectrum (39x4112) by bringing the sky spectrum to the wl of the stellar spectrum'''
    
    if timit:
        start_time = time.time()  
    
    assert f.shape == f_sky.shape, 'ERROR: stellar spectrum and sky spectrum do not have the same shape!!!'
    assert f.shape == err.shape, 'ERROR: dimensions of flux and error arrays do not agree!!!'
    assert err.shape == err_sky.shape, 'ERROR: uncertainties in stellar spectrum and sky spectrum do not have the same shape!!!'
    assert f.shape == wl.shape, 'ERROR: dimensions of flux and wavelength arrays do not agree!!!'
    assert wl.shape == wl_sky.shape, 'ERROR: wavelengh arrays for stellar spectrum and sky spectrum do not have the same shape!!!'
    
    # use the stellar wl as the reference
    ref_wl = wl.copy()
    
#     #     if fibs.lower() == 'stellar':
#     if fibs in ['stellar', 'Stellar', 'STELLAR']:
#         fib_userange = np.arange(3,22)
#         # let's use the wl-solution for the central fibre (index 12) as our reference wl solution for the stellar fibres
#         ref_wl = wl[:,ref,:]
#     elif fibs in ['sky', 'Sky', 'SKY']:
#         fib_userange = [1, 2, 22, 23, 24]
#         # let's use the wl-solution for the central fibre (index 12) as our reference wl solution for the stellar fibres
#         ref_wl = wl[:,ref,:]
#     else:
#         fib_userange = fibs[:]
#         # let's use the wl-solution for the central fibre (index 12) as our reference wl solution for the stellar fibres
#         ref_wl = wl[:,ref,:]

    # prepare some arrays
    os_f = np.zeros((f.shape[0], osf*f.shape[1]))
    os_err = np.zeros((f.shape[0], osf*f.shape[1]))
    os_f_sky = np.zeros((f_sky.shape[0], osf*f_sky.shape[1]))
    os_err_sky = np.zeros((f_sky.shape[0], osf*f_sky.shape[1]))
    f_ss = np.zeros(f.shape)
    err_ss = np.zeros(f.shape)
    
    # loop over orders
    for ord in range(f.shape[0]):
        
        # make oversampled wl-grid for this order
        ord_ref_wl = ref_wl[ord,:]
        os_wlgrid_sorted = np.linspace(np.min(ord_ref_wl), np.max(ord_ref_wl), osf*len(ord_ref_wl))
    
        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(ord_ref_wl) < 0).any():
            ord_wl_sorted = wl[ord,::-1]
            ord_f_sorted = f[ord,::-1]
            ord_err_sorted = err[ord,::-1]
            ord_wl_sky_sorted = wl_sky[ord,::-1]
            ord_f_sky_sorted = f_sky[ord,::-1]
            ord_err_sky_sorted = err_sky[ord,::-1]
        else:
            ord_wl_sorted = wl[ord,:].copy()
            ord_f_sorted = f[ord,:].copy()
            ord_err_sorted = err[ord,:].copy()
            ord_wl_sky_sorted = wl_sky[ord,:].copy()
            ord_f_sky_sorted = f_sky[ord,:].copy()
            ord_err_sky_sorted = err_sky[ord,:].copy()
        
        # rebin spectra in individual fibres onto oversampled wavelength grid
        spl_ref_f = interp.InterpolatedUnivariateSpline(ord_wl_sorted, ord_f_sorted, k=3)    # slightly slower than linear, but best performance for cubic spline
        os_f[ord,:] = spl_ref_f(os_wlgrid_sorted)
        spl_ref_err = interp.InterpolatedUnivariateSpline(ord_wl_sorted, ord_err_sorted, k=3)    # slightly slower than linear, but best performance for cubic spline
        os_err[ord,:] = spl_ref_err(os_wlgrid_sorted)
        spl_ref_f_sky = interp.InterpolatedUnivariateSpline(ord_wl_sky_sorted, ord_f_sky_sorted, k=3)    # slightly slower than linear, but best performance for cubic spline
        os_f_sky[ord,:] = spl_ref_f_sky(os_wlgrid_sorted)
        spl_ref_err_sky = interp.InterpolatedUnivariateSpline(ord_wl_sky_sorted, ord_err_sky_sorted, k=3)    # slightly slower than linear, but best performance for cubic spline
        os_err_sky[ord,:] = spl_ref_err_sky(os_wlgrid_sorted)
    
    # now that everything is on the same wl-grid, we can subtract sky flux from the stellar flux
    os_f_ss = os_f - os_f_sky
    os_err_ss = np.sqrt(os_err**2 + os_err_sky**2)
    
    # downsample to original wl grid again and turn around once more
    for ord in range(f.shape[0]):
        
        # make oversampled wl-grid for this order
        ord_ref_wl = ref_wl[ord,:]
        os_wlgrid_sorted = np.linspace(np.min(ord_ref_wl), np.max(ord_ref_wl), osf*len(ord_ref_wl))
        
        spl_ref_f = interp.InterpolatedUnivariateSpline(os_wlgrid_sorted, os_f_ss[ord,:], k=3)    # slightly slower than linear, but best performance for cubic spline
        f_ss[ord,:] = spl_ref_f(ref_wl[ord,::-1])[::-1]
        spl_ref_err = interp.InterpolatedUnivariateSpline(os_wlgrid_sorted, os_err_ss[ord,:], k=3)    # slightly slower than linear, but best performance for cubic spline
        err_ss[ord,:] = spl_ref_err(ref_wl[ord,::-1])[::-1]
        
    if timit:
        delta_t = time.time() - start_time
        print('Time elapsed: ' + str(np.round(delta_t,1)) + ' seconds')
        
    return f_ss, err_ss, ref_wl



def median_fibres(f, err, wl, osf=5, fibs='sky', ref=12, timit=False):
    
    if timit:
        start_time = time.time()  
    
    assert f.shape == (39,26,4112), 'ERROR: unknown format encountered (f does NOT have dimensions of (39,26,4112) !!!'
    assert f.shape == err.shape, 'ERROR: dimensions of flux and error arrays do not agree!!!'
    assert f.shape == wl.shape, 'ERROR: dimensions of flux and wavelength arrays do not agree!!!'
    
#     if fibs.lower() == 'stellar':
    if fibs in ['stellar', 'Stellar', 'STELLAR']:
        fib_userange = np.arange(3,22)
        # let's use the wl-solution for the central fibre (index 12) as our reference wl solution for the stellar fibres
        ref_wl = wl[:,ref,:]
    elif fibs in ['sky', 'Sky', 'SKY']:
        fib_userange = [1, 2, 22, 23, 24]
        # let's use the wl-solution for the central fibre (index 12) as our reference wl solution for the stellar fibres
        ref_wl = wl[:,ref,:]
    else:
        fib_userange = fibs[:]
        # let's use the wl-solution for the central fibre (index 12) as our reference wl solution for the stellar fibres
        ref_wl = wl[:,ref,:]

    
    # prepare some arrays
    os_f = np.zeros((f.shape[0], f.shape[1], osf*f.shape[2]))
    os_err = np.zeros((f.shape[0], f.shape[1], osf*f.shape[2]))
    comb_f = np.zeros((f.shape[0], f.shape[2]))
    comb_err = np.zeros((f.shape[0], f.shape[2]))
    
    # loop over orders
    for ord in range(f.shape[0]):
        
        # make oversampled wl-grid for this order
        ord_ref_wl = ref_wl[ord,:]
        os_wlgrid_sorted = np.linspace(np.min(ord_ref_wl), np.max(ord_ref_wl), osf*len(ord_ref_wl))
    
        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(ord_ref_wl) < 0).any():
            ord_wl_sorted = wl[ord,:,::-1]
            ord_f_sorted = f[ord,:,::-1]
            ord_err_sorted = err[ord,:,::-1]
        else:
            ord_wl_sorted = wl[ord,:,:].copy()
            ord_f_sorted = f[ord,:,:].copy()
            ord_err_sorted = err[ord,:,:].copy()
        
        # loop over fibres
        for fib in fib_userange:
            # rebin spectra in individual fibres onto oversampled wavelength grid
            spl_ref_f = interp.InterpolatedUnivariateSpline(ord_wl_sorted[fib,:], ord_f_sorted[fib,:], k=3)    # slightly slower than linear, but best performance for cubic spline
            os_f[ord,fib,:] = spl_ref_f(os_wlgrid_sorted)
            spl_ref_err = interp.InterpolatedUnivariateSpline(ord_wl_sorted[fib,:], ord_err_sorted[fib,:], k=3)    # slightly slower than linear, but best performance for cubic spline
            os_err[ord,fib,:] = spl_ref_err(os_wlgrid_sorted)
    
    # now that everything is on the same wl-grid, we can add the flux-/error-arrays of the individual fibres up (or in this case take the median...)
    os_f_med = np.nanmedian(os_f[:,fib_userange,:], axis=1)
    nfib = len(fib_userange)     # number of sky fibres 
    # err_master = 1.253 * np.std(allimg, axis=0) / np.sqrt(nfib-1)     # normally it would be sigma/sqrt(n), but np.std is dividing by sqrt(n), not by sqrt(n-1)
    os_err_med = 1.253 * np.nanstd(os_f[:,fib_userange,:] , axis=1) / np.sqrt(nfib-1)
    
    # downsample to original wl grid again and turn around once more
    for ord in range(f.shape[0]):
        
        # make oversampled wl-grid for this order
        ord_ref_wl = ref_wl[ord,:]
        os_wlgrid_sorted = np.linspace(np.min(ord_ref_wl), np.max(ord_ref_wl), osf*len(ord_ref_wl))
        
        spl_ref_f = interp.InterpolatedUnivariateSpline(os_wlgrid_sorted, os_f_med[ord,:], k=3)    # slightly slower than linear, but best performance for cubic spline
        comb_f[ord,:] = spl_ref_f(ref_wl[ord,::-1])[::-1]
        spl_ref_err = interp.InterpolatedUnivariateSpline(os_wlgrid_sorted, os_err_med[ord,:], k=3)    # slightly slower than linear, but best performance for cubic spline
        comb_err[ord,:] = spl_ref_err(ref_wl[ord,::-1])[::-1]
        
    if timit:
        delta_t = time.time() - start_time
        print('Time elapsed: ' + str(np.round(delta_t,1)) + ' seconds')
        
    return comb_f, comb_err, ref_wl



def combine_exposures(f_list, err_list, wl_list, osf=5, remove_cosmics=True, thresh=7, low_thresh=3, debug_level=0, timit=False):
    
    if timit:
        start_time = time.time()  
    
    n_exp = len(f_list)
    
    assert len(f_list) == len(err_list), 'ERROR: list of different lengths provided for flux and error!!!'
    assert len(f_list) == len(wl_list), 'ERROR: list of different lengths provided for flux and wl!!!'
    
    # OK, let's use the wl-solution for the first of the provided exposures as our reference wl solution 
    ref_wl = wl_list[0]

    # convert the lists to numpy arrays
    f_arr = np.stack(f_list, axis=0)
    err_arr = np.stack(err_list, axis=0)
    wl_arr = np.stack(wl_list, axis=0)
    cleaned_f_arr = f_arr.copy()
    
    if remove_cosmics:
        # LOOP OVER ORDERS
        for ord in range(f_list[0].shape[0]):
            if debug_level > 0:
                if ord == 0:
                    print('Cleaning cosmics from order ' + str(ord+1).zfill(2)),
                elif ord == f_list[0].shape[0]-1:
                    print(' ' + str(ord+1).zfill(2))
                else:
                    print(' ' + str(ord+1).zfill(2)),
            if n_exp == 1:
                # we have to do it the hard way...
                print('coffee???')
            elif n_exp == 2:
                scales = np.nanmedian(f_arr[:,ord,1000:3000]/f_arr[0,ord,1000:3000], axis=1)
                # take minimum image after scaling 
                min_spec = np.minimum(f_arr[0,ord,:]/scales[0], f_arr[1,ord,:]/scales[1])
                # make sure we don't have negative values for the SQRT (can happen eg b/c of bad pixels in bias subtraction)
                min_spec = np.clip(min_spec, 0, None)
                # "expected" STDEV for the minimum image (NOT the proper error of the median); (from LB Eq 2.1)
                min_sig_arr = np.sqrt(min_spec + 20**2)   # 20 ~ sqrt(19)*4.5 is the equivalent of read noise here, but that's really random; we just dont want to clean noise
                # get array containing deviations from the minimum spectrum for each exposure
                diff_spec_arr = f_arr[:,ord,:] / scales.reshape(n_exp, 1) - min_spec
                # identify cosmic-ray affected pixels
                cosmics = diff_spec_arr > thresh * min_sig_arr
                # replace cosmic-ray affected pixels by the (scaled) pixel values in the median image
                ord_cleaned_f_arr = f_arr[:,ord,:].copy()
                min_spec_arr = np.vstack([min_spec] * n_exp)
                ord_cleaned_f_arr[cosmics] = (min_spec_arr * scales.reshape(n_exp, 1))[cosmics]
                # "grow" the cosmics by 1 pixel in each direction (as in LACosmic)
                growkernel = np.zeros((3,3))
                growkernel[1,:] = np.ones(3)
                extended_cosmics = np.cast['bool'](ndimage.convolve(np.cast['float32'](cosmics), growkernel))
                cosmic_edges = np.logical_xor(cosmics, extended_cosmics)
                # now check only for these pixels surrounding the cosmics whether they are affected (but use lower threshold)
                bad_edges = np.logical_and(diff_spec_arr > low_thresh * min_sig_arr, cosmic_edges)
                ord_cleaned_f_arr[bad_edges] = (min_spec_arr * scales.reshape(n_exp, 1))[bad_edges]
                cleaned_f_arr[:,ord,:] = ord_cleaned_f_arr.copy()
            else:
                scales = np.nanmedian(f_arr[:,ord,1000:3000]/f_arr[0,ord,1000:3000], axis=1)
                # take median after scaling 
                med_spec = np.median(f_arr[:,ord,:] / scales.reshape(n_exp, 1), axis=0)
                # make sure we don't have negative values for the SQRT (can happen eg b/c of bad pixels in bias subtraction)
                med_spec = np.clip(med_spec, 0, None)
                # "expected" STDEV for the median spectrum (NOT the proper error of the median); (from LB Eq 2.1)
                med_sig_arr = np.sqrt(med_spec + 20**2)   # 20 ~ sqrt(19)*4.5 is the equivalent of read noise here, but that's really random; we just dont want to clean noise
                # get array containing deviations from the median spectrum for each exposure
                diff_spec_arr = f_arr[:,ord,:] / scales.reshape(n_exp, 1) - med_spec
                # identify cosmic-ray affected pixels
                cosmics = diff_spec_arr > thresh * med_sig_arr
                # replace cosmic-ray affected pixels by the (scaled) pixel values in the median image
                ord_cleaned_f_arr = f_arr[:,ord,:].copy()
                med_spec_arr = np.vstack([med_spec] * n_exp)
                ord_cleaned_f_arr[cosmics] = (med_spec_arr * scales.reshape(n_exp, 1))[cosmics]
                # "grow" the cosmics by 1 pixel in each direction (as in LACosmic)
                growkernel = np.zeros((3,3))
                growkernel[1,:] = np.ones(3)
                extended_cosmics = np.cast['bool'](ndimage.convolve(np.cast['float32'](cosmics), growkernel))
                cosmic_edges = np.logical_xor(cosmics, extended_cosmics)
                # now check only for these pixels surrounding the cosmics whether they are affected (but use lower threshold)
                bad_edges = np.logical_and(diff_spec_arr > low_thresh * med_sig_arr, cosmic_edges)
                ord_cleaned_f_arr[bad_edges] = (med_spec_arr * scales.reshape(n_exp, 1))[bad_edges]
                cleaned_f_arr[:,ord,:] = ord_cleaned_f_arr.copy()
    
    
    # prepare some arrays
    os_f = np.zeros((f_arr.shape[0], f_list[0].shape[0], osf*f_list[0].shape[1]))
    os_err = np.zeros((f_arr.shape[0], f_list[0].shape[0], osf*f_list[0].shape[1]))
    comb_f = np.zeros(f_list[0].shape)
    comb_err = np.zeros(f_list[0].shape)
    
    # loop over orders
    for ord in range(f_list[0].shape[0]):
        
        # make oversampled wl-grid for this order
        ord_ref_wl = ref_wl[ord,:]
        os_wlgrid_sorted = np.linspace(np.min(ord_ref_wl), np.max(ord_ref_wl), osf*len(ord_ref_wl))
    
        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(ord_ref_wl) < 0).any():
            ord_wl_sorted = wl_arr[:,ord,::-1]
            ord_f_sorted = cleaned_f_arr[:,ord,::-1]
            ord_err_sorted = err_arr[:,ord,::-1]
        else:
            ord_wl_sorted = wl_arr[:,ord,:].copy()
            ord_f_sorted = cleaned_f_arr[:,ord,:].copy()
            ord_err_sorted = err_arr[:,ord,:].copy()
        
        # loop over individual exposures
        for exp in range(f_arr.shape[0]):
            # rebin spectra in individual fibres onto oversampled wavelength grid
            spl_ref_f = interp.InterpolatedUnivariateSpline(ord_wl_sorted[exp,:], ord_f_sorted[exp,:], k=3)    # slightly slower than linear, but best performance for cubic spline
            os_f[exp,ord,:] = spl_ref_f(os_wlgrid_sorted)
            spl_ref_err = interp.InterpolatedUnivariateSpline(ord_wl_sorted[exp,:], ord_err_sorted[exp,:], k=3)    # slightly slower than linear, but best performance for cubic spline
            os_err[exp,ord,:] = spl_ref_err(os_wlgrid_sorted)
    
    # now that everything is on the same wl-grid, we can add the flux-/error-arrays of the individual fibres up
    os_f_sum = np.sum(os_f, axis=0)
    os_err_sum = np.sqrt(np.sum(os_err**2, axis=0))
    
    # downsample to original wl grid again and turn around once more
    for ord in range(f_list[0].shape[0]):
        
        # make oversampled wl-grid for this order
        ord_ref_wl = ref_wl[ord,:]
        os_wlgrid_sorted = np.linspace(np.min(ord_ref_wl), np.max(ord_ref_wl), osf*len(ord_ref_wl))
        
        spl_ref_f = interp.InterpolatedUnivariateSpline(os_wlgrid_sorted, os_f_sum[ord,:], k=3)    # slightly slower than linear, but best performance for cubic spline
        comb_f[ord,:] = spl_ref_f(ref_wl[ord,::-1])[::-1]
        spl_ref_err = interp.InterpolatedUnivariateSpline(os_wlgrid_sorted, os_err_sum[ord,:], k=3)    # slightly slower than linear, but best performance for cubic spline
        comb_err[ord,:] = spl_ref_err(ref_wl[ord,::-1])[::-1]        
        
    if timit:
        delta_t = time.time() - start_time
        print('Time elapsed: ' + str(np.round(delta_t,1)) + ' seconds')
        
    return comb_f, comb_err, ref_wl



def main_script_for_sarah(date = '20190722', skysub=False):
    # Gaia DR2 ID dictionary for Zucker July 2019 targets
    hipnum_dict = {'10144': 7588, 
                   '121263': 68002,
                   '175191': 92855,
                   'HD 206739 (std)': 107337}
    
    gaia_dict = {'105435': 6126469654573573888,
                 '118716': 6065752767077605504,
                 '120324': 6109087784487667712,
                 '132058': 5908509891187794176,
                 '143018': 6235406071206201600,
                 '157246': 5922299343254265088,
                 '209952': 6560604777053880704,    
                 'HD 140283 (std)': 6268770373590148224,
                 'HE 0015+0048': 2547143725127991168,
                 'HE 0107-5240': 4927204800008334464,
                 'CS 29504-006': 5010164739030492032,
                 'CS 22958-042': 4718427642340545408,
                 'HE 1133-0555': 3593627144045992832,
                 'HE 1249-3121': 6183795820024064256,
                 'HE 1310-0536': 3635533208672382592,
                 'CS 22877-001': 3621673727165280384,
                 'HE 1327-2326': 6194815228636688768,
                 'G64-12': 3662741860852094208,
                 'G64-37': 3643857920443831168,
                 'HE 1410+0213': 3667206118578976896,
                 'BD-18 5550': 6867802519062194560,
                 'BPS CS 30314-067': 6779790049231492096,
                 'CS 29498-043': 6788448668941293952,
                 'HE 2139-5432 ': 6461736966363075200,
                 'BPS CS 29502-092': 2629500925618285952,
                 'HE 2302-2154a': 2398202677437168384,
                 'CD-24 17504': 2383484851010749568,
                 'HE 2318-1621': 2406023396270909440,
                 'HE 2319-5228': 6501398446721935744,
                 '6182748015506372480': 6182748015506372480,
                 '6192933650707925376': 6192933650707925376,
                 '6192500855443308160': 6192500855443308160,
                 '6194706681927050496': 6194706681927050496,
                 '6190169375397005824': 6190169375397005824,
                 '6190736590253462784': 6190736590253462784,
                 '151008003501121': 2558459589561967232,
                 '141031003601274': 2977723336242924544,
                 '140311007101309': 5363629792898912512,
                 '150408004101222': 5398144047005910656,
                 '170130004601208': 3698111844248492160,
                 '170506003901265': 6140829138994504960,
                 '160403004701275': 3616785740848955776,
                 '140310004701055': 3673146848623371264,
                 '160520004901236': 5818849184718555392,
                 '170711003001241': 4377886454310583168,
                 '140711001901267': 5809854183164908928,
                 '170615004401258': 6702907209758894848,
                 '170912002401113': 6888748417431916928,
                 '160724002601324': 1733472307022576384,
                 '140810004201231': 6579952677010742272,
                 '171106002401258': 2668887906026528000,
                 '140812004401091': 6397474768030945152,
                 '140810005301179': 6406537325120547456,
                 '140805004201070': 6381051156688800896,
                 '170711005801135': 6485376840021854848}
    

    # assign wl-solutions to stellar spectra by linear interpolation between a library of fibThXe dispsols
    path = '/Volumes/BERGRAID/data/veloce/reduced/' + date + '/'
    
    air_wl_list = glob.glob(path + 'fibth_dispsols/' + '*air*.fits')
    air_wl_list.sort()
    vac_wl_list = glob.glob(path + 'fibth_dispsols/' + '*vac*.fits')
    vac_wl_list.sort()
     
    fibth_obsnames = [fn.split('_air_')[0][-10:] for fn in air_wl_list]
    # arc_list = glob.glob(path + 'calibs/' + 'ARC*optimal*.fits')   
#     used_fibth_list = [path + 'calibs/' + date + '_ARC - ThAr_' + obsname + '_optimal3a_extracted.fits' for obsname in fibth_obsnames]
    used_fibth_list = [path + 'calibs/' + 'ARC - ThAr_' + obsname + '_optimal3a_extracted.fits' for obsname in fibth_obsnames]
    stellar_list = glob.glob(path + 'stellar_only/' + '*optimal*.fits')
    stellar_list.sort()
    # stellar_list_quick = glob.glob(path + 'stellar_only/' + '*quick*.fits')
    # stellar_list_quick.sort()
     
    t_calibs = np.array([pyfits.getval(fn, 'UTMJD') + 0.5*pyfits.getval(fn, 'ELAPSED')/86400. for fn in used_fibth_list])
    # t_stellar = [pyfits.getval(fn, 'UTMJD') + 0.5*pyfits.getval(fn, 'ELAPSED')/86400. for fn in stellar_list]
    
    
    ### STEP 1: create (39 x 26 x 4112) wavelength solution for every stellar observation by linearly interpolating between wl-solutions of surrounding fibre ThXe exposures
    # loop over all stellar observations
    for i,file in enumerate(stellar_list):
        if i==0:
            print('STEP 1: wavelength solutions')
        print(str(i+1)+'/'+str(len(stellar_list)))
        # get observation midpoint in time
        tobs = pyfits.getval(file, 'UTMJD') + 0.5*pyfits.getval(file, 'ELAPSED')/86400.
        
        # find the indices of the ARC files bracketing the stellar observations
        above = np.argmax(t_calibs > tobs)   # first occurrence where t_calibs are larger than tobs
        below = above - 1
        # get obstimes and wl solutions for these ARC exposures
        t1 = t_calibs[below]
        t2 = t_calibs[above] 
        wl1 = pyfits.getdata(air_wl_list[below])
        wl2 = pyfits.getdata(air_wl_list[above])
        # do a linear interpolation to find the wl-solution at t=tobs
        wl = interpolate_dispsols(wl1, wl2, t1, t2, tobs)
        # append this wavelength solution to the extracted spectrum FITS files
        pyfits.append(file, wl, overwrite=True)
    
    
    ### STEP 2: append barycentric correction!?!?!?
    # loop over all stellar observations
    for i,file in enumerate(stellar_list):
        if i==0:
            print
            print('STEP 3: appending barycentric correction')
        print(str(i+1)+'/'+str(len(stellar_list)))
        
        # get object name
        object = pyfits.getval(file, 'OBJECT').split('+')[0]
        # get observation midpoint in time (in JD)
        jd = pyfits.getval(file, 'UTMJD') + 0.5*pyfits.getval(file, 'ELAPSED')/86400. + 2.4e6 + 0.5
        # get Gaia DR2 ID from object
        if object in gaia_dict.keys():
            gaia_dr2_id = gaia_dict[object]
            # get barycentric correction from Gaia DR2 ID and obstime
            try:
                bc = get_bc_from_gaia(gaia_dr2_id, jd)[0]
            except:
                bc = get_bc_from_gaia(gaia_dr2_id, jd)
        else:
            hipnum = hipnum_dict[object]
            bc = barycorrpy.get_BC_vel(JDUTC=jd, hip_id=hipnum, obsname='AAO', ephemeris='de430')[0][0]
        
        bc = np.round(bc,2)
        assert not np.isnan(bc), 'ERROR: could not calculate barycentric correction for '+file
        print('barycentric correction for object ' + object + ' :  ' + str(bc) + ' m/s')
        
        # write the barycentric correction into the FITS header of both the quick-extracted and the optimal-extracted reduced spectrum files
        pyfits.setval(file, 'BARYCORR', value=bc, comment='barycentric velocity correction [m/s]')
    
    
    ### STEP 3: combine the flux in all fibres for each exposure (by going to a common wl-grid (by default the one for the central fibre) and get median sky spectrum
    # loop over all stellar observations
    for i,file in enumerate(stellar_list):
        if i==0:
            print
            print('STEP 2: combining fibres')
        print(str(i+1)+'/'+str(len(stellar_list)))
    
        # read in extracted spectrum file
        f = pyfits.getdata(file, 0)
        err = pyfits.getdata(file, 1)
        wl = pyfits.getdata(file, 2)
        h = pyfits.getheader(file, 0)
        h_err = pyfits.getheader(file, 1)
        
        # combine the stellar fibres
        comb_f, comb_err, ref_wl = combine_fibres(f, err, wl, osf=5, fibs='stellar')
        
        # combine sky fibres (4 if LC was on, 5 otherwise), then take the median
        h = pyfits.getheader(file)
        assert 'LCNEXP' in h.keys(), 'ERROR: not the latest version of the FITS headers !!! (from May 2019 onwards)'
        if ('LCEXP' in h.keys()) or ('LCMNEXP' in h.keys()):   # this indicates the LFC actually was actually exposed (either automatically or manually)
            comb_f_sky, comb_err_sky, ref_wl_sky = median_fibres(f, err, wl, osf=5, fibs=[1, 2, 22, 23])   # we don't want to use the sky fibre right next to the LFC if the LFC was on!!!
        else:
            comb_f_sky, comb_err_sky, ref_wl_sky = median_fibres(f, err, wl, osf=5, fibs='sky'
        
        # save to new FITS file
        outpath = path + 'fibres_combined/'
        fname = file.split('/')[-1]
        new_fn = outpath + fname.split('.')[0] + '_stellar_fibres_combined.fits'
        pyfits.writeto(new_fn, comb_f, h, overwrite=True)
        pyfits.append(new_fn, comb_err, h_err, overwrite=True)
        pyfits.append(new_fn, ref_wl, overwrite=True)
        sky_fn = outpath + fname.split('.')[0] + '_median_sky.fits'
        pyfits.writeto(sky_fn, comb_f_sky, h, overwrite=True)
        pyfits.append(sky_fn, comb_err_sky, h_err, overwrite=True)
        pyfits.append(sky_fn, ref_wl_sky, overwrite=True)
    
    
    ### STEP 4: combine all single-shot exposures for each target and do sky-subtraction, and flux-weighting of barycentric correction
    # first we need to make a new list for the combined-fibre spectra 
    fc_stellar_list = glob.glob(path + 'fibres_combined/' + '*optimal*stellar*.fits')
    fc_stellar_list.sort()
    sky_list = glob.glob(path + 'fibres_combined/' + '*optimal*sky*.fits')
    sky_list.sort()
    
    object_list = [pyfits.getval(file, 'OBJECT').split('+')[0] for file in fc_stellar_list]
    
    # loop over all stellar observations
    for i,(file,skyfile) in enumerate(zip(fc_stellar_list, sky_list)):
        if i==0:
            print
            print('STEP 4: combining single-shot exposures')
        print(str(i+1) + '/' + str(len(fc_stellar_list)))
        
        # get headers
        h = pyfits.getheader(file, 0)
        h_err = pyfits.getheader(file, 1)
        
        # get object name
        object = pyfits.getval(file, 'OBJECT').split('+')[0]
        
        # make list that keeps a record of which observations feed into the combined final one
        used_obsnames = [(fn.split('/')[-1]).split('_')[1] for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        # add this information to the fits headers
        h['N_EXP'] = (len(used_obsnames), 'number of single-shot exposures')
        h_err['N_EXP'] = (len(used_obsnames), 'number of single-shot exposures')
        for j in range(len(used_obsnames)):
            h['EXP_' + str(j+1)] = (used_obsnames[j], 'name of single-shot exposure')
            h_err['EXP_' + str(j+1)] = (used_obsnames[j], 'name of single-shot exposure')
        
#         # make lists containing the (sky-subtracted) flux, error, and wl-arrays for the fibre-combined optimal extracted spectra  
#         f_list = [pyfits.getdata(fn,0) - 19*pyfits.getdata(skyfn,0) for fn,skyfn,obj in zip(fc_stellar_list, sky_list, object_list) if obj == object]
#         err_list = [np.sqrt(pyfits.getdata(fn,1)**2 + 19*pyfits.getdata(skyfn,1)**2) for fn,skyfn,obj in zip(fc_stellar_list, sky_list, object_list) if obj == object]
#         wl_list = [pyfits.getdata(fn,2) for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
#         # combine the single-shot exposures
#         comb_f, comb_err, ref_wl = combine_exposures(f_list, err_list, wl_list, osf=5, remove_cosmics=True, thresh=7, low_thresh=3, debug_level=0, timit=False)
        
        if skysub:
            f_list = []
            err_list = []
            wl_list = []
            for fn,skyfn,obj in zip(fc_stellar_list, sky_list, object_list):
                if obj == object:
                    F, ERR, WL = subtract_sky(pyfits.getdata(fn,0), pyfits.getdata(fn,1), pyfits.getdata(fn,2), 19*pyfits.getdata(skyfn,0), np.sqrt(19)*pyfits.getdata(skyfn,1), pyfits.getdata(skyfn,2))
                    f_list.append(F)
                    err_list.append(ERR)
                    wl_list.append(WL)
        else: 
            f_list = [pyfits.getdata(fn,0) for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
            err_list = [pyfits.getdata(fn,1) for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
            wl_list = [pyfits.getdata(fn,2) for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        # combine the single-shot exposures
        comb_f, comb_err, ref_wl = combine_exposures(f_list, err_list, wl_list, osf=5, remove_cosmics=True, thresh=7, low_thresh=3, debug_level=0, timit=False)
        
        # make list of the barycentric correction and exposure for every single-shot exposure
        bc_list = [pyfits.getval(fn, 'BARYCORR') for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        texp_list = [pyfits.getval(fn, 'ELAPSED') for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        # now assign weights based on exposure time and get weighted mean for b.c. (that only works well if the seeing was roughly constant and there were no clouds, as it should really be FLUX-weighted)
        wm_bc = np.average(bc_list, weights=texp_list)  
        
        # save to new FITS file(s)
#         outpath = path + 'final_combined_spectra/'
#         new_fn = outpath + object + '_final_combined.fits'
#         pyfits.writeto(new_fn, comb_f, h, overwrite=True)
#         pyfits.append(new_fn, comb_err, h_err, overwrite=True)
#         pyfits.append(new_fn, ref_wl, overwrite=True)
        # write the barycentric correction into the FITS header of both the quick-extracted and the optimal-extracted reduced spectrum files
#         pyfits.setval(new_fn, 'BARYCORR', value=wm_bc, comment='barycentric velocity correction [m/s]')

        # save to new FITS file(s)
        outpath = path + 'final_combined_spectra/'
        if skysub:
            new_fn = outpath + object + '_final_combined_sky_subtracted.fits'
            h['SKYSUB'] = 'TRUE'
        else:
            new_fn = outpath + object + '_final_combined.fits'
            h['SKYSUB'] = 'FALSE'
        pyfits.writeto(new_fn, comb_f, h, overwrite=True)
        pyfits.append(new_fn, comb_err, h_err, overwrite=True)
        pyfits.append(new_fn, ref_wl, overwrite=True)
        # write the barycentric correction into the FITS header of both the quick-extracted and the optimal-extracted reduced spectrum files
        pyfits.setval(new_fn, 'BARYCORR', value=wm_bc, comment='barycentric velocity correction [m/s]')

    return 1





def main_script_for_timtim(date = '20190621'):
    
    print(date)
    
    # barycentric correction is now already written into the FITS headers of the reduced spectra
    
    # assign wl-solutions to stellar spectra by linear interpolation between a library of fibThXe dispsols
    path = '/Volumes/BERGRAID/data/veloce/white_and_bedding/' + date + '/'
    
    air_wl_list = glob.glob(path + 'fibth_dispsols/' + '*air*.fits')
    air_wl_list.sort()
    vac_wl_list = glob.glob(path + 'fibth_dispsols/' + '*vac*.fits')
    vac_wl_list.sort()
     
    fibth_obsnames = [fn.split('_air_')[0][-10:] for fn in air_wl_list]
    # arc_list = glob.glob(path + 'calibs/' + 'ARC*optimal*.fits')   
    used_fibth_list = [path + 'calibs/' + 'ARC - ThAr_' + obsname + '_optimal3a_extracted.fits' for obsname in fibth_obsnames]
    stellar_list = glob.glob(path + 'stellar_only/' + '*optimal*.fits')
    stellar_list.sort()
    # stellar_list_quick = glob.glob(path + 'stellar_only/' + '*quick*.fits')
    # stellar_list_quick.sort()
     
    t_calibs = np.array([pyfits.getval(fn, 'UTMJD') + 0.5*pyfits.getval(fn, 'ELAPSED')/86400. for fn in used_fibth_list])
    # t_stellar = [pyfits.getval(fn, 'UTMJD') + 0.5*pyfits.getval(fn, 'ELAPSED')/86400. for fn in stellar_list]
    
    
    ### STEP 1: create (39 x 26 x 4112) wavelength solution for every stellar observation by linearly interpolating between wl-solutions of surrounding fibre ThXe exposures
    # loop over all stellar observations
    for i,file in enumerate(stellar_list):
        if i==0:
            print('STEP 1: wavelength solutions')
        print(str(i+1)+'/'+str(len(stellar_list)))
        # get observation midpoint in time
        tobs = pyfits.getval(file, 'UTMJD') + 0.5*pyfits.getval(file, 'ELAPSED')/86400.
        
        # find the indices of the ARC files bracketing the stellar observations
        above = np.argmax(t_calibs > tobs)   # first occurrence where t_calibs are larger than tobs
        below = above - 1
        # get obstimes and wl solutions for these ARC exposures
        t1 = t_calibs[below]
        t2 = t_calibs[above] 
        wl1 = pyfits.getdata(air_wl_list[below])
        wl2 = pyfits.getdata(air_wl_list[above])
        # do a linear interpolation to find the wl-solution at t=tobs
        wl = interpolate_dispsols(wl1, wl2, t1, t2, tobs)
        # append this wavelength solution to the extracted spectrum FITS files
        pyfits.append(file, wl, overwrite=True)
    
    
    #     ### STEP 2: append barycentric correction!?!?!?
    #     # loop over all stellar observations
    #     for i,file in enumerate(stellar_list):
    #         if i==0:
    #             print
    #             print('STEP 3: appending barycentric correction')
    #         print(str(i+1)+'/'+str(len(stellar_list)))
    #         
    #         # get object name
    #         object = pyfits.getval(file, 'OBJECT').split('+')[0]
    #         # get observation midpoint in time (in JD)
    #         jd = pyfits.getval(file, 'UTMJD') + 0.5*pyfits.getval(file, 'ELAPSED')/86400. + 2.4e6 + 0.5
    #         # get Gaia DR2 ID from object
    #         if object in gaia_dict.keys():
    #             gaia_dr2_id = gaia_dict[object]
    #             # get barycentric correction from Gaia DR2 ID and obstime
    #             try:
    #                 bc = get_bc_from_gaia(gaia_dr2_id, jd)[0]
    #             except:
    #                 bc = get_bc_from_gaia(gaia_dr2_id, jd)
    #         else:
    #             hipnum = hipnum_dict[object]
    #             bc = barycorrpy.get_BC_vel(JDUTC=jd, hip_id=hipnum, obsname='AAO', ephemeris='de430')[0][0]
    #         
    #         bc = np.round(bc,2)
    #         assert not np.isnan(bc), 'ERROR: could not calculate barycentric correction for '+file
    #         print('barycentric correction for object ' + object + ' :  ' + str(bc) + ' m/s')
    #         
    #         # write the barycentric correction into the FITS header of both the quick-extracted and the optimal-extracted reduced spectrum files
    #         pyfits.setval(file, 'BARYCORR', value=bc, comment='barycentric velocity correction [m/s]')
    
    
    ### STEP 3: combine the flux in all fibres for each exposure (by going to a common wl-grid (by default the one for the central fibre) and get median sky spectrum
    # loop over all stellar observations
    for i,file in enumerate(stellar_list):
        if i==0:
            print
            print('STEP 2: combining fibres')
        print(str(i+1) + '/' + str(len(stellar_list)))
    
        # read in extracted spectrum file
        f = pyfits.getdata(file, 0)
        err = pyfits.getdata(file, 1)
        wl = pyfits.getdata(file, 2)
        h = pyfits.getheader(file, 0)
        h_err = pyfits.getheader(file, 1)
        
        # combine sky fibres (3, if LFC & ThXe were on, 4 if LC was on, 5 otherwise), then take the median
        h = pyfits.getheader(file)
        if file in [path + 'stellar_only/HD222496_21jun30131_optimal3a_extracted.fits', path + 'stellar_only/HD222496_21jun30132_optimal3a_extracted.fits']:
            # both simcalibs were on
            comb_f_sky, comb_err_sky, ref_wl_sky = median_fibres(f, err, wl, osf=5, fibs=[2, 22, 23])
        elif file in [path + 'stellar_only/HD17693_22jun30231_optimal3a_extracted.fits', path + 'stellar_only/HD199247_15apr30270_optimal3a_extracted.fits', path + 'stellar_only/HD200835_15apr30274_optimal3a_extracted.fits']:
            # only LFC was on
            comb_f_sky, comb_err_sky, ref_wl_sky = median_fibres(f, err, wl, osf=5, fibs=[1, 2, 22, 23])
        else:
            # neither simcalib was on
            comb_f_sky, comb_err_sky, ref_wl_sky = median_fibres(f, err, wl, osf=5, fibs='sky')
            
        # sky subtraction (for stellar fibres only)
        f_ss = np.zeros(f.shape)
        err_ss = np.zeros(f.shape)
        for o in range(f.shape[0]):
            for fib in range(3,22):
                f_ss[o,fib,:] = f[o,fib,:] -  comb_f_sky[o,:]
                err_ss[o,fib,:] = np.sqrt(err[o,fib,:]**2 + comb_err_sky[o,:]**2)
        
        # remove cosmics (from stellar fibres only)
        f_clean = np.zeros(f.shape)
        for o in range(f.shape[0]):
            for fib in range(3,22):
                f_clean[o,fib,:],ncos = onedim_medfilt_cosmic_ray_removal(f_ss[o,fib,:], err_ss[o,fib,:], w=31, thresh=5., low_thresh=3.)
                
        # now combine the sky-subtracted and cosmic-cleaned stellar fibres
        comb_f, comb_err, ref_wl = combine_fibres(f_clean, err_ss, wl, osf=5, fibs='stellar')
        
        # save to new FITS file
        outpath = path + 'fibres_combined/'
        fname = file.split('/')[-1]
        new_fn = outpath + fname.split('.')[0] + '_stellar_fibres_combined.fits'
        pyfits.writeto(new_fn, comb_f, h, overwrite=True)
        pyfits.append(new_fn, comb_err, h_err, overwrite=True)
        pyfits.append(new_fn, ref_wl, overwrite=True)
        sky_fn = outpath + fname.split('.')[0] + '_median_sky.fits'
        pyfits.writeto(sky_fn, comb_f_sky, h, overwrite=True)
        pyfits.append(sky_fn, comb_err_sky, h_err, overwrite=True)
        pyfits.append(sky_fn, ref_wl_sky, overwrite=True)
    
    
    ### STEP 4: combine all single-shot exposures for each target and do sky-subtraction, and flux-weighting of barycentric correction
    # first we need to make a new list for the combined-fibre spectra 
    fc_stellar_list = glob.glob(path + 'fibres_combined/' + '*optimal*stellar*.fits')
    fc_stellar_list.sort()
#     sky_list = glob.glob(path + 'fibres_combined/' + '*optimal*sky*.fits')
#     sky_list.sort()
     
    object_list = [pyfits.getval(file, 'OBJECT').split('+')[0] for file in fc_stellar_list]
    
    # speed of light in m/s
    c = 2.99792458e8
     
    # loop over all stellar observations
#     for i,(file,skyfile) in enumerate(zip(fc_stellar_list, sky_list)):
    for i,file in enumerate(fc_stellar_list):
        if i==0:
            print
            print('STEP 4: combining single-shot exposures')
        print(str(i+1)+'/'+str(len(fc_stellar_list)))
         
        # get headers
        h = pyfits.getheader(file, 0)
        h_err = pyfits.getheader(file, 1)
         
        # get object name
        object = pyfits.getval(file, 'OBJECT').split('+')[0]
         
        # make list that keeps a record of which observations feed into the combined final one
        used_obsnames = [(fn.split('/')[-1]).split('_')[1] for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        # add this information to the fits headers
        h['N_EXP'] = (len(used_obsnames), 'number of single-shot exposures')
        h_err['N_EXP'] = (len(used_obsnames), 'number of single-shot exposures')
        for j in range(len(used_obsnames)):
            h['EXP_' + str(j+1)] = (used_obsnames[j], 'name of single-shot exposure')
            h_err['EXP_' + str(j+1)] = (used_obsnames[j], 'name of single-shot exposure')
         
        # make lists containing the flux, error, wl-arrays, barycentric correction, and exposure times for the fibre-combined optimal extracted spectra  
        f_list = [pyfits.getdata(fn,0) for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        err_list = [pyfits.getdata(fn,1) for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        wl_list = [pyfits.getdata(fn,2) for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        bc_list = [pyfits.getval(fn, 'BARYCORR') for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        texp_list = [pyfits.getval(fn, 'ELAPSED') for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
         
        # only do this if there are actually 2 or more single-shot exposures
        if len(used_obsnames) > 1:
            # now shift the wl-grids by the difference in b.c. (use the first one in the list as the reference one)
            wl_list_shifted = [wl*(1. + (bc-bc_list[0])/c) for wl,bc in zip(wl_list, bc_list)]
            # combine the single-shot exposures (uses wl-grid of first exposure in list)
            comb_f, comb_err, ref_wl = combine_exposures(f_list, err_list, wl_list_shifted, osf=5, remove_cosmics=True, thresh=7, low_thresh=3, debug_level=0, timit=False)
        else:
            comb_f = f_list[0].copy()
            comb_err = err_list[0].copy()
            ref_wl = wl_list[0].copy()
         
        # we used the barycentric correction of the first spectrum as the reference, so need to write that to FITS header of combined multi-shot spectrum
        bc = bc_list[0]  
         
        # save to new FITS file(s)
        outpath = path + 'final_combined_spectra/'
        new_fn = outpath + object + '_final_combined.fits'
        pyfits.writeto(new_fn, comb_f, h, overwrite=True)
        pyfits.append(new_fn, comb_err, h_err, overwrite=True)
        pyfits.append(new_fn, ref_wl, overwrite=True)
        # write the barycentric correction into the FITS header of both the quick-extracted and the optimal-extracted reduced spectrum files
        pyfits.setval(new_fn, 'BARYCORR', value=bc, comment='barycentric velocity correction [m/s]')

    return 1





def main_script_for_bouma(date = '30200130', skysub=False):
    
    print(date)
    
    # barycentric correction is now already written into the FITS headers of the reduced spectra
    
    # assign wl-solutions to stellar spectra by linear interpolation between a library of fibThXe dispsols
    path = '/Volumes/BERGRAID/data/veloce/reduced/' + date + '/'
    
    air_wl_list = glob.glob(path + 'fibth_dispsols/' + '*air*.fits')
    air_wl_list.sort()
    vac_wl_list = glob.glob(path + 'fibth_dispsols/' + '*vac*.fits')
    vac_wl_list.sort()
     
    fibth_obsnames = [fn.split('_air_')[0][-10:] for fn in air_wl_list]
    # arc_list = glob.glob(path + 'calibs/' + 'ARC*optimal*.fits')   
    used_fibth_list = [path + 'calibs/' + date + '_ARC - ThAr_' + obsname + '_optimal3a_extracted.fits' for obsname in fibth_obsnames]
    stellar_list = glob.glob(path + 'stellar_only/' + '*optimal*.fits')
    stellar_list.sort()
    stellar_obsnames = [(fn.split('/')[-1]).split('_')[2] for fn in stellar_list]
    # stellar_list_quick = glob.glob(path + 'stellar_only/' + '*quick*.fits')
    # stellar_list_quick.sort()
     
    t_calibs = np.array([pyfits.getval(fn, 'UTMJD') + 0.5*pyfits.getval(fn, 'ELAPSED')/86400. for fn in used_fibth_list])
    # t_stellar = [pyfits.getval(fn, 'UTMJD') + 0.5*pyfits.getval(fn, 'ELAPSED')/86400. for fn in stellar_list]
    
    
    ### STEP 1: create (39 x 26 x 4112) wavelength solution for every stellar observation by linearly interpolating between wl-solutions of surrounding fibre ThXe exposures
    # loop over all stellar observations
    for i,file in enumerate(stellar_list):
        if i==0:
            print('STEP 1: wavelength solutions')
        print(str(i+1)+'/'+str(len(stellar_list)))
        # get observation midpoint in time
        tobs = pyfits.getval(file, 'UTMJD') + 0.5*pyfits.getval(file, 'ELAPSED')/86400.
        
        # find the indices of the ARC files bracketing the stellar observations
        above = np.argmax(t_calibs > tobs)   # first occurrence where t_calibs are larger than tobs
        below = above - 1
        # get obstimes and wl solutions for these ARC exposures
        t1 = t_calibs[below]
        t2 = t_calibs[above] 
        wl1 = pyfits.getdata(air_wl_list[below])
        wl2 = pyfits.getdata(air_wl_list[above])
        # do a linear interpolation to find the wl-solution at t=tobs
        wl = interpolate_dispsols(wl1, wl2, t1, t2, tobs)
        # append this wavelength solution to the extracted spectrum FITS files
        pyfits.append(file, wl, overwrite=True)
    
# #     # alternatively, if there are LFC wl-solutions available
#     for filename,obsname in zip(stellar_list[:5], stellar_obsnames[:5]):
#         wldict, wl = get_dispsol_for_all_fibs_3(obsname, date='20190126')
#         pyfits.append(filename, wl[:-1,:,:])
#  
#     for filename,obsname in zip(stellar_list[5:], stellar_obsnames[5:]):
#         wldict, wl = get_dispsol_for_all_fibs_3(obsname, date='20190522')
#         pyfits.append(filename, wl[:-1,:,:])
    
    
    
    
#     ### STEP 2: append barycentric correction!?!?!?
#     # loop over all stellar observations
#     for i,file in enumerate(stellar_list):
#         if i==0:
#             print
#             print('STEP 3: appending barycentric correction')
#         print(str(i+1)+'/'+str(len(stellar_list)))
#         
#         # get object name
#         object = pyfits.getval(file, 'OBJECT').split('+')[0]
#         # get observation midpoint in time (in JD)
#         jd = pyfits.getval(file, 'UTMJD') + 0.5*pyfits.getval(file, 'ELAPSED')/86400. + 2.4e6 + 0.5
#         # get Gaia DR2 ID from object
#         if object in gaia_dict.keys():
#             gaia_dr2_id = gaia_dict[object]
#             # get barycentric correction from Gaia DR2 ID and obstime
#             try:
#                 bc = get_bc_from_gaia(gaia_dr2_id, jd)[0]
#             except:
#                 bc = get_bc_from_gaia(gaia_dr2_id, jd)
#         else:
#             hipnum = hipnum_dict[object]
#             bc = barycorrpy.get_BC_vel(JDUTC=jd, hip_id=hipnum, obsname='AAO', ephemeris='de430')[0][0]
#         
#         bc = np.round(bc,2)
#         assert not np.isnan(bc), 'ERROR: could not calculate barycentric correction for '+file
#         print('barycentric correction for object ' + object + ' :  ' + str(bc) + ' m/s')
#         
#         # write the barycentric correction into the FITS header of both the quick-extracted and the optimal-extracted reduced spectrum files
#         pyfits.setval(file, 'BARYCORR', value=bc, comment='barycentric velocity correction [m/s]')
    
    
    ### STEP 3: combine the flux in all fibres for each exposure (by going to a common wl-grid (by default the one for the central fibre) and get median sky spectrum
    # loop over all stellar observations
    for i,file in enumerate(stellar_list):
        if i==0:
            print
            print('STEP 2: combining fibres')
        print(str(i+1) + '/' + str(len(stellar_list)))
    
        # read in extracted spectrum file
        f = pyfits.getdata(file, 0)
        err = pyfits.getdata(file, 1)
        wl = pyfits.getdata(file, 2)
        h = pyfits.getheader(file, 0)
        h_err = pyfits.getheader(file, 1)
        
        # combine the stellar fibres
        comb_f, comb_err, ref_wl = combine_fibres(f, err, wl, osf=5, fibs='stellar')
        
        # combine sky fibres (N=3, exclude the ones next to SimTh (only SimTh was on for 20200130 and 20200131)), then take the median        
        f_sky, err_sky, wl_sky = median_fibres(f, err, wl, osf=5, fibs=[22, 23, 24])
                                          
        # save to new FITS files
        outpath = path + 'fibres_combined/'
        fname = file.split('/')[-1]
        new_fn = outpath + fname.split('.')[0] + '.' + fname.split('.')[1] + '_stellar_fibres_combined.fits'
#         new_fn = outpath + fname.split('.')[0] + '_stellar_fibres_combined.fits'
        pyfits.writeto(new_fn, comb_f, h, overwrite=True)
        pyfits.append(new_fn, comb_err, h_err, overwrite=True)
        pyfits.append(new_fn, ref_wl, overwrite=True)
        sky_fn = outpath + fname.split('.')[0] + '.' + fname.split('.')[1] + '_median_sky.fits'
#         sky_fn = outpath + fname.split('.')[0] + '_median_sky.fits'
        pyfits.writeto(sky_fn, f_sky, h, overwrite=True)
        pyfits.append(sky_fn, err_sky, h_err, overwrite=True)
        pyfits.append(sky_fn, wl_sky, overwrite=True)
    
    
    ### STEP 4: combine all single-shot exposures for each target and do sky-subtraction, and flux-weighting of barycentric correction
    # first we need to make a new list for the combined-fibre spectra 
    fc_stellar_list = glob.glob(path + 'fibres_combined/' + '*optimal*stellar*.fits')
    fc_stellar_list.sort()
    sky_list = glob.glob(path + 'fibres_combined/' + '*optimal*sky*.fits')
    sky_list.sort()
     
    object_list = [pyfits.getval(file, 'OBJECT').split('+')[0] for file in fc_stellar_list]
    
    # speed of light in m/s
    c = 2.99792458e8
     
    # loop over all stellar observations
    for i,(file,skyfile) in enumerate(zip(fc_stellar_list, sky_list)):
        if i==0:
            print
            print('STEP 3: combining single-shot exposures')
        print(str(i+1) + '/' + str(len(fc_stellar_list)))
         
        # get headers
        h = pyfits.getheader(file, 0)
        h_err = pyfits.getheader(file, 1)
         
        # get object name
        object = pyfits.getval(file, 'OBJECT').split('+')[0]
         
        # make list that keeps a record of which observations feed into the combined final one
#         used_obsnames = [(fn.split('/')[-1]).split('_')[3] for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        used_obsnames = [(fn.split('/')[-1]).split('_')[2] for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        
        # add this information to the fits headers
        h['N_EXP'] = (len(used_obsnames), 'number of single-shot exposures')
        h_err['N_EXP'] = (len(used_obsnames), 'number of single-shot exposures')
        for j in range(len(used_obsnames)):
            h['EXP_' + str(j+1)] = (used_obsnames[j], 'name of single-shot exposure')
            h_err['EXP_' + str(j+1)] = (used_obsnames[j], 'name of single-shot exposure')
         
        # make lists containing the (sky-subtracted) flux, error, wl-arrays, barycentric correction, and exposure times for the fibre-combined optimal extracted spectra  
#         bc_list = [pyfits.getval(fn, 'BARYCORR') for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
#         texp_list = [pyfits.getval(fn, 'ELAPSED') for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
        if skysub:
            f_list = []
            err_list = []
            wl_list = []
            for fn,skyfn,obj in zip(fc_stellar_list, sky_list, object_list):
                if obj == object:
                    F, ERR, WL = subtract_sky(pyfits.getdata(fn,0), pyfits.getdata(fn,1), pyfits.getdata(fn,2), 19*pyfits.getdata(skyfn,0), np.sqrt(19)*pyfits.getdata(skyfn,1), pyfits.getdata(skyfn,2))
                    f_list.append(F)
                    err_list.append(ERR)
                    wl_list.append(WL)
        else: 
            f_list = [pyfits.getdata(fn,0) for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
            err_list = [pyfits.getdata(fn,1) for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
            wl_list = [pyfits.getdata(fn,2) for fn,obj in zip(fc_stellar_list, object_list) if obj == object]
         
        # only do this if there are actually 2 or more single-shot exposures
        if len(used_obsnames) > 1:
            # now shift the wl-grids by the difference in b.c. (use the first one in the list as the reference one)
#             wl_list_shifted = [wl*(1. + (bc-bc_list[0])/c) for wl,bc in zip(wl_list, bc_list)]
            wl_list_shifted = [wl * np.sqrt((1 + (bc-bc_list[0]) / c) / (1 - (bc-bc_list[0]) / c)) for wl,bc in zip(wl_list, bc_list)]
            # combine the single-shot exposures (uses wl-grid of first exposure in list)
            comb_f, comb_err, ref_wl = combine_exposures(f_list, err_list, wl_list_shifted, osf=5, remove_cosmics=True, thresh=7, low_thresh=3, debug_level=0, timit=False)
        else:
            comb_f = f_list[0].copy()
            comb_err = err_list[0].copy()
            ref_wl = wl_list[0].copy()
         
        # we used the barycentric correction of the first spectrum as the reference, so need to write that to FITS header of combined multi-shot spectrum
        bc = bc_list[0]  
         
        # save to new FITS file(s)
        outpath = path + 'final_combined_spectra/'
        if skysub:
            new_fn = outpath + object + '_final_combined_sky_subtracted.fits'
            h['SKYSUB'] = 'TRUE'
        else:
            new_fn = outpath + object + '_final_combined.fits'
            h['SKYSUB'] = 'FALSE'
        pyfits.writeto(new_fn, comb_f, h, overwrite=True)
        pyfits.append(new_fn, comb_err, h_err, overwrite=True)
        pyfits.append(new_fn, ref_wl, overwrite=True)
        # write the barycentric correction into the FITS header of both the quick-extracted and the optimal-extracted reduced spectrum files
        pyfits.setval(new_fn, 'BARYCORR', value=bc, comment='barycentric velocity correction [m/s]')

    return 1
