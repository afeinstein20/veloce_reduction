'''
Created on 12 Jul. 2018

@author: christoph
'''

import numpy as np
import time
import datetime
import astropy.io.fits as pyfits
import os

from veloce_reduction.veloce_reduction.helper_functions import fibmodel_with_amp, make_norm_profiles_6, short_filenames
from veloce_reduction.veloce_reduction.spatial_profiles import fit_single_fibre_profile
from veloce_reduction.veloce_reduction.linalg import linalg_extract_column
from veloce_reduction.veloce_reduction.order_tracing import flatten_single_stripe, flatten_single_stripe_from_indices, extract_stripes
from veloce_reduction.veloce_reduction.relative_intensities import get_relints





def quick_extract(stripes, err_stripes, slit_height=30, skip_first_order=False, debug_level=0, timit=False):
    """
    This routine performs a quick-look reduction of an echelle spectrum, by simply adding up the flux in a pixel column
    perpendicular to the dispersion direction. Similar to the tramline extraction in "collapse_extract", but even sloppier
    as edge effects (ie fractional pixels) are not taken into account.
    
    INPUT:
    'stripes'     : dictionary (keys = orders) containing the 2-dim stripes (ie the to-be-extracted regions centred on the orders) of the spectrum
    'err_stripes' : dictionary (keys = orders) containing the errors in the 2-dim stripes (ie the to-be-extracted regions centred on the orders) of the spectrum    
    'slit_height' : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'skip_first_order'   : boolean - do you want to skip order 01 (causes problems as not fully on chip, and especially b/c LFC trace is rubbish)
    'verbose'     : boolean - for debugging...
    'timit'       : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'pixnum'  : dictionary (keys = orders) containing the pixel numbers (in dispersion direction)
    'flux'    : dictionary (keys = orders) containing the summed up (ie collapsed) flux
    'err'     : dictionary (keys = orders) containing the uncertainty in the summed up (ie collapsed) flux (including photon noise and read-out noise)
    """
    
    if debug_level > 1:
        print('Performing quick-look extraction of orders...')
    
    if timit:    
        start_time = time.time()
    
    flux = {}
    err = {}
    pixnum = {}

    useful_orders = sorted(stripes.keys())
    if skip_first_order:
        del useful_orders[0]

    # loop over all orders
    for ord in useful_orders:
        if debug_level > 1:
            print('OK, now processing order '+str(ord)+'...')
        if timit:
            order_start_time = time.time()
        
        # define stripe
        stripe = stripes[ord]
        err_stripe = err_stripes[ord]
        # find and fill the "order-box"
        sc, sr = flatten_single_stripe(stripe, slit_height=slit_height, timit=False)
        err_sc, err_sr = flatten_single_stripe(err_stripe, slit_height=slit_height, timit=False)
        # get dimensions of the box
        ny, nx = sc.shape
        
        flux[ord] = np.sum(sc, axis=0)
        err[ord] = np.sqrt(np.sum(err_sc*err_sc, axis=0))
        pixnum[ord] = np.arange(nx) + 1
    
        if debug_level > 0:
            print('Time taken for quick-look extraction of '+ord+': '+str(time.time() - order_start_time)+' seconds')
    
    
    if timit:
        print('Time taken for quick-look extraction of spectrum: '+str(time.time() - start_time)+' seconds')
    
    if debug_level > 1:
        print('Extraction complete! Coffee time...')
    
    return pixnum,flux,err





def quick_extract_from_indices(img, err_img, stripe_indices, slit_height=30, skip_first_order=False, debug_level=0, timit=False):
    """
    This routine performs a quick-look reduction of an echelle spectrum, by simply adding up the flux in a pixel column
    perpendicular to the dispersion direction. Similar to the tramline extraction in "collapse_extract", but even sloppier
    as edge effects (ie fractional pixels) are not taken into account.
    
    INPUT:
    'img'            : 2-dim input array
    'err_img'        : 2-dim array of the corresponding errors
    'stripe_indices' : dictionary (keys = orders) containing the indices of the pixels that are identified as the "stripes" (ie the to-be-extracted regions centred on the orders)
    'slit_height'    : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'skip_first_order'   : boolean - do you want to skip order 01 (causes problems as not fully on chip, and especially b/c LFC trace is rubbish)
    'debug_level'    : boolean - for debugging...
    'timit'          : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'pixnum'  : dictionary (keys = orders) containing the pixel numbers (in dispersion direction)
    'flux'    : dictionary (keys = orders) containing the summed up (ie collapsed) flux
    'err'     : dictionary (keys = orders) containing the uncertainty in the summed up (ie collapsed) flux (including photon noise and read-out noise)
    """
    
    if debug_level > 0:
        print('ATTENTION: This routine works fine, but consider using "quick_extract", as it is a factor of ~2 faster...')
    
    if debug_level > 1:
        print('Performing quick-look extraction of orders...')
    
    if timit:    
        start_time = time.time()
    
    flux = {}
    err = {}
    pixnum = {}

    useful_orders = sorted(stripe_indices.keys())
    if skip_first_order:
        del useful_orders[0]

    # loop over all orders
    for ord in useful_orders:
        if debug_level > 1:
            print('OK, now processing order '+str(ord)+'...')
        if timit:
            order_start_time = time.time()
        
        # define indices
        indices = stripe_indices[ord]
        # find and fill the "order-box"
        sc,sr = flatten_single_stripe_from_indices(img,indices,slit_height=slit_height,timit=False)
        err_sc,err_sr = flatten_single_stripe_from_indices(err_img,indices,slit_height=slit_height,timit=False)
        # get dimensions of the box
        ny,nx = sc.shape
        
        flux[ord] = np.sum(sc,axis=0)
        err[ord] = np.sqrt(np.sum(err_sc*err_sc,axis=0))
        pixnum[ord] = np.arange(nx) + 1
    
        if debug_level > 0:
            print('Time taken for quick-look extraction of '+ord+': '+str(time.time() - order_start_time)+' seconds')
    
    
    if timit:
        print('Time taken for quick-look extraction of spectrum: '+str(time.time() - start_time)+' seconds')
    
    if debug_level > 1:
        print('Extraction complete! Coffee time...')
    
    return pixnum,flux,err





def collapse_extract_single_cutout(cutout, err_cutout, top, bottom):
    
    x = np.arange(len(cutout))
    inner_range = np.logical_and(x >= np.ceil(bottom), x <= np.floor(top))
    top_frac = top - np.floor(top)
    bottom_frac = np.ceil(bottom) - bottom
    flux = np.sum(cutout[inner_range]) + top_frac * cutout[int(np.ceil(top))] + bottom_frac * cutout[int(np.floor(bottom))] 
    err = np.sqrt(np.sum(err_cutout*err_cutout))
#     n = np.sum(inner_range)     # as in my thesis; sum is fine because inner_range is boolean
#     w = n + 2                   # as in my thesis
    
    return flux, err





def collapse_extract_order(ordnum, data, err_data, row_ix, upper_boundary, lower_boundary):
    
    flux,err = (np.zeros(len(upper_boundary)),np.zeros(len(upper_boundary)))
    pixnum = []
    
    for i in range(data.shape[1]):
        pixnum.append(ordnum+str(i+1).zfill(4))
        cutout = data[:,i]
        err_cutout = err_data[:,i]
        top = upper_boundary[i] - row_ix[0,i]
        bottom = lower_boundary[i] - row_ix[0,i]
        if top>=0 and bottom>=0:
            if top<=data.shape[0] and bottom <=data.shape[0] and top>bottom:
                # this is the normal case, where the entire cutout lies on the chip
                f,e = collapse_extract_single_cutout(cutout, err_cutout, top, bottom)
            else:
                print('ERROR: Tramlines are not properly defined!!!')
                quit()
        else:
            # just output zero if any bit of the cutout falls outside the chip
            f,e = (0.,0.)
        flux[i] = f
        err[i] = e
    
    return pixnum, flux, err





def collapse_extract(stripes, err_stripes, tramlines, slit_height=25, verbose=False, timit=False):
    
    if verbose:
        print('Collapsing and extracting orders...')
    
    if timit:    
        start_time = time.time()
    
    flux = {}
    err = {}
    pixnum = {}
    
#     for ord in stripes.keys():
    for ord in tramlines.keys():
        if verbose:
            print('OK, now processing order '+str(ord)+'...')
        if timit:
            order_start_time = time.time()
        
        #order number
        ordnum = ord[-2:]
        
        # define stripe
        stripe = stripes[ord]
        err_stripe = err_stripes[ord]
        # find the "order-box"
        sc,sr = flatten_single_stripe(stripe, slit_height=slit_height, timit=False)
        err_sc,err_sr = flatten_single_stripe(err_stripe, slit_height=slit_height, timit=False)
        
        # define upper and lower extraction boundaries
        upper_boundary = tramlines[ord]['upper_boundary']
        lower_boundary = tramlines[ord]['lower_boundary']
        # call extraction routine for this order 
        pix,f,e = collapse_extract_order(ordnum, sc, err_sc, sr, upper_boundary, lower_boundary)
        
        flux[ord] = f
        err[ord] = e
        pixnum[ord] = pix
    
        if timit:
            print('Time taken for extraction of '+ord+': '+str(time.time() - order_start_time)+' seconds')
    
    
    if timit:
        print('Time taken for extraction of spectrum: '+str(time.time() - start_time)+' seconds')
    
    if verbose:
        print('Extraction complete! Have a coffee now...')
    
    return pixnum,flux,err





def collapse_extract_from_indices(img, err_img, stripe_indices, tramlines, slit_height=25, verbose=False, timit=False):
    
    if verbose:
        print('Collapsing and extracting orders...')
    
    if timit:    
        start_time = time.time()
    
    flux = {}
    err = {}
    pixnum = {}
    
#     for ord in stripes.keys():
    for ord in tramlines.keys():
        if verbose:
            print('OK, now processing order '+str(ord)+'...')
        if timit:
            order_start_time = time.time()
        
        #order number
        ordnum = ord[-2:]
        
        # define stripe
        #stripe = stripes[ord]
        indices = stripe_indices[ord]
        # find the "order-box"
        #sc,sr = flatten_single_stripe(stripe,slit_height=slit_height,timit=False)
        sc,sr = flatten_single_stripe_from_indices(img, indices, slit_height=slit_height, timit=False)
        err_sc,err_sr = flatten_single_stripe_from_indices(err_img, indices, slit_height=slit_height, timit=False)
        
        # define upper and lower extraction boundaries
        upper_boundary = tramlines[ord]['upper_boundary']
        lower_boundary = tramlines[ord]['lower_boundary']
        # call extraction routine for this order 
        pix,f,e = collapse_extract_order(ordnum, sc, err_sc, sr, upper_boundary, lower_boundary, RON=RON, gain=gain)
        
        flux[ord] = f
        err[ord] = e
        pixnum[ord] = pix
    
        if timit:
            print('Time taken for extraction of '+ord+': '+str(time.time() - order_start_time)+' seconds')
    
    
    if timit:
        print('Time taken for extraction of spectrum: '+str(time.time() - start_time)+' seconds')
    
    if verbose:
        print('Extraction complete! Have a coffee now...')
    
    return pixnum,flux,err





def optimal_extraction(stripes, err_stripes=None, ron_stripes=None, slit_height=30, date=None, pathdict=None, fibs='all', relints=None, skip_first_order=False,
                       simu=False, phi_onthefly=False, individual_fibres=True, combined_profiles=False, integrate_profiles=False,
                       slope=False, offset=False, collapse=False, debug_level=0, timit=False):

    """
    This routine performs the optimal extraction of an echelle spectrum following the formalism described in Sharp & Birchall 2010, PASA, 27:91.
    Output is saved in dictionaries ("pix", "flux", "err").
    The exact behaviour can be controlled via a variety of keywords (see more description of (3a), (3b) & (3c) in "extract_spectrum").

    INPUT:
    'stripes'            : dictionary (keys = orders) containing the 2-dim stripes (ie the to-be-extracted regions centred on the orders) of the spectrum
    'err_stripes'        : dictionary (keys = orders) containing the errors in the 2-dim stripes (ie the to-be-extracted regions centred on the orders) of the spectrum
    'ron_stripes'        : dictionary (keys = orders) containing the read-out noise in the 2-dim stripes (ie the to-be-extracted regions centred on the orders) of the spectrum
    'slit_height'        : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'date'               : the date ('YYYYMMDD') the obervations were taken (so that the sorresponding (pre-determined) fibre profiles can be loaded)
    'pathdict'           : dictionary containing all directories relevant to the reduction
    'fibs'               : which fibres are you using? ['all', 'stellar', 'sky2', 'sky3', 'allsky']
    'relints'            : an array of the relative intensities in the stellar fibres (only needed if 'individual_fibres' == FALSE and 'combined_profiles' == TRUE)
    'skip_first_order'   : boolean - do you want to skip order 01 (causes problems as not fully on chip, and especially b/c LFC trace is rubbish)
    'simu'               : boolean - are you using simulated spectra?
    'phi_onthefly'       : boolean - CODING RELIC - TO BE REMOVED; creates fibre profiles on the fly
    'individual_fibres'  : boolean - do you want to extract spectra for each fibre individually?
    'combined_profiles'  : boolean - do you want to use a combined profile for each 'object' (stellar / sky / laser / thxe)? ignored if 'individual_fibres' is set to TRUE
    'integrate_profiles' : boolean - do you want to integrate the fibre profiles (better but MUCH slower), or just evaluate them at the pixel centres?
    'slope'              : boolean - do you want to include a slope as an "extra fibre"?
    'offset'             : boolean - do you want to include an offset as an "extra fibre"?
    'collapse'           : boolean - set this keyword to simply do a collapse extract (not recommended - this is a CODING RELIC - TO BE REMOVED; use routine "quick_extract" instead)
    'debug_level'        : for debugging...
    'timit'              : boolean - do you want to measure execution run time?

    OUTPUT:
    'pix'   : dictionary (keys = orders) containing the pixel numbers (in dispersion direction)
    'flux'  : dictionary (keys = orders) containing the extracted flux (ie the eta's in Sharp & Birchall)
    'err'   : dictionary (keys = orders) containing the uncertainty in the extracted flux

    TODO:
    incorporate the sim ThXe profiles

    """

    if timit:
        start_time = time.time()

    if date is None:
        date = raw_input("Please enter date of observations 'YYYYMMDD': ")

    if err_stripes is None:
        print('WARNING: errors not provided! Using sqrt(RON**2 + flux) as an estimate...')

    if fibs.lower() == 'all':
#         nfib = 24
        nfib = 26
    elif fibs.lower() == 'stellar':
        nfib = 19
    elif fibs.lower() == 'sky2':
        nfib = 2
    elif fibs.lower() == 'sky3':
        nfib = 3
    elif fibs.lower() == 'allsky':
        nfib = 5
    elif fibs.lower() in ['thxe', 'simth']:
        nfib = 1
    elif fibs.lower() in ['lfc', 'laser']:
        nfib = 1
    elif fibs.lower() == 'calibs':
        nfib = 2
    else:
        print('WARNING: input for "fibs" not recognized - using all 26 fibres as a default...')
#         nfib = 24
        nfib = 26

    # read in polynomial coefficients of best-fit individual-fibre-profile parameters
    if simu:
        fibparms = np.load(pathdict['fp'] + 'sim/fibparms_by_ord.npy').item()
    else:
        if date not in ['20181116', '20190127', '20190201']:
#             fibparms = np.load(pathdict['fp'] + 'archive/fibre_profile_fits_' + date + '.npy').item()
            fibparms = np.load(pathdict['fp'] + 'archive/combined_fibre_profile_fits_' + date + '.npy').item()
            if debug_level > 0:
                print('OK, loading fibre profiles for ' + date + '...')
        else:
            # have to laod this crutch, as the first order fits were crap for 20181116 / 20190127 / 20190201, so just for order 01 I replaced them with the parms from the following night
#             fibparms = np.load(pathdict['fp'] + 'archive/fibre_profile_fits_' + date + '_crutch.npy').item()
            fibparms = np.load(pathdict['fp'] + 'archive/combined_fibre_profile_fits_' + date + '_crutch.npy').item()
            if debug_level > 0:
                print('OK, loading fibre profiles (CRUTCH!!!) for ' + date + '...')

    flux = {}
    err = {}
    pix = {}

    useful_orders = sorted(stripes.keys())
    if skip_first_order:
        del useful_orders[0]

    # loop over all orders
    for ord in useful_orders:

        if timit and (debug_level > 0):
            order_start_time = time.time()

        # order number
        ordnum = ord[-2:]

        if ord == useful_orders[0]:
            print('OK, now processing order: ' + ordnum),
        elif ord == useful_orders[-1]:
            print(' ' + ordnum)
        else:
            print(' ' + ordnum),

        # fibre profile parameters for that order
        fppo = fibparms[ord]

        # define stripe
        stripe = stripes[ord]
        ron_stripe = ron_stripes[ord]

        # find the "order-box"
        sc, sr = flatten_single_stripe(stripe, slit_height=slit_height, timit=False)
        ron_sc, ron_sr = flatten_single_stripe(ron_stripe, slit_height=slit_height, timit=False)

        if err_stripes is not None:
            err_stripe = err_stripes[ord]
            err_sc, err_sr = flatten_single_stripe(err_stripe, slit_height=slit_height, timit=False)

        npix = sc.shape[1]

        flux[ord] = {}
        err[ord] = {}
        pix[ord] = []
        if not phi_onthefly and not collapse:
            if individual_fibres:
                for j in range(nfib):
                    fib = 'fibre_' + str(j + 1).zfill(2)
                    flux[ord][fib] = []
                    err[ord][fib] = []
            else:
                flux[ord]['laser'] = []
                err[ord]['laser'] = []
                flux[ord]['sky'] = []
                err[ord]['sky'] = []
                flux[ord]['stellar'] = []
                err[ord]['stellar'] = []
                flux[ord]['thxe'] = []
                err[ord]['thxe'] = []

        #         if individual_fibres:
        #             f_ord = np.zeros((nfib,npix))
        #             e_ord = np.zeros((nfib,npix))
        #         else:
        #             f_ord = np.zeros(npix)
        #             e_ord = np.zeros(npix)

        goodrange = np.arange(npix)
        if simu and ord == 'order_01':
            goodrange = np.arange(1300, 4096)
            for j in range(1300):
                pix[ord].append(ordnum + str(j + 1).zfill(4))

        for i in goodrange:
            if debug_level > 0:
                print('pixel ' + str(i + 1) + '/' + str(npix))
            pix[ord].append(ordnum + str(i + 1).zfill(4))
            z = sc[:, i].copy()
            if simu:
                z -= 1.  # note the minus 1 is because we added 1 artificially at the beginning in order for "extract_stripes" to work properly
            roncol = ron_sc[:, i].copy()

            # if error is not provided, estimate it (NOT RECOMMENDED!!!)
            if err_stripes is None:
                pixerr = np.sqrt(ron_sc[:, i] ** 2 + np.abs(z))
            else:
                pixerr = err_sc[:, i].copy()

            # assign weights for flux (and take care of NaNs and INFs)
            pix_w = 1. / (pixerr * pixerr)

            ###initially I thought this was clearly rubbish as it down-weights the central parts
            # and that we really want to use the relative errors, ie w_i = 1/(relerr_i)**2
            # relerr = pixerr / z     ### the pixel err
            # pix_w = 1. / (relerr)**2
            # HOWEVER: this is not true, and the optimal extraction linalg routine requires absolute errors!!!

            # check for NaNs etc
            pix_w[np.isinf(pix_w)] = 0.

            if phi_onthefly:
                quickfit = fit_single_fibre_profile(sr[:, i], z)
                bestparms = np.array([quickfit.best_values['mu'], quickfit.best_values['sigma'],
                                      quickfit.best_values['amp'], quickfit.best_values['beta']])
                phi = fibmodel_with_amp(sr[:, i], *bestparms)
                phi /= np.max([np.sum(phi), 0.001])  # we can do this because grid-stepsize = 1; also make sure that we do not divide by zero
                phi = phi.reshape(len(phi), 1)  # stupid python...
            else:
                # get normalized profiles for all fibres for this cutout
                if combined_profiles:
                    phi_laser = np.sum(make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, fibs='laser'), axis=1)
                    phi_thxe = np.sum(make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, fibs='thxe'), axis=1)
                    phis_sky3 = make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, fibs='sky3')
                    phi_sky3 = np.sum(phis_sky3, axis=1) / 3.
                    phis_stellar = make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, fibs='stellar')
                    phi_stellar = np.sum(phis_stellar * relints, axis=1)
                    phis_sky2 = make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, fibs='sky2')
                    phi_sky2 = np.sum(phis_sky2, axis=1) / 2.
                    phi_sky = (phi_sky3 + phi_sky2) / 2.
                    phi = np.vstack((phi_laser, phi_sky, phi_stellar, phi_thxe)).T
                else:
                    # phi = make_norm_profiles(sr[:,i], ord, i, fibparms)
                    # phi = make_norm_profiles_temp(sr[:,i], ord, i, fibparms)
                    # phi = make_norm_single_profile_temp(sr[:,i], ord, i, fibparms)
                    phi = make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, slope=slope, offset=offset, fibs=fibs)

            # print('WARNING: TEMPORARY offset correction is not commented out!!!')
            # # subtract the median as the offset if BG is not properly corrected for
            # z -= np.median(z)

            # do the optimal extraction
            if not collapse:
                if np.sum(phi) == 0:
                    # f,v = (0.,np.sqrt(len(phi)*RON*RON))
                    f, v = (0., np.sqrt(np.sum(pixerr * pixerr)))
                else:
                    # THIS IS THE NORMAL CASE!!!
                    # NOTE: take the read-out noise as the average of the individual-pixel read-out noise values over
                    # the cutout, as it can change if we cross a quadrant boundary!
                    f, v = linalg_extract_column(z, pix_w, phi, RON=np.mean(roncol))
            else:
                # f,v = (np.sum(z-np.median(z)), np.sum(z-np.median(z)) + len(phi)*RON*RON)   ### background should already be taken care of here...
                # f,v = (np.sum(z), np.sum(z) + len(phi)*RON*RON)
                f, v = (np.sum(z), np.sqrt(np.sum(pixerr * pixerr)))

            # e = np.sqrt(v)
            # model = np.sum(f*phi,axis=1)

            # fill output arrays depending on the selected method
            if not phi_onthefly and not collapse:

                # theoretically there should not be negative values, but we will allow them (bias and dark subtraction can produce those)
                # f[f < 0] = 0.
                # however, their errors are treated below
                # not sure if this is the proper way to do this, but we can't have negative variance
                # v[np.logical_or(v<=0,f<=0)] = RON*RON
                # v[v<RON*RON] = np.maximum(RON*RON,1.)   #just a stupid fix so that variance is never below 1
                v[np.logical_or(v <= 0, f <= 0)] = np.nanmean(roncol) ** 2
                v[v < np.nanmean(roncol) ** 2] = np.maximum(np.nanmean(roncol) ** 2, 1.)  # just a stupid fix so that variance is never below 1

                if individual_fibres:
                    ### THIS IS METHOD (3a) - PREFERRED OPTION! ###
                    # fill flux- and error- output arrays for individual fibres
                    for j in range(nfib):
                        fib = 'fibre_' + str(j + 1).zfill(2)
                        flux[ord][fib].append(f[j])
                        err[ord][fib].append(np.sqrt(v[j]))

                elif combined_profiles:
                    ### THIS IS METHOD (3c) ###
                    # fill flux- and error- output arrays for all objects (Laser, Sky, Stellar, ThXe)
                    # Laser
                    flux[ord]['laser'].append(f[0])
                    err[ord]['laser'].append(np.sqrt(v[0]))
                    # Sky
                    flux[ord]['sky'].append(f[1])
                    err[ord]['sky'].append(np.sqrt(v[1]))
                    # Stellar
                    flux[ord]['stellar'].append(f[2])
                    err[ord]['stellar'].append(np.sqrt(v[2]))
                    # ThXe
                    flux[ord]['thxe'].append(f[3])
                    err[ord]['thxe'].append(np.sqrt(v[3]))

                else:
                    ### THIS IS METHOD (3b) ###
                    # Optimal extraction was done for all fibres individually, but now add up the respective "eta's"
                    # for the different "objects"
                    # fill flux- and error- output arrays for all objects (Laser, Sky, Stellar, ThXe)
                    # Laser
                    flux[ord]['laser'].append(f[0])
                    err[ord]['laser'].append(np.sqrt(v[0]))
                    # Sky
                    flux[ord]['sky'].append(np.sum(f[1:4]) + np.sum(f[25:27]))
                    err[ord]['sky'].append(np.sqrt(np.sum(v[1:4]) + np.sum(v[25:27])))
                    # Stellar
                    flux[ord]['stellar'].append(np.sum(f[5:24]))
                    err[ord]['stellar'].append(np.sqrt(np.sum(v[5:24])))
                    # ThXe
                    flux[ord]['thxe'].append(f[27])
                    err[ord]['thxe'].append(np.sqrt(v[27]))

            else:
                # flux[ord].append(np.max([f, 0.]))
                flux[ord].append(f)
                # if f <= 0 or v <= 0:
                if v <= 0:
                    # err[ord].append(np.sqrt(len(phi)*RON*RON))
                    err[ord].append(np.sqrt(np.sum(pixerr * pixerr)))
                else:
                    err[ord].append(np.sqrt(v))


        # fix for order_01
        if ord == 'order_01':
            for fib in sorted(flux[ord].keys()):
                flux['order_01'][fib] = np.r_[np.repeat(0., 900), flux['order_01'][fib]]
                err['order_01'][fib] = np.r_[np.repeat(0., 900), err['order_01'][fib]]

        if timit:
            print('Time taken for extraction of ' + ord + ': ' + str(time.time() - order_start_time) + ' seconds')

    if timit:
        print('Time elapsed for optimal extraction of entire spectrum: ' + str(
            time.time() - start_time) + ' seconds...')

    return pix, flux, err





def optimal_extraction_from_indices(img, stripe_indices, err_img=None, ronmask=None, slit_height=30, date=None, pathdict=None, fibs='all',
                                    relints=None, skip_first_order=False, simu=False, phi_onthefly=False, individual_fibres=True,
                                    combined_profiles=False, integrate_profiles=False, slope=False, offset=False,
                                    collapse=False, debug_level=0, timit=False):
    """
    This routine performs the optimal extraction of an echelle spectrum following the formalism described in Sharp & Birchall 2010, PASA, 27:91.
    Output is saved in dictionaries ("pix", "flux", "err").
    The exact behaviour can be controlled via a variety of keywords (see more description of (3a), (3b) & (3c) in "extract_spectrum").

    INPUT:
    'img'                : 2-dim flux array
    'stripe_indices'     : dictionary (keys = orders) containing the indices of the pixels that are identified as the "stripes" (ie the to-be-extracted regions centred on the orders)
    'err_img'            : 2-dim array of the corresponding errors
    'ronmask'            : read-noise mask (same dimension as img) in e-/pix
    'slit_height'        : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'date'               : the date ('YYYYMMDD') the obervations were taken (so that the sorresponding (pre-determined) fibre profiles can be loaded)
    'pathdict'           : dictionary containing all directories relevant to the reduction
    'fibs'               : which fibres are you using? ['all', 'stellar', 'sky2', 'sky3', 'allsky']
    'relints'            : an array of the relative intensities in the stellar fibres (only needed if 'individual_fibres' == FALSE and 'combined_profiles' == TRUE)
    'simu'               : boolean - are you using simulated spectra?
    'skip_first_order'   : boolean - do you want to skip order 01 (causes problems as not fully on chip, and especially b/c LFC trace is rubbish)
    'phi_onthefly'       : boolean - CODING RELIC - TO BE REMOVED; creates fibre profiles on the fly
    'individual_fibres'  : boolean - do you want to extract spectra for each fibre individually?
    'combined_profiles'  : boolean - do you want to use a combined profile for each 'object' (stellar / sky / laser / thxe)? ignored if 'individual_fibres' is set to TRUE
    'integrate_profiles' : boolean - do you want to integrate the fibre profiles (better but MUCH slower), or just evaluate them at the pixel centres?
    'slope'              : boolean - do you want to include a slope as an "extra fibre"?
    'offset'             : boolean - do you want to include an offset as an "extra fibre"?
    'collapse'           : boolean - set this keyword to simply do a collapse extract (not recommended - this is a CODING RELIC - TO BE REMOVED; use routine "quick_extract" instead)
    'debug_level'        : for debugging...
    'timit'              : boolean - do you want to measure execution run time?

    OUTPUT:
    'pix'   : dictionary (keys = orders) containing the pixel numbers (in dispersion direction)
    'flux'  : dictionary (keys = orders) containing the extracted flux (ie the eta's in Sharp & Birchall)
    'err'   : dictionary (keys = orders) containing the uncertainty in the extracted flux

    TODO:
    - incorporate the sim ThXe profiles

    """



    if timit:
        start_time = time.time()

    if date is None:
        date = raw_input("Please enter date of observations 'YYYYMMDD': ")

    if err_img is None:
        print('WARNING: errors not provided! Using sqrt(flux + RON**2) as an estimate...')

    if fibs.lower() == 'all':
#         nfib = 24
        nfib = 26
    elif fibs.lower() == 'stellar':
        nfib = 19
    elif fibs.lower() == 'sky2':
        nfib = 2
    elif fibs.lower() == 'sky3':
        nfib = 3
    elif fibs.lower() == 'allsky':
        nfib = 5
    elif fibs.lower() in ['thxe', 'simth']:
        nfib = 1
    elif fibs.lower() in ['lfc', 'laser']:
        nfib = 1
    elif fibs.lower() == 'calibs':
        nfib = 2
    else:
        print('WARNING: input for "fibs" not recognized - using all 26 fibres as a default...')
#         nfib = 24
        nfib = 26

    # read in polynomial coefficients of best-fit individual-fibre-profile parameters
    if simu:
        fibparms = np.load(pathdict['fp'] + 'sim/fibparms_by_ord.npy').item()
    else:
        if date not in ['20181116', '20190127', '20190201']:
#             fibparms = np.load(pathdict['fp'] + 'archive/fibre_profile_fits_' + date + '.npy').item()
            fibparms = np.load(pathdict['fp'] + 'archive/combined_fibre_profile_fits_' + date + '.npy').item()
            if debug_level > 0:
                print('OK, loading fibre profiles for ' + date + '...')
        else:
            # have to laod this crutch, as the first order fits were crap for 20181116 / 20190127 / 20190201, so just for order 01 I replaced them with the parms from the following night
#             fibparms = np.load(pathdict['fp'] + 'archive/fibre_profile_fits_' + date + '_crutch.npy').item()
            fibparms = np.load(pathdict['fp'] + 'archive/combined_fibre_profile_fits_' + date + '_crutch.npy').item()
            if debug_level > 0:
                print('OK, loading fibre profiles (CRUTCH!!!) for ' + date + '...')

    flux = {}
    err = {}
    pix = {}

    useful_orders = sorted(stripe_indices.keys())
    if skip_first_order:
        del useful_orders[0]

    # loop over all orders
    for ord in useful_orders:
        if timit and (debug_level > 0):
            order_start_time = time.time()

        # order number
        ordnum = ord[-2:]

        if ord == useful_orders[0]:
            print('OK, now processing order: ' + ordnum),
        elif ord == useful_orders[-1]:
            print(' ' + ordnum)
        else:
            print(' ' + ordnum),

        # fibre profile parameters for that order
        fppo = fibparms[ord]

        # define stripe indices
        indices = stripe_indices[ord]

        # find the "order-box"
        sc, sr = flatten_single_stripe_from_indices(img, indices, slit_height=slit_height, timit=False)
        if ronmask is None:
            ronmask = np.ones(img.shape) * 3.
        ron_sc, ron_sr = flatten_single_stripe_from_indices(ronmask, indices, slit_height=slit_height, timit=False)
        if err_img is not None:
            err_sc, err_sr = flatten_single_stripe_from_indices(err_img, indices, slit_height=slit_height, timit=False)

        npix = sc.shape[1]

        flux[ord] = {}
        err[ord] = {}
        pix[ord] = []
        if not phi_onthefly and not collapse:
            if individual_fibres:
                for j in range(nfib):
                    fib = 'fibre_' + str(j + 1).zfill(2)
                    flux[ord][fib] = []
                    err[ord][fib] = []
            else:
                flux[ord]['laser'] = []
                err[ord]['laser'] = []
                flux[ord]['sky'] = []
                err[ord]['sky'] = []
                flux[ord]['stellar'] = []
                err[ord]['stellar'] = []
                flux[ord]['thxe'] = []
                err[ord]['thxe'] = []

        #         if individual_fibres:
        #             f_ord = np.zeros((nfib,npix))
        #             e_ord = np.zeros((nfib,npix))
        #         else:
        #             f_ord = np.zeros(npix)
        #             e_ord = np.zeros(npix)

        # exclude the range of order_01 (m = 65) that does fall off the chip!
        goodrange = np.arange(npix)
        if simu and ord == 'order_01':
            # goodrange = goodrange[fibparms[ord]['fibre_21']['onchip']]
            goodrange = np.arange(1301, 4096)
            for j in range(1301):
                pix[ord].append(ordnum + str(j + 1).zfill(4))
        if not simu and ord == 'order_01':
            # goodrange = goodrange[fibparms[ord]['fibre_21']['onchip']]
            goodrange = np.arange(900, 4112)
            for j in range(900):
                pix[ord].append(ordnum + str(j + 1).zfill(4))


        for i in goodrange:
            if debug_level > 0:
                print('pixel ' + str(i + 1) + '/' + str(npix))
            pix[ord].append(ordnum + str(i + 1).zfill(4))
            z = sc[:, i].copy()
            if simu:
                z -= 1.  # note the minus 1 is because we added 1 artificially at the beginning in order for "extract_stripes" to work properly
            roncol = ron_sc[:, i].copy()

            # if error is not provided, estimate it here (NOT RECOMMENDED!!!)
            if err_img is None:
                pixerr = np.sqrt(ron_sc[:, i] ** 2 + np.abs(z))
            else:
                pixerr = err_sc[:, i].copy()

            # assign weights for flux (and take care of NaNs and INFs)
            pix_w = 1. / (pixerr * pixerr)

            # Initially I thought this was clearly rubbish as it down-weights the central parts
            # and that we really want to use the relative errors, ie w_i = 1/(relerr_i)**2
            # relerr = pixerr / z     ### the pixel err
            # pix_w = 1. / (relerr)**2
            # HOWEVER: this is not true, and the optimal extraction linalg routine requires absolute errors!!!

            # check for NaNs etc
            pix_w[np.isinf(pix_w)] = 0.

            if phi_onthefly:
                quickfit = fit_single_fibre_profile(sr[:, i], z)
                bestparms = np.array([quickfit.best_values['mu'], quickfit.best_values['sigma'],
                                      quickfit.best_values['amp'], quickfit.best_values['beta']])
                phi = fibmodel_with_amp(sr[:, i], *bestparms)
                phi /= np.max([np.sum(phi), 0.001])  # we can do this because grid-stepsize = 1; also make sure that we do not divide by zero
                phi = phi.reshape(len(phi), 1)  # stupid python...
            else:
                # get normalized profiles for all fibres for this cutout
                if combined_profiles:
                    phi_laser = np.sum(make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, fibs='laser'), axis=1)
                    phi_thxe = np.sum(make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, fibs='thxe'), axis=1)
                    phis_sky3 = make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, fibs='sky3')
                    phi_sky3 = np.sum(phis_sky3, axis=1) / 3.
                    phis_stellar = make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, fibs='stellar')
                    phi_stellar = np.sum(phis_stellar * relints, axis=1)
                    phis_sky2 = make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, fibs='sky2')
                    phi_sky2 = np.sum(phis_sky2, axis=1) / 2.
                    phi_sky = (phi_sky3 + phi_sky2) / 2.
                    phi = np.vstack((phi_laser, phi_sky, phi_stellar, phi_thxe)).T
                else:
                    # phi = make_norm_profiles(sr[:,i], ord, i, fibparms)
                    # phi = make_norm_profiles_temp(sr[:,i], ord, i, fibparms)
                    # phi = make_norm_single_profile_temp(sr[:,i], ord, i, fibparms)
                    phi = make_norm_profiles_6(sr[:, i], i, fppo, integrate=integrate_profiles, slope=slope, offset=offset, fibs=fibs)

            # print('WARNING: TEMPORARY offset correction is not commented out!!!')
            # # subtract the median as the offset if BG is not properly corrected for
            # z -= np.median(z)

            # do the optimal extraction
            if not collapse:
                if np.sum(phi) == 0:
                    # f,v = (0.,np.sqrt(len(phi)*RON*RON))
                    f, v = (0., np.sqrt(np.sum(pixerr * pixerr)))
                else:
                    # THIS IS THE NORMAL CASE!!!
                    # NOTE: take the read-out noise as the average of the individual-pixel read-out noise values over
                    # the cutout, as it can change if we cross a quadrant boundary!
                    f, v = linalg_extract_column(z, pix_w, phi, RON=np.mean(roncol))
            else:
                # f,v = (np.sum(z-np.median(z)), np.sum(z-np.median(z)) + len(phi)*RON*RON)   ### background should already be taken care of here...
                # f,v = (np.sum(z), np.sum(z) + len(phi)*RON*RON)
                f, v = (np.sum(z), np.sqrt(np.sum(pixerr * pixerr)))

            # e = np.sqrt(v)
            # model = np.sum(f*phi,axis=1)

            # fill output arrays depending on the selected method
            if not phi_onthefly and not collapse:

                # theoretically there should not be negative values, but we will allow them (bias and dark subtraction can produce those)
                # f[f < 0] = 0.
                # however, their errors are treated below
                # not sure if this is the proper way to do this, but we can't have negative variance
                # v[np.logical_or(v<=0,f<=0)] = RON*RON
                # v[v<RON*RON] = np.maximum(RON*RON,1.)   # just a stupid fix so that variance is never below 1
                v[np.logical_or(v <= 0, f <= 0)] = np.nanmean(roncol) ** 2
                v[v < np.nanmean(roncol) ** 2] = np.maximum(np.nanmean(roncol) ** 2, 1.)  # just a stupid fix so that variance is never below 1

                if individual_fibres:
                    ### THIS IS METHOD (3a) - PREFERRED OPTION! ###
                    # fill flux- and error- output arrays for individual fibres
                    for j in range(nfib):
                        fib = 'fibre_' + str(j + 1).zfill(2)
                        flux[ord][fib].append(f[j])
                        err[ord][fib].append(np.sqrt(v[j]))

                elif combined_profiles:
                    ### THIS IS METHOD (3c) ###
                    # fill flux- and error- output arrays for all objects (Laser, Sky, Stellar, ThXe)
                    # Laser
                    flux[ord]['laser'].append(f[0])
                    err[ord]['laser'].append(np.sqrt(v[0]))
                    # Sky
                    flux[ord]['sky'].append(f[1])
                    err[ord]['sky'].append(np.sqrt(v[1]))
                    # Stellar
                    flux[ord]['stellar'].append(f[2])
                    err[ord]['stellar'].append(np.sqrt(v[2]))
                    # ThXe
                    flux[ord]['thxe'].append(f[3])
                    err[ord]['thxe'].append(np.sqrt(v[3]))

                else:
                    ### THIS IS METHOD (3b) ###
                    # Optimal extraction was done for all fibres individually, but now add up the respective "eta's"
                    # for the different "objects"
                    # fill flux- and error- output arrays for all objects (Laser, Sky, Stellar, ThXe)
                    # Laser
                    flux[ord]['laser'].append(f[0])
                    err[ord]['laser'].append(np.sqrt(v[0]))
                    # Sky
                    flux[ord]['sky'].append(np.sum(f[1:4]) + np.sum(f[25:27]))
                    err[ord]['sky'].append(np.sqrt(np.sum(v[1:4]) + np.sum(v[25:27])))
                    # Stellar
                    flux[ord]['stellar'].append(np.sum(f[5:24]))
                    err[ord]['stellar'].append(np.sqrt(np.sum(v[5:24])))
                    # ThXe
                    flux[ord]['thxe'].append(f[27])
                    err[ord]['thxe'].append(np.sqrt(v[27]))

            else:
                # flux[ord].append(np.max([f, 0.]))
                flux[ord].append(f)
                # if f <= 0 or v <= 0:
                if v <= 0:
                    # err[ord].append(np.sqrt(len(phi)*RON*RON))
                    err[ord].append(np.sqrt(np.sum(pixerr * pixerr)))
                else:
                    err[ord].append(np.sqrt(v))


        # fix for order_01
        if ord == 'order_01':
            for fib in sorted(flux[ord].keys()):
                flux['order_01'][fib] = np.r_[np.repeat(0., 900), flux['order_01'][fib]]
                err['order_01'][fib] = np.r_[np.repeat(0., 900), err['order_01'][fib]]

        if timit and (debug_level > 0):
            print('Time taken for extraction of ' + ord + ': ' + str(time.time() - order_start_time) + ' seconds')

    if timit:
        print('Time elapsed for optimal extraction of entire spectrum: ' + str(
            time.time() - start_time) + ' seconds...')

    return pix, flux, err





def extract_spectrum(stripes, err_stripes, ron_stripes, method='optimal', individual_fibres=True, combined_profiles=False, integrate_profiles=False, slope=False,
                     offset=False, fibs='all', slit_height=30, savefile=False, filetype='fits', obsname=None, date=None, pathdict=None, lamp_config=None,
                     skip_first_order=False, simu=False, verbose=False, timit=False, debug_level=0):
    """
    This routine is simply a wrapper code for the different extraction methods. There are a total FIVE (1,2,3a,3b,3c) different extraction methods implemented, 
    which can be selected by a combination of the 'method', individual_fibres', and 'combined_profile' keyword arguments.
    
    (1) QUICK EXTRACTION: 
        A quick-look reduction of an echelle spectrum, by simply adding up the flux in a pixel column perpendicular to the dispersion direction
        
    (2) TRAMLINE EXTRACTION:
        Similar to quick extraction, but takes more care in defining the (non-constant) width of the extraction slit. Also uses partial pixels at both ends
        of the extraction slit.
        
    (3) OPTIMAL EXTRACTION:
        This follows the formalism of Sharp & Birchall, 2010, PASA, 27:91. One can choose between three different sub-methods:
        
        (3a) Extract a 1-dim spectrum for each fibre (individual_fibres = True). This is most useful when the fibres are well-separated in cross-dispersion direction)
             and/or the fibres are significantly offset with respect to each other (in dispersion direction).
        
        (3b) Extract ONE 1-dim spectrum for each object (individual_fibres = False  &&  combined_profile = False). Objects are 'stellar', 'sky', 'laser', and 'thxe'.
             Calculates "eta's" for each individual fibre, but then adds them up within each respective object.
        
        (3c) Extract ONE 1-dim spectrum (individual_fibres = False  &&  combined_profiles = True). Performs the optimal extraction linear algebra for one combined
             profile for each object.
             
    
    INPUT:
    'stripes'      : dictionary (keys = orders) containing the order-stripes (from "extract_stripes")
    'err_stripes'  : dictionary (keys = orders) containing the errors in the order-stripes (from "extract_stripes")
    'ron_stripes'  : dictionary (keys = orders) containing the read-out noise stripes (from "extract_stripes")
    
    OPTIONAL INPUT / KEYWORDS:
    'method'             : method for extraction - valid options are ["quick" / "tramline" / "optimal"]
    'individual_fibres'  : boolean - set to TRUE for method (3a); set to FALSE for methods (3b) or (3c) ; ignored if method is not set to 'optimal'
    'combined_profiles'  : boolean - set to TRUE for method (3c); set to FALSE for method (3b) ; only takes effect if 'individual_fibres' is set to FALSE; ignored if method is not set to 'optimal'
    'integrate_profiles' : boolean - set to TRUE if you want to (CORRECTLY but MUCH SLOWER) integrate over the (highly non-linear) functional form describing the profiles, rather than evaluating
                           the function at the discrete values corresponding to the pixel centres (which is only a good approximation if the function varies slowly)
    'slope'              : boolean - do you want to include a slope (along the slit) as an extra 'fibre' in the optimal extraction?
    'offset'             : boolean - do you want to include an offset as an extra 'fibre' in the optimal extraction?
    'fibs'               : which fibres do you want to include in the profile creation for the optimal extraction? ['all', 'stellar', 'sky2', 'sky3', 'allsky', 'lfc', 'simth', 'calibs']
    'slit_height'        : height of the extraction slit is 2*slit_height pixels
    'gain'               : gain
    'savefile'           : boolean - do you want to save the extracted spectrum to a file? 
    'filetype'           : if 'savefile' is set to TRUE: do you want to save it as a 'fits' file, or as a 'dict' (python disctionary), or 'both'
    'obsname'            : (short) name of observation file
    'date'               : the date of the observations to be extracted in format 'YYYYMMDD' (needed for the optimal extraction routine to select the right fibre profiles/traces)
    'pathdict'           : dictionary containing all directories relevant to the reduction
    'lamp_config'        : simcalib lamp configuration (only needed for output filename determination for simcalib frames)
    'skip_first_order'   : boolean - do you want to skip order 01 (causes problems as not fully on chip, and especially b/c LFC trace is rubbish)
    'simu'               : boolean - are you using ES-simulated spectra???
    'verbose'            : boolean - for debugging...
    'timit'              : boolean - do you want to measure execution run time?
    'debug_level'        : for debugging...
    
    OUTPUT:
    'pix'     : dictionary (keys = orders) containing the pixel numbers (in dispersion direction)
    'flux'    : dictionary (keys = orders) containing the extracted flux
    'err'     : dictionary (keys = orders) containing the uncertainty in the extracted flux (including photon noise and read-out noise)
    
    NOTE: Depending on 'method', 'flux' and 'err' can contain several keys each (either the individual fibres (method 3a), or the different 'object' (method 3b & 3c)!!!
    
    MODHIST:
    13/07/18 - CMB create
    02/08/18 - added 'savefile', 'path', and 'obsname' keywords - save output as FITS file
    22/04/20 - using pathdict instead of single path variable
    """

    assert pathdict is not None, 'ERROR: pathdict nor provided!!!'
    path = pathdict['raw']

    if date is None:
        date = raw_input("Please enter date of observations 'YYYYMMDD': ")
    
    while method not in ["quick", "tramline", "optimal"]:
        print('ERROR: extraction method not recognized!')
        method = raw_input('Which method do you want to use (valid options are ["quick" / "tramline" / "optimal"] )?')
        
    if method.lower() == 'quick':
        pix, flux, err = quick_extract(stripes, err_stripes, slit_height=slit_height, skip_first_order=skip_first_order, debug_level=debug_level, timit=timit)
    elif method.lower() == 'tramline':
        print('WARNING: need to update tramline finding routine first for new IFU layout - use method="quick" in the meantime')
        return
        #tramlines = find_tramlines(fibre_profiles_02, fibre_profiles_03, fibre_profiles_21, fibre_profiles_22, mask_02, mask_03, mask_21, mask_22)
        #pix,flux,err = collapse_extract(stripes, err_stripes, tramlines, slit_height=slit_height, verbose=verbose, timit=timit, debug_level=debug_level)
    elif method.lower() == 'optimal':
        pix,flux,err = optimal_extraction(stripes, err_stripes=err_stripes, ron_stripes=ron_stripes, slit_height=slit_height, individual_fibres=individual_fibres,
                                          skip_first_order=skip_first_order, combined_profiles=combined_profiles, integrate_profiles=integrate_profiles, slope=slope, 
                                          offset=offset, fibs=fibs, date=date, pathdict=pathdict, simu=simu, timit=timit, debug_level=debug_level)
    else:
        print('ERROR: Nightmare! That should never happen  --  must be an error in the Matrix...')
        return    
    
    # now save to FITS file or PYTHON DICTIONARY if desired
    if savefile:
        
        # which extraction method was used exactly?
        submethod = ''
        if method.lower() == 'optimal':
            if individual_fibres:
                submethod = '3a'
            else:
                if combined_profiles:
                    submethod = '3c'
                else:
                    submethod = '3b'
            
        # name of object
        try:
            starname = pyfits.getval(path + obsname + '.fits', 'OBJECT').split('+')[0]
        except:
            starname = ''

        # fix output filename issue for simcalib spectra with both lamps on
        if starname.lower() in ["lc", "lc-only", "lfc", "lfc-only", "simlc", "thxe", "thxe-only", "simth", "thxe+lfc", "lfc+thxe", "lc+simthxe", "lc+thxe"]):
            assert lamp_config.lower() in ['lfc', 'thxe', 'both'], 'ERROR: lamp_config not valid!!!'
            if lamp_config.lower() == 'lfc':
                starname = 'SimLC'
            elif lamp_config.lower() == 'thxe':
                starname = 'SimTh'
            elif lamp_config.lower() == 'both':
                starname = 'SimLC_plus_SimTh'

        if path is None:
            print('ERROR: path to output directory not provided!!!')
            return
        elif obsname is None:
            print('ERROR: "obsname" not provided!!!')
            return
        else:
            while filetype not in ["fits", "dict", "both"]:
                print('ERROR: file type for output file not recognized!')
                filetype = raw_input('Which file type do you want to use (valid options are ["fits" / "dict" / "both"] )?') 
            if filetype in ['fits', 'both']:
                fluxarr = np.zeros((len(pix), len(flux[flux.keys()[0]]), len(pix[pix.keys()[0]])))
                errarr = np.zeros((len(pix), len(flux[flux.keys()[0]]), len(pix[pix.keys()[0]])))
                for i,o in enumerate(sorted(pix.keys())):
                    fluxarr[i,:] = flux[o]
                    errarr[i,:] = err[o]
                # try and get header from previously saved files
                if os.path.exists(path + date + '_' + obsname + '_BD_CR_BG_FF.fits'):
                    h = pyfits.getheader(path + date + '_' + obsname + '_BD_CR_BG_FF.fits')
                elif os.path.exists(path + date + '_' + obsname + '_BD_CR_BG.fits'):
                    h = pyfits.getheader(path + date + '_' + obsname + '_BD_CR_BG.fits')
                elif os.path.exists(path + date + '_' + obsname + '_BD_CR.fits'):
                    h = pyfits.getheader(path + date + '_' + obsname + '_BD_CR.fits')
                elif os.path.exists(path + date + '_' + obsname + '_BD.fits'):
                    h = pyfits.getheader(path + date + '_' + obsname + '_BD.fits')
                else:
                    h = pyfits.getheader(path + obsname + '.fits')   
                # update the header and write to file
                h['HISTORY'] = '   EXTRACTED SPECTRUM - created ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ' (GMT)'
                h['EXMETH'] = (method, 'extraction method used')
                topord = sorted(pix.keys())[0]
                topordnum = int(topord[-2:])
                botord = sorted(pix.keys())[-1]
                botordnum = int(botord[-2:])
                h['FIRSTORD'] = (topordnum, 'order number of first (top) order')
                h['LASTORD'] = (botordnum, 'order number of last (bottom) order')
                if method.lower() == 'optimal':
                    h['EXMETH2'] = (submethod, 'exact optimal extraction method used')
                    h['SLOPE'] = (slope,)
                    h['OFFSET'] = (offset,)
                # write to FITS file
                if starname == '':
                    outfn = path + date + '_' + starname + obsname + '_' + method.lower() + submethod + '_extracted.fits'
                else:
                    outfn = path + date + '_' + starname + '_' + obsname + '_' + method.lower() + submethod + '_extracted.fits'
                pyfits.writeto(outfn, np.float32(fluxarr), h, overwrite=True)
                # now append the corresponding error array
                h_err = h.copy()
                h_err['HISTORY'] = 'estimated uncertainty in EXTRACTED SPECTRUM - created ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ' (GMT)'
                pyfits.append(outfn, np.float32(errarr), h_err, overwrite=True)
                
            if filetype in ['dict', 'both']:
                # OK, save as a python dictionary
                # create combined dictionary
                extracted = {}
                extracted['pix'] = pix
                extracted['flux'] = flux
                extracted['err'] = err
                np.save(path + date + '_' + starname + '_' + obsname + '_' + method.lower() + submethod + '_extracted.npy', extracted)
        
    return pix,flux,err





def extract_spectrum_from_indices(img, err_img, stripe_indices, ronmask=None, method='optimal', individual_fibres=True, combined_profiles=False, integrate_profiles=False, slope=False,
                                  offset=False, fibs='all', slit_height=30, savefile=False, filetype='fits', obsname=None, date=None, pathdict=None, lamp_config=None,
                                  skip_first_order=False, simu=False, verbose=False, timit=False, debug_level=0):
    """
    CLONE OF 'extract_spectrum'! 
    This routine is simply a wrapper code for the different extraction methods. There are a total FIVE (1,2,3a,3b,3c) different extraction methods implemented, 
    which can be selected by a combination of the 'method', individual_fibres', and 'combined_profile' keyword arguments.
    
    (1) QUICK EXTRACTION: 
        A quick-look reduction of an echelle spectrum, by simply adding up the flux in a pixel column perpendicular to the dispersion direction
        
    (2) TRAMLINE EXTRACTION:
        Similar to quick extraction, but takes more care in defining the (non-constant) width of the extraction slit. Also uses partial pixels at both ends
        of the extraction slit.
        
    (3) OPTIMAL EXTRACTION:
        This follows the formalism of Sharp & Birchall, 2010, PASA, 27:91. One can choose between three different sub-methods:
        
        (3a) Extract a 1-dim spectrum for each fibre (individual_fibres = True). This is most useful when the fibres are well-separated in cross-dispersion direction)
             and/or the fibres are significantly offset with respect to each other (in dispersion direction).
        
        (3b) Extract ONE 1-dim spectrum for each object (individual_fibres = False  &&  combined_profile = False). Objects are 'stellar', 'sky', 'laser', and 'thxe'.
             Calculates "eta's" for each individual fibre, but then adds them up within each respective object.
        
        (3c) Extract ONE 1-dim spectrum (individual_fibres = False  &&  combined_profiles = True). Performs the optimal extraction linear algebra for one combined
             profile for each object.
             
    
    INPUT:
    'img'            : 2-dim input array
    'err_img'        : 2-dim array of the corresponding errors
    'stripe_indices' : dictionary (keys = orders) containing the indices of the pixels that are identified as the "stripes" (ie the to-be-extracted regions centred on the orders)
    'ronmask'        : read-noise mask in e-/pix (same dimensions as img)

    OPTIONAL INPUT / KEYWORDS:
    'method'             : method for extraction - valid options are ["quick" / "tramline" / "optimal"]
    'individual_fibres'  : boolean - set to TRUE for method (3a); set to FALSE for methods (3b) or (3c) ; ignored if method is not set to 'optimal'
    'combined_profiles'  : boolean - set to TRUE for method (3c); set to FALSE for method (3b) ; only takes effect if 'individual_fibres' is set to FALSE; ignored if method is not set to 'optimal'
    'integrate_profiles' : boolean - set to TRUE if you want to (CORRECTLY but MUCH SLOWER) integrate over the (highly non-linear) functional form describing the profiles, rather than evaluating
                           the function at the discrete values corresponding to the pixel centres (which is only a good approximation if the function varies slowly)
    'slope'              : boolean - do you want to include a slope (along the slit) as an extra 'fibre' in the optimal extraction?
    'offset'             : boolean - do you want to include an offset as an extra 'fibre' in the optimal extraction?
    'fibs'               : which fibres do you want to include in the profile creation for the optimal extraction? ['all', 'stellar', 'sky2', 'sky3', 'allsky', 'lfc', 'simth', 'calibs']
    'slit_height'        : height of the extraction slit is 2*slit_height pixels
    'savefile'           : boolean - do you want to save the extracted spectrum to a file? 
    'filetype'           : if 'savefile' is set to TRUE: do you want to save it as a 'fits' file, or as a 'dict' (python disctionary)
    'obsname'            : (short) name of observation file
    'date'               : the date of the observations to be extracted in format 'YYYYMMDD' (needed for the optimal extraction routine to select the right fibre profiles/traces)
    'pathdict'           : dictionary containing all directories relevant to the reduction
    'lamp_config'        : simcalib lamp configuration (only needed for output filename determination for simcalib frames)
    'skip_first_order'   : boolean - do you want to skip order 01 (causes problems as not fully on chip, and especially b/c LFC trace is rubbish)
    'simu'               : boolean - are you using ES-simulated spectra???
    'verbose'            : boolean - for debugging...
    'timit'              : boolean - do you want to measure execution run time?
    'debug_level'        : for debugging...
    
    OUTPUT:
    'pix'     : dictionary (keys = orders) containing the pixel numbers (in dispersion direction)
    'flux'    : dictionary (keys = orders) containing the extracted flux
    'err'     : dictionary (keys = orders) containing the uncertainty in the extracted flux (including photon noise and read-out noise)
    
    NOTE: Depending on 'method', 'flux' and 'err' can contain several keys each (either the individual fibres (method 3a), or the different 'object' (method 3b & 3c)!!!
    
    MODHIST:
    17/07/18 - CMB create
    22/04/20 - using pathdict instead of single path variable
    """

    assert pathdict is not None, 'ERROR: pathdict nor provided!!!'
    path = pathdict['raw']

    if date is None:
        date = raw_input("Please enter date of observations 'YYYYMMDD': ")        

    while method not in ["quick", "tramline", "optimal"]:
        print('ERROR: extraction method not recognized!')
        method = raw_input('Which method do you want to use (valid options are ["quick" / "tramline" / "optimal"] )?')
        
    if method.lower() == 'quick':
        pix, flux, err = quick_extract_from_indices(img, err_img, stripe_indices, slit_height=slit_height, skip_first_order=skip_first_order, debug_level=debug_level, timit=timit)
    elif method.lower() == 'tramline':
        print('WARNING: need to update tramline finding routine first for new IFU layout - use method="quick" in the meantime')
        return
#         tramlines = find_tramlines(fibre_profiles_02, fibre_profiles_03, fibre_profiles_21, fibre_profiles_22, mask_02, mask_03, mask_21, mask_22)
#         pix,flux,err = collapse_extract_from_indices(img, err_img, stripe_indices, tramlines, slit_height=slit_height, verbose=verbose, timit=timit, debug_level=debug_level)
    elif method.lower() == 'optimal':
        pix,flux,err = optimal_extraction_from_indices(img, stripe_indices, err_img=err_img, ronmask=ronmask, slit_height=slit_height, individual_fibres=individual_fibres,
                                                       combined_profiles=combined_profiles, integrate_profiles=integrate_profiles, slope=slope, offset=offset, fibs=fibs, 
                                                       skip_first_order=skip_first_order, date=date, pathdict=pathdict, simu=simu, timit=timit, debug_level=debug_level)
    else:
        print('ERROR: Nightmare! That should never happen  --  must be an error in the Matrix...')
        return    
        
    # now save to FITS file or PYTHON DICTIONARY if desired
    if savefile:
        
        # which extraction method was used exactly?
        submethod = ''
        if method.lower() == 'optimal':
            if individual_fibres:
                submethod = '3a'
            else:
                if combined_profiles:
                    submethod = '3c'
                else:
                    submethod = '3b'
        
        # name of object
        try:
            starname = pyfits.getval(path + obsname + '.fits', 'OBJECT').split('+')[0]
        except:
            starname = ''

        # fix output filename issue for simcalib spectra with both lamps on
        if starname.lower() in ["lc", "lc-only", "lfc", "lfc-only", "simlc", "thxe", "thxe-only", "simth", "thxe+lfc", "lfc+thxe", "lc+simthxe", "lc+thxe"]):
            assert lamp_config.lower() in ['lfc', 'thxe', 'both'], 'ERROR: lamp_config not valid!!!'
            if lamp_config.lower() == 'lfc':
                starname = 'SimLC'
            elif lamp_config.lower() == 'thxe':
                starname = 'SimTh'
            elif lamp_config.lower() == 'both':
                starname = 'SimLC_plus_SimTh'

        if path is None:
            print('ERROR: path to output directory not provided!!!')
            return
        elif obsname is None:
            print('ERROR: "obsname" not provided!!!')
            return
        else:
            while filetype not in ["fits", "dict", "both"]:
                print('ERROR: file type for output file not recognized!')
                filetype = raw_input('Which file type do you want to use (valid options are ["fits" / "dict" / "both"] )?') 
            if filetype in ['fits', 'both']:
                if method.lower() == 'optimal' and submethod == '3a':      # ie for individual-fibre extraction
                    fluxarr = np.zeros((len(pix), len(flux[flux.keys()[0]]), len(pix[pix.keys()[0]])))
                    errarr = np.zeros((len(pix), len(flux[flux.keys()[0]]), len(pix[pix.keys()[0]])))
                    for i,o in enumerate(sorted(pix.keys())):
                        for j,fib in enumerate(sorted(flux[o].keys())):
                            fluxarr[i,j,:] = flux[o][fib]
                            errarr[i,j,:] = err[o][fib]
                else:
                    fluxarr = np.zeros((len(pix), len(pix['order_01'])))
                    errarr = np.zeros((len(pix), len(pix['order_01'])))
                    for i,o in enumerate(sorted(pix.keys())):
                        fluxarr[i,:] = flux[o]
                        errarr[i,:] = err[o]
                # try and get header from previously saved files
                if os.path.exists(path + date + '_' + obsname + '_BD_CR_BG_FF.fits'):
                    h = pyfits.getheader(path + date + '_' + obsname + '_BD_CR_BG_FF.fits')
                elif os.path.exists(path + date + '_' + obsname + '_BD_CR_BG.fits'):
                    h = pyfits.getheader(path + date + '_' + obsname + '_BD_CR_BG.fits')
                elif os.path.exists(path + date + '_' + obsname + '_BD_CR.fits'):
                    h = pyfits.getheader(path + date + '_' + obsname + '_BD_CR.fits')
                elif os.path.exists(path + date + '_' + obsname + '_BD.fits'):
                    h = pyfits.getheader(path + date + '_' + obsname + '_BD.fits')
                else:
                    h = pyfits.getheader(path + obsname + '.fits')   
                # update the header and write to file
                h['HISTORY'] = '   EXTRACTED SPECTRUM - created ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ' (GMT)'
                h['EXMETH'] = (method, 'extraction method used')
                topord = sorted(pix.keys())[0]
                topordnum = int(topord[-2:])
                botord = sorted(pix.keys())[-1]
                botordnum = int(botord[-2:])
                h['FIRSTORD'] = (topordnum, 'order number of first (top) order')
                h['LASTORD'] = (botordnum, 'order number of last (bottom) order')
                if method.lower() == 'optimal':
                    h['EXMETH2'] = (submethod, 'exact optimal extraction method used')
                    h['SLOPE'] = (slope,)
                    h['OFFSET'] = (offset,)
                # write to FITS file    
                if starname == '':
                    # outfn = path + date + '_' + starname + obsname + '_' + method.lower() + submethod + '_extracted.fits'
                    outfn = path + obsname + '_' + method.lower() + submethod + '_extracted.fits'
                else:
                    outfn = path + date + '_' + starname + '_' + obsname + '_' + method.lower() + submethod + '_extracted.fits'
                pyfits.writeto(outfn, np.float32(fluxarr), h, overwrite=True)
                # now append the corresponding error array
                h_err = h.copy()
                h_err['HISTORY'] = 'estimated uncertainty in EXTRACTED SPECTRUM - created ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ' (GMT)'
                pyfits.append(outfn, np.float32(errarr), h_err, overwrite=True)
                
            if filetype in ['dict', 'both']:
                # OK, save as a python dictionary
                # create combined dictionary
                extracted = {}
                extracted['pix'] = pix
                extracted['flux'] = flux
                extracted['err'] = err
                np.save(path + date + '_' + starname + '_' + obsname + '_' + method.lower() + submethod + '_extracted.npy', extracted)
        
    return pix,flux,err





def extract_spectra(filelist, P_id, mask, method='optimal', save_files=True, outpath=None, verbose=False):
    """
    DUMMY ROUTINE: not currently in use
    """
    
    
    #get short filenames for files in 'filelist'
    obsnames = short_filenames(filelist)    
    
    #make default output directory if none is provided
    if outpath is None:
        dum = filelist[0].split('/')
        path = filelist[0][0:-len(dum[-1])]
        outpath = path + 'reduced/'
        os.makedirs(outpath)

    #loop over all observations in 'filelist'
    for obsname,imgname in zip(obsnames,filelist):
        img = pyfits.getdata(imgname) + 1.
        h = pyfits.getheader(imgname)
        #extract stripes
        stripes,stripe_indices = extract_stripes(img, P_id, return_indices=True, slit_height=25)
        #perform the extraction!
        pix,flux,err = extract_spectrum(stripes, method=method, slit_height=25, RON=0., gain=1.,simu=True)
        #get relative intensities in fibres
        relints = get_relints(P_id, stripes, mask=mask, sampling_size=25, slit_height=25)
        
        #save output files
        if save_files:
            if verbose:
                print('Saving output files...')
            #save stripes
            np.save(outpath + obsname + '_stripes.npy', stripes)
            #save stripe indices
            np.save(outpath + obsname + '_stripe_indices.npy', stripe_indices)
            #save extracted spectrum (incl BC and wl-solution) to FITS file
            
            #save extracted spectrum (incl BC and wl-solution) to ASCII file
            #h['BARYCORR'] = (bc, 'barycentric correction in m/s')
            #h['DISPSOL'] = (xxx,'wavelength solution as calculated from laser frequency comb')
            now = datetime.datetime.utcnow()
            h['HISTORY'] = (str(now)+' UTC  -  stumpfer kommentar hier...')
    
    
    return













