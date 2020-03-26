

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

from veloce_reduction.veloce_reduction.wavelength_solution import find_suitable_peaks
from veloce_reduction.veloce_reduction.helper_functions import multi_fibmodel_with_amp, CMB_multi_gaussian, \
    central_parts_of_mask, multi_fibmodel_with_amp_and_offset, CMB_multi_gaussian_with_offset
from veloce_reduction.veloce_reduction.order_tracing import flatten_single_stripe, flatten_single_stripe_from_indices




def get_multiple_fibre_profiles_single_order(sc, sr, err_sc, ordpol, ordmask=None, nfib=24, sampling_size=25, step_size=None,
                                             varbeta=True, offset=True, return_snr=True, lfc=False, debug_level=0, timit=False):
    """
    INPUT:
    'sc'             : the flux in the extracted, flattened stripe
    'sr'             : row-indices (ie in spatial direction) of the cutouts in 'sc'
    'err_sc'         : the error in the extracted, flattened stripe
    'ordpol'         : set of polynomial coefficients from P_id for that order (ie p = P_id[ord])
    'ordmask'        : gives user the option to provide a mask (eg from "find_stripes")
    'nfib'           : number of fibres for which to retrieve the relative intensities (default is 24, b/c there are
                       19 object fibres plus 5 sky fibres for Veloce Rosso)
    'sampling_size'  : how many pixels (in dispersion direction) either side of current i-th pixel do you want to
                       consider? (ie stack profiles for a total of 2*sampling_size+1 pixels in dispersion direction...)
    'step_size'      : only calculate the relative intensities every so and so many pixels (should not change, plus it
                       takes ages...)
    'varbeta'        : boolean - if set to TRUE, use Gauss-like function for fitting, if set to FALSE use plain Gaussian
    'offset'         : boolean - do you want to fit an offset as well?
    'return_snr'     : boolean - do you want to return SNR of the collapsed super-pixel at each location in 'userange'?
    'lfc'            : boolean - is this LFC data? (in which case we only want to use the pixels near a peak to determine the fibre profiles)
    'debug_level'    : for debugging...
    'timit'          : boolean - do you want to measure execution run-time?

    OUTPUT:
    'relints'        : relative intensities in the fibres
    'relints_norm'   : relative intensities in the fibres re-normalized to a sum of 1
    'full_model'     : a list containing the full model of every cutout (only if 'return_full' is set to TRUE)
    'modgrid'        : a list containing the grid on which the full model is evaluated (only if 'return_full' is
                       set to TRUE)


    MODHIST:
    23/09/2018 - CMB create (essentially a mix of "determine_spatial_profiles_single_order" and
                 "fit_profiles_single_order")

    """

    if timit:
        start_time = time.time()
    if debug_level >= 1:
        print('Fitting fibre profiles for one order...')

    npix = sc.shape[1]
    xx = np.arange(npix)

    if ordmask is None:
        ordmask = np.ones(npix, dtype='bool')

    if step_size is None:
        step_size = 2 * sampling_size
    userange = np.arange(np.arange(npix)[ordmask][0] + sampling_size, np.arange(npix)[ordmask][-1], step_size)
    # don't use pixel columns 200 pixels from either end of the chip
    userange = userange[np.logical_and(userange > 200, userange < npix - 200)]
    # now we also need to mask out a region of +/- 100 pixels around the "maximum" of the order trace,
    # i.e. where the order has close to zero curvature
    order_peak_location = np.argwhere(ordpol(xx) == np.max(ordpol(xx)))[0]
    userange = userange[np.logical_or(userange < order_peak_location - 100, userange > order_peak_location + 100)]

    # prepare output dictionary
    fibre_profiles_ord = {}
    fibre_profiles_ord['mu'] = np.zeros((len(userange), nfib))
    fibre_profiles_ord['sigma'] = np.zeros((len(userange), nfib))
    fibre_profiles_ord['amp'] = np.zeros((len(userange), nfib))
    fibre_profiles_ord['pix'] = []
    if varbeta:
        fibre_profiles_ord['beta'] = np.zeros((len(userange), nfib))
    if offset:
        fibre_profiles_ord['offset'] = []
    if return_snr:
        fibre_profiles_ord['SNR'] = []

    # loop over all columns for one order and do the profile fitting
    for i, pix in enumerate(userange):

        if debug_level >= 1:
            print('pix = ', str(pix))

        # keep a record of the pixel number where this particular profile measurement was made
        fibre_profiles_ord['pix'].append(pix)

        # estimate SNR of collapsed super-pixel at this location (including RON)
        if return_snr:
            #snr.append(np.sqrt(np.sum(sc[:,pix])))
            #snr.append(np.sum(sc[:, pix]) / np.sqrt(np.sum(err_sc[:, pix] ** 2)))
            snr = np.sum(sc[:, pix]) / np.sqrt(np.sum(err_sc[:, pix] ** 2))
            fibre_profiles_ord['SNR'].append(snr)

        # check if that particular cutout falls fully onto CCD
        checkprod = np.product(sr[1:, pix].astype(float))  # exclude the first row number, as that can legitimately be zero
        # NOTE: This also covers row numbers > ny, as in these cases 'sr' is set to zero in "flatten_single_stripe(_from_indices)"
        if ordmask[pix] == False:
            print('WARNING: this pixel column was masked out during order tracing!!!')
            fibre_profiles_ord['mu'][i, :] = np.repeat(-1., nfib)
            fibre_profiles_ord['sigma'][i, :] = np.repeat(-1., nfib)
            fibre_profiles_ord['amp'][i, :] = np.repeat(-1., nfib)
            if varbeta:
                fibre_profiles_ord['beta'][i, :] = np.repeat(-1., nfib)
            if offset:
                fibre_profiles_ord['offset'].append(-1.)
#         elif snr < nfib * 15.:
        elif snr < nfib * 5:
            print('WARNING: SNR too low!!!')
            fibre_profiles_ord['mu'][i, :] = np.repeat(-1., nfib)
            fibre_profiles_ord['sigma'][i, :] = np.repeat(-1., nfib)
            fibre_profiles_ord['amp'][i, :] = np.repeat(-1., nfib)
            if varbeta:
                fibre_profiles_ord['beta'][i, :] = np.repeat(-1., nfib)
            if offset:
                fibre_profiles_ord['offset'].append(-1.)
        elif checkprod == 0:
            checksum = np.sum(sr[:, pix])
            if checksum == 0:
                print('WARNING: the entire cutout lies outside the chip!!!')
                # best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
            else:
                print('WARNING: parts of the cutout lie outside the chip!!!')
                # best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
            fibre_profiles_ord['mu'][i, :] = np.repeat(-1., nfib)
            fibre_profiles_ord['sigma'][i, :] = np.repeat(-1., nfib)
            fibre_profiles_ord['amp'][i, :] = np.repeat(-1., nfib)
            if varbeta:
                fibre_profiles_ord['beta'][i, :] = np.repeat(-1., nfib)
            if offset:
                fibre_profiles_ord['offset'].append(-1.)
        else:
            # this is the NORMAL case, where the entire cutout lies on the chip
            grid = []
            # data = []
            normdata = []
            # errors = []
            weights = []
            refpos = ordpol(pix)
            for j in np.arange(np.max([0, pix - sampling_size]), np.min([npix - 1, pix + sampling_size]) + 1):
                
                colcounts = np.sum(sc[:, j])           
                
                if (colcounts >= 500) or (not lfc):
                    grid.append(sr[:, j] - ordpol(j) + refpos)
                    # data.append(sc[:,j])
                    normdata.append(sc[:, j] / np.sum(sc[:, j]))
                    # make sure we do not divide by zero by adding 1 plus the min. colcounts across all j's, so that the smallest number we are dividing by is 1
                    # normdata.append(sc[:, j] / (np.sum(sc[:, j]) - np.min(np.sum(sc[:, np.arange(np.max([0, pix - sampling_size]), np.min([npix - 1, pix + sampling_size]) + 1)], axis=0)) + 1))
                    # assign weights for flux (and take care of NaNs and INFs)
                    # normerr = np.sqrt(sc[:,j] + RON**2) / np.sum(sc[:,j])
                    normerr = err_sc[:, j] / np.sum(sc[:, j])
                    # normerr = err_sc[:, j] / (np.sum(sc[:, j]) - np.min(np.sum(sc[:, np.arange(np.max([0, pix - sampling_size]), np.min([npix - 1, pix + sampling_size]) + 1)], axis=0)) + 1)
                    pix_w = 1. / (normerr * normerr)
                    pix_w[np.isinf(pix_w)] = 0.
                    weights.append(pix_w)
                    ### initially I thought this was clearly rubbish as it down-weights the central parts
                    ### and that we really want to use the relative errors, ie w_i = 1/(relerr_i)**2
                    ### HOWEVER: this is not true, and the optimal extraction linalg routine requires absolute errors!!!
                    # weights.append(1./((np.sqrt(sc[:,j] + RON**2)) / sc[:,j])**2)
                    if debug_level >= 3:
                        # plt.plot(sr[:,j] - ordpol(j),sc[:,j],'.')
                        plt.plot(sr[:, j] - ordpol(j), sc[:, j] / np.sum(sc[:, j]), '.')
                        # plt.xlim(-5,5)
                        plt.xlim(-sc.shape[0] / 2, sc.shape[0] / 2)

            # data = np.array(data).flatten()
            normdata = np.array(normdata).flatten()
            weights = np.array(weights).flatten()
            grid = np.array(grid).flatten()
            # data = data[grid.argsort()]
            normdata = normdata[grid.argsort()]
            weights = weights[grid.argsort()]
            grid = grid[grid.argsort()]

            # if debug_level >= 2:
            #     plt.plot(grid,normdata)

            dynrange = np.max(normdata) - np.min(normdata)
            goodpeaks, mostpeaks, allpeaks = find_suitable_peaks(normdata, thresh=np.min(normdata) + 0.3*dynrange, bgthresh=np.min(normdata) + 0.15*dynrange,
                                                                 clip_edges=False, gauss_filter_sigma=10, slope=1e-8)

            if debug_level >=1 :
                print('Number of fibres found: ',len(goodpeaks))

            if len(goodpeaks) != nfib :
            # if (len(allpeaks) != nfib) and (len(goodpeaks) != nfib):
                print('ERROR: not all peaks found!!!')
                print('pix = ', pix)

                fibre_profiles_ord['mu'][i, :] = np.repeat(-1., nfib)
                fibre_profiles_ord['sigma'][i, :] = np.repeat(-1., nfib)
                fibre_profiles_ord['amp'][i, :] = np.repeat(-1., nfib)
                if varbeta:
                    fibre_profiles_ord['beta'][i, :] = np.repeat(-1., nfib)
                if offset:
                    fibre_profiles_ord['offset'].append(-1.)

            else:

                #do the fitting
                peaks = np.r_[grid[goodpeaks]]

                npeaks = len(peaks)

                guess = []
                lower_bounds = []
                upper_bounds = []
                for n in range(npeaks):
                    if varbeta:
                        guess.append(np.array([peaks[n], 0.7, normdata[goodpeaks[n]], 2.]))
                        lower_bounds.append([peaks[n] - 1, 0, 0, 1])
                        upper_bounds.append([peaks[n] + 1, np.inf, np.inf, 4])
                    else:
                        guess.append(np.array([peaks[n], 1., normdata[goodpeaks[n]]]))
                        lower_bounds.append([peaks[n] - 1, 0, 0])
                        upper_bounds.append([peaks[n] + 1, np.inf, np.inf])
                guess = np.array(guess).flatten()
                lower_bounds = np.array(lower_bounds).flatten()
                upper_bounds = np.array(upper_bounds).flatten()
                if offset:
                    if varbeta:
                        popt, pcov = op.curve_fit(multi_fibmodel_with_amp_and_offset, grid, normdata, p0=np.r_[guess,0],
                                                  bounds=(np.r_[lower_bounds,0], np.r_[upper_bounds,np.max(normdata)]), maxfev=100000)
                    else:
                        popt, pcov = op.curve_fit(CMB_multi_gaussian_with_offset, grid, normdata, p0=np.r_[guess,0],
                                                  bounds=(np.r_[lower_bounds,0], np.r_[upper_bounds, np.max(normdata)]), maxfev=100000)
                else:
                    if varbeta:
                        popt, pcov = op.curve_fit(multi_fibmodel_with_amp, grid, normdata, p0=guess,
                                                  bounds=(lower_bounds, upper_bounds), maxfev=100000)
                    else:
                        popt, pcov = op.curve_fit(CMB_multi_gaussian, grid, normdata, p0=guess,
                                                  bounds=(lower_bounds, upper_bounds), maxfev=100000)


                if offset:
                    if varbeta:
                        popt_arr = np.reshape(popt[:-1], (nfib, 4))
                    else:
                        popt_arr = np.reshape(popt[:-1], (nfib, 3))
                else:
                    if varbeta:
                        popt_arr = np.reshape(popt, (nfib, 4))
                    else:
                        popt_arr = np.reshape(popt, (nfib, 3))

                # fibre numbers here increase from red to blue, ie from ThXe side to LFC side, as in:
                # pseudo-slit layout:   ThXe S5  S2   X   7  18  17   6  16  15   5  14  13   1  12  11   4  10   9   3   8  19   2   X  S4  S3  S1 LFC
                # array indices     :         0   1       2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20      21  22  23

                fibre_profiles_ord['mu'][i, :] = popt_arr[:,0]
                fibre_profiles_ord['sigma'][i, :] = popt_arr[:, 1]
                fibre_profiles_ord['amp'][i, :] = popt_arr[:, 2]
                if varbeta:
                    fibre_profiles_ord['beta'][i, :] = popt_arr[:, 3]
                if offset:
                    fibre_profiles_ord['offset'].append(popt[-1])


                # full_model = np.zeros(grid.shape)
                # if varbeta:
                #     full_model += multi_fibmodel_with_amp(grid,*popt)
                # else:
                #     full_model += CMB_pure_gaussian(grid, *popt)


    if timit:
        print('Elapsed time for retrieving relative intensities: ' + np.round(time.time() - start_time, 2).astype(str) + ' seconds...')

    return fibre_profiles_ord



def fit_multiple_profiles(P_id, stripes, err_stripes, mask=None, slit_height=25, nfib=24, varbeta=True, offset=True,
                          debug_level=0, timit=False):
    """
    This routine determines the profiles of the fibres in spatial direction. This is an extremely crucial step, as the
    pre-defined profiles are then used during the optimal extraction, as well as during the determination of the
    relative fibre intensities!!!

    INPUT:
    'P_id'          : dictionary of the form of {order: np.poly1d, ...} (as returned by "identify_stripes")
    'stripes'       : dictionary containing the flux in the extracted stripes (keys = orders)
    'err_stripes'   : dictionary containing the errors in the extracted stripes (keys = orders)
    'mask'          : dictionary of boolean masks (keys = orders) from "find_stripes" (masking out regions of very low signal)
    'slit_height'   : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'nfib'          : number of fibres
    'varbeta'       : boolean - if set to TRUE, use Gauss-like function for fitting, if set to FALSE use plain Gaussian
    'offset'        : boolean - do you want to fit an offset as well?
    'debug_level'   : for debugging...
    'timit'         : boolean - do you want to measure execution run-time?

    OUTPUT:
    'fibre_profiles'  : dictionary (keys=orders) containing the calculated spatial-direction fibre profiles
    """

    print('Fitting fibre profiles...')

    if timit:
        start_time = time.time()

    # create "global" parameter dictionary for entire chip
    fibre_profiles = {}

    # make sure we have a proper mask
    if mask is None:
        cenmask = {}
    else:
        #we also only want to use the central TRUE parts of the masks, ie want ONE consecutive stretch per order
        cenmask = central_parts_of_mask(mask)

    # loop over all orders
    #for ord in sorted(P_id.iterkeys()):
    for ord in sorted(P_id.keys()):
        print('OK, now processing ' + str(ord))

        ordpol = P_id[ord]

        # define stripe
        stripe = stripes[ord]
        err_stripe = err_stripes[ord]
        # find the "order-box"
        sc, sr = flatten_single_stripe(stripe, slit_height=slit_height, timit=False)
        err_sc, err_sr = flatten_single_stripe(err_stripe, slit_height=slit_height, timit=False)

        if mask is None:
            cenmask[ord] = np.ones(sc.shape[1], dtype='bool')

        # fit profile for single order and save result in "global" parameter dictionary for entire chip
        fpo = get_multiple_fibre_profiles_single_order(sc, sr, err_sc, ordpol, ordmask=cenmask[ord], nfib=nfib,
                                                       sampling_size=25, varbeta=varbeta, offset=offset,
                                                       return_snr=True, debug_level=debug_level, timit=timit)

        # if stacking:
        #     colfits = determine_spatial_profiles_single_order(sc, sr, err_sc, ordpol, ordmask=mask[ord], model=model,
        #                                                       return_stats=return_stats, timit=timit)
        # else:
        #     colfits = fit_profiles_single_order(sr, sc, ordpol, osf=1, silent=True, timit=timit)

        fibre_profiles[ord] = fpo

    if timit:
        print('Time elapsed: ' + str(int(time.time() - start_time)) + ' seconds...')

    return fibre_profiles



def fit_multiple_profiles_from_indices(P_id, img, err_img, stripe_indices, mask=None, slit_height=25, nfib=24, lfc=False,
                                       sampling_size=25, step_size=None, varbeta=True, offset=True, debug_level=0, timit=False):
    """
    This routine determines the profiles of the fibres in spatial direction. This is an extremely crucial step, as the
    pre-defined profiles are then used during the optimal extraction, as well as during the determination of the
    relative fibre intensities!!!

    CLONE OF "fit_multiple_profiles", but using stripe-indices, rather than stripes...

    INPUT:
    'P_id'          : dictionary of the form of {order: np.poly1d, ...} (as returned by "identify_stripes")
    'img'           : 2-dim input array/image
    'err_img'       : estimated uncertainties in the 2-dim input array/image
    'mask'          : dictionary of boolean masks (keys = orders) from "find_stripes" (masking out regions of very low signal)
    'slit_height'   : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'nfib'          : number of fibres
    'lfc'           : boolean - is this LFC data? (in which case we only want to use the pixels near a peak to determine the fibre profiles)
    'sampling_size' : how many pixels (in dispersion direction) either side of current i-th pixel do you want to
                      consider? (ie stack profiles for a total of 2*sampling_size+1 pixels in dispersion direction...)
    'step_size'     : only calculate the relative intensities every so and so many pixels (should not change, plus it takes ages...)
    'varbeta'       : boolean - if set to TRUE, use Gauss-like function for fitting, if set to FALSE use plain Gaussian
    'offset'        : boolean - do you want to fit an offset as well?
    'debug_level'   : for debugging...
    'timit'         : boolean - do you want to measure execution run time?

    OUTPUT:
    'fibre_profiles'  : dictionary (keys=orders) containing the calculated spatial-direction fibre profiles
    """

    print('Fitting fibre profiles...')

    if timit:
        start_time = time.time()

    # create "global" parameter dictionary for entire chip
    fibre_profiles = {}

    # make sure we have a proper mask
    if mask is None:
        cenmask = {}
    else:
        # we also only want to use the central TRUE parts of the masks, ie want ONE consecutive stretch per order
        cenmask = central_parts_of_mask(mask)

    # loop over all orders
    for ord in sorted(P_id.iterkeys()):
        print('OK, now processing ' + str(ord))

        ordpol = P_id[ord]

        # define stripe
        indices = stripe_indices[ord]
        # find the "order-box"
        sc, sr = flatten_single_stripe_from_indices(img, indices, slit_height=slit_height, timit=False)
        err_sc, err_sr = flatten_single_stripe_from_indices(err_img, indices, slit_height=slit_height, timit=False)

        if mask is None:
            cenmask[ord] = np.ones(sc.shape[1], dtype='bool')

        # fit profile for single order and save result in "global" parameter dictionary for entire chip
        fpo = get_multiple_fibre_profiles_single_order(sc, sr, err_sc, ordpol, ordmask=cenmask[ord], nfib=nfib, lfc=lfc,
                                                       sampling_size=sampling_size, step_size=step_size, varbeta=varbeta, offset=offset,
                                                       return_snr=True, debug_level=debug_level, timit=timit)

        # if stacking:
        #     colfits = determine_spatial_profiles_single_order(sc, sr, err_sc, ordpol, ordmask=mask[ord], model=model,
        #                                                       return_stats=return_stats, timit=timit)
        # else:
        #     colfits = fit_profiles_single_order(sr, sc, ordpol, osf=1, silent=True, timit=timit)

        fibre_profiles[ord] = fpo

    if timit:
        print('Time elapsed: ' + str(int(time.time() - start_time)) + ' seconds...')

    return fibre_profiles








