'''
Created on 14 Jun. 2018

@author: christoph
'''

import matplotlib.pyplot as plt
import time
import numpy as np
import astropy.io.fits as pyfits
import scipy.optimize as op

# from scipy import signal
# from scipy import ndimage
# from lmfit import Parameters, Model
# from lmfit.models import *
# from lmfit.minimizer import *

from veloce_reduction.veloce_reduction.linalg import linalg_extract_column
from veloce_reduction.veloce_reduction.helper_functions import make_norm_profiles_2, central_parts_of_mask, CMB_pure_gaussian
from veloce_reduction.veloce_reduction.order_tracing import flatten_single_stripe, flatten_single_stripe_from_indices
from veloce_reduction.veloce_reduction.wavelength_solution import find_suitable_peaks




def get_relints_single_order(sc, sr, err_sc, ordpol, fppo, ordmask=None, nfib=19, sampling_size=25, step_size=None, return_full=False, return_snr=True, debug_level=0, timit=False):
    """
    INPUT:
    'sc'             : the flux in the extracted, flattened stripe
    'sr'             : row-indices (ie in spatial direction) of the cutouts in 'sc'
    'err_sc'         : the error in the extracted, flattened stripe
    'ordpol'         : set of polynomial coefficients from P_id for that order (ie p = P_id[ord])
    'fppo'           : Fibre Profile Parameters by Order (dictionary containing the fitted fibre profile parameters)
    'ordmask'        : gives user the option to provide a mask (eg from "find_stripes")
    'nfib'           : number of fibres for which to retrieve the relative intensities (default is 19, b/c there are 19
                       object fibres for Veloce Rosso)
    'slit_height'    : height of the 'extraction slit' is 2*slit_height
    'sampling_size'  : how many pixels (in dispersion direction) either side of current i-th pixel do you want to consider? 
                       (ie stack profiles for a total of 2*sampling_size+1 pixels in dispersion direction...)
    'step_size'      : only calculate the relative intensities every so and so many pixels (should not change, plus it takes ages...)
    'return_full'    : boolean - do you want to also return the full model (if FALSE, then only the relative intensities are returned)
    'return_snr'     : boolean - do you want to return the SNR of the collapsed super-pixel at each location in 'userange'?
    'debug_level'    : for debugging...
    
    OUTPUT:
    'relints'        : relative intensities in the fibres
    'relints_norm'   : relative intensities in the fibres re-normalized to a sum of 1
    'full_model'     : a list containing the full model of every cutout (only if 'return_full' is set to TRUE)
    'modgrid'        : a list containing the grid on which the full model is evaluated (only if 'return_full' is set to TRUE)
    
    
    MODHIST:
    14/06/2018 - CMB create (essentially a clone of "determine_spatial_profiles_single_order", but removed all the parts concerning the testing of different models)
    25/06/2018 - CMB added call to "linalg_extract_column" rather than naively fitting some model
    28/06/2018 - CMB added 'return_snr' keyword
    03/08/2018 - CMB added proper error treatment    
    """
    
    if timit:
        start_time = time.time()
    if debug_level >= 1:
        print('Fitting fibre profiles for one order...') 
        
    npix = sc.shape[1]
    
    if ordmask is None:
        ordmask = np.ones(npix, dtype='bool')
    
    if step_size is None:
        step_size = 2 * sampling_size
    userange = np.arange(np.arange(npix)[ordmask][0]+sampling_size, np.arange(npix)[ordmask][-1], step_size)
    #prepare output arrays
    relints = np.zeros((len(userange),nfib))
    relints_norm = np.zeros((len(userange),nfib))
    if return_full:
        #full_model = np.zeros((len(userange),(2*sampling_size+1)*(2*slit_height)))
        full_model = []
        modgrid = []
    if return_snr:
        snr = []
    
    #loop over all columns for one order and do the profile fitting
    for i,pix in enumerate(userange):
        
        if debug_level >= 1:
            print('pix = ',str(pix))
    
        #fail-check variable    
        fu = 0
        
        #calculate SNR of collapsed super-pixel at this location
        if return_snr:
#             snr.append(np.sqrt(np.sum(sc[:,pix])))
            snr.append( np.sum(sc[:,pix]) / np.sqrt(np.sum(err_sc[:,pix]**2)) )     
        
        #check if that particular cutout falls fully onto CCD
        checkprod = np.product(sr[1:,pix].astype(float))    #exclude the first row number, as that can legitimately be zero
        #NOTE: This also covers row numbers > ny, as in these cases 'sr' is set to zero in "flatten_single_stripe(_from_indices)"
        if ordmask[pix]==False:
            fu = 1
        elif checkprod == 0:
            fu = 1
            checksum = np.sum(sr[:,pix])
            if checksum == 0:
                print('WARNING: the entire cutout lies outside the chip!!!')
                #best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
            else:
                print('WARNING: parts of the cutout lie outside the chip!!!')
                #best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
        else:    
            #this is the NORMAL case, where the entire cutout lies on the chip
            grid = []
            #data = []
            normdata = []
            #errors = []
            weights = []
            refpos = ordpol(pix)
            for j in np.arange(np.max([0,pix-sampling_size]),np.min([npix-1,pix+sampling_size])+1):
                grid.append(sr[:,j] - ordpol(j) + refpos)
                #data.append(sc[:,j])
                normdata.append(sc[:,j] / np.sum(sc[:,j]))
                # assign weights for flux (and take care of NaNs and INFs)
                # normerr = np.sqrt(sc[:,j] + RON**2) / np.sum(sc[:,j])
                # normerr = err_sc[:,j] / np.sum(sc[:,j])
                # using relative errors for weights in the fit !?!?!?
                normerr = (err_sc[:, j] / sc[:, j]) / np.sum(sc[:, j])
                pix_w = 1./(normerr*normerr)  
                pix_w[np.isinf(pix_w)] = 0.
                weights.append(pix_w)    
                ### initially I thought this was clearly rubbish as it down-weights the central parts
                ### and that we really want to use the relative errors, ie w_i = 1/(relerr_i)**2
                ### HOWEVER: this is not true, and the optimal extraction linalg routine requires absolute errors!!!    
                #weights.append(1./((np.sqrt(sc[:,j] + RON**2)) / sc[:,j])**2)    
                if debug_level >= 2:
                    #plt.plot(sr[:,j] - ordpol(j),sc[:,j],'.')
                    plt.plot(sr[:,j] - ordpol(j),sc[:,j]/np.sum(sc[:,j]),'.')
                    #plt.xlim(-5,5)
                    plt.xlim(-sc.shape[0]/2,sc.shape[0]/2)
                
            # data = np.array(data)
            normdata = np.array(normdata)
            weights = np.array(weights)
            grid = np.array(grid)
            # data = data[grid.argsort()]
            normdata = normdata[grid.argsort()]
            weights = weights[grid.argsort()]
            grid = grid[grid.argsort()]
            
            #adjust to flux level in actual pixel position
            #normdata = normdata * np.sum(sc[:,pix])
            #I THINK WE DON'T NEED THIS IN THE RELINT CASE
            
            #get fibre profiles for i-th pixel column 
            #phi = make_norm_profiles_2(grid, pix, fppo, fibs='stellar')
            phi = make_norm_profiles_2(grid, pix, fppo, fibs='all')     #(alternatively, we could pass fibs='stellar' and then not select f[5:24] and v[5:24] below)
            
            #now perform the actual amplitude fit, which is really just the optimal extraction linear algebra
            #NOTE: the RON=0 approximation is OK, because we are using normalized data - worst case we are overestimating the variance a little bit!
            #NOTE: also, we don't really care about the variance much in this case...
            f,v = linalg_extract_column(normdata, weights, phi, RON=0)     
            #now only select 'stellar' fibres 
            f = f[5:24]
            v = v[5:24]
            #that's the model if we need it (the combined model is "np.sum(fmodel,axis=1)" )
            if return_full:
                fmodel = f * phi
                #full_model[i,:] = np.sum(fmodel,axis=1)
                full_model.append(np.sum(fmodel,axis=1))
                modgrid.append(grid)
            
            #fill output array
            relints[i,:] = f
            fnorm = f/np.sum(f)
            relints_norm[i,:] = fnorm
            
            
    if timit:
        print('Elapsed time for retrieving relative intensities: '+np.round(time.time() - start_time,2).astype(str)+' seconds...')
        
    if return_full:
        if return_snr:
            return relints,relints_norm,full_model,modgrid,snr
        else:
            return relints,relints_norm,full_model,modgrid
    else:
        if return_snr:
            return relints,relints_norm,snr
        else:
            return relints,relints_norm





def old_get_relints_single_order(sc, sr, ordpol, fppo, ordmask=None, nfib=19, slit_height=25, sampling_size=25, step_size=None, RON=0., gain=1., return_full=False, return_snr=True, debug_level=0, timit=False):
    """
    INPUT:
    'sc'             : the flux in the extracted, flattened stripe
    'sr'             : row-indices (ie in spatial direction) of the cutouts in 'sc'
    'ordpol'         : set of polynomial coefficients from P_id for that order (ie p = P_id[ord])
    'fppo'           : Fibre Profile Parameters by Order (dictionary containing the fitted fibre profile parameters)
    'ordmask'        : gives user the option to provide a mask (eg from "find_stripes")
    'nfib'           : number of fibres for which to retrieve the relative intensities (default is 19, b/c there are 19 object fibres for VeloceRosso)
    'slit_height'    : height of the 'extraction slit' is 2*slit_height
    'sampling_size'  : how many pixels (in dispersion direction) either side of current i-th pixel do you want to consider? 
                       (ie stack profiles for a total of 2*sampling_size+1 pixels in dispersion direction...)
    'step_size'      : only calculate the relative intensities every so and so many pixels (should not change, plus it takes ages...)
    'RON'            : read-out noise per pixel
    'gain'           : gain
    'return_full'    : boolean - do you want to also return the full model (if FALSE, then only the relative intensities are returned)
    'return_snr'     : boolean - do you want to return the SNR of the collapsed super-pixel at each location in 'userange'?
    'debug_level'    : for debugging...
    
    OUTPUT:
    'relints'        : relative intensities in the fibres
    'relints_norm'   : relative intensities in the fibres re-normalized to a sum of 1
    'full_model'     : a list containing the full model of every cutout (only if 'return_full' is set to TRUE)
    'modgrid'        : a list containing the grid on which the full model is evaluated (only if 'return_full' is set to TRUE)
    
    
    MODHIST:
    14/06/2018 - CMB create (essentially a clone of "determine_spatial_profiles_single_order", but removed all the parts concerning the testing of different models)
    25/06/2018 - CMB added call to "linalg_extract_column" rather than naively fitting some model
    28/06/2018 - CMB added 'return_snr' keyword
    
    TODO:
    (1) use proper errors for SNR calculation (instead of just the SQRT)
    """
    
    if timit:
        start_time = time.time()
    if debug_level >= 1:
        print('Fitting fibre profiles for one order...') 
        
    npix = sc.shape[1]
    
    if ordmask is None:
        ordmask = np.ones(npix, dtype='bool')
    
    if step_size is None:
        step_size = 2 * sampling_size
    userange = np.arange(np.arange(npix)[ordmask][0]+sampling_size, np.arange(npix)[ordmask][-1], step_size)
    #prepare output arrays
    relints = np.zeros((len(userange),nfib))
    relints_norm = np.zeros((len(userange),nfib))
    if return_full:
        #full_model = np.zeros((len(userange),(2*sampling_size+1)*(2*slit_height)))
        full_model = []
        modgrid = []
    if return_snr:
        snr = []
    
    #loop over all columns for one order and do the profile fitting
    for i,pix in enumerate(userange):
        
        if debug_level >= 1:
            print('pix = ',str(pix))
    
        #fail-check variable    
        fu = 0
        
        #calculate SNR of collapsed super-pixel at this location
        if return_snr:
            snr.append(np.sqrt(np.sum(sc[:,pix])))
        
        #check if that particular cutout falls fully onto CCD
        checkprod = np.product(sr[1:,pix].astype(float))    #exclude the first row number, as that can legitimately be zero
        #NOTE: This also covers row numbers > ny, as in these cases 'sr' is set to zero in "flatten_single_stripe(_from_indices)"
        if ordmask[pix]==False:
            fu = 1
        elif checkprod == 0:
            fu = 1
            checksum = np.sum(sr[:,pix])
            if checksum == 0:
                print('WARNING: the entire cutout lies outside the chip!!!')
                #best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
            else:
                print('WARNING: parts of the cutout lie outside the chip!!!')
                #best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
        else:    
            #this is the NORMAL case, where the entire cutout lies on the chip
            grid = []
            #data = []
            normdata = []
            #errors = []
            weights = []
            refpos = ordpol(pix)
            for j in np.arange(np.max([0,pix-sampling_size]),np.min([npix-1,pix+sampling_size])+1):
                grid.append(sr[:,j] - ordpol(j) + refpos)
                #data.append(sc[:,j])
                normdata.append(sc[:,j] / np.sum(sc[:,j]))
                #assign weights for flux (and take care of NaNs and INFs)
                normerr = np.sqrt(sc[:,j] + RON**2) / np.sum(sc[:,j])
                pix_w = 1./(normerr*normerr)  
                pix_w[np.isinf(pix_w)] = 0.
                weights.append(pix_w)    
                ### initially I thought this was clearly rubbish as it down-weights the central parts
                ### and that we really want to use the relative errors, ie w_i = 1/(relerr_i)**2
                ### HOWEVER: this is not true, and the optimal extraction linalg routine requires absolute errors!!!
                #weights.append(1./((np.sqrt(sc[:,j] + RON**2)) / sc[:,j])**2)        
                if debug_level >= 2:
                    #plt.plot(sr[:,j] - ordpol(j),sc[:,j],'.')
                    plt.plot(sr[:,j] - ordpol(j),sc[:,j]/np.sum(sc[:,j]),'.')
                    #plt.xlim(-5,5)
                    plt.xlim(-sc.shape[0]/2,sc.shape[0]/2)
                
            # data = np.array(data)
            normdata = np.array(normdata)
            weights = np.array(weights)
            grid = np.array(grid)
            # data = data[grid.argsort()]
            normdata = normdata[grid.argsort()]
            weights = weights[grid.argsort()]
            grid = grid[grid.argsort()]
            
            #adjust to flux level in actual pixel position
            #normdata = normdata * np.sum(sc[:,pix])
            #I THINK WE DON'T NEED THIS IN THE RELINT CASE
            
            #get fibre profiles for i-th pixel column 
            #phi = make_norm_profiles_2(grid, pix, fppo, fibs='stellar')
            phi = make_norm_profiles_2(grid, pix, fppo, fibs='all')
            
            #now perform the actual amplitude fit, which is really just the optimal extraction linear algebra
            f,v = linalg_extract_column(normdata, weights, phi, RON=RON)
            #now only select 'stellar' fibres
            f = f[5:24]
            v = v[5:24]
            #that's the model if we need it (the combined model is "np.sum(fmodel,axis=1)" )
            if return_full:
                fmodel = f * phi
                #full_model[i,:] = np.sum(fmodel,axis=1)
                full_model.append(np.sum(fmodel,axis=1))
                modgrid.append(grid)
            
            #fill output array
            relints[i,:] = f
            fnorm = f/np.sum(f)
            relints_norm[i,:] = fnorm
            
            
    if timit:
        print('Elapsed time for retrieving relative intensities: '+np.round(time.time() - start_time,2).astype(str)+' seconds...')
        
    if return_full:
        if return_snr:
            return relints,relints_norm,full_model,modgrid,snr
        else:
            return relints,relints_norm,full_model,modgrid
    else:
        if return_snr:
            return relints,relints_norm,snr
        else:
            return relints,relints_norm





def get_relints(P_id, stripes, err_stripes, mask=None, sampling_size=25, slit_height=32, return_full=False, simu=False, debug_level=0, timit=False):
    """
    This routine computes the relative intensities in the individual fibres of a Veloce spectrum.
    
    INPUT:
    'P_id'           : dictionary of the form of {order: np.poly1d, ...} (as returned by "identify_stripes")
    'stripes'        : dictionary containing the flux in the extracted stripes (keys = orders)
    'err_stripes'    : dictionary containing the errors in the extracted stripes (keys = orders)
    'mask'           : dictionary of boolean masks (keys = orders) from "find_stripes" (masking out regions of very low signal)
    'pathdict'       : dictionary containing all directories relevant to the reduction
    'sampling_size'  : 'sampling_size'  : how many pixels (in dispersion direction) either side of current i-th pixel do you want to consider?
                       (ie stack profiles for a total of 2*sampling_size+1 pixels in dispersion direction...)
    'slit_height'    : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'return_full'    : boolean - do you want to return the full model as well?
    'simu'           : boolean - are you using simulated spectra?
    'debug_level'    : for debugging...
    'timit'          : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'wm_relints'    : weighted mean (averaged over all pixel columns of all orders) of the relative intensities in the individual fibres
    'relints'       : relative intensities in the individual fibres (only if 'return_full' is set to TRUE)
    'relints_norm'  : normalized relative intensities in the individual fibres (only if 'return_full' is set to TRUE)
    'fmodel'        : full model (only if 'return_full' is set to TRUE)
    'model_grid'    : "x-grid" for the full model (only if 'return_full' is set to TRUE)
    """

    assert pathdict is not None, 'ERROR: pathdict not provided!!!'
    fibparms_path = pathdict['fp']

    print('Fitting relative intensities of fibres...')
    
    #read in polynomial coefficients of best-fit individual-fibre-profile parameters
    if simu:
        fibparms = np.load(fibparms_path + 'sim/fibparms_by_ord.npy').item()
    else:
        #fibparms = np.load(fibparms_path + 'real/first_real_veloce_test_fps.npy').item()
        fibparms = np.load(fibparms_path + 'real/from_master_white_40orders.npy').item()
    
    if timit:
        start_time = time.time()
    
    #create output dictionaries
    relints = {}
    snr = {}
    relints_norm = {}
    if return_full:
        fmodel = {}
        model_grid = {}
    
    if mask is None:
        cenmask = {}
    else:
        #we also only want to use the central TRUE parts of the masks, ie want ONE consecutive stretch per order
        cenmask = central_parts_of_mask(mask)
        
    #loop over all orders
    for ord in sorted(P_id.iterkeys()):
        print('OK, now processing '+str(ord))
        
        #fibre profile parameters for that order
        fppo = fibparms[ord]
        
        ordpol = P_id[ord]
        
        # define stripe
        stripe = stripes[ord]
        err_stripe = err_stripes[ord]
        # find the "order-box"
        sc,sr = flatten_single_stripe(stripe, slit_height=slit_height, timit=False)
        err_sc,err_sr = flatten_single_stripe(err_stripe, slit_height=slit_height, timit=False)
        
        if mask is None:
            cenmask[ord] = np.ones(sc.shape[1], dtype='bool')
        
        # fit profile for single order and save result in "global" parameter dictionary for entire chip
        if return_full:
            relints_ord,relints_ord_norm,fmodel_ord,modgrid_ord,snr_ord = get_relints_single_order(sc, sr, err_sc, ordpol, fppo, ordmask=cenmask[ord], 
                                                                                                   nfib=19, sampling_size=sampling_size, return_full=return_full)
        else:
            relints_ord,relints_ord_norm,snr_ord = get_relints_single_order(sc, sr, err_sc, ordpol, fppo, ordmask=cenmask[ord], nfib=19, 
                                                                            sampling_size=sampling_size, return_full=return_full)
        
        if debug_level >= 2:
            #try to find cause for NaNs
            print('n_elements should be:   len(relints_ord)*19 = '+str(len(relints_ord)*19))
            print('relints_ord   : ' + str(np.sum(relints_ord == relints_ord)))
            print('relints_ord_norm   : ' + str(np.sum(relints_ord_norm == relints_ord_norm)))
            print('n_elements(SNR) should be:   '+str(len(snr_ord))+'   ;   '+str(np.sum(snr_ord == snr_ord).all()))
                  
        
        relints[ord] = relints_ord
        snr[ord] = snr_ord
        relints_norm[ord] = relints_ord_norm
        if return_full:
            fmodel[ord] = fmodel_ord
            model_grid[ord] = modgrid_ord
            
    
    #get weighted mean of all relints (weights = SNRs)
    allsnr = []
    #allrelints = np.zeros([])
    for ord in sorted(snr.keys()):
        allsnr.append(snr[ord])
        try:
            allrelints = np.vstack((allrelints, relints_norm[ord]))
        except:
            allrelints = relints_norm[ord]
    wm_relints = np.average(allrelints, axis=0, weights=np.array(allsnr), returned=False)
        
    
    if timit:
        print('Time elapsed: '+str(int(time.time() - start_time))+' seconds...')  
          
    if return_full:
        return wm_relints, relints, relints_norm, fmodel, model_grid
    else:
        return wm_relints





def get_relints_from_indices(P_id, img, err_img, stripe_indices, mask=None, pathdict=None, sampling_size=25, slit_height=32, return_full=False, simu=False, debug_level=0, timit=False):
    """
    This routine computes the relative intensities in the individual fibres of a Veloce spectrum.
    
    CLONE OF 'get_relints', but uses stripe_indices, rather than the stripes themselves.
    
    INPUT:
    'P_id'           : dictionary of the form of {order: np.poly1d, ...} (as returned by "identify_stripes")
    'img'            : 2-dim input array/image
    'err_img'        : estimated uncertainties in the 2-dim input array/image
    'stripe_indices' : dictionary (keys = orders) containing the indices of the pixels that are identified as the "stripes" (ie the to-be-extracted regions centred on the orders)
    'mask'           : dictionary of boolean masks (keys = orders) from "find_stripes" (masking out regions of very low signal)
    'pathdict'       : dictionary containing all directories relevant to the reduction
    'sampling_size'  : 'sampling_size'  : how many pixels (in dispersion direction) either side of current i-th pixel do you want to consider?
                       (ie stack profiles for a total of 2*sampling_size+1 pixels in dispersion direction...)
    'slit_height'    : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'return_full'    : boolean - do you want to return the full model as well?
    'simu'           : boolean - are you using simulated spectra?
    'debug_level'    : for debugging...
    'timit'          : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'wm_relints'    : weighted mean (averaged over all pixel columns of all orders) of the relative intensities in the individual fibres
    'relints'       : relative intensities in the individual fibres (only if 'return_full' is set to TRUE)
    'relints_norm'  : normalized relative intensities in the individual fibres (only if 'return_full' is set to TRUE)
    'fmodel'        : full model (only if 'return_full' is set to TRUE)
    'model_grid'    : "x-grid" for the full model (only if 'return_full' is set to TRUE)
    """

    assert pathdict is not None, 'ERROR: pathdict not provided!!!'
    fibparms_path = pathdict['fp']

    print('Fitting relative intensities of fibres...')

    if timit:
        start_time = time.time()
        
    #read in polynomial coefficients of best-fit individual-fibre-profile parameters
    if simu:
        fibparms = np.load(fibparms_path + 'sim/fibparms_by_ord.npy').item()
    else:
        #fibparms = np.load(fibparms_path + 'real/first_real_veloce_test_fps.npy').item()
        fibparms = np.load(fibparms_path + 'real/from_master_white_40orders.npy').item()
        
    #create output dictionaries
    relints = {}
    snr = {}
    relints_norm = {}
    if return_full:
        fmodel = {}
        model_grid = {}
    
    if mask is None:
        cenmask = {}
    else:
        #we also only want to use the central TRUE parts of the masks, ie want ONE consecutive stretch per order
        cenmask = central_parts_of_mask(mask)
        
    #loop over all orders
    for ord in sorted(P_id.iterkeys()):
        print('OK, now processing '+str(ord))
        
        #fibre profile parameters for that order
        fppo = fibparms[ord]
        
        ordpol = P_id[ord]
        
        # define stripe
        indices = stripe_indices[ord]
        # find the "order-box"
        sc,sr = flatten_single_stripe_from_indices(img, indices, slit_height=slit_height, timit=False)
        err_sc,err_sr = flatten_single_stripe_from_indices(err_img, indices, slit_height=slit_height, timit=False)
        
        if mask is None:
            cenmask[ord] = np.ones(sc.shape[1], dtype='bool')
        
        # fit profile for single order and save result in "global" parameter dictionary for entire chip
        if return_full:
            relints_ord,relints_ord_norm,fmodel_ord,modgrid_ord,snr_ord = get_relints_single_order(sc, sr, err_sc, ordpol, fppo, ordmask=cenmask[ord], nfib=19, 
                                                                                                   sampling_size=sampling_size, return_full=return_full)
        else:
            relints_ord,relints_ord_norm,snr_ord = get_relints_single_order(sc, sr, err_sc, ordpol, fppo, ordmask=cenmask[ord], nfib=19, 
                                                                            sampling_size=sampling_size, return_full=return_full)
        
        
        if debug_level >= 2:
            #try to find cause for NaNs
            print('n_elements should be:   len(relints_ord)*19 = '+str(len(relints_ord)*19))
            print('relints_ord   : ' + str(np.sum(relints_ord == relints_ord)))
            print('relints_ord_norm   : ' + str(np.sum(relints_ord_norm == relints_ord_norm)))
            print('n_elements(SNR) should be:   '+str(len(snr_ord))+'   ;   '+str(np.sum(snr_ord == snr_ord).all()))
        
        
        relints[ord] = relints_ord
        snr[ord] = snr_ord
        relints_norm[ord] = relints_ord_norm
        if return_full:
            fmodel[ord] = fmodel_ord
            model_grid[ord] = modgrid_ord
            
    
    #get weighted mean of all relints (weights = SNRs)
    allsnr = []
    #allrelints = np.zeros([])
    for ord in sorted(snr.keys()):
        allsnr.append(snr[ord])
        try:
            allrelints = np.vstack((allrelints,relints_norm[ord]))
        except:
            allrelints = relints_norm[ord]
    wm_relints = np.average(allrelints, axis=0, weights=np.array(allsnr), returned=False)
            
    
    if timit:
        print('Time elapsed: '+str(int(time.time() - start_time))+' seconds...')  
          
    if return_full:
        return wm_relints, relints, relints_norm, fmodel, model_grid
    else:
        return wm_relints





def get_relints_single_order_gaussian(sc, sr, err_sc, ordpol, ordmask=None, nfib=24, sampling_size=25, step_size=None,
                                      return_snr=True, debug_level=0, timit=False):
    """
    INPUT:
    'sc'             : the flux in the extracted, flattened stripe
    'sr'             : row-indices (ie in spatial direction) of the cutouts in 'sc'
    'err_sc'         : the error in the extracted, flattened stripe
    'ordpol'         : set of polynomial coefficients from P_id for that order (ie p = P_id[ord])
    'fppo'           : Fibre Profile Parameters by Order (dictionary containing the fitted fibre profile parameters)
    'ordmask'        : gives user the option to provide a mask (eg from "find_stripes")
    'nfib'           : number of fibres for which to retrieve the relative intensities (default is 24, b/c there are 19 object fibres plus 5 sky fibres for VeloceRosso)
    'slit_height'    : height of the 'extraction slit' is 2*slit_height
    'sampling_size'  : how many pixels (in dispersion direction) either side of current i-th pixel do you want to consider?
                       (ie stack profiles for a total of 2*sampling_size+1 pixels in dispersion direction...)
    'step_size'      : only calculate the relative intensities every so and so many pixels (should not change, plus it takes ages...)
    'return_full'    : boolean - do you want to also return the full model (if FALSE, then only the relative intensities are returned)
    'return_snr'     : boolean - do you want to return the SNR of the collapsed super-pixel at each location in 'userange'?
    'debug_level'    : for debugging...

    OUTPUT:
    'relints'        : relative intensities in the fibres
    'relints_norm'   : relative intensities in the fibres re-normalized to a sum of 1
    'full_model'     : a list containing the full model of every cutout (only if 'return_full' is set to TRUE)
    'modgrid'        : a list containing the grid on which the full model is evaluated (only if 'return_full' is set to TRUE)


    MODHIST:
    14/06/2018 - CMB create (essentially a clone of "determine_spatial_profiles_single_order", but removed all the parts concerning the testing of different models)
    25/06/2018 - CMB added call to "linalg_extract_column" rather than naively fitting some model
    28/06/2018 - CMB added 'return_snr' keyword
    03/08/2018 - CMB added proper error treatment
    """

    fitwidth = 30

    if timit:
        start_time = time.time()
    if debug_level >= 1:
        print('Fitting fibre profiles (pure Gaussians) for one order...')

    npix = sc.shape[1]

    if ordmask is None:
        ordmask = np.ones(npix, dtype='bool')

    if step_size is None:
        step_size = 2 * sampling_size
    userange = np.arange(np.arange(npix)[ordmask][0] + sampling_size, np.arange(npix)[ordmask][-1], step_size)
    # don't use pixel columns 200 pixels from either end of the chip
    userange = userange[np.logical_and(userange > 200, userange < npix - 200)]

    # prepare output arrays
    positions = np.zeros((len(userange), nfib))
    relints = np.zeros((len(userange), nfib))
    relints_norm = np.zeros((len(userange), nfib))
    if return_snr:
        snr = []

    # loop over all columns for one order and do the profile fitting
    for i, pix in enumerate(userange):

        if debug_level >= 1:
            print('pix = ', str(pix))

        # fail-check variable
        fu = 0

        # calculate SNR of collapsed super-pixel at this location
        if return_snr:
            #snr.append(np.sqrt(np.sum(sc[:,pix])))    # ignoring RON
            snr.append(np.sum(sc[:, pix]) / np.sqrt(np.sum(err_sc[:, pix] ** 2)))

        # check if that particular cutout falls fully onto CCD
        checkprod = np.product(sr[1:, pix].astype(float))  # exclude the first row number, as that can legitimately be zero
        # NOTE: This also covers row numbers > ny, as in these cases 'sr' is set to zero in "flatten_single_stripe(_from_indices)"
        if ordmask[pix] == False:
            fu = 1
            line_pos_fitted = np.repeat(-1, nfib)
            line_amp_fitted = np.repeat(-1, nfib)
            line_sigma_fitted = np.repeat(-1, nfib)
        elif checkprod == 0:
            fu = 1
            checksum = np.sum(sr[:, pix])
            if checksum == 0:
                print('WARNING: the entire cutout lies outside the chip!!!')
                # best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
            else:
                print('WARNING: parts of the cutout lie outside the chip!!!')
                # best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
            line_pos_fitted = np.repeat(-1, nfib)
            line_amp_fitted = np.repeat(-1, nfib)
            line_sigma_fitted = np.repeat(-1, nfib)
        else:
            # this is the NORMAL case, where the entire cutout lies on the chip
            grid = []
            # data = []
            normdata = []
            # errors = []
            weights = []
            refpos = ordpol(pix)
            for j in np.arange(np.max([0, pix - sampling_size]), np.min([npix - 1, pix + sampling_size]) + 1):
                grid.append(sr[:, j] - ordpol(j) + refpos)
                # data.append(sc[:,j])
                normdata.append(sc[:, j] / np.sum(sc[:, j]))
                # assign weights for flux (and take care of NaNs and INFs)
                # normerr = np.sqrt(sc[:,j] + RON**2) / np.sum(sc[:,j])
                # normerr = err_sc[:, j] / np.sum(sc[:, j])
                # using relative errors for weights in the fit !?!?!?
                normerr = (err_sc[:, j] / sc[:, j]) / np.sum(sc[:, j])
                pix_w = 1. / (normerr * normerr)
                pix_w[np.isinf(pix_w)] = 0.
                weights.append(pix_w)
                ### initially I thought this was clearly rubbish as it down-weights the central parts
                ### and that we really want to use the relative errors, ie w_i = 1/(relerr_i)**2
                ### HOWEVER: this is not true, and the optimal extraction linalg routine requires absolute errors!!!
                # weights.append(1./((np.sqrt(sc[:,j] + RON**2)) / sc[:,j])**2)
                if debug_level >= 2:
                    # plt.plot(sr[:,j] - ordpol(j),sc[:,j],'.')
                    plt.plot(sr[:, j] - ordpol(j), sc[:, j] / np.sum(sc[:, j]), '.')
                    # plt.xlim(-5,5)
                    plt.xlim(-sc.shape[0] / 2, sc.shape[0] / 2)

            # data = np.array(data)
            # normdata = np.array(normdata)
            # weights = np.array(weights)
            # grid = np.array(grid)
            # need 1-dim arrays
            normdata = np.array(normdata).flatten()
            weights = np.array(weights).flatten()
            grid = np.array(grid).flatten()
            # data = data[grid.argsort()]
            normdata = normdata[grid.argsort()]
            weights = weights[grid.argsort()]
            grid = grid[grid.argsort()]

            if debug_level >= 2:
                plt.plot(grid,normdata)

            # smooth data to make sure we are not finding noise peaks, and add tiny slope to make sure peaks are found even when pixel-values are like [...,3,6,18,41,41,21,11,4,...]
            # this uses snippets from "find_suitable_peaks" and "fit_emission_lines"
            # xx = np.arange(len(normdata))
            # filtered_data = ndimage.gaussian_filter(normdata.astype(np.float), 5.) + xx * 1e-8
            # allpeaks = signal.argrelextrema(filtered_data, np.greater)[0]
            # mostpeaks = allpeaks.copy()
            # goodpeaks = allpeaks.copy()
            dynrange = np.max(normdata) - np.min(normdata)
            goodpeaks, mostpeaks, allpeaks = find_suitable_peaks(normdata, thresh=np.min(normdata)+0.5*dynrange, bgthresh=np.min(normdata)+0.25*dynrange,
                                                                 clip_edges=False, gauss_filter_sigma=10, slope=1e-8)
            #print('Number of fibres found: ',len(goodpeaks))
            if len(goodpeaks) != 24 :
                print('FUUUUU!!! pix = ',pix)
                line_pos_fitted = np.repeat(-1, nfib)
                line_amp_fitted = np.repeat(-1, nfib)
                line_sigma_fitted = np.repeat(-1, nfib)
            else:
                line_pos_fitted = []
                line_amp_fitted = []
                line_sigma_fitted = []


                # #do the fitting
                # peaks = np.r_[grid[goodpeaks]]
                #
                # npeaks = len(peaks)
                # #xrange = xx[np.max([0, peaks[0] - fitwidth]): np.min([peaks[-1] + fitwidth + 1, len(data) - 1])]  # this should satisfy: len(xrange) == len(checkrange) - 2*fitwidth + len(peaks)
                #
                # guess = []
                # lower_bounds = []
                # upper_bounds = []
                # for i in range(npeaks):
                #     if varbeta:
                #         guess.append(np.array([peaks[i], 1., normdata[goodpeaks[i]], 2.]))
                #         lower_bounds.append([peaks[i] - 1, 0, 0, 1])
                #         upper_bounds.append([peaks[i] + 1, np.inf, np.inf, 4])
                #     else:
                #         guess.append(np.array([peaks[i], 1., normdata[goodpeaks[i]]]))
                #         lower_bounds.append([peaks[i] - 1, 0, 0])
                #         upper_bounds.append([peaks[i] + 1, np.inf, np.inf])
                # guess = np.array(guess).flatten()
                # lower_bounds = np.array(lower_bounds).flatten()
                # upper_bounds = np.array(upper_bounds).flatten()
                # if varbeta:
                #     popt, pcov = op.curve_fit(multi_fibmodel_with_amp, grid, normdata, p0=guess,
                #                               bounds=(lower_bounds, upper_bounds))
                # else:
                #     popt, pcov = op.curve_fit(CMB_multi_gaussian, grid, normdata, p0=guess,
                #                               bounds=(lower_bounds, upper_bounds))


                global_model = np.zeros(grid.shape)

                # if debug_level >= 2:
                #    plt.plot(grid, normdata)

                for xguess in goodpeaks:
                    ################################################################################################################################################################################
                    # METHOD 1 (using curve_fit; slightly faster than method 2, but IDK how to make sure the fit converged (as with .ier below))

                    peaks = np.r_[grid[xguess]]

                    npeaks = len(peaks)
                    #xrange = xx[peaks[0] - fitwidth: peaks[-1] + fitwidth + 1]
                    xrange = grid[xguess - fitwidth: xguess + fitwidth + 1]

                    guess = np.array([grid[xguess], 0.6, normdata[xguess]])
                    popt, pcov = op.curve_fit(CMB_pure_gaussian, xrange, normdata[xguess - fitwidth: xguess + fitwidth + 1], p0=guess,
                                              bounds=([grid[xguess] - 1, 0, 0], [grid[xguess] + 1, np.inf, np.inf]))
                    fitted_pos = popt[0]
                    fitted_sigma = popt[1]
                    fitted_amp = popt[2]

                    line_pos_fitted.append(fitted_pos)
                    line_sigma_fitted.append(fitted_sigma)
                    line_amp_fitted.append(fitted_amp)

                    global_model += CMB_pure_gaussian(grid,*popt)

                    if debug_level >= 2:
                        plt.plot(xrange, CMB_pure_gaussian(xrange, *popt))



        # fill output array
        positions[i,:] = line_pos_fitted
        relints[i, :] = line_amp_fitted
        fnorm = line_amp_fitted / np.sum(line_amp_fitted)
        relints_norm[i, :] = fnorm

    if timit:
        print('Elapsed time for retrieving relative intensities: ' + np.round(time.time() - start_time, 2).astype(str) + ' seconds...')

    if return_snr:
        return relints, relints_norm, positions, snr
    else:
        return relints, relints_norm, positions





def get_relints_from_indices_gaussian(P_id, img, err_img, stripe_indices, mask=None, nfib=24, sampling_size=25, slit_height=32,
                                      debug_level=0, timit=False):
    """
    This routine computes the relative intensities in the individual fibres of a Veloce spectrum.

    CLONE OF 'get_relints', but uses stripe_indices, rather than the stripes themselves.

    INPUT:
    'P_id'           : dictionary of the form of {order: np.poly1d, ...} (as returned by "identify_stripes")
    'img'            : 2-dim input array/image
    'err_img'        : estimated uncertainties in the 2-dim input array/image
    'stripe_indices' : dictionary (keys = orders) containing the indices of the pixels that are identified as the "stripes" (ie the to-be-extracted regions centred on the orders)
    'mask'           : dictionary of boolean masks (keys = orders) from "find_stripes" (masking out regions of very low signal)
    'nfib'           : number of fibres (default is 24, ie 19 stellar + 5 sky)
    'sampling_size'  : 'sampling_size'  : how many pixels (in dispersion direction) either side of current i-th pixel do you want to consider?
                       (ie stack profiles for a total of 2*sampling_size+1 pixels in dispersion direction...)
    'slit_height'    : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'debug_level'    : for debugging...
    'timit'          : boolean - do you want to measure execution run time?

    OUTPUT:
    'wm_relints'    : weighted mean (averaged over all pixel columns of all orders) of the relative intensities in the individual fibres
    'relints'       : relative intensities in the individual fibres (only if 'return_full' is set to TRUE)
    'relints_norm'  : normalized relative intensities in the individual fibres (only if 'return_full' is set to TRUE)
    """

    print('Fitting relative intensities of fibres using simple Gaussians...')

    if timit:
        start_time = time.time()

    # create output dictionaries
    pos = {}
    relints = {}
    snr = {}
    relints_norm = {}

    if mask is None:
        cenmask = {}
    else:
        # we also only want to use the central TRUE parts of the masks, ie want ONE consecutive stretch per order
        cenmask = central_parts_of_mask(mask)

    # loop over all orders
    for ord in sorted(P_id.iterkeys()):
        print('OK, now processing ' + str(ord))

        # # fibre profile parameters for that order
        # fppo = fibparms[ord]

        ordpol = P_id[ord]

        # define stripe
        indices = stripe_indices[ord]

        # find the "order-box"
        sc, sr = flatten_single_stripe_from_indices(img, indices, slit_height=slit_height, timit=False)
        err_sc, err_sr = flatten_single_stripe_from_indices(err_img, indices, slit_height=slit_height, timit=False)


        if mask is None:
            cenmask[ord] = np.ones(sc.shape[1], dtype='bool')

        # fit profile for single order and save result in "global" parameter dictionary for entire chip
        relints_ord, relints_ord_norm, positions, snr_ord = get_relints_single_order_gaussian(sc, sr, err_sc, ordpol, ordmask=cenmask[ord], nfib=nfib, sampling_size=sampling_size)
        
        # debugging...
        if debug_level >= 2:
            # try to find cause for NaNs
            print('n_elements should be:   len(relints_ord)*nfib = ' + str(len(relints_ord) * 19))
            print('relints_ord   : ' + str(np.sum(relints_ord == relints_ord)))
            print('relints_ord_norm   : ' + str(np.sum(relints_ord_norm == relints_ord_norm)))
            print('n_elements(SNR) should be:   ' + str(len(snr_ord)) + '   ;   ' + str(
                np.sum(snr_ord == snr_ord).all()))

        pos[ord] = positions
        relints[ord] = relints_ord
        snr[ord] = snr_ord
        relints_norm[ord] = relints_ord_norm
        # if return_full:
        #     fmodel[ord] = fmodel_ord
        #     model_grid[ord] = modgrid_ord

    # get weighted mean of all relints (weights = SNRs)
    allsnr = []
    # allrelints = np.zeros([])
    for ord in sorted(snr.keys()):
        allsnr.append(snr[ord])
        try:
            allrelints = np.vstack((allrelints, relints_norm[ord]))
        except:
            allrelints = relints_norm[ord]
    flattened_allsnr = [snr for sublist in allsnr for snr in sublist]
    wm_relints = np.average(allrelints, axis=0, weights=np.array(flattened_allsnr), returned=False)

    if timit:
        print('Time elapsed: ' + str(int(time.time() - start_time)) + ' seconds...')

    # if return_full:
    #     return wm_relints, relints, relints_norm, fmodel, model_grid
    # else:
    #     return wm_relints
    return wm_relints, relints, relints_norm, pos, allsnr





def append_relints_to_FITS(relints, fn):
    
    nfib = len(relints)
    
    assert nfib in [19,24], 'ERROR: relative intensities provided are not for exactly 19 or exactly 24 fibres!!!'
    
    # numbering of fibres as per Ion Lawrence's diagram (email from 22/02/2018
    # ie correspoding to the following layout:
    # LC S1 S3 S4 X I O2 O1 I O2 O1 I O2 O1 C O2 O1 I O2 O1 I O2 O1 I X S2 S5 ThXe
    # LC S1 S3 S4 X 2 19  8 3  9 10 4 11 12 1 13 14 5 15 16 6 17 18 7 X S2 S5 ThXe
    # here 'O1' is the outer ring that is slightly further away from the centre!!!
    
    if nfib == 19:    
        # prepare information on fibres
        fibinfo = ['inner ring', 'outer ring (2)', 'outer ring (1)', 'inner ring', 'outer ring (2)', 'outer ring (1)', 'inner ring', 'outer ring (2)', 'outer ring (1)',
                   'central', 'outer ring (2)', 'outer ring (1)', 'inner ring', 'outer ring (2)', 'outer ring (1)', 'inner ring', 'outer ring (2)', 'outer ring (1)', 'inner ring']
        fibnums = ['02', '19', '08', '03', '09', '10', '04', '11', '12', '01', '13', '14', '05', '15', '16', '06', '17', '18', '07']
        # the pseudo-slit is reversed w.r.t. my simulations - we therefore turn the fibnums array around
        fibnums = fibnums[::-1]   # so this goes from simThXe to LFC now
    elif nfib == 24:
        # prepare information on fibres
        fibinfo = ['sky', 'sky', 'sky', 'inner ring', 'outer ring (2)', 'outer ring (1)', 'inner ring', 'outer ring (2)', 'outer ring (1)', 'inner ring', 'outer ring (2)', 'outer ring (1)',
                   'central', 'outer ring (2)', 'outer ring (1)', 'inner ring', 'outer ring (2)', 'outer ring (1)', 'inner ring', 'outer ring (2)', 'outer ring (1)', 'inner ring', 'sky', 'sky']
        fibnums = ['S1', 'S3', 'S4', '02', '19', '08', '03', '09', '10', '04', '11', '12', '01', '13', '14', '05', '15', '16', '06', '17', '18', '07', 'S2', 'S5']
        # the pseudo-slit is reversed w.r.t. my simulations - we therefore turn the fibnums array around
        fibnums = fibnums[::-1]   # so this goes from simThXe to LFC now
    
    #loop over all fibres
    for i in np.arange(nfib):
        pyfits.setval(fn, 'RELINT'+str(i+1).zfill(2), value=relints[i], comment='fibre '+str(fibnums[i])+'   ('+fibinfo[i]+' fibre)')
        
    return








