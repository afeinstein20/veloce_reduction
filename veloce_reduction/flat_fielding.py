'''
Created on 26 Apr. 2018

@author: christoph
'''

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import medfilt
from scipy import ndimage
import astropy.io.fits as pyfits




# xdisp_boxsize = 1
# disp_boxsize = 15
# medfiltered_flat = medfilt(MW,[xdisp_boxsize,disp_boxsize])
# OR
# dum, filtered_flat = make_model_stripes(...) --> see spatial_profiles.py
#
# pix_sens_image = MW / medfiltered_flat   #should be roughly flat & scattering around 1
# 
# smoothed_MW = MW / pix_sens_image    #ie for the flat fields that means that smoothed_MW = filtered flat...
# smoothed_img = img / pix_sens_image
#

# maybe the not-fitted offsets in the model_stripes from "make_model_stripes_gausslike" are causing a Fubar in the division here!?!?!?




def onedim_pixtopix_variations_spline(flat, knots=9, guess=13, savefits=True, path=None, debug_level=0):
    """
    This routine fits a smoothing spline to an observed flat field in order to determine the pixel-to-pixel sensitivity variations
    as well as the fringing pattern in the red orders. This is done in 1D, ie for the already extracted spectrum.
    
    INPUT:
    'flat'          : dictionary / np.array containing the extracted flux from the flat field (master white) (keys = orders)
    'knots'         : the desired number of knots for the smoothing spline
    'guess'         : starting guess for the "smoothing factor" that determines the number of knots:  s = 10^guess  (see online documentation for scipy.interpolate.UnivariateSpline)
    
    OUTPUT:
    'pix_sens'      : dictionary / np.array of the pixel-to-pixel sensitivities (keys = orders)
    'smoothed_flat' : dictionary / np.array of the smoothed (ie filtered) whites (keys = orders)
    
    MODHIST:
    13/12/2019 - CMB create (clone of onedim_pixtopix_variations)
    """
    
    if savefits:
        assert path is not None, 'ERROR: path variable is not set!'
    
    # check whether it's a numpy array (eg from FITS file), or a python dictionary
    if flat.__class__ == np.ndarray:
        
        # make sure that they are either quick-extracted spectra (order, pixel), or optimal-extracted spectra (order, fibre, pixel)
        assert len(flat.shape) in [2,3], 'ERROR: shape of flat not recognized!!!'
        assert flat.shape[-1] == 4112, 'ERROR: there are NOT exactly 4112 pixels in dispersion direction'
        
        if debug_level >= 1:
            print('Fitting smoothing spline with ' + str(knots) + ' knots to...')
        
        xx = np.arange(flat.shape[-1])
        pix_sens = np.zeros(flat.shape) - 1.
        smoothed_flat = np.zeros(flat.shape) - 1.
        
        # loop over all orders
        for o in range(flat.shape[0]):
            if debug_level >= 1:
                print('Order_'+str(o+1).zfill(2))
            # are they optimal 3a extracted spectra?
            if len(flat.shape) == 3:
                # loop over all fibres
                for fib in range(flat.shape[1]): 
                    if debug_level >= 2:
                        print('Fibre '+str(fib+1).zfill(2))
                    # determine smoothing parameter to get desired number of knots
                    spl = UnivariateSpline(xx, flat[o,fib,:], s=10**guess)
                    nk = len(spl.get_knots())
                    if nk < knots:
                        add = 0
                        while nk < knots:
                            add -= 0.01
                            spl = UnivariateSpline(xx, flat[o,fib,:], s=10**(guess + add))
                            nk = len(spl.get_knots())
#                             print(add, nk)
                    elif nk > knots:
                        add = 0
                        while nk > knots:
                            add += 0.01
                            spl = UnivariateSpline(xx, flat[o,fib,:], s=10**(guess + add))
                            nk = len(spl.get_knots())
#                             print(add, nk)
                    smoothed_flat[o,fib,:] = spl(xx)
                    pix_sens[o,fib,:] = flat[o,fib,:] / smoothed_flat[o,fib,:]
            # or are they just quick-extracted spectra
            else:
                # determine smoothing parameter to get desired number of knots
                spl = UnivariateSpline(xx, flat[o,:], s=10**guess)
                nk = len(spl.get_knots())
                if nk < knots:
                    add = 0
                    while nk < knots:
                        add -= 0.01
                        spl = UnivariateSpline(xx, flat[o,:], s=10**(guess + add))
                        nk = len(spl.get_knots())
#                         print(add, nk)
                elif nk > knots:
                    add = 0
                    while nk > knots:
                        add += 0.01
                        spl = UnivariateSpline(xx, flat[o,:], s=10**(guess + add))
                        nk = len(spl.get_knots())
#                         print(add, nk)
                smoothed_flat[o,:] = spl(xx)
                pix_sens[o,:] = flat[o,:] / smoothed_flat[o,:]
        
        if savefits:
            date = path.split('/')[-2]
            # get header from master white
            h = pyfits.getheader(path + date + '_master_white.fits')
            # add requested number of knots to header
            h['N_KNOTS'] = (knots, 'number of knots used in smoothing spline')
            pyfits.writeto(path + date + '_smoothed_flat.fits', smoothed_flat, h)
            pyfits.writeto(path + date + '_pixel_sensitivity.fits', pix_sens, h)
                
    elif flat.__class__ == dict:
        print('This has not been implemented yet for dictionaries...')
        return
#         pix_sens = {}
#         smoothed_flat = {}
#         # loop over all orders
#         for ord in sorted(flat.keys()): 
            
    else:
        print('ERROR: data type / variable class not recognized')
        return   
    
    if debug_level >= 1:
        print('DONE!!!')
        
    return smoothed_flat, pix_sens





def onedim_pixtopix_variations(flat, filt='gaussian', filter_width=25):
    """
    This routine applies a filter ('gaussian' / 'savgol' / 'median') to an observed flat field in order to determine the pixel-to-pixel sensitivity variations
    as well as the fringing pattern in the red orders. This is done in 1D, ie for the already extracted spectrum.
    
    INPUT:
    'flat'          : dictionary / np.array containing the extracted flux from the flat field (master white) (keys = orders)
    'filt'          : method of filtering ('gaussian' / 'savgol' / 'median') - WARNING: ONLY GAUSSIAN FILTER HAS BEEN IMPLEMENTED SO FAR!!!
    'filter_width'  : the width of the kernel for the filtering in pixels; defined differently for the different types of filters (see description of scipy.ndimage)
    
    OUTPUT:
    'pix_sens'      : dictionary of the pixel-to-pixel sensitivities (keys = orders)
    'smoothed_flat' : dictionary of the smoothed (ie filtered) whites (keys = orders)
    
    MODHIST:
    24/05/2018 - CMB create
    15/01/2019 - CMB added choice of dict / np-array as input
    """
    
    while filt.lower() not in ['g','gaussian','s','savgol','m','median']:
        print("ERROR: filter choice not recognised!")
        filt = raw_input("Please try again: ['(G)aussian','(S)avgol','(M)edian']")
    
    # check whether it's a numpy array (eg from FITS file), or a python dictionary
    if flat.__class__ == np.ndarray:
        
        # make sure that they are either quick-extracted spectra (order, pixel), or optimal-extracted spectra (order, fibre, pixel)
        assert len(flat.shape) in [2,3], 'ERROR: shape of flat not recognized!!!'
        
        pix_sens = np.zeros(flat.shape) - 1.
        smoothed_flat = np.zeros(flat.shape) - 1.
        # loop over all orders
        for o in range(flat.shape[0]):
            # are they optimal 3a extracted spectra?
            if len(flat.shape) == 3:
                # loop over all fibres
                for f in range(flat.shape[1]): 
                    if filt.lower() in ['g','gaussian']:
                        #Gaussian filter
                        smoothed_flat[o,f,:] = ndimage.gaussian_filter(flat[o,f,:], filter_width)    
                        pix_sens[o,f,:] = flat[o,f,:] / smoothed_flat[o,f,:]
                    elif filt.lower() in ['s','savgol']:
                        print('WARNING: SavGol filter not implemented yet!!!')
                        break
                    elif filt.lower() in ['m','median']:
                        print('WARNING: Median filter not implemented yet!!!')
                        break
                    else:
                        #This should never happen!!!
                        print("ERROR: filter choice still not recognised!")
                        break
            # or are they just quick-extracted spectra
            else:
                if filt.lower() in ['g','gaussian']:
                    #Gaussian filter
                    smoothed_flat[o,:] = ndimage.gaussian_filter(flat[o,:], filter_width)    
                    pix_sens[o,:] = flat[o,:] / smoothed_flat[o,:]
                elif filt.lower() in ['s','savgol']:
                    print('WARNING: SavGol filter not implemented yet!!!')
                    break
                elif filt.lower() in ['m','median']:
                    print('WARNING: Median filter not implemented yet!!!')
                    break
                else:
                    #This should never happen!!!
                    print("ERROR: filter choice still not recognised!")
                    break
                
    elif flat.__class__ == dict:
        pix_sens = {}
        smoothed_flat = {}
        # loop over all orders
        for ord in sorted(flat.keys()): 
            if filt.lower() in ['g','gaussian']:
                #Gaussian filter
                smoothed_flat[ord] = ndimage.gaussian_filter(flat[ord], filter_width)    
                pix_sens[ord] = flat[ord] / smoothed_flat[ord]
            elif filt.lower() in ['s','savgol']:
                print('WARNING: SavGol filter not implemented yet!!!')
                break
            elif filt.lower() in ['m','median']:
                print('WARNING: Median filter not implemented yet!!!')
                break
            else:
                #This should never happen!!!
                print("ERROR: filter choice still not recognised!")
                break
    else:
        print('ERROR: data type / variable class not recognized')
        return   
        
    return smoothed_flat, pix_sens
    
   
    
    
    
def onedim_pixtopix_variations_single_order(f_flat, filt='gaussian', filter_width=25):
    """
    This routine applies a filter ('gaussian' / 'savgol' / 'median') to an observed flat field in order to determine the pixel-to-pixel sensitivity variations
    as well as the fringing pattern in the red orders. This is done in 1D, ie for the already extracted spectrum.
    
    INPUT:
    'f_flat'        : 1-dim array containing the extracted flux from the flat field (master white) for one order
    'filt'          : method of filtering ('gaussian' / 'savgol' / 'median') - WARNING: ONLY GAUSSIAN FILTER HAS BEEN IMPLEMENTED SO FAR!!!
    'filter_width'  : the width of the kernel for the filtering in pixels; defined differently for the different types of filters (see description of scipy.ndimage....)
    
    OUTPUT:
    'pix_sens'      : dictionary of the pixel-to-pixel sensitivities (keys = orders)
    'smoothed_flat' : dictionary of the smoothed (ie filtered) whites (keys = orders)
    
    MODHIST:
    05/10/2018 - CMB create   (clone of "onedim_pixtopix_variations")
    """
    
    while filt.lower() not in ['g','gaussian','s','savgol','m','median']:
        print("ERROR: filter choice not recognised!")
        filt = raw_input("Please try again: ['(G)aussian','(S)avgol','(M)edian']")
    
    if filt.lower() in ['g','gaussian']:
        #Gaussian filter
        smoothed_flat = ndimage.gaussian_filter(f_flat, filter_width)    
        pix_sens = f_flat / smoothed_flat
    elif filt.lower() in ['s','savgol']:
        print('WARNING: SavGol filter not implemented yet!!!')
        return
    elif filt.lower() in ['m','median']:
        print('WARNING: Median filter not implemented yet!!!')
        return
    else:
        #This should never happen!!!
        print("ERROR: filter choice still not recognised!")
        return
        
    return smoothed_flat, pix_sens    
    
    
    
    
    
def deblaze_orders(f, flat, err=None, mask=None, wl=None, degpol=1, gauss_filter_sigma=3., maxfilter_size=100,
                   combine_fibres=False, skip_first_order=False, debug_level=0):
    
    assert f.shape == flat.shape, 'Shapes of "flux" and "flat" do not agree!!!'
# wl has shape (40,4112), whereas flux has (39, 4112)
#     assert f.shape == wl.shape, 'Shapes of "flux" and "wl" do not agree!!!'
    if wl is not None:
        assert wl.__class__ == flat.__class__, '"wl" and "flat" are not the same class object!!!' 
        assert f.__class__ == wl.__class__, '"flux" and "wl" are not the same class object!!!' 
    if err is not None:
        assert f.__class__ == err.__class__, '"flux" and "err" are not the same class object!!!'
        if f.__class__ == np.ndarray:
            assert f.shape == err.shape, 'Shapes of "flux" and "error" do not agree!!!' 
        
    # make dummy mask of all ones if none is provided
    if mask is None:
        mask = {}
        for o in range(f.shape[0]):
            mask['order_'+str(o+1)] = np.ones(f.shape[-1])
    
    # if everything comes as numpy arrays
    if flat.__class__ == np.ndarray:
        f_dblz = np.zeros(f.shape)
        if err is not None:
            err_dblz = np.zeros(err.shape)
    
        # if using cross-correlation to get RVs, we really should de-blaze the spectra first...
        # loop over all orders
        for o in range(f.shape[0]):
            
            if (not skip_first_order) or (o > 0):
            
                # make sure that they are either quick-extracted spectra (order, pixel), or optimal-extracted spectra (order, fibre, pixel)
                assert len(f.shape) in [2,3], 'ERROR: shape of flux-array not recognized!!!'
                
                ord = 'order_'+str(o+1).zfill(2)
                
                if debug_level >= 1:
                    print('o = ' + str(o) + ' / ' + ord)
                
                # are they optimal 3a extracted spectra?
                if len(f.shape) == 3:
                    # loop over all fibres
                    for fib in range(f.shape[1]): 
                        
                        if debug_level >= 2:
                                print('fib = ' + str(fib))
                        
                        # first, divide by the "blaze-function", ie the flat, which we got from filtering the MASTER WHITE
                        if not combine_fibres:
                            f_dblz[o,fib,:] = f[o,fib,:] / (flat[o,fib,:] / np.nanmax(flat[o,fib,:]))
                            # we don't need the mask here
                            # f_dblz[o,fib,:][mask[ord]] = f[o,fib,:][mask[ord]] / (flat[o,fib,:][mask[ord]] / np.nanmax(flat[o,fib,:][mask[ord]]))
                        else:
                            f_dblz[o,fib,:] = f[o,fib,:] / (np.nansum(flat[o,:,:],axis=0) / np.nanmax(np.nansum(flat[o,:,:],axis=0)))
                            # we don't need the mask here
                            # f_dblz[o,fib,:][mask[ord]] = f[o,fib,:][mask[ord]] / (np.nansum(flat[o,:,:],axis=0)[mask[ord]] / np.nanmax(np.nansum(flat[o,:,:],axis=0)[mask[ord]]))
                        # get rough continuum shape by performing a series of filters
                        cont_rough = ndimage.maximum_filter(ndimage.gaussian_filter(f_dblz[o,fib,:], gauss_filter_sigma), size=maxfilter_size)
                        # now fit polynomial to that rough continuum and divide by that polynomial
                        # if wl is provided do that fit in wl space, otherwise just in pixel space)
                        if wl is not None:
                            p = np.poly1d(np.polyfit(wl[o,fib,:][mask[ord]], cont_rough[mask[ord]], degpol))
                            f_dblz[o,fib,:] = f_dblz[o,fib,:] / (p(wl[o,fib,:]) / np.nanmedian(p(wl[o,fib,:])[mask[ord]]))
                        else:
                            x = np.arange(f.shape[-1])[mask[ord]]
                            y = cont_rough[mask[ord]]
                            goodix = ~np.isnan(x) & ~np.isnan(y)
                            p = np.poly1d(np.polyfit(x[goodix], y[goodix], degpol)) 
                            f_dblz[o,fib,:] = f_dblz[o,fib,:] / (p(np.arange(f.shape[-1])) / np.nanmedian(p(np.arange(f.shape[-1])[mask[ord]])))
                        # need to treat the error arrays in the same way, as need to keep relative error the same
                        if err is not None:
                            if not combine_fibres:
                                err_dblz[o,fib,:] = err[o,fib,:] / (flat[o,fib,:]/np.nanmax(flat[o,fib,:]))
                                # we don't need the mask here
                                # err_dblz[o,fib,:][mask[ord]] = err[o,fib,:][mask[ord]] / (flat[o,fib,:][mask[ord]] / np.nanmax(flat[o,fib,:][mask[ord]]))
                            else:
                                err_dblz[o,fib,:] = err[o,fib,:] / (np.nansum(flat[o,:,:],axis=0) / np.nanmax(np.nansum(flat[o,:,:],axis=0)))
                                # we don't need the mask here
                                # err_dblz[o,fib,:][mask[ord]] = err[o,fib,:][mask[ord]] / (np.nansum(flat[o,:,:],axis=0)[mask[ord]] / np.nanmax(np.nansum(flat[o,:,:],axis=0)[mask[ord]]))
                            if wl is not None:    
                                err_dblz[o,fib,:] = err_dblz[o,fib,:] / (p(wl[o,fib,:]) / np.nanmedian(p(wl[o,fib,:])[mask[ord]]))
                            else:
                                err_dblz[o,fib,:] = err_dblz[o,fib,:] / (p(np.arange(f.shape[-1])) / np.nanmedian(p(np.arange(f.shape[-1])[mask[ord]])))
                # or are they just quick-extracted spectra
                else:
                    # first, divide by the "blaze-function", ie the flat, which we got from filtering the MASTER WHITE
                    f_dblz[o,:] = f[o,:] / (flat[o,:] / np.nanmax(flat[o,:]))
                    # get rough continuum shape by performing a series of filters
                    cont_rough = ndimage.maximum_filter(ndimage.gaussian_filter(f_dblz[o,:], gauss_filter_sigma), size=maxfilter_size)
                    # now fit polynomial to that rough continuum
                    # then divide by that polynomial
                    if wl is not None:
                        p = np.poly1d(np.polyfit(wl[o,:][mask[ord]], cont_rough[mask[ord]], degpol))
                        f_dblz[o,:] = f_dblz[o,:] / (p(wl[o,:]) / np.nanmedian(p(wl[o,:])[mask[ord]]))
                    else:
                        p = np.poly1d(np.polyfit(np.arange(f.shape[-1])[mask[ord]], cont_rough[mask[ord]], degpol)) 
                        f_dblz[o,:] = f_dblz[o,:] / (p(np.arange(f.shape[-1])) / np.nanmedian(p(np.arange(f.shape[-1])[mask[ord]])))
                    # need to treat the error arrays in the same way, as need to keep relative error the same
                    if err is not None:
                        err_dblz[o,:] = err[o,:] / (flat[o,:]/np.nanmax(flat[o,:]))
                        if wl is not None:
                            err_dblz[o,:] = err_dblz[o,:] / (p(wl[o,:]) / np.nanmedian(p(wl[o,:])[mask[ord]]))
                        else:
                            err_dblz[o,:] = err_dblz[o,:] / (p(np.arange(f.shape[-1])) / np.nanmedian(p(np.arange(f.shape[-1])[mask[ord]])))
                    
    # if everything comes as dictionaries
    elif flat.__class__ == dict:
        print('WARNING: THIS VERSION IS NOT UP TO DATE FOR DICTIONARY FORMATS...')
        f_dblz = {}
        if err is not None:
            err_dblz = {}
        
        # if using cross-correlation to get RVs, we need to de-blaze the spectra first
        for o in f.keys():
            # first, divide by the "blaze-function", ie the flat, which we got from filtering the MASTER WHITE
            f_dblz[o] = f[o] / (flat[o]/np.max(flat[o]))
            # get rough continuum shape by performing a series of filters
            cont_rough = ndimage.maximum_filter(ndimage.gaussian_filter(f_dblz[o],gauss_filter_sigma), size=maxfilter_size)
            # now fit polynomial to that rough continuum
            p = np.poly1d(np.polyfit(wl[o][mask[o]], cont_rough[mask[o]], degpol))
            # divide by that polynomial
            f_dblz[o] = f_dblz[o] / (p(wl[o]) / np.median(p(wl[o])[mask[o]]))
            # need to treat the error arrays in the same way, as need to keep relative error the same
            if err is not None:
                err_dblz[o] = err[o] / (flat[o]/np.max(flat[o]))
                err_dblz[o] = err_dblz[o] / (p(wl[o]) / np.median(p(wl[o])[mask[o]]))
    
    else:
        print('ERROR: data type / variable class not recognized')
        return

    if err is not None:
        return f_dblz, err_dblz
    else:
        return f_dblz




