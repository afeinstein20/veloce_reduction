'''
Created on 5 Feb. 2018

@author: christoph
'''

import numpy as np
import time
import scipy.sparse as sparse
from scipy.ndimage import label
import astropy.io.fits as pyfits

from .helper_functions import polyfit2d, polyval2d, fit_poly_surface_2D, sigma_clip

__all__ = ['make_dummy_background', 'remove_background', 'remove_background_pid', 'extract_background', 'extract_background_pid',
           'fit_background', 'fit_background_astropy']

def make_dummy_background(nx=4112, ny=4096):
    # make simulated background
    x = np.repeat(np.arange(nx) - nx/2,nx)
    y = np.tile(np.arange(ny) - ny/2,ny)
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), np.linspace(y.min(), y.max(), ny))
    #m =    [a00,a01,a02,  a03  ,a10,a11,a12,a13,  a20 ,a21,a22,a23, a30 ,a31,a32,a33] for order=3
    parms = [90., 0., 0., 1.5e-9, 0., 0., 0., 0., -4e-9, 0., 0., 0., 1e-9, 0., 0., 0.]
    #parms = np.array([1000., 0., -5.e-5, 0., 0., 0., 0., 0., -5.e-5, 0., 0., 0., 0., 0., 0., 0.])
    zz_nf = polyval2d(xx, yy, parms)
    #add white noise
    noise = np.resize(np.random.normal(0, 1, nx*ny),(ny,nx))
    scaled_noise = noise * np.sqrt(zz_nf)
    zz = zz_nf + scaled_noise
    imgtest = img + zz
    return imgtest



def remove_background(img, bgmask, obsname, path, degpol=5, save_bg=True, savefile=True, save_err=False, timit=False):
    """
    Top-level wrapper function to identify, extract, fit, and subtract the background for a given image.
    
    INPUT:
    'img'                     : input image (2-dim numpy array)
    'bgmask'                  : 2D boolean mask identifying the background pixels (= chipmask['bg'])
    'obsname'                 : the obsname in "obsname.fits"
    'path'                    : the directory of the files
    'degpol'                  : degree (in each direction) of the 2-dim polynomial surface fit
    'slit_height'             : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'save_bg'                 : boolean - do you want to save the background image?
    'savefile'                : boolean - do you want to save the background-corrected science image?
    'save_err'                : boolean - do you want to save the corresponding error array as well? (remains unchanged though)
    'timit'                   : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'corrected_image'         : the background-corrected science image
    """
    
    if timit:
        start_time = time.time()
    
    #(1) identify and extract background
    bg = extract_background(img, bgmask, timit=timit)
    #(2) fit background
    bg_coeffs, bg_img = fit_background(bg, deg=degpol, return_full=True, timit=timit)
    #(3) subtract background
    corrected_image = img - bg_img
    #what about errors??????
    
    #save background image
    if save_bg:
        outfn = path+obsname+'_BG_img.fits'
        #get header from the BIAS- & DARK-subtracted & cosmic-ray corrected image if it exists
        try:
            h = pyfits.getheader(path+obsname+'_BD_CR.fits')
        except:
            #otherwise try to get header from the BIAS- & DARK-subtracted image; otherwise from the original image FITS file
            try: 
                h = pyfits.getheader(path+obsname+'_BD.fits')
            except:
                h = pyfits.getheader(path+obsname+'.fits')
                h['UNITS'] = 'ELECTRONS'
        h['HISTORY'] = '   BACKGROUND image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.writeto(outfn, np.float32(bg_img), h, overwrite=True)
        
    #save background-corrected image
    if savefile:
        outfn = path+obsname+'_BD_CR_BG.fits'
        #get header from the BIAS- & DARK-subtracted & cosmic-ray corrected image if it exists
        try:
            h = pyfits.getheader(path+obsname+'_BD_CR.fits')
        except:
            #otherwise try to get header from the BIAS- & DARK-subtracted image; otherwise from the original image FITS file
            try: 
                h = pyfits.getheader(path+obsname+'_BD.fits')
            except:
                h = pyfits.getheader(path+obsname+'.fits')
                h['UNITS'] = 'ELECTRONS'
        h['HISTORY'] = '   BACKGROUND-corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.writeto(outfn, np.float32(corrected_image), h, overwrite=True)
        #also save the error array if desired
        if save_err:
            try:
                err = pyfits.getdata(path+obsname+'_BD_CR.fits', 1)
                h_err = h.copy()
                h_err['HISTORY'] = 'estimated uncertainty in BACKGROUND-corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
                pyfits.append(outfn, np.float32(err), h_err, overwrite=True)
            except:
                try:
                    err = pyfits.getdata(path+obsname+'_BD.fits', 1)
                    h_err = h.copy()
                    h_err['HISTORY'] = 'estimated uncertainty in BACKGROUND-corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
                    pyfits.append(outfn, np.float32(err), h_err, overwrite=True)
                except:
                    print('WARNING: error array not found - cannot save error array')
    
    
    if timit:
        print('Total time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')
    
    return corrected_image



def remove_background_pid(img, P_id, obsname, path, degpol=5, slit_height=25, save_bg=True, savefile=True, save_err=False, exclude_top_and_bottom=False, timit=False):
    """
    Top-level wrapper function to identify, extract, fit, and subtract the background for a given image.
    
    INPUT:
    'img'                     : input image (2-dim numpy array)
    'P_id'                    : dictionary of the form of {order: np.poly1d, ...} (as returned by identify_stripes)
    'obsname'                 : the obsname in "obsname.fits"
    'path'                    : the directory of the files
    'degpol'                  : degree (in each direction) of the 2-dim polynomial surface fit
    'slit_height'             : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'save_bg'                 : boolean - do you want to save the background image?
    'savefile'                : boolean - do you want to save the background-corrected science image?
    'save_err'                : boolean - do you want to save the corresponding error array as well? (remains unchanged though)
    'exclude_top_and_bottom'  : boolean - do you want to exclude the areas at top and bottom of chip, ie outside the useful orders but still containing some incomplete orders?
    'timit'                   : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'corrected_image'         : the background-corrected science image
    """
    
    if timit:
        start_time = time.time()
    
    #(1) identify and extract background
    bg = extract_background_pid(img, P_id, slit_height=slit_height, exclude_top_and_bottom=exclude_top_and_bottom, timit=timit)
    #(2) fit background
    bg_coeffs,bg_img = fit_background(bg, deg=degpol, return_full=True, timit=timit)
    #(3) subtract background
    corrected_image = img - bg_img
    #what about errors??????
    
    #save background image
    if save_bg:
        outfn = path+obsname+'_BG_img.fits'
        #get header from the BIAS- & DARK-subtracted & cosmic-ray corrected image if it exists
        try:
            h = pyfits.getheader(path+obsname+'_BD_CR.fits')
        except:
            #otherwise try to get header from the BIAS- & DARK-subtracted image; otherwise from the original image FITS file
            try: 
                h = pyfits.getheader(path+obsname+'_BD.fits')
            except:
                h = pyfits.getheader(path+obsname+'.fits')
                h['UNITS'] = 'ELECTRONS'
        h['HISTORY'] = '   BACKGROUND image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.writeto(outfn, np.float32(bg_img), h, overwrite=True)
        
    #save background-corrected image
    if savefile:
        outfn = path+obsname+'_BD_CR_BG.fits'
        #get header from the BIAS- & DARK-subtracted & cosmic-ray corrected image if it exists
        try:
            h = pyfits.getheader(path+obsname+'_BD_CR.fits')
        except:
            #otherwise try to get header from the BIAS- & DARK-subtracted image; otherwise from the original image FITS file
            try: 
                h = pyfits.getheader(path+obsname+'_BD.fits')
            except:
                h = pyfits.getheader(path+obsname+'.fits')
                h['UNITS'] = 'ELECTRONS'
        h['HISTORY'] = '   BACKGROUND-corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.writeto(outfn, np.float32(corrected_image), h, overwrite=True)
        #also save the error array if desired
        if save_err:
            try:
                err = pyfits.getdata(path+obsname+'_BD_CR.fits', 1)
                h_err = h.copy()
                h_err['HISTORY'] = 'estimated uncertainty in BACKGROUND-corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
                pyfits.append(outfn, np.float32(err), h_err, overwrite=True)
            except:
                try:
                    err = pyfits.getdata(path+obsname+'_BD.fits', 1)
                    h_err = h.copy()
                    h_err['HISTORY'] = 'estimated uncertainty in BACKGROUND-corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
                    pyfits.append(outfn, np.float32(err), h_err, overwrite=True)
                except:
                    print('WARNING: error array not found - cannot save error array')
    
    
    if timit:
        print('Total time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')
    
    return corrected_image





def extract_background(img, bgmask, timit=False):
    """
    Extracts the background (ie the inter-order regions = everything outside the order stripes - as identified by a boolean mask)
    from the original 2D spectrum to a sparse matrix containing only relevant pixels.
    
    INPUT:
    'img'                     : 2D echelle spectrum [np.array]
    'bgmask'                  : 2D boolean mask identifying the background pixels (= chipmask['bg'])
    'timit'                   : for timing tests...
    
    OUTPUT:
    'mat.tocsc()'  : scipy.sparse_matrix containing the locations and values of the inter-order regions    
    """
    
    if timit:
        start_time = time.time()
    
#     logging.info('Extracting background...')
    print('Extracting background...')
    
    ny, nx = img.shape
    xx = np.arange(nx, dtype='f8')
    yy = np.arange(ny, dtype='f8')
    x_grid, y_grid = np.meshgrid(xx, yy, copy=False)
    
    mat = sparse.coo_matrix((np.squeeze(np.array(img[bgmask])), (y_grid[bgmask], x_grid[bgmask])), shape=img.shape)
    
    if timit:
        print('Elapsed time: ',time.time() - start_time,' seconds')
    
    return mat.tocsc()



def extract_background_pid(img, P_id, slit_height=25, return_mask=False, exclude_top_and_bottom=True, timit=False):
    """
    This function marks all relevant pixels for extraction. Extracts the background (ie the inter-order regions = everything outside the order stripes)
    from the original 2D spectrum to a sparse matrix containing only relevant pixels.
    
    INPUT:
    'img'                     : 2D echelle spectrum [np.array]
    'P_id'                    : dictionary of the form of {order: np.poly1d} (as returned by make_P_id / identify_stripes)
    'slit_height'             : half the total slit height in pixels
    'return_mask'             : boolean - do you want to return the mask of the background locations as well?
    'exclude_top_and_bottom'  : boolean - do you want to exclude the top and bottom bits (where there are usually incomplete orders)
    'timit'                   : for timing tests...
    
    OUTPUT:
    'mat.tocsc()'  : scipy.sparse_matrix containing the locations and values of the inter-order regions    
    'bg_mask'      : boolean array containing the location of what is considered background (ie the inter-order space)
    """
    
    if timit:
        start_time = time.time()
    
#     logging.info('Extracting background...')
    print('Extracting background...')

    ny, nx = img.shape
    xx = np.arange(nx, dtype='f8')
    yy = np.arange(ny, dtype='f8')
    x_grid, y_grid = np.meshgrid(xx, yy, copy=False)
    bg_mask = np.ones((ny, nx), dtype=bool)

    for o, p in sorted(P_id.items()):
        
        # order trace
        y = np.poly1d(p)(xx)
        
        # distance from order trace
        distance = y_grid - y.repeat(ny).reshape((nx, ny)).T
        indices = abs(distance) > slit_height
        
        # include in global mask
        bg_mask *= indices
        
    
    final_bg_mask = bg_mask.copy()
    # in case we want to exclude the top and bottom parts where incomplete orders are located
    if exclude_top_and_bottom:
        print('WARNING: this fix works for the current Veloce CCD layout only!!!')
        labelled_mask,nobj = label(bg_mask)
        # WARNING: this fix works for the current Veloce CCD layout only!!!
        topleftnumber = labelled_mask[ny-1,0]
        toprightnumber = labelled_mask[ny-1,nx-1]
#         bottomleftnumber = labelled_mask[0,0]
        bottomrightnumber = labelled_mask[0,nx-1]
        final_bg_mask[labelled_mask == topleftnumber] = False
        final_bg_mask[labelled_mask == toprightnumber] = False
        final_bg_mask[labelled_mask == bottomrightnumber] = False
    
    
    mat = sparse.coo_matrix((np.squeeze(np.array(img[final_bg_mask])), (y_grid[final_bg_mask], x_grid[final_bg_mask])), shape=(ny, nx))
#     return mat.tocsr()
    
    if timit:
        print('Elapsed time: ',time.time() - start_time,' seconds')
    
    if not return_mask:
        return mat.tocsc()
    else:
        return mat.tocsc(), final_bg_mask





def fit_background(bg, deg=5, clip=10, return_full=True, timit=False):
    """ 
    INPUT:
    'bg'                      : sparse matrix containing the inter-order regions of the 2D image
    'deg'                     : the order of the polynomials to use in the fit (for both dimensions)
    'clip'                    : threshold for sigma clipping (needed to get rid of hot pixels etc)
    'return_full'             : boolean - if TRUE, then the full image of the background model is returned; otherwise just the set of coefficients that describe it
    'timit'                   : time it...
    
    OUTPUT:
    'coeffs'    : polynomial coefficients that describe the background model
    'bkgd_img'  : full background image constructed from best-fit model (only if 'return_full' is set to TRUE)
    
    TODO:
    figure out how to properly use weights here
    """
    
    if timit:
        start_time = time.time()
        
    # find the non-zero parts of the sparse matrix
    # format is:
    # contents[0] = row indices, ie y-values
    # contents[1] = column indices, ie x-values
    # contents[2] = values
    contents = sparse.find(bg)
    
    ny, nx = bg.todense().shape
    
    # perform sigma-clipping to get rid of hot pixels etc
    z_clean, goodix, badix = sigma_clip(contents[2], clip, return_indices=True)
    
    # re-normalize to [-1,+1] - otherwise small errors in parms have huge effects
#     x_norm = (contents[0] / ((nx-1)/2.)) - 1.
#     y_norm = (contents[1] / ((ny-1)/2.)) - 1.
#     z = contents[2]
    y_norm = (contents[0][goodix] / ((nx-1)/2.)) - 1.
    x_norm = (contents[1][goodix] / ((ny-1)/2.)) - 1.
#     z_clean_2 = contents[2][goodix]   # I double-checked and it really is the same as output from sigma_clip!!!
    
    
#     m = polyfit2d(contents[0]-int(ny/2), contents[1]-int(nx/2), contents[2], order=deg)
    coeffs = polyfit2d(x_norm, y_norm, z_clean, order=deg)
    # The result (m) is an array of the polynomial coefficients in the model f  = sum_i sum_j a_ij x^i y^j, 
    # eg:    m = [a00,a01,a02,a03,a10,a11,a12,a13,a20,.....,a33] for order=3
    
    
    if timit:
        print('Time taken for fitting background model: ' + np.round(time.time() - start_time, 2).astype(str) + ' seconds...')
        start_time_2 = time.time()
    
    if return_full:
        xx = np.arange(nx)          
        xxn = (xx / ((nx-1)/2.)) - 1.
        yy = np.arange(ny)
        yyn = (yy / ((ny-1)/2.)) - 1.
        X,Y = np.meshgrid(xxn,yyn)
        bkgd_img = polyval2d(X,Y,coeffs)
#         bkgd_img = polyval2d(Y,X,coeffs)    # IDKY, but the indices are the wrong way around if I do it like in the line above!!!!! 
#         Ha!!! I know why! I had x_norm and y_norm mixed up above...
    
    if timit:
        print('Time taken for constructing full background image: ' + np.round(time.time() - start_time_2, 2).astype(str) + ' seconds...')
        print('Total time elapsed: ' + np.round(time.time() - start_time, 2).astype(str) + ' seconds...')
    
    if return_full:
        return coeffs, bkgd_img
    else:
        return coeffs
 
 
 
 
 
def fit_background_astropy(bg, poly_deg=5, clip=10, polytype='chebyshev', return_full=True, timit=False):
    """ 
    WARNING: While this works just fine, it is MUCH MUCH slower than 'fit_background' above. The astropy fitting/modelling must be to blame...
    
    INPUT:
    'bg'               : sparse matrix containing the inter-order regions of the 2D image
    'poly_deg'         : the order of the polynomials to use in the fit (for both dimensions)
    'clip'             : threshold for sigma clipping (needed to get rid of hot pixels etc)
    'polytype'         : either 'polynomial' (default), 'legendre', or 'chebyshev' are accepted
    'return_full'      : boolean - if TRUE, then the background model for each pixel for each order is returned; otherwise just the set of coefficients that describe it
    'timit'            : time it...
    'debug_level'      : for debugging only
    
    OUTPUT:
    EITHER
    'bkgd_coeffs'      : functional form of the coefficients that describe the background model
    OR 
    'bkgd_img'         : full background image constructed from best-fit model
    (selection between outputs is controlled by the 'return_full' keyword)
    
    TODO:
    figure out how to properly use weights here
    """
    
    if timit:
        start_time = time.time()
    
    # find the non-zero parts of the sparse matrix
    # format is:
    # contents[0] = row indices, ie y-values
    # contents[1] = column indices, ie x-values
    # contents[2] = values
    contents = sparse.find(bg)
    
    ny, nx = bg.todense().shape
    
    # perform sigma-clipping to get rid of hot pixels etc
    z_clean, goodix, badix = sigma_clip(contents[2], clip, return_indices=True)
    
    # re-normalize to [-1,+1] - otherwise small errors in parms have huge effects
#     x_norm = (contents[0] / ((nx-1)/2.)) - 1.
#     y_norm = (contents[1] / ((ny-1)/2.)) - 1.
#     z = contents[2]
    y_norm = (contents[0][goodix] / ((nx-1)/2.)) - 1.
    x_norm = (contents[1][goodix] / ((ny-1)/2.)) - 1.
#     z_clean_2 = contents[2][goodix]   # I double-checked and it really is the same as output from sigma_clip!!!
           
    # call the surface fitting routine
    bkgd_coeffs = fit_poly_surface_2D(x_norm, y_norm, z_clean, weights=None, polytype = polytype, poly_deg=poly_deg)    
#     cheb_coeffs = fit_poly_surface_2D(x_norm, y_norm, z, weights=None, polytype = 'c', poly_deg=3, timit=True)  
#     lege_coeffs = fit_poly_surface_2D(x_norm, y_norm, z, weights=None, polytype = 'l', poly_deg=3, timit=True)  
#     poly_coeffs = fit_poly_surface_2D(x_norm, y_norm, z, weights=None, polytype = 'p', poly_deg=3, timit=True)

    if return_full:
        xx = np.arange(nx)          
        xxn = (xx / ((nx-1)/2.)) - 1.
        yy = np.arange(ny)
        yyn = (yy / ((ny-1)/2.)) - 1.
        X,Y = np.meshgrid(xxn,yyn)
        bkgd_img = bkgd_coeffs(X,Y)
#         bkgd_img = bkgd_coeffs(Y,X)     #IDKY, but the indices are the wrong way around if I do it like in the line above!!!!!
#         Ha!!! I know why! I had x_norm and y_norm mixed up above...

    if timit:
        print('Time elapsed: '+np.round(time.time() - start_time,2).astype(str)+' seconds...')    

    if return_full:
        return bkgd_coeffs,bkgd_img
    else:
        return bkgd_coeffs


