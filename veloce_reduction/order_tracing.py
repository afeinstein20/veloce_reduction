'''
Created on 4 Sep. 2017

@author: christoph
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy.sparse as sparse
import time, sys

from .helper_functions import sigma_clip


__all__ =  ['find_stripes', 'find_gaps', 'make_P_id_old', 'make_P_id', 'make_mask_dict', 
            'make_order_traces_from_fibparms', 'extract_single_stripe', 'extract_stripes', 
            'flatten_single_stripe', 'flatten_single_stripe_from_indices', 'flatten_stripes', 
            'find_tramlines_single_order', 'find_tramlines', 'find_laser_tramlines_single_order', 'find_laser_tramlines']



def find_stripes(flat, deg_polynomial=2, gauss_filter_sigma=3., min_peak=0.05, maskthresh=100., weighted_fits=True, slowmask=False, simu=False, timit=False, debug_level=0):
    """
    BASED ON JULIAN STUERMER'S MAROON_X PIPELINE:
    
    Locates and fits stripes (ie orders) in a flat field echelle spectrum.
    
    Starting in the central column, the algorithm identifies peaks and traces each stripe to the edge of the detector
    by following the brightest pixels along each order. It then fits a polynomial to each stripe.
    To improve algorithm stability, the image is first smoothed with a Gaussian filter. It not only eliminates noise, but
    also ensures that the cross-section profile of the flat becomes peaked in the middle, which helps to identify the
    center of each stripe. Choose gauss_filter accordingly.
    To avoid false positives, only peaks above a certain (relative) intensity threshold are used.
      
    :param flat: dark-corrected flat field spectrum
    :type flat: np.ndarray
    :param deg_polynomial: degree of the polynomial fit
    :type deg_polynomial: int
    :param gauss_filter_sigma: sigma of the gaussian filter used to smooth the image.
    :type gauss_filter_sigma: float
    :param min_peak: minimum relative peak height 
    :type min_peak: float
    :param debug_level: debug level flag
    :type debug_level: int
    :return: list of polynomial fits (np.poly1d)
    :rtype: list
    """
    
    if timit:
        start_time = time.time()
    
    #logging.info('Finding stripes...')
    print("Finding stripes...")
    ny, nx = flat.shape

    # smooth image slightly for noise reduction
    filtered_flat = ndimage.gaussian_filter(flat.astype(np.float), gauss_filter_sigma)
    
    # find peaks in center column
    data = filtered_flat[:, int(nx / 2)]
    peaks = np.r_[True, data[1:] > data[:-1]] & np.r_[data[:-1] > data[1:], True]
    #troughs = np.r_[True, data[1:] < data[:-1]] & np.r_[data[:-1] < data[1:], True]

    if debug_level > 1:
        plt.figure()
        plt.title('Local maxima')
        plt.plot(data)
        plt.scatter(np.arange(ny)[peaks], data[peaks],s=25)
        plt.show()

    idx = np.logical_and(peaks, data > min_peak * np.max(data))
    maxima = np.arange(ny)[idx]

    # filter out maxima too close to the boundary to avoid problems
    maxima = maxima[maxima > 3]
    maxima = maxima[maxima < ny - 3]

    if debug_level > 1:
        #labels, n_labels = ndimage.measurements.label(data > min_peak * np.max(data))
        plt.figure()
        plt.title('Order peaks')
        plt.plot(data)
        plt.scatter(np.arange(ny)[maxima], data[maxima],s=25, c='red')
        #plt.plot((labels[0] > 0) * np.max(data))   #what does this do???
        plt.show()

    n_order = len(maxima)
    #logging.info('Number of stripes found: %d' % n_order)
    print('Number of stripes found: %d' % n_order)

    orders = np.zeros((n_order, nx))
    # because we only want to use good pixels in the fit later on
    mask = np.ones((n_order, nx), dtype=bool)

    # walk through to the left and right along the maximum of the order
    # loop over all orders:
    for m, row in enumerate(maxima):
        column = int(nx / 2)
        orders[m, column] = row
        start_row = row
        # walk right
        while (column + 1 < nx):
            column += 1
            args = np.array(np.linspace(max(1, start_row - 1), min(start_row + 1, ny - 1), 3), dtype=int)
            args = args[np.logical_and(args < ny, args > 0)]     #deal with potential edge effects
            p = filtered_flat[args, column]
            # new maximum (apply only when there are actually flux values in p, ie not when eg p=[0,0,0]), otherwise leave start_row unchanged
            if ~(p[0]==p[1] and p[0]==p[2]):
                start_row = args[np.argmax(p)]
            orders[m, column] = start_row
            #build mask - exclude pixels at upper/lower end of chip; also exclude peaks that do not lie at least 5 sigmas above rms of 3-sigma clipped background (+/- cliprange pixels from peak location)
            #if ((p < 10).all()) or ((column > 3500) and (mask[m,column-1]==False)) or (start_row in (0,nx-1)) or (m==42 and (p < 100).all()):
            if slowmask:
                cliprange = 25
                bg = filtered_flat[start_row-cliprange:start_row+cliprange+1, column]
                clipped = sigma_clip(bg,3.)
                if (filtered_flat[start_row,column] - np.median(clipped) < 5.*np.std(clipped)) or (start_row in (0,nx-1)):
                    mask[m,column] = False
            else:
                if ((p < maskthresh).all()) or (start_row in (0,ny-1)):
                    mask[m,column] = False
        # walk left
        column = int(nx / 2)
        start_row = row
        while (column > 0):
            column -= 1
            args = np.array(np.linspace(max(1, start_row - 1), min(start_row + 1, ny - 1), 3), dtype=int)
            args = args[np.logical_and(args < ny, args > 0)]     #deal with potential edge effects
            p = filtered_flat[args, column]
            # new maximum (apply only when there are actually flux values in p, ie not when eg p=[0,0,0]), otherwise leave start_row unchanged
            if ~(p[0]==p[1] and p[0]==p[2]):
                start_row = args[np.argmax(p)]
            orders[m, column] = start_row
            #build mask - exclude pixels at upper/lower end of chip; also exclude peaks that do not lie at least 5 sigmas above rms of 3-sigma clipped bcakground (+/- cliprange pixels from peak location)
            #if ((p < 10).all()) or ((column < 500) and (mask[m,column+1]==False)) or (start_row in (0,nx-1)) or (m==42 and (p < 100).all()):
            if slowmask:
                cliprange = 25
                bg = filtered_flat[start_row-cliprange:start_row+cliprange+1, column]
                clipped = sigma_clip(bg,3.)
                if (filtered_flat[start_row,column] - np.median(clipped) < 5.*np.std(clipped)) or (start_row in (0,nx-1)):
                    mask[m,column] = False
            else:
                if ((p < maskthresh).all()) or (start_row in (0,ny-1)) or (simu==True and m==0 and column < 1300) or (simu==False and m==0 and column < 900):
                    mask[m,column] = False
    # do Polynomial fit for each order
    #logging.info('Fit polynomial of order %d to each stripe' % deg_polynomial)
    print('Fit polynomial of order %d to each stripe...' % deg_polynomial)
    P = []
    xx = np.arange(nx)
    for i in range(len(orders)):
        if not weighted_fits:
            #unweighted
            p = np.poly1d(np.polyfit(xx[mask[i,:]], orders[i,mask[i,:]], deg_polynomial))
        else:
            #weighted
            filtered_flux_along_order = np.zeros(nx)
            for j in range(nx):
                #filtered_flux_along_order[j] = filtered_flat[o[j].astype(int),j]    #that was when the loop reas: "for o in orders:"
                filtered_flux_along_order[j] = filtered_flat[orders[i,j].astype(int),j]
            filtered_flux_along_order[filtered_flux_along_order < 1] = 1   
            #w = 1. / np.sqrt(filtered_flux_along_order)   this would weight the order centres less!!!
            w = np.sqrt(filtered_flux_along_order)
            p = np.poly1d(np.polyfit(xx[mask[i,:]], orders[i,mask[i,:]], deg_polynomial, w=w[mask[i,:]]))
        P.append(p)

    if debug_level > 0:
        plt.figure()
        plt.imshow(filtered_flat, interpolation='none', vmin=np.min(flat), vmax=0.9 * np.max(flat), cmap=plt.get_cmap('gray'))
        for p in P:
            plt.plot(xx, p(xx), 'g', alpha=1)
        plt.ylim((0, ny))
        plt.xlim((0, nx))
        plt.show()    
        
    if timit:
        print('Elapsed time: '+str(time.time() - start_time)+' seconds')

    return P,mask



def find_gaps(flat, deg_polynomial=2, gauss_filter_sigma=3., min_peak=0.05, maskthresh=100., weighted_fits=True, slowmask=False, simu=False, timit=False, debug_level=0):
    """
    BASED ON JULIAN STUERMER'S MAROON_X PIPELINE:
    
    Locates and fits stripes (ie orders) in a flat field echelle spectrum.
    
    Starting in the central column, the algorithm identifies peaks and traces each stripe to the edge of the detector
    by following the brightest pixels along each order. It then fits a polynomial to each stripe.
    To improve algorithm stability, the image is first smoothed with a Gaussian filter. It not only eliminates noise, but
    also ensures that the cross-section profile of the flat becomes peaked in the middle, which helps to identify the
    center of each stripe. Choose gauss_filter accordingly.
    To avoid false positives, only peaks above a certain (relative) intensity threshold are used.
      
    :param flat: dark-corrected flat field spectrum
    :type flat: np.ndarray
    :param deg_polynomial: degree of the polynomial fit
    :type deg_polynomial: int
    :param gauss_filter_sigma: sigma of the gaussian filter used to smooth the image.
    :type gauss_filter_sigma: float
    :param min_peak: minimum relative peak height 
    :type min_peak: float
    :param debug_level: debug level flag
    :type debug_level: int
    :return: list of polynomial fits (np.poly1d)
    :rtype: list
    """
    
    if timit:
        start_time = time.time()
    
    #logging.info('Finding stripes...')
    print("Finding stripes...")
    ny, nx = flat.shape

    # smooth image slightly for noise reduction
    filtered_flat = ndimage.gaussian_filter(flat.astype(np.float), gauss_filter_sigma)
    
    # find peaks in center column
    data = filtered_flat[:, int(nx / 2)]
    #peaks = np.r_[True, data[1:] > data[:-1]] & np.r_[data[:-1] > data[1:], True]
    troughs = np.r_[True, data[1:] < data[:-1]] & np.r_[data[:-1] < data[1:], True]

    if debug_level > 1:
        plt.figure()
        plt.title('Local maxima')
        plt.plot(data)
        plt.scatter(np.arange(ny)[troughs], data[troughs],s=25)
        plt.show()

    # fiddly fix, so we only find the troughs between stellar and sky fibres
    idx = np.logical_and(np.logical_and(troughs, data > 50), data < 2000.)
    idx[:40] = False
    idx[4016:] = False
    
    minima = np.arange(ny)[idx]

    # filter out maxima too close to the boundary to avoid problems
    minima = minima[minima > 3]
    minima = minima[minima < ny - 3]

    if debug_level > 1:
        #labels, n_labels = ndimage.measurements.label(data > min_peak * np.max(data))
        plt.figure()
        plt.title('Order gaps (stellar / sky)')
        plt.plot(data)
        plt.scatter(np.arange(ny)[minima], data[minima],s=25, c='red')
        #plt.plot((labels[0] > 0) * np.max(data))   #what does this do???
        plt.show()

    n_gaps = len(minima)
    #logging.info('Number of stripes found: %d' % n_order)
    print('Number of gaps found: %d' % n_gaps)

    gaps = np.zeros((n_gaps, nx))
    # because we only want to use good pixels in the fit later on
    mask = np.ones((n_gaps, nx), dtype=bool)

    # walk through to the left and right along the maximum of the order
    # loop over all orders:
    for m, row in enumerate(minima):
        column = int(nx / 2)
        gaps[m, column] = row
        start_row = row
        # walk right
        while (column + 1 < nx):
            column += 1
            args = np.array(np.linspace(max(1, start_row - 1), min(start_row + 1, ny - 1), 3), dtype=int)
            args = args[np.logical_and(args < ny, args > 0)]     #deal with potential edge effects
            p = filtered_flat[args, column]
            # new maximum (apply only when there are actually flux values in p, ie not when eg p=[0,0,0]), otherwise leave start_row unchanged
            if ~(p[0]==p[1] and p[0]==p[2]):
                start_row = args[np.argmin(p)]
            gaps[m, column] = start_row
            #build mask - exclude pixels at upper/lower end of chip; also exclude peaks that do not lie at least 5 sigmas above rms of 3-sigma clipped background (+/- cliprange pixels from peak location)
            #if ((p < 10).all()) or ((column > 3500) and (mask[m,column-1]==False)) or (start_row in (0,nx-1)) or (m==42 and (p < 100).all()):
            if slowmask:
                cliprange = 25
                bg = filtered_flat[start_row-cliprange:start_row+cliprange+1, column]
                clipped = sigma_clip(bg,3.)
                if (filtered_flat[start_row,column] - np.median(clipped) < 5.*np.std(clipped)) or (start_row in (0,nx-1)):
                    mask[m,column] = False
            else:
                if ((p < maskthresh).all()) or (start_row in (0,ny-1)):
                    mask[m,column] = False
        # walk left
        column = int(nx / 2)
        start_row = row
        while (column > 0):
            column -= 1
            args = np.array(np.linspace(max(1, start_row - 1), min(start_row + 1, ny - 1), 3), dtype=int)
            args = args[np.logical_and(args < ny, args > 0)]     #deal with potential edge effects
            p = filtered_flat[args, column]
            # new maximum (apply only when there are actually flux values in p, ie not when eg p=[0,0,0]), otherwise leave start_row unchanged
            if ~(p[0]==p[1] and p[0]==p[2]):
                start_row = args[np.argmin(p)]
            gaps[m, column] = start_row
            #build mask - exclude pixels at upper/lower end of chip; also exclude peaks that do not lie at least 5 sigmas above rms of 3-sigma clipped bcakground (+/- cliprange pixels from peak location)
            #if ((p < 10).all()) or ((column < 500) and (mask[m,column+1]==False)) or (start_row in (0,nx-1)) or (m==42 and (p < 100).all()):
            if slowmask:
                cliprange = 25
                bg = filtered_flat[start_row-cliprange:start_row+cliprange+1, column]
                clipped = sigma_clip(bg,3.)
                if (filtered_flat[start_row,column] - np.median(clipped) < 5.*np.std(clipped)) or (start_row in (0,nx-1)):
                    mask[m,column] = False
            else:
                if ((p < maskthresh).all()) or (start_row in (0,ny-1)) or (simu==True and m==0 and column < 1300) or (simu==False and m==0 and column < 900):
                    mask[m,column] = False
    
    # do Polynomial fit for each order
    #logging.info('Fit polynomial of order %d to each stripe' % deg_polynomial)
    print('Fit polynomial of order %d to each stripe...' % deg_polynomial)
    P = []
    xx = np.arange(nx)
    for i in range(len(gaps)):
        if not weighted_fits:
            #unweighted
            p = np.poly1d(np.polyfit(xx[mask[i,:]], gaps[i,mask[i,:]], deg_polynomial))
        else:
            #weighted
            filtered_flux_along_order = np.zeros(nx)
            for j in range(nx):
                #filtered_flux_along_order[j] = filtered_flat[o[j].astype(int),j]    #that was when the loop reas: "for o in orders:"
                filtered_flux_along_order[j] = filtered_flat[gaps[i,j].astype(int),j]
            filtered_flux_along_order[filtered_flux_along_order < 1] = 1   
            #w = 1. / np.sqrt(filtered_flux_along_order)   this would weight the order centres less!!!
            w = np.sqrt(filtered_flux_along_order)
            p = np.poly1d(np.polyfit(xx[mask[i,:]], gaps[i,mask[i,:]], deg_polynomial, w=w[mask[i,:]]))
        P.append(p)

    if debug_level > 0:
        plt.figure()
        plt.imshow(filtered_flat, interpolation='none', vmin=np.min(flat), vmax=0.9 * np.max(flat), cmap=plt.get_cmap('gray'))
        for p in P:
            plt.plot(xx, p(xx), 'g', alpha=1)
        plt.ylim((0, ny))
        plt.xlim((0, nx))
        plt.show()    
        
    if timit:
        print('Elapsed time: '+str(time.time() - start_time)+' seconds')

    return P,mask



def make_P_id_old(P):
    """OLD ROUTINE, NOT CURRENTLY IN USE!!!"""
    Ptemp = {}
    ordernames = []
    for i in range(1,10):
        ordernames.append('order_0%i' % i)
    for i in range(10,len(P)+1):
        ordernames.append('order_%i' % i)
    #the array parms comes from the "find_stripes" function
    for i in range(len(P)):
        Ptemp.update({ordernames[i]:P[i]})
    P_id = {'fibre_01': Ptemp}
     
    return P_id



def make_P_id(P):
    '''This routine brings the results from "find_stripes" into the right format (python dictionary, orders=keys)!'''
    P_id = {}
    ordernames = []
    for i in range(len(P)):    
        ordernames.append('order_'+str(i+1).zfill(2))
        P_id.update({ordernames[i]:P[i]})
    
    return P_id



def make_mask_dict(tempmask):
    mask = {}
    ordernames = []
    for i in range(len(tempmask)):
        ordernames.append('order_'+str(i+1).zfill(2))
        mask.update({ordernames[i]:tempmask[i,:]})
    return mask



def make_order_traces_from_fibparms(fibparms, from_simcalibs=False, degpol=7):
    
    traces = {}
    traces['lfc'] = {}
    traces['simth'] = {}
    traces['stellar'] = {}
    traces['allfib'] = {}
    
    nx = len(fibparms['order_02']['fibre_02']['mu_fit'])
    xx = np.arange(nx)
    
    for ord in fibparms.keys():
        if from_simcalibs:
            allfib_trace = 0.5 * (fibparms[ord]['fibre_01']['mu_fit'] + fibparms[ord]['fibre_28']['mu_fit'])
        else:
            # I think the best way is to use the outermost sky fibres, as the LFC trace is often a bit dodgy, so that's the default
            allfib_trace = 0.5 * (fibparms[ord]['fibre_02']['mu_fit'] + fibparms[ord]['fibre_27']['mu_fit'])

        lfc_trace = fibparms[ord]['fibre_01']['mu_fit']
        simth_trace = fibparms[ord]['fibre_28']['mu_fit']
        stellar_trace = 0.5 * (fibparms[ord]['fibre_06']['mu_fit'] + fibparms[ord]['fibre_24']['mu_fit'])


        lfc_trace_fit = np.poly1d(np.polyfit(xx, lfc_trace, degpol))
        simth_trace_fit = np.poly1d(np.polyfit(xx, simth_trace, degpol))
        stellar_trace_fit = np.poly1d(np.polyfit(xx, stellar_trace, degpol))
        allfib_trace_fit = np.poly1d(np.polyfit(xx, allfib_trace, degpol))

        traces['lfc'][ord] = lfc_trace_fit
        traces['simth'][ord] = simth_trace_fit
        traces['stellar'][ord] = stellar_trace_fit
        traces['allfib'][ord] = allfib_trace_fit
    
    return traces
        
        

def extract_single_stripe(img, p, slit_height=25, return_indices=False, indonly=False, debug_level=0):
    """
    Extracts single stripe from 2d image.

    This function returns a sparse matrix containing all relevant pixel for a single stripe for a given polynomial p
    and a given slit height.

    :param img: 2d echelle spectrum
    :type img: np.ndarray
    :param P: polynomial coefficients
    :type P: np.ndarray
    :param slit_height: height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    :type slit_height: double
    :param debug_level: debug level
    :type debug_level: int
    :return: extracted spectrum
    :rtype: scipy.sparse.csc_matrix
    """
    
    #start_time = time.time()
    
    ny, nx = img.shape
    #xx = np.arange(nx, dtype=img.dtype)
    xx = np.arange(nx, dtype='f8')
    #yy = np.arange(ny, dtype=img.dtype)
    yy = np.arange(ny, dtype='f8')

    y = np.poly1d(p)(xx)
    x_grid, y_grid = np.meshgrid(xx, yy, copy=False)

    distance = y_grid - y.repeat(ny).reshape((nx, ny)).T
    indices = abs(distance) <= slit_height

    if debug_level >= 2:
        plt.figure()
        plt.imshow(img)
        plt.imshow(indices, origin='lower', alpha=0.5)
        plt.show()

    mat = sparse.coo_matrix((img[indices], (y_grid[indices], x_grid[indices])), shape=(ny, nx))
    # return mat.tocsr()
    
    #print('Elapsed time: ',time.time() - start_time,' seconds')
    
    if indonly:
        return indices
    else:
        if return_indices:
            return mat.tocsc(),indices
        else:
            return mat.tocsc()



def extract_stripes(img, P_id, slit_height=25, return_indices=True, savefiles=False, obsname=None, path=None, debug_level=0, timit=False):
    """
    Extracts the stripes from the original 2D spectrum to a sparse array, containing only relevant pixels.
    
    This function marks all relevant pixels for extraction. Using the provided dictionary P_id it iterates over all
    stripes in the image and saves a sparse matrix for each stripe.
    
    INPUT:
    'img'             : 2-dim image
    'P_id'            : dictionary of the form of {order: np.poly1d, ...} (as returned by "identify_stripes")
    'slit_height'     : height of the extraction slit (ie the pixel columns are 2*slit_height pixels long)
    'return_indices'  : boolean - do you also want to return the indices (ie x-&y-coordinates) of the pixels in the stripes? 
    'savefiles'       : boolean - do you want to save the extracted stripes and stripe-indices to files? [as a dictionary stored in a numpy file]
    'obsname'         : (short) name of observation file
    'path'            : directory to the destination of the output file
    'debug_level'     : for debugging...
    'timit'           : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'stripes'         : dictionary containing the extracted stripes (keys = orders)
    'stripe_indices'  : dictionary containing the indices (ie x-&y-coordinates) of the pixels in the extracted stripes (keys = orders)
    """
    
    if timit:
        overall_start_time = time.time()
    
    #logging.info('Extract stripes...')
    if return_indices:
        print('Extracting stripes and identifying stripe pixel-offsets...')
    else:
        print('Extracting stripes...')
    stripes = {}
    if return_indices:
        stripe_indices = {}
#     if isinstance(P_id, str):
#         # get all fibers
#         P_id = utils.load_dict_from_hdf5(P_id, 'extraction_parameters/')
#         # copy extraction parameters to result file
#         if output_file is not None:
#             utils.save_dict_to_hdf5(P_id, output_file, 'extraction_parameters/')
    
    # loop over all orders
    for o, p in sorted(P_id.items()):
        if return_indices:
            stripe,indices = extract_single_stripe(img, p, slit_height=slit_height, return_indices=True, debug_level=debug_level)
        else:
            stripe = extract_single_stripe(img, p, slit_height=slit_height, debug_level=debug_level)
            
#         if o in stripes:
        stripes.update({o: stripe})
        if return_indices:
            stripe_indices.update({o: indices})
#         else:
#              stripes = {o: stripe}

    if savefiles:
        if path is None:
            print('ERROR: path to output directory not provided!!!')
            return
        elif obsname is None:
            print('ERROR: "obsname" not provided!!!')
            return
        else:
            np.save(path + obsname + '_stripes.npy', stripes)
            if return_indices:
                np.save(path + obsname + '_stripe_indices.npy', stripe_indices)
#         for f in stripes.keys():
#             for o in stripes[f].keys():
#                 utils.store_sparse_mat(stripes[f][o], 'extracted_stripes/%s/%s' % (f, o), output_file)

    if timit:
        print('Total time taken for "EXTRACT_STRIPES": ',time.time() - overall_start_time,' seconds')
        
    if return_indices:
        return stripes,stripe_indices
    else:
        return stripes



def flatten_single_stripe(stripe, slit_height=25, timit=False):
    """
    CMB 06/09/2017
    
    This function stores the non-zero values of the sparse matrix "stripe" in a rectangular array, ie
    take out the curvature of the order/stripe, potentially useful for further processing.

    INPUT:
    "stripe"  : sparse 4096x4096 matrix, non-zero for just one stripe/order of user defined width (=2*slit_height in "extract_stripes")
    
    OUTPUT:
    "stripe_columns" : dense rectangular matrix containing only the non-zero elements of "stripe". This has
                       dimensions of (2*slit_height, 4096)
    "stripe_rows" :    row indices (ie rows being in dispersion direction) of the original image for the columns (ie the "cutouts")
    """
    # input is sth like this: stripe = stripes['fibre_01']['order_01']
    # TODO: sort the dictionary by order number...
    
    if timit:
        start_time = time.time()    
    
    ny, nx = stripe.todense().shape
    
    # find the non-zero parts of the sparse matrix
    # format is:
    # contents[0] = row indices
    # contents[1] = column indices
    # contents[2] = values
    contents = sparse.find(stripe)
    # the individual columns correspond to the unique values of the x-indices, stored in contents[1]    
    col_values, col_indices, counts = np.unique(contents[1], return_index=True, return_counts=True)     
#     stripe_columns = np.zeros((int(len(contents[0]) / len(col_indices)),len(col_indices)))
#     stripe_flux = np.zeros((2*slit_height, 4096))
    stripe_flux = np.zeros((2*slit_height, nx)) - 1.     #negative flux can be used to identify pixel-positions that fall outside the chip later
#     stripe_rows = np.zeros((int(len(contents[0]) / len(col_indices)),len(col_indices)))
    stripe_rows = np.zeros((2*slit_height, nx))
    
    # check if whole order falls on CCD in dispersion direction
    if len(col_indices) != nx:
        print('WARNING: Not the entire order falls on the CCD:'),    
        # parts of order missing on LEFT side of chip?
        if contents[1][0] != 0:
            print('some parts of the order are missing on LEFT side of chip...')
            #this way we also know how many pixels are defined (ie on the CCD) for each column of the stripe
            for i,coli,ct in zip(col_values,col_indices,counts):
                if i == np.max(col_values):
                    flux_temp = contents[2][coli:]         #flux
                    rownum_temp = contents[0][coli:]       #row number
                else:
                    #this is confusing, but: coli = col_indices[i-np.min(col_values)]
                    flux_temp = contents[2][col_indices[i-np.min(col_values)]:col_indices[i-np.min(col_values)+1]]         #flux
                    rownum_temp = contents[0][col_indices[i-np.min(col_values)]:col_indices[i-np.min(col_values)+1]]       #row number        
                #now, because the valid pixels are supposed to be at the bottom of the cutout, we need to roll them to the end
                if ct != (2*slit_height):
                    flux = np.zeros(2*slit_height) - 1.     #negative flux can be used to identify these pixels later
                    flux[0:ct] = flux_temp
                    flux = np.roll(flux,2*slit_height - ct)
                    rownum = np.zeros(2*slit_height,dtype='int32')    #zeroes can be used to identify these pixels later
                    rownum[0:ct] = rownum_temp
                    rownum = np.roll(rownum,2*slit_height - ct)
                else:
                    flux = flux_temp
                    rownum = rownum_temp
                stripe_flux[:,i] = flux
                stripe_rows[:,i] = rownum
        
        
        
        #parts of order missing on RIGHT side of chip?
        elif contents[1][-1] != nx-1:
            print('some parts of the order are missing on RIGHT side of chip...NIGHTMARE, THIS HAS NOT BEEN IMPLEMENTED YET') 
            sys.exit()
#             for i in col_values:
#                 #is there a value for all pixels across the i-th cutout?
#                 if i == np.max(col_values):
#                     flux = contents[2][col_indices[i]:]         #flux
#                     rownum = contents[0][col_indices[i]:]       #row number
#                 else:
#                     flux = contents[2][col_indices[i]:col_indices[i+1]]         #flux
#                     rownum = contents[0][col_indices[i]:col_indices[i+1]]       #row number
#                 stripe_flux[:,i] = flux
#                 stripe_rows[:,i] = rownum 
    
    
    #check if whole order falls on CCD in spatial direction
    elif ~(counts == 2*slit_height).all():
        print('WARNING: Not the entire order falls on the CCD:'),
        #parts of order missing at the top?
        if np.max(contents[0]) == ny-1:
            print('some parts of the order are missing on at the TOP of the chip...') 
            #this way we also know how many pixels are defined (ie on the CCD) for each column of the stripe
            for i,coli,ct in zip(col_values,col_indices,counts):
                
                if i == np.max(col_values):
                    flux_temp = contents[2][coli:]         #flux
                    rownum_temp = contents[0][coli:]       #row number
                else:
                    #this is confusing, but: coli = col_indices[i-np.min(col_values)]
                    flux_temp = contents[2][col_indices[i-np.min(col_values)]:col_indices[i-np.min(col_values)+1]]         #flux
                    rownum_temp = contents[0][col_indices[i-np.min(col_values)]:col_indices[i-np.min(col_values)+1]]       #row number        
                
                #now, because the valid pixels are supposed to be at the top of the cutout, we do NOT need to roll them to the end
                if ct != (2*slit_height):
                    flux = np.zeros(2*slit_height) - 1.         #negative flux can be used to identify these pixels layer
                    flux[0:ct] = flux_temp
                    #flux = np.roll(flux,2*slit_height - ct)
                    rownum = np.zeros(2*slit_height,dtype='int32')   #zeroes can be used to identify these pixels later
                    rownum[0:ct] = rownum_temp
                    #rownum = np.roll(rownum,2*slit_height - ct)
                else:
                    flux = flux_temp
                    rownum = rownum_temp
                stripe_flux[:,i] = flux
                stripe_rows[:,i] = rownum
        
        #parts of order missing at the bottom?
        elif np.min(contents[0]) == 0:
            print('some parts of the order are missing on at the BOTTOM of the chip...NIGHTMARE, THIS HAS NOT BEEN IMPLEMENTED YET') 
            sys.exit()
        
    else:    
        #this is the "normal", easy part, where all (2*slit_height,4096) pixels of the stripe lie on the CCD 
        for i in range(len(col_indices)):
            #this is the cutout from the original image
            if i == len(col_indices)-1:
                flux = contents[2][col_indices[i]:]         #flux
                rownum = contents[0][col_indices[i]:]       #row number
            else:
                flux = contents[2][col_indices[i]:col_indices[i+1]]         #flux
                rownum = contents[0][col_indices[i]:col_indices[i+1]]       #row number
            stripe_flux[:,i] = flux
            stripe_rows[:,i] = rownum
    
    if timit:
        delta_t = time.time() - start_time
        print('Time taken for "flattening" stripe: '+str(delta_t)+' seconds...')
            
    return stripe_flux,stripe_rows.astype(int)
   
   

def flatten_single_stripe_from_indices(img, indices, slit_height=25, timit=False):
    """
    CMB 07/03/2018
    
    This function stores the non-zero values of the sparse matrix "stripe" in a rectangular array, ie
    take out the curvature of the order/stripe, potentially useful for further processing.

    INPUT:
    "img": the image from the FITS file
    "indices": indices of img, that correspond to the stripe identified in 'extract_single_stripe'
    
    OUTPUT:
    "stripe_columns": dense rectangular matrix containing only the non-zero elements of "stripe". This has
                      dimensions of (2*slit_height, ~4096)
    "stripe_rows":    row indices (ie rows being in dispersion direction) of the original image for the columns (ie the "cutouts")
    """
    #input is sth like this
    #stripe = stripes['fibre_01']['order_01']
    #TODO: sort the dictionary by order number...
    
    if timit:
        start_time = time.time()    
    
    ny, nx = img.shape
    
    #stripe_flux = np.zeros((2*slit_height, 4096))
    stripe_flux = np.zeros((2*slit_height, nx))
    #stripe_rows = np.zeros((int(len(contents[0]) / len(col_indices)),len(col_indices)))
    stripe_rows = np.zeros((2*slit_height, nx))
    
    for i in range(nx):
        #this is the cutout from the original image
        flux = img[:,i][indices[:,i]]
        rownum = (indices[:,i] * np.arange(ny))[indices[:,i]]
        #now check if the full cutout lies on the image
        #cutout completely off the chip?
        if len(rownum) == 0:
            n_miss = 2 * slit_height
            flux = np.repeat([-1],n_miss)
            rownum = np.repeat([0],n_miss)
        #parts missing at BOTTOM???
        elif rownum[0] == 0:
            #how many missing?
            n_miss = 2*slit_height - len(rownum)
            #append flux and rownum
            flux = np.append(np.repeat([-1],n_miss),flux)
            rownum = np.append(np.repeat([0],n_miss),rownum)
        #parts missing at TOP???
        elif rownum[-1] == ny-1:
            #how many missing?
            n_miss = 2*slit_height - len(rownum)
            #append flux and rownum
            flux = np.append(flux,np.repeat([-1],n_miss))
            rownum = np.append(rownum,np.repeat([0],n_miss))
    
        stripe_flux[:,i] = flux
        stripe_rows[:,i] = rownum
    
    if timit:
        delta_t = time.time() - start_time
        print('Time taken for "flattening" stripe: '+str(delta_t)+' seconds...')
            
    return stripe_flux,stripe_rows.astype(int)
     
    

def flatten_stripes(stripes, slit_height=25):
    """
    CMB 27/09/2017
    
    For each stripe (ie order), this function stores the non-zero values of the sparse matrix "stripe" ("stripes" contains one "stripe" for each order)
    in a rectangular array, ie take out the curvature of the order/stripe, potentially only useful for further processing.

    INPUT:
    "stripes": dictionary containing the output from "extract_stripes", ie a sparse matrix for each stripe containing the respective non-zero elements
    
    OUTPUT:
    """

    order_boxes = {}
    
    # loop over all orders
    for ord in stripes.keys():
        #print(ord)
        stripe = stripes[ord]
        sc,sr = flatten_single_stripe(stripe,slit_height=slit_height)
        order_boxes[ord] = {'rows':sr, 'cols':sc}
    
    return order_boxes
        




def find_tramlines_single_order(uu, ul, lu, ll, mask_uu, mask_ul, mask_lu, mask_ll):
    
    #make sure they're all the same length
    if (len(uu) != len(ul)) or (len(uu) != len(lu)) or (len(uu) != len(ll)):
        print('ERROR: Dimensions of input arrays do not agree!!!')
        sys.exit()
        
    #fit 5th-order polynomial to each peak-array, with the weights of the fit being the RMS of the initial fit to the fibre profiles!?!?!?
    #WARNING: do unweighted for now...
    xx = np.arange(len(uu))
    
    # for the subsequent fit we only want to take the parts of the order, for which all the order traces lie on the chip
    # (this works b/c all fit parameters are set to -1 if any part of the cutout lies outside the chip; this is done in "fit_profiles_single_order")
    #good = np.logical_and(np.logical_and(uu>=0, ul>=0), np.logical_and(lu>=0, ll>=0))
    upper_good = np.logical_and(uu>=0, ul>=0)     #this removes the parts of the order where the fitted mu-values of any of the two upper fibres fall outside the chip
    lower_good = np.logical_and(lu>=0, ll>=0)     #this removes the parts of the order where the fitted mu-values of any of the two lower fibres fall outside the chip
    upper_mask = np.logical_and(upper_good, np.logical_and(mask_uu,mask_ul))     #this uses the mask from "find_stripes", ie also removes the low-flux regions at either side of some orders
    lower_mask = np.logical_and(lower_good, np.logical_and(mask_lu,mask_ll))     #this uses the mask from "find_stripes", ie also removes the low-flux regions at either side of some orders
    
    #w = np.sqrt(amp)
    #p = np.poly1d(np.polyfit(xx, mu03, 5,w=w))
    p_uu = np.poly1d(np.polyfit(xx[upper_mask], uu[upper_mask], 5))
    p_ul = np.poly1d(np.polyfit(xx[upper_mask], ul[upper_mask], 5))
    p_lu = np.poly1d(np.polyfit(xx[lower_mask], lu[lower_mask], 5))
    p_ll = np.poly1d(np.polyfit(xx[lower_mask], ll[lower_mask], 5))
    
#     res02 = p02(xx) - mu02
#     res03 = p03(xx) - mu03
#     res21 = p21(xx) - mu21
#     res22 = p22(xx) - mu22
#      
#     rms02 = np.sqrt( np.sum(res02*res02)/len(res02) )
#     rms03 = np.sqrt( np.sum(res03*res03)/len(res03) )
#     rms21 = np.sqrt( np.sum(res21*res21)/len(res21) )
#     rms22 = np.sqrt( np.sum(res22*res22)/len(res22) )
    
    #but define the boundaries for the entire order, ie also for the "bad" parts
    upper_boundary = 0.5 * (p_uu(xx) + p_ul(xx))
    lower_boundary = 0.5 * (p_lu(xx) + p_ll(xx))
    
    return upper_boundary, lower_boundary





def find_tramlines(fp_uu, fp_ul, fp_lu, fp_ll, mask_uu, mask_ul, mask_lu, mask_ll, debug_level=0, timit=False):
    '''
    INPUT: 
    P_id
    four single-fibre fibre-profiles-dictionaries, from fitting the single fibres for each order and cutout
    '''
    
    if timit:
        start_time = time.time()
    
    #make sure they're all the same length
    if (len(fp_uu) != len(fp_ul)) or (len(fp_uu) != len(fp_lu)) or (len(fp_uu) != len(fp_ll)):
        print('ERROR: Dimensions of input dictionaries do not agree!!!')
        sys.exit()
    
    tramlines = {}
    
    for ord in sorted(fp_uu.iterkeys()):
        uu = np.array(fp_uu[ord]['mu'])
        ul = np.array(fp_ul[ord]['mu'])
        lu = np.array(fp_lu[ord]['mu'])
        ll = np.array(fp_ll[ord]['mu'])
        upper_boundary, lower_boundary = find_tramlines_single_order(uu, ul, lu, ll, mask_uu[ord], mask_ul[ord], mask_lu[ord], mask_ll[ord])
        tramlines[ord] = {'upper_boundary':upper_boundary, 'lower_boundary':lower_boundary}
    
    if debug_level >= 1:
        xx = np.arange(4096)
        plt.figure()
        #plt.imshow(img, origin='lower', norm=LogNorm())
        for ord in tramlines.keys():
            plt.plot(xx, tramlines[ord]['upper_boundary'],'y-')
            plt.plot(xx, tramlines[ord]['lower_boundary'],'r-')
    
    if timit:
        print('Time taken for finding extraction tramlines: '+str(time.time() - start_time)+' seconds...')    
    
    return tramlines





def find_laser_tramlines_single_order(mu, mask):
        
    #fit 5th-order polynomial to each peak-array, with the weights of the fit being the RMS of the initial fit to the fibre profiles!?!?!?
    #WARNING: do unweighted for now...
    xx = np.arange(len(mu))
    
    # for the subsequent fit we only want to take the parts of the order, for which all the order traces lie on the chip
    # (this works b/c all fit parameters are set to -1 if any part of the cutout lies outside the chip; this is done in "fit_profiles_single_order")
    #good = np.logical_and(np.logical_and(uu>=0, ul>=0), np.logical_and(lu>=0, ll>=0))
    good = np.logical_and(mu>=0, mask)     #this removes the parts of the order where the fitted mu-values of any of the two upper fibres fall outside the chip
    
    #w = np.sqrt(amp)
    #p = np.poly1d(np.polyfit(xx, mu03, 5,w=w))
    p_mu = np.poly1d(np.polyfit(xx[good], mu[good], 5))
    
#     res02 = p02(xx) - mu02
#     res03 = p03(xx) - mu03
#     res21 = p21(xx) - mu21
#     res22 = p22(xx) - mu22
#      
#     rms02 = np.sqrt( np.sum(res02*res02)/len(res02) )
#     rms03 = np.sqrt( np.sum(res03*res03)/len(res03) )
#     rms21 = np.sqrt( np.sum(res21*res21)/len(res21) )
#     rms22 = np.sqrt( np.sum(res22*res22)/len(res22) )
    
    #but define the boundaries for the entire order, ie also for the "bad" parts
    upper_boundary = p_mu(xx) + 3
    lower_boundary = p_mu(xx) - 3
    
    
    return upper_boundary, lower_boundary





def find_laser_tramlines(fp, mask, debug_level=0, timit=False):
    
    if timit:
        start_time = time.time()
    
    tramlines = {}
    
    for ord in sorted(fp.iterkeys()):
        mu = np.array(fp[ord]['mu'])
        upper_boundary, lower_boundary = find_laser_tramlines_single_order(mu, mask[ord])
        tramlines[ord] = {'upper_boundary':upper_boundary, 'lower_boundary':lower_boundary}
    
    if debug_level >= 1:
        xx = np.arange(4096)
        plt.figure()
        #plt.imshow(img, origin='lower', norm=LogNorm())
        for ord in tramlines.keys():
            plt.plot(xx, tramlines[ord]['upper_boundary'],'y-')
            plt.plot(xx, tramlines[ord]['lower_boundary'],'r-')
    
    if timit:
        print('Time taken for finding extraction tramlines: '+str(time.time() - start_time)+' seconds...')    
    
    return tramlines




