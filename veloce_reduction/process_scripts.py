'''
Created on 25 Jul. 2018

@author: christoph
'''

import astropy.io.fits as pyfits
import numpy as np
import time
import os
import glob

from .helper_functions import binary_indices, laser_on, thxe_on
from .calibration import correct_for_bias_and_dark_from_filename
from .cosmic_ray_removal import remove_cosmics, median_remove_cosmics
from .background import extract_background, extract_background_pid, fit_background
from .order_tracing import extract_stripes
from .extraction import extract_spectrum, extract_spectrum_from_indices
from .relative_intensities import get_relints, get_relints_from_indices, append_relints_to_FITS
from .barycentric_correction import get_barycentric_correction




def process_whites(white_list, MB=None, ronmask=None, MD=None, gain=None, P_id=None, scalable=False, fancy=False, remove_bg=True, clip=5., savefile=True, saveall=False, diffimg=False, path=None, debug_level=0, timit=False, date=None):
    """
    This routine processes all whites from a given list of files. It corrects the orientation of the image and crops the overscan regions,
    and subtracts both the MASTER BIAS frame [in ADU], and the MASTER DARK frame [in e-] from every image before combining them to create a MASTER WHITE frame.
    NOTE: the input image has units of ADU, but the output image has units of electrons!!!
    
    INPUT:
    'white_list'  : list of filenames of raw white images (incl. directories)
    'MB'          : the master bias frame (bias only, excluding OS levels) [ADU]
    'ronmask'     : the read-noise mask (or frame) [e-]
    'MD'          : the master dark frame [e-]
    'gain'        : the gains for each quadrant [e-/ADU]
    'P_id'        : order tracing dictionary (only needed if remove_bg is set to TRUE)
    'scalable'    : boolean - do you want to normalize the dark current to an exposure time of 1s? (ie do you want to make it "scalable"?)
    'fancy'       : boolean - do you want to use the 'fancy' method for creating the master white frame? (otherwise a simple median image will be used)
    'remove_bg'   : boolean - do you want to remove the background from the output master white?
    'clip'        : number of 'expected-noise sigmas' a pixel has to deviate from the median pixel value across all images to be considered an outlier when using the 'fancy' method
    'savefile'    : boolean - do you want to save the master white frame as a FITS file?
    'saveall'     : boolean - do you want to save all individual bias- & dark-corrected images as well?
    'diffimg'     : boolean - do you want to save the difference image (ie containing the outliers)? only used if 'fancy' is set to TRUE
    'path'        : path to the output file directory (only needed if savefile is set to TRUE)
    'debug_level' : for debugging...
    'timit'       : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'master'      : the master white image [e-] (also has been brought to 'correct' orientation, overscan regions cropped, and (if desired) bg-corrected) 
    'err_master'  : the corresponding uncertainty array [e-]    
    
    """


    if timit:
        start_time = time.time()

    if debug_level >= 1:
        print('Creating master white frame from '+str(len(white_list))+' fibre flats...')

    # if INPUT arrays are not given, read them from default files
    if path is None:
        print('WARNING: output file directory not provided!!!')
        print('Using same directory as input file...')
        dum = white_list[0].split('/')
        path = white_list[0][0:-len(dum[-1])]
    
    date = path.split('/')[-2]
        
    if MB is None:
        # no need to fix orientation, this is already a processed file [ADU]
#         MB = pyfits.getdata(path+'master_bias.fits')
        MB = pyfits.getdata(path + date + '_median_bias.fits')
    if ronmask is None:
        # no need to fix orientation, this is already a processed file [e-]
        ronmask = pyfits.getdata(path + date + '_read_noise_mask.fits')
    if MD is None:
        if scalable:
            # no need to fix orientation, this is already a processed file [e-]
            MD = pyfits.getdata(path + date + '_master_dark_scalable.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_scalable.fits', 1)
        else:
            # no need to fix orientation, this is already a processed file [e-]
            texp = pyfits.getval(white_list[0])
            MD = pyfits.getdata(path + date + '_master_dark_t' + str(int(np.round(texp,0))) + '.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_t'+str(int(np.round(texp,0)))+'.fits', 1)


    # prepare arrays
    allimg = []
    allerr = []

    # loop over all files in "white_list"; correct for bias and darks on the fly
    for n,fn in enumerate(sorted(white_list)):
        if debug_level >=1:
            print('Now processing file ' + str(n+1) + '/' + str(len(white_list)) + '   (' + fn + ')')

        # call routine that does all the bias and dark correction stuff and converts from ADU to e-
        if scalable:
            # if the darks have a different exposure time than the whites, then we need to re-scale the master dark
            texp = pyfits.getval(white_list[0], 'ELAPSED')
            img = correct_for_bias_and_dark_from_filename(fn, MB, MD*texp, gain=gain, scalable=scalable, savefile=saveall,
                                                          path=path, timit=timit)     #these are now bias- & dark-corrected images; units are e-
        else:
            img = correct_for_bias_and_dark_from_filename(fn, MB, MD, gain=gain, scalable=scalable, savefile=saveall,
                                                          path=path, timit=timit)     # these are now bias- & dark-corrected images; units are e-

        if debug_level >=2:
            print('min(img) = ' + str(np.min(img)))
        allimg.append(img)
#         err_img = np.sqrt(img + ronmask*ronmask)   # [e-]
        # TEMPFIX: (how should I be doing this properly???)
        err_img = np.sqrt(np.clip(img,0,None) + ronmask*ronmask)   # [e-]
        allerr.append(err_img)

    # list of individual exposure times for all whites (should all be the same, but just in case...)
    texp_list = [pyfits.getval(file, 'ELAPSED') for file in white_list]
    # scale to the median exposure time
    tscale = np.array(texp_list) / np.median(texp_list)


    #########################################################################
    ### now we do essentially what "CREATE_MASTER_IMG" does for whites... ###
    #########################################################################
    # add individual-image errors in quadrature (need it either way, not only for fancy method)
    err_summed = np.sqrt(np.sum((np.array(allerr)**2), axis=0))
#     # get plain median image
#     medimg = np.median(np.array(allimg), axis=0)
    # take median after scaling to median exposure time 
    print(tscale)
    medimg = np.median(np.array(allimg), axis=0) # / tscale.reshape(len(allimg), 1, 1), axis=0)    

    if fancy:
        # need to create a co-added frame if we want to do outlier rejection the fancy way
        summed = np.sum((np.array(allimg)), axis=0)
        if diffimg:
            diff = np.zeros(summed.shape)

        master_outie_mask = np.zeros(summed.shape, dtype='int')

        # make sure we do not have any negative pixels for the sqrt
        medimgpos = medimg.copy()
        medimgpos[medimgpos < 0] = 0.
        med_sig_arr = np.sqrt(medimgpos + ronmask*ronmask)       # expected STDEV for the median image (from LB Eq 2.1); still in ADUs
        for n,img in enumerate(allimg):
            # outie_mask = np.abs(img - medimg) > clip*med_sig_arr
            outie_mask = (img - medimg) > clip*med_sig_arr      # do we only want HIGH outliers, ie cosmics?
            # save info about which image contributes the outlier pixel using unique binary numbers technique
            master_outie_mask += (outie_mask * 2**n).astype(int)
        # see which image(s) produced the outlier(s) and replace outies by mean of pixel value from remaining images
        n_outie = np.sum(master_outie_mask > 0)
        print('Correcting '+str(n_outie)+' outliers...')
        # loop over all outliers
        for i,j in zip(np.nonzero(master_outie_mask)[0],np.nonzero(master_outie_mask)[1]):
            # access binary numbers and retrieve component(s)
            outnum = binary_indices(master_outie_mask[i,j])   # these are the indices (within allimg) of the images that contain outliers
            dumix = np.arange(len(white_list))
            # remove the images containing the outliers in order to compute mean from the remaining images
            useix = np.delete(dumix,outnum)
            if diffimg:
                diff[i,j] = summed[i,j] - ( len(outnum) * np.mean( np.array([allimg[q][i,j] for q in useix]) ) + np.sum( np.array([allimg[q][i,j] for q in useix]) ) )
            # now replace value in master image by the sum of all pixel values in the unaffected pixels
            # plus the number of affected images times the mean of the pixel values in the unaffected images
            summed[i,j] = len(outnum) * np.mean( np.array([allimg[q][i,j] for q in useix]) ) + np.sum( np.array([allimg[q][i,j] for q in useix]) )
        # once we have finished correcting the outliers, we want to "normalize" (ie divide by number of frames) the master image and the corresponding error array
        master = summed / len(white_list)
        err_master = err_summed / len(white_list)
    else:
        # ie not fancy, just take the median image to remove outliers       
        # now set master image equal to median image
        master = medimg.copy()
        nw = len(white_list)     # number of whites 
#         # estimate of the corresponding error array (estimate only!!!)
#         err_master = err_summed / nw     # I don't know WTF I was thinking here...
        # if roughly Gaussian distribution of values: error of median ~= 1.253*error of mean
        # err_master = 1.253 * np.std(allimg, axis=0) / np.sqrt(nw-1)     # normally it would be sigma/sqrt(n), but np.std is dividing by sqrt(n), not by sqrt(n-1)
        # need to rescale by exp time here, too
        if nw == 1:
            err_master = allerr[0]
        else:
            err_master = 1.253 * np.std(np.array(allimg), axis=0) / np.sqrt(nw-1)# / tscale.reshape(len(allimg), 1, 1), axis=0) / np.sqrt(nw-1)     # normally it would be sigma/sqrt(n), but np.std is dividing by sqrt(n), not by sqrt(n-1)
        # err_master = np.sqrt( np.sum( (np.array(allimg) - np.mean(np.array(allimg), axis=0))**2 / (nw*(nw-1)) , axis=0) )   # that is equivalent, but slower
    
    
    # now subtract background (errors remain unchanged)
    if remove_bg:
        # identify and extract background
        print(master.shape)
        bg = extract_background_pid(master, P_id, slit_height=30, exclude_top_and_bottom=True, timit=timit)
        # fit background
        bg_coeffs, bg_img = fit_background(bg, clip=10, return_full=True, timit=timit)
        # subtract background
        master = master - bg_img


    # now save master white to file
    if savefile:
        outfn = os.path.join(path, '{0}_master_white.fits'.format(date))
#        outfn = path + date + '_master_white.fits'
        pyfits.writeto(outfn, np.float32(master), overwrite=True)
        pyfits.setval(outfn, 'HISTORY', value='   MASTER WHITE frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
        # pyfits.setval(outfn, 'EXPTIME', value=texp, comment='exposure time [s]')
        pyfits.setval(outfn, 'UNITS', value='ELECTRONS')
        if fancy:
            pyfits.setval(outfn, 'METHOD', value='fancy', comment='method to create master white & remove outliers')
        else:
            pyfits.setval(outfn, 'METHOD', value='median', comment='method to create master white & remove outliers')
        h = pyfits.getheader(outfn)
        h_err = h.copy()
        h_err['HISTORY'] = 'estimated uncertainty in MASTER WHITE frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.append(outfn, np.float32(err_master), h_err, overwrite=True)

    # also save the difference image if desired
    if diffimg:
        hdiff = h.copy()
        hdiff['HISTORY'] = '   MASTER WHITE DIFFERENCE IMAGE - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.writeto(path + date + '_master_white_diffimg.fits', diff, hdiff, overwrite=True)

    if timit:
        print('Total time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return master, err_master



def process_science_images(imglist, P_id, chipmask, mask=None, stripe_indices=None, quick_indices=None,
                           sampling_size=25, slit_height=32, qsh=23, gain=[1., 1., 1., 1.], MB=None, ronmask=None,
                           MD=None, scalable=False, saveall=False, pathdict=None, ext_method='optimal', path=None,
                           from_indices=True, slope=True, offset=True, fibs='all', date=None, timit=False):
    """
    Process all science / calibration lamp images. This includes:

    (1) bias and dark subtraction
    (2) cosmic ray removal
    (3) background extraction and estimation
    (4) flat-fielding (ie removal of pixel-to-pixel sensitivity variations)
    =============================
    (5) extraction of stripes
    (6) extraction of 1-dim spectra
    (7) get relative intensities of different fibres
    (8) wavelength solution
    (9) barycentric correction (for stellar observations only)
    """

    print('WARNING: I commented out BARCYRORR')
    # cont = raw_input('Do you still want to continue?')
    cont = 'y'
    assert cont.lower() == 'y', 'You chose to quit!'

#    assert pathdict is not None, 'ERROR: pathdict not provided!!!'
    if pathdict is None:
        path = path
    else:
        path = pathdict['raw']

    if timit:
        start_time = time.time()

    # sort image list, just in case
    imglist.sort()

    # get a list with the object names
    object_list = [pyfits.getval(file, 'OBJECT').split('+')[0] for file in imglist]
    if object_list[0][:3] == 'ARC':
        obstype = 'ARC'
    elif object_list[0].lower() in ["lc", "lc-only", "lfc", "lfc-only", "simlc", "thxe", "thxe-only", "simth",
                                    "thxe+lfc", "lfc+thxe", "lc+simthxe", "lc+thxe"]:
        obstype = 'simcalib'
    else:
        obstype = 'stellar'
    if obstype in ['stellar', 'ARC']:
        # and the indices where the object changes (to figure out which observations belong to one epoch)
        changes = np.where(np.array(object_list)[:-1] != np.array(object_list)[1:])[
                      0] + 1  # need the plus one to get the indices of the first occasion of a new object
        # list of indices for individual epochs - there's gotta be a smarter way to do this...
        all_epoch_list = []
        if len(changes) > 0:
            all_epoch_list.append(np.arange(0, changes[0]))
            for j in range(len(changes) - 1):
                all_epoch_list.append(np.arange(changes[j], changes[j + 1]))
            all_epoch_list.append(np.arange(changes[-1], len(object_list)))
        else:
            all_epoch_list.append(np.arange(0, len(object_list)))

    #####################################
    ### (1) bias and dark subtraction ###
    #####################################

    # if INPUT arrays are not given, read them from default files
    if path is None:
        print('WARNING: output file directory not provided!!!')
        print('Using same directory as input file...')
        dum = imglist[0].split('/')
        path = imglist[0][0: -len(dum[-1])]
    if date is None:
        date = path.split('/')[-2]
    if MB is None:
        # no need to fix orientation, this is already a processed file [ADU]
        #         MB = pyfits.getdata(path + 'master_bias.fits')
        MB = pyfits.getdata(path + 'median_bias.fits')
    if ronmask is None:
        # no need to fix orientation, this is already a processed file [e-]
        ronmask = pyfits.getdata(path + 'read_noise_mask.fits')
    if MD is None:
        if scalable:
            # no need to fix orientation, this is already a processed file [e-]
            MD = pyfits.getdata(path + 'master_dark_scalable.fits', 0)
        #             err_MD = pyfits.getdata(path + 'master_dark_scalable.fits', 1)
        else:
            # no need to fix orientation, this is already a processed file [e-]
            print('WARNING: scalable KW not properly implemented (stellar_list can have different exposure times...)')
            texp = 600.
            MD = pyfits.getdata(path + 'master_dark_t' + str(int(np.round(texp, 0))) + '.fits', 0)
    #             err_MD = pyfits.getdata(path + 'master_dark_t' + str(int(np.round(texp, 0))) + '.fits', 1)

    if not from_indices:
        ron_stripes = extract_stripes(ronmask, P_id, return_indices=False, slit_height=slit_height, savefiles=False,
                                      timit=True)

    # loop over all files
    for i, filename in enumerate(imglist):

        # (0) do some housekeeping with filenames, and check if there are multiple exposures for a given epoch of a star
        dum = filename.split('/')
        dum2 = dum[-1].split('.')
        obsname = dum2[0]
        obsnum = int(obsname[-5:])
        object = pyfits.getval(filename, 'OBJECT').split('+')[0]
        object_indices = np.where(object == np.array(object_list))[0]
        texp = pyfits.getval(filename, 'ELAPSED')
        # check if this exposure belongs to the same epoch as the previous one
        if obstype in ['stellar', 'ARC']:

            # list of all the observations belonging to this epoch
            epoch_ix = [sublist for sublist in all_epoch_list if
                        i in sublist]  # different from object_indices, as epoch_ix contains only indices for this particular epoch if there are multiple epochs of a target in a given night
            epoch_list = list(np.array(imglist)[epoch_ix])
            # make sublists according to the four possible calibration lamp configurations
            epoch_sublists = {'lfc': [], 'thxe': [], 'both': [], 'neither': []}

            if i > 0:
                if filename in epoch_list:
                    new_epoch = False
                else:
                    new_epoch = True
                    # delete existing temp bg files so we don't accidentally load them for a wrong epoch
                    if os.path.isfile(path + 'temp_bg_lfc.fits'):
                        os.remove(path + 'temp_bg_lfc.fits')
                    if os.path.isfile(path + 'temp_bg_thxe.fits'):
                        os.remove(path + 'temp_bg_thxe.fits')
                    if os.path.isfile(path + 'temp_bg_both.fits'):
                        os.remove(path + 'temp_bg_both.fits')
                    if os.path.isfile(path + 'temp_bg_neither.fits'):
                        os.remove(path + 'temp_bg_neither.fits')
            else:
                new_epoch = True
                # delete existing temp bg files so we don't accidentally load them for a wrong epoch
                if os.path.isfile(path + 'temp_bg_lfc.fits'):
                    os.remove(path + 'temp_bg_lfc.fits')
                if os.path.isfile(path + 'temp_bg_thxe.fits'):
                    os.remove(path + 'temp_bg_thxe.fits')
                if os.path.isfile(path + 'temp_bg_both.fits'):
                    os.remove(path + 'temp_bg_both.fits')
                if os.path.isfile(path + 'temp_bg_neither.fits'):
                    os.remove(path + 'temp_bg_neither.fits')
        else:
            if i == 0:
                new_epoch = True
            else:
                new_epoch = False

        print('Extracting ' + obstype + ' spectrum ' + str(i + 1) + '/' + str(len(imglist)) + ': ' + obsname)

        if obstype in ['stellar', 'ARC']:

            # nasty temp fix to make sure we are always looking at the 2D images until the header keywords are reliable
            checkdate = '1' + date[1:]

            if int(checkdate) < 20190503:
                # look at the actual 2D image (using chipmasks for LFC and simThXe) to determine which calibration lamps fired
                for file in epoch_list:
                    img = correct_for_bias_and_dark_from_filename(file, MB, MD, gain=gain, scalable=scalable,
                                                                  savefile=saveall, path=path)
                    lc = laser_on(img, chipmask)
                    thxe = thxe_on(img, chipmask)
                    if (not lc) and (not thxe):
                        epoch_sublists['neither'].append(file)
                    elif (lc) and (thxe):
                        epoch_sublists['both'].append(file)
                    else:
                        if lc:
                            epoch_sublists['lfc'].append(file)
                        elif thxe:
                            epoch_sublists['thxe'].append(file)
                # now check the calibration lamp configuration for the main observation in question
                img = correct_for_bias_and_dark_from_filename(filename, MB, MD, gain=gain, scalable=scalable,
                                                              savefile=saveall, path=path)
                lc = laser_on(img, chipmask)
                thxe = thxe_on(img, chipmask)
                if (not lc) and (not thxe):
                    lamp_config = 'neither'
                elif (lc) and (thxe):
                    lamp_config = 'both'
                else:
                    if lc:
                        lamp_config = 'lfc'
                    elif thxe:
                        lamp_config = 'thxe'
            else:
                # since May 2019 the header keywords are (mostly) correct, so could just check for LFC / ThXe in header, as that is MUCH faster
                for file in epoch_list:
                    lc = 0
                    thxe = 0
                    h = pyfits.getheader(file)
                    if 'LCNEXP' in h.keys():  # this indicates the latest version of the FITS headers (from May 2019 onwards)
                        if ('LCEXP' in h.keys()) or (
                                'LCMNEXP' in h.keys()):  # this indicates the LFC actually was actually exposed (either automatically or manually)
                            lc = 1
                    else:  # if not, just go with the OBJECT field
                        if ('LC' in pyfits.getval(filename, 'OBJECT').split('+')) or (
                                'LFC' in pyfits.getval(filename, 'OBJECT').split('+')):
                            lc = 1
                    if (h['SIMCALTT'] > 0) and (h['SIMCALN'] > 0) and (h['SIMCALSE'] > 0):
                        thxe = 1
                    assert lc + thxe in [0, 1,
                                         2], 'ERROR: could not establish status of LFC and simultaneous ThXe for ' + obsname + '.fits !!!'
                    if lc + thxe == 0:
                        epoch_sublists['neither'].append(file)
                    elif lc + thxe == 1:
                        if lc == 1:
                            epoch_sublists['lfc'].append(file)
                        else:
                            epoch_sublists['thxe'].append(file)
                    elif lc + thxe == 2:
                        epoch_sublists['both'].append(file)
                # now check the calibration lamp configuration for the main observation in question
                lc = 0
                thxe = 0
                h = pyfits.getheader(filename)
                if 'LCNEXP' in h.keys():  # this indicates the latest version of the FITS headers (from May 2019 onwards)
                    if ('LCEXP' in h.keys()) or (
                            'LCMNEXP' in h.keys()):  # this indicates the LFC actually was actually exposed (either automatically or manually)
                        lc = 1
                else:  # if not latest header version, just go with the OBJECT field
                    if ('LC' in pyfits.getval(filename, 'OBJECT').split('+')) or (
                            'LFC' in pyfits.getval(filename, 'OBJECT').split('+')):
                        lc = 1
                if h['SIMCALTT'] > 0:
                    thxe = 1
                if lc + thxe == 0:
                    lamp_config = 'neither'
                elif lc + thxe == 1:
                    if lc == 1:
                        lamp_config = 'lfc'
                    else:
                        lamp_config = 'thxe'
                elif lc + thxe == 2:
                    lamp_config = 'both'
        else:
            # for sim. calibration images we don't need to check for the calibration lamp configuration for all exposures (done external to this function)!
            # just for the file in question and then create a dummy copy of the image list so that it is in the same format that is expected for stellar observations

            # nasty temp fix to make sure we are always looking at the 2D images until the header keywords are reliable
            checkdate = '1' + date[1:]

            if int(checkdate) < 20190503:
                # now check the calibration lamp configuration for the main observation in question
                img = correct_for_bias_and_dark_from_filename(filename, MB, MD, gain=gain, scalable=scalable,
                                                              savefile=saveall, path=path)
                lc = laser_on(img, chipmask)
                thxe = thxe_on(img, chipmask)
                if (not lc) and (not thxe):
                    lamp_config = 'neither'
                elif (lc) and (thxe):
                    lamp_config = 'both'
                else:
                    if lc:
                        lamp_config = 'lfc'
                    elif thxe:
                        lamp_config = 'thxe'
            else:
                # now check the calibration lamp configuration for the main observation in question
                lc = 0
                thxe = 0
                h = pyfits.getheader(filename)
                if 'LCNEXP' in h.keys():  # this indicates the latest version of the FITS headers (from May 2019 onwards)
                    if ('LCEXP' in h.keys()) or (
                            'LCMNEXP' in h.keys()):  # this indicates the LFC actually was actually exposed (either automatically or manually)
                        lc = 1
                else:  # if not latest header version, just go with the OBJECT field
                    if ('LC' in pyfits.getval(filename, 'OBJECT').split('+')) or (
                            'LFC' in pyfits.getval(filename, 'OBJECT').split('+')):
                        lc = 1
                if h['SIMCALTT'] > 0:
                    thxe = 1
                if lc + thxe == 0:
                    lamp_config = 'neither'
                elif lc + thxe == 1:
                    if lc == 1:
                        lamp_config = 'lfc'
                    else:
                        lamp_config = 'thxe'
                elif lc + thxe == 2:
                    lamp_config = 'both'

            epoch_sublists = {}
            epoch_sublists[lamp_config] = imglist[:]

        # (1) call routine that does all the overscan-, bias- & dark-correction stuff and proper error treatment
        img = correct_for_bias_and_dark_from_filename(filename, MB, MD, gain=gain, scalable=scalable, savefile=saveall,
                                                      path=path)  # [e-]
        # err = np.sqrt(img + ronmask*ronmask)   # [e-]
        # TEMPFIX: (how should I be doing this properly???)
        err_img = np.sqrt(np.clip(img, 0, None) + ronmask * ronmask)  # [e-]

        ## (2) remove cosmic rays from background, then fit and remove background
        ## check if there are multiple exposures for this epoch (if yes, we can do the much simpler "median_remove_cosmics")
        if len(epoch_sublists[lamp_config]) == 1:
            # do it the hard way using LACosmic
            # identify and extract background
            bg_raw = extract_background(img, chipmask['bg'], timit=timit)
            # remove cosmics, but only from background
            cosmic_cleaned_img = remove_cosmics(bg_raw.todense(), ronmask, obsname, path, Flim=3.0, siglim=5.0,
                                                maxiter=1, savemask=False, savefile=False, save_err=False, verbose=True,
                                                timit=True)  # [e-]
            # identify and extract background from cosmic-cleaned image
            bg = extract_background(cosmic_cleaned_img, chipmask['bg'], timit=timit)
            #             bg = extract_background_pid(cosmic_cleaned_img, P_id, slit_height=30, exclude_top_and_bottom=True, timit=timit)
            # fit background
            bg_coeffs, bg_img = fit_background(bg, clip=10, return_full=True, timit=timit)
        elif len(epoch_sublists[lamp_config]) == 2:
            if new_epoch or not os.path.isfile(path + 'temp_bg_' + lamp_config + '.fits'):
                # list of individual exposure times for this epoch
                subepoch_texp_list = [pyfits.getval(file, 'ELAPSED') for file in epoch_sublists[lamp_config]]
                tscale = np.array(subepoch_texp_list) / texp
                # get background from the element-wise minimum-image of the two images
                img1 = correct_for_bias_and_dark_from_filename(epoch_sublists[lamp_config][0], MB, MD, gain=gain,
                                                               scalable=scalable, savefile=False)
                img2 = correct_for_bias_and_dark_from_filename(epoch_sublists[lamp_config][1], MB, MD, gain=gain,
                                                               scalable=scalable, savefile=False)
                min_img = np.minimum(img1 / tscale[0], img2 / tscale[1])
                # identify and extract background from the minimum-image
                bg = extract_background(min_img, chipmask['bg'], timit=timit)
                #             bg = extract_background_pid(min_img, P_id, slit_height=30, exclude_top_and_bottom=True, timit=timit)
                del min_img
                # fit background
                bg_coeffs, bg_img = fit_background(bg, clip=10, return_full=True, timit=timit)
                # save background image to temporary file for re-use later (when reducing the next file of this sublist)
                pyfits.writeto(path + 'temp_bg_' + lamp_config + '.fits', np.float32(bg_img), overwrite=True)
            else:
                # no need to re-compute background, just load it from file
                print('Loading background image for this epoch and lamp configuration...')
                bg_img = pyfits.getdata(path + 'temp_bg_' + lamp_config + '.fits')
        else:
            if new_epoch or not os.path.isfile(path + 'temp_bg_' + lamp_config + '.fits'):
                # make sure this sublist is not too long (otherwise we might run out of memory in this step)
                if len(epoch_sublists[lamp_config]) > 10:
                    mainix = epoch_sublists[lamp_config].index(filename)
                    if mainix < 5:
                        epoch_sublists[lamp_config] = epoch_sublists[lamp_config][:11]
                    elif mainix > len(epoch_sublists[lamp_config]) - 6:
                        epoch_sublists[lamp_config] = epoch_sublists[lamp_config][-11:]
                    else:
                        epoch_sublists[lamp_config] = epoch_sublists[lamp_config][mainix - 5:mainix + 6]
                # list of individual exposure times for this epoch
                subepoch_texp_list = [pyfits.getval(file, 'ELAPSED') for file in epoch_sublists[lamp_config]]
                tscale = np.array(subepoch_texp_list) / texp
                # make list of actual images
                img_list = []
                for file in epoch_sublists[lamp_config]:
                    img_list.append(correct_for_bias_and_dark_from_filename(file, MB, MD, gain=gain, scalable=scalable,
                                                                            savefile=False))
                #             # index indicating which one of the files in the epoch list is the "main" one
                #             main_index = np.where(np.array(epoch_ix) == i)[0][0]
                # take median after scaling to same exposure time as main exposure
                med_img = np.median(np.array(img_list) / tscale.reshape(len(img_list), 1, 1), axis=0)
                del img_list
                # identify and extract background from the median image
                bg = extract_background(med_img, chipmask['bg'], timit=timit)
                #             bg = extract_background_pid(med_img, P_id, slit_height=30, exclude_top_and_bottom=True, timit=timit)
                del med_img
                # fit background
                bg_coeffs, bg_img = fit_background(bg, clip=10, return_full=True, timit=timit)
                # save background image to temporary file for re-use later (when reducing the next file of this sublist)
                pyfits.writeto(path + 'temp_bg_' + lamp_config + '.fits', np.float32(bg_img), overwrite=True)
            else:
                # no need to re-compute background, just load it from file
                print('Loading background image for this epoch and lamp configuration...')
                bg_img = pyfits.getdata(path + 'temp_bg_' + lamp_config + '.fits')

        # now actually subtract the background model
        bg_corrected_img = img - bg_img

        # # save background model to file (or APPEND TO RAW FILE???)
        # bg_fn= path + obsname + '_BG_model.fits'
        # pyfits.writeto(bg_fn, bg_img)

        #       cosmic_cleaned_img = median_remove_cosmics(img_list, main_index=main_index, scales=scaled_texp, ronmask=ronmask, debug_level=1, timit=True)

        # (3) fit and remove background (ERRORS REMAIN UNCHANGED)
        # bg_corrected_img = remove_background(cosmic_cleaned_img, P_id, obsname, path, degpol=5, slit_height=slit_height, save_bg=True, savefile=True, save_err=False,
        #                                      exclude_top_and_bottom=True, verbose=True, timit=True)   # [e-]
        # bg_corrected_img = remove_background(img, P_id, obsname, path, degpol=5, slit_height=slit_height, save_bg=False, savefile=True, save_err=False,
        #                                      exclude_top_and_bottom=True, verbose=True, timit=True)   # [e-]
        # adjust errors?

        # (4) remove pixel-to-pixel sensitivity variations (2-dim)
        # XXXXXXXXXXXXXXXXXXXXXXXXXXX
        # TEMPFIX
        final_img = bg_corrected_img.copy()  # [e-]
        #         final_img = img.copy()   # [e-]
        # adjust errors?

        # (5) extract stripes
        if not from_indices:
            stripes, stripe_indices = extract_stripes(final_img, P_id, return_indices=True, slit_height=slit_height,
                                                      savefiles=saveall, obsname=obsname, path=path, timit=True)
            err_stripes = extract_stripes(err_img, P_id, return_indices=False, slit_height=slit_height,
                                          savefiles=saveall, obsname=obsname + '_err', path=path, timit=True)
        if stripe_indices is None:
            # this is just to get the stripe indices in case we forgot to provide them (DONE ONLY ONCE, if at all...)
            stripes, stripe_indices = extract_stripes(final_img, P_id, return_indices=True, slit_height=slit_height,
                                                      savefiles=False, obsname=obsname, path=path, timit=True)

        # (6) perform extraction of 1-dim spectrum
        if from_indices:
            pix, flux, err = extract_spectrum_from_indices(final_img, err_img, quick_indices, method='quick',
                                                           slit_height=qsh, ronmask=ronmask, savefile=True,
                                                           filetype='fits', obsname=obsname, date=date,
                                                           pathdict=pathdict, lamp_config=lamp_config, timit=True)
            pix, flux, err = extract_spectrum_from_indices(final_img, err_img, stripe_indices, method=ext_method,
                                                           slope=slope, offset=offset, fibs=fibs,
                                                           slit_height=slit_height,
                                                           ronmask=ronmask, savefile=True, filetype='fits',
                                                           obsname=obsname, date=date, pathdict=pathdict,
                                                           lamp_config=lamp_config, timit=True)
        else:
            pix, flux, err = extract_spectrum(stripes, err_stripes=err_stripes, ron_stripes=ron_stripes, method='quick',
                                              slit_height=qsh, ronmask=ronmask, savefile=True,
                                              filetype='fits', obsname=obsname, date=date, pathdict=pathdict,
                                              lamp_config=lamp_config, timit=True)
            pix, flux, err = extract_spectrum(stripes, err_stripes=err_stripes, ron_stripes=ron_stripes,
                                              method=ext_method, slope=slope, offset=offset, fibs=fibs,
                                              slit_height=slit_height, ronmask=ronmask, savefile=True, filetype='fits',
                                              obsname=obsname, date=date, pathdict=pathdict, lamp_config=lamp_config,
                                              timit=True)

    #         # (7) get relative intensities of different fibres
    #         if from_indices:
    #             relints = get_relints_from_indices(P_id, final_img, err_img, stripe_indices, mask=mask, sampling_size=sampling_size, slit_height=slit_height, return_full=False, timit=True)
    #         else:
    #             relints = get_relints(P_id, stripes, err_stripes, mask=mask, sampling_size=sampling_size, slit_height=slit_height, return_full=False, timit=True)
    #
    #
    #         # (8) get wavelength solution
    #         #XXXXX

    # # (9) get barycentric correction
    # if obstype == 'stellar':
    #     bc = get_barycentric_correction(filename)
    #     bc = np.round(bc,2)
    #     if np.isnan(bc):
    #         bc = ''
    #     # write the barycentric correction into the FITS header of both the quick-extracted and the optimal-extracted reduced spectrum files
    #     outfn_list = glob.glob(path + '*' + obsname + '*extracted*')
    #     for outfn in outfn_list:
    #         pyfits.setval(outfn, 'BARYCORR', value=bc, comment='barycentric velocity correction [m/s]')

    #         #now append relints, wl-solution, and barycorr to extracted FITS file header
    #         outfn = path + obsname + '_extracted.fits'
    #         if os.path.isfile(outfn):
    #             #relative fibre intensities
    #             dum = append_relints_to_FITS(relints, outfn, nfib=19)
    #             #wavelength solution
    #             #pyfits.setval(fn, 'RELINT' + str(i + 1).zfill(2), value=relints[i], comment='fibre #' + str(fibnums[i]) + ' - ' + fibinfo[i] + ' fibre')

    if timit:
        print('Total time elapsed: ' + str(np.round(time.time() - start_time, 1)) + ' seconds')

    return






























