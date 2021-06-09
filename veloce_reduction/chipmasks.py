'''
Created on 8 Oct. 2018

@author: christoph
'''
import numpy as np
import time
from scipy.ndimage import label


__all__ = ['make_single_chipmask', 'old_make_single_chipmask', 'get_mean_fibre_separation', 'make_chipmask']


def make_single_chipmask(fibparms, meansep, masktype='stellar', exclude_top_and_bottom=True, nx=4112, ny=4096,
                         use_lfc=False, debug_level=0, timit=False):
    if timit:
        start_time = time.time()

    # some housekeeping...
    while masktype.lower() not in ["stellar", "sky2", "sky3", "lfc", "thxe", "background", "bg"]:
        print('ERROR: chipmask "type" not recognized!!!')
        masktype = raw_input(
            'Please enter "type" - valid options are ["object" / "sky2" / "sky3" / "LFC" / "ThXe" / "background" or "bg"]:  ')

    # ny, nx = img.shape
    chipmask = np.zeros((ny, nx))
    xx = np.arange(nx)
    yy = np.arange(ny)
    XX, YY = np.meshgrid(xx, yy)

    # now identify regions of the chip
    for order in sorted(fibparms.keys()):

        if debug_level >= 1:
            print('Checking ' + order)

        if masktype.lower() == 'stellar':
            # for the object-fibres chipmask, take the middle between the last object fibre and first sky fibre at each end (ie the "gaps")
            f_upper = 0.5 * (fibparms[order]['fibre_04']['mu_fit'] + fibparms[order]['fibre_06']['mu_fit'])
            f_lower = 0.5 * (fibparms[order]['fibre_24']['mu_fit'] + fibparms[order]['fibre_26']['mu_fit'])
        elif masktype.lower() == 'sky2':
            # for the 2 sky fibres near the ThXe, we use the "gap" as the upper bound, and as the lower bound either (i) the midpoint between the lowest sky2 fibre and
            # the simThXe fibre, or (ii) the trace of the lowermost sky fibre minus half the average fibre separation for this order and pixel location
            f_upper = 0.5 * (fibparms[order]['fibre_24']['mu_fit'] + fibparms[order]['fibre_26']['mu_fit'])
            try:
                f_lower = 0.5 * (fibparms[order]['fibre_27']['mu_fit'] + fibparms[order]['fibre_28']['mu_fit'])
            except:
                f_lower = 1 * fibparms[order]['fibre_27']['mu_fit']     # the multiplication with one acts like a copy
                f_lower -= 0.5 * meansep[order]
        elif masktype.lower() == 'sky3':
            # for the 3 sky fibres near the LFC, we use the "gap" as the lower bound, and as the upper bound either (i) the midpoint between the uppermost sky3 fibre and
            # the LFC fibre, or (ii) the trace of the uppermost sky fibre plus half the average fibre separation for this order and pixel location
            if use_lfc:
                try:
                    f_upper = 0.5 * (fibparms[order]['fibre_02']['mu_fit'] + fibparms[order]['fibre_01']['mu_fit'])
                except:
                    f_upper = 1 * fibparms[order]['fibre_02']['mu_fit']     # the multiplication with one acts like a copy
                    f_upper += 0.5 * meansep[order]
            else:
                f_upper = 1 * fibparms[order]['fibre_02']['mu_fit']     # the multiplication with one acts like a copy
                f_upper += 0.5 * meansep[order]
            f_lower = 0.5 * (fibparms[order]['fibre_04']['mu_fit'] + fibparms[order]['fibre_06']['mu_fit'])
        elif masktype.lower() == 'lfc':
            # for the LFC fibre, we assume as the lower bound either (i) the midpoint between the uppermost sky3 fibre and the LFC fibre, or (ii) the trace of the uppermost 
            # sky fibre plus half the average fibre separation for this order and pixel location; as the upper bound either (i) the trace of the LFC fibre plus half the average 
            # fibre separation for this order and pixel location, or (ii) the trace of the uppermost sky3 fibre plus one and a half times the average fibre separation for this
            # order and pixel location
            if use_lfc:
                try:
                    f_upper = 1 * fibparms[order]['fibre_01']['mu_fit']     # the multiplication with one acts like a copy
                    f_upper += 0.5 * meansep[order]
                    f_lower = 0.5 * (fibparms[order]['fibre_01']['mu_fit'] + fibparms[order]['fibre_02']['mu_fit'])
                except:
                    f_upper = 1 * fibparms[order]['fibre_02']['mu_fit']     # the multiplication with one acts like a copy
                    f_upper += 1.5 * meansep[order]
                    f_lower = 1 * fibparms[order]['fibre_02']['mu_fit']     # the multiplication with one acts like a copy
                    f_lower += 0.5 * meansep[order]
            else:
                f_upper = 1 * fibparms[order]['fibre_02']['mu_fit']     # the multiplication with one acts like a copy
                f_upper += 1.5 * meansep[order]
                f_lower = 1 * fibparms[order]['fibre_02']['mu_fit']     # the multiplication with one acts like a copy
                f_lower += 0.5 * meansep[order]
        elif masktype.lower() == 'thxe':
            # for the simThXe fibre, we assume as the lower bound either (i) the trace of the simThXe fibre minus half the average fibre separation for this order and pixel 
            # location, or (ii) the trace of the lowermost sky2 fibre minus one and a half times the average fibre separation for this order and pixel location; as the upper
            # bound either (i) the midpoint between the lowermost sky2 fibre and the simThXe fibre, or (ii) the trace of the lowermost sky2 fibre minus one a half times the 
            # average fibre separation for this order and pixel location
            try:
                f_upper = 0.5 * (fibparms[order]['fibre_27']['mu_fit'] + fibparms[order]['fibre_28']['mu_fit'])
                f_lower = 1 * fibparms[order]['fibre_28']['mu_fit']     # the multiplication with one acts like a copy
                f_lower -= 0.5 * meansep[order]
            except:
                f_upper = 1 * fibparms[order]['fibre_27']['mu_fit']     # the multiplication with one acts like a copy
                f_upper -= 0.5 * meansep[order]
                f_lower = 1 * fibparms[order]['fibre_27']['mu_fit']     # the multiplication with one acts like a copy
                f_lower -= 1.5 * meansep[order]
        elif masktype.lower() in ['background', 'bg']:
            # could either do sth like 1. - np.sum(chipmask_i), but we can also just use the lower bound of ThXe and the upper bound of LFC
            # identify what is NOT background, and later "invert" that
            if use_lfc:
                try:
                    f_upper = 1 * fibparms[order]['fibre_01']['mu_fit']     # the multiplication with one acts like a copy
                    f_upper += 1. * meansep[order]     # use whole meansep here to avoid contamination from the calib sources a bit more
                    f_lower = 1 * fibparms[order]['fibre_28']['mu_fit']     # the multiplication with one acts like a copy
                    f_lower -= 1. * meansep[order]     # use whole meansep here to avoid contamination from the calib sources a bit more
                except:
                    f_upper = 1 * fibparms[order]['fibre_02']['mu_fit']     # the multiplication with one acts like a copy
                    f_upper += 2. * meansep[order]     # use 2 here to avoid contamination from the calib sources a bit more
                    f_lower = 1 * fibparms[order]['fibre_27']['mu_fit']     # the multiplication with one acts like a copy
                    f_lower -= 2. * meansep[order]     # use 2 here to avoid contamination from the calib sources a bit more
            else:
                f_upper = 1 * fibparms[order]['fibre_02']['mu_fit']     # the multiplication with one acts like a copy
                f_upper += 2. * meansep[order]     # use 2 here to avoid contamination from the calib sources a bit more
                try:
                    f_lower = 1 * fibparms[order]['fibre_28']['mu_fit']     # the multiplication with one acts like a copy
                    f_lower -= 1. * meansep[order]  
                except:
                    f_lower = 1 * fibparms[order]['fibre_27']['mu_fit']     # the multiplication with one acts like a copy
                    f_lower -= 2. * meansep[order]     # use 2 here to avoid contamination from the calib sources a bit more

        # get indices of the pixels that fall into the respective regions
        order_stripe = (YY < f_upper) & (YY > f_lower)
        chipmask = np.logical_or(chipmask, order_stripe)

    # for the background we have to invert that mask and (optionally) exclude the top and bottom regions
    # which still include fainter orders etc (same as in "extract_background")
    if masktype.lower() in ['bg', 'background']:
        chipmask = np.invert(chipmask)
        if exclude_top_and_bottom:
            if debug_level >= 1:
                print('WARNING: this fix works for the current Veloce CCD layout only!!!')
            labelled_mask, nobj = label(chipmask)
            # WARNING: this fix works for the current Veloce CCD layout only!!!
            topleftnumber = labelled_mask[ny - 1, 0]
            # toprightnumber = labelled_mask[ny-1,nx-1]
            # bottomleftnumber = labelled_mask[0,0]
            bottomrightnumber = labelled_mask[0, nx - 1]
            chipmask[labelled_mask == topleftnumber] = False
            # chipmask[labelled_mask == toprightnumber] = False
            chipmask[labelled_mask == bottomrightnumber] = False

    return chipmask



def old_make_single_chipmask(fibparms, meansep, masktype='stellar', exclude_top_and_bottom=False, nx=4112, ny=4096):

    """OLD ROUTINE, NOT CURRENTLY IN USE!!!"""

    # some housekeeping...
    while masktype.lower() not in ["stellar", "sky2", "sky3", "lfc", "thxe", "background", "bg"]:
        print('ERROR: chipmask "type" not recognized!!!')
        masktype = raw_input(
            'Please enter "type" - valid options are ["object" / "sky2" / "sky3" / "LFC" / "ThXe" / "background" or "bg"]:  ')

    # ny, nx = img.shape
    chipmask = np.zeros((ny, nx))

    # now actually make the "chipmask"
    for pix in np.arange(nx):

        if (pix + 1) % 100 == 0:
            print('Pixel column ' + str(pix + 1) + '/4112')

        for order in sorted(fibparms.keys()):

            if masktype.lower() == 'stellar':
                # for the object-fibres chipmask, take the middle between the last object fibre and first sky fibre at each end (ie the "gaps")
                f_upper = 0.5 * (fibparms[order]['fibre_04']['mu_fit'] + fibparms[order]['fibre_06']['mu_fit'])
                f_lower = 0.5 * (fibparms[order]['fibre_24']['mu_fit'] + fibparms[order]['fibre_26']['mu_fit'])
            elif masktype.lower() == 'sky2':
                # for the 2 sky fibres near the ThXe, we use the "gap" as the upper bound, and as the lower bound the trace of the lowermost sky fibre
                # minus half the average fibre separation for this order and pixel location, ie halfway between the lowermost sky fibre and the ThXe fibre
                f_upper = 0.5 * (fibparms[order]['fibre_24']['mu_fit'] + fibparms[order]['fibre_26']['mu_fit'])
                f_lower = 1 * fibparms[order]['fibre_27']['mu_fit']  # the multiplication with one acts like a copy
                f_lower.coefficients[-1] -= 0.5 * meansep[order][pix]
            elif masktype.lower() == 'sky3':
                # for the 3 sky fibres near the LFC, we use the "gap" as the lower bound, and as the upper bound the trace of the uppermost sky fibre
                # plus half the average fibre separation for this order and pixel location, ie halfway between the uppermost sky fibre and the LFC fibre
                f_upper = 1 * fibparms[order]['fibre_02']['mu_fit']  # the multiplication with one acts like a copy
                f_upper.coefficients[-1] += 0.5 * meansep[order][pix]
                f_lower = 0.5 * (fibparms[order]['fibre_04']['mu_fit'] + fibparms[order]['fibre_06']['mu_fit'])
            elif masktype.lower() == 'lfc':
                # for the LFC fibre, we assume as the lower bound the trace of the uppermost sky fibre plus half the average fibre separation for this order and pixel location,
                # ie halfway between the uppermost sky fibre and the LFC fibre, and as the upper bound the trace of the uppermost sky fibre plus two times the average
                # fibre separation for this order and pixel location
                f_upper = 1 * fibparms[order]['fibre_02']['mu_fit']
                f_upper.coefficients[-1] += 2. * meansep[order][pix]
                f_lower = 1 * fibparms[order]['fibre_02']['mu_fit']  # the multiplication with one acts like a copy
                f_lower.coefficients[-1] += 0.5 * meansep[order][pix]
            elif masktype.lower() == 'thxe':
                # for the ThXe fibre, we assume as the upper bound the trace of the lowermost sky fibre minus half the average fibre separation for this order and pixel location,
                # ie halfway between the lowermost sky fibre and the ThXe fibre, and as the lower bound the trace of the lowermost sky fibre minus two times the average
                # fibre separation for this order and pixel location
                f_upper = 1 * fibparms[order]['fibre_27']['mu_fit']
                f_upper.coefficients[-1] -= 0.5 * meansep[order][pix]
                f_lower = 1 * fibparms[order]['fibre_27']['mu_fit']  # the multiplication with one acts like a copy
                f_lower.coefficients[-1] -= 2. * meansep[order][pix]
            elif masktype.lower() in ['background', 'bg']:
                # could either do sth like 1. - np.sum(chipmask_i), but can also just use the lower bound of ThXe and the upper bound of LFC
                f_upper = 1 * fibparms[order]['fibre_02']['mu_fit']
                f_upper.coefficients[-1] += 2. * meansep[order][pix]
                f_lower = 1 * fibparms[order]['fibre_27']['mu_fit']  # the multiplication with one acts like a copy
                f_lower.coefficients[-1] -= 2. * meansep[order][pix]
            else:
                print('ERROR: Nightmare! That should never happen  --  must be an error in the Matrix...')
                return

            ymin = f_lower(pix)
            ymax = f_upper(pix)

            # these are the pixels that fall completely in the range
            # NOTE THAT THE CO-ORDINATES ARE CENTRED ON THE PIXELS, HENCE THE 0.5s...
            full_range = np.arange(np.maximum(np.ceil(ymin + 0.5), 0),
                                   np.minimum(np.floor(ymax - 0.5) + 1, ny - 1)).astype(int)
            if len(full_range) > 0:
                chipmask[full_range, pix] = 1.

            # bottom edge pixel
            if ymin > -0.5 and ymin < ny - 1 + 0.5:
                qlow = np.ceil(ymin - 0.5) - ymin + 0.5
                chipmask[np.floor(ymin + 0.5).astype(int), pix] = qlow

            # top edge pixel
            if ymax > -0.5 and ymax < ny - 1 + 0.5:
                qtop = ymax - np.floor(ymax - 0.5) - 0.5
                chipmask[np.ceil(ymax - 0.5).astype(int), pix] = qtop

    # for the background we have to invert that mask and (optionally) exclude the top and bottom regions
    # which still include fainter orders etc (same as in "extract_background")
    if masktype.lower() == 'background':
        chipmask = 1. - chipmask
        if exclude_top_and_bottom:
            print('WARNING: this fix works for the current Veloce CCD layout only!!!')
            labelled_mask, nobj = label(chipmask)
            # WARNING: this fix works for the current Veloce CCD layout only!!!
            topleftnumber = labelled_mask[ny - 1, 0]
            # toprightnumber = labelled_mask[ny-1,nx-1]
            # bottomleftnumber = labelled_mask[0,0]
            bottomrightnumber = labelled_mask[0, nx - 1]
            chipmask[labelled_mask == topleftnumber] = False
            # chipmask[labelled_mask == toprightnumber] = False
            chipmask[labelled_mask == bottomrightnumber] = False

    return chipmask



def get_mean_fibre_separation(fibparms, nx=4112):
    # nord = len(fibparms)
    meansep = {}
    xx = np.arange(nx)
    nfib = len(fibparms[fibparms.keys()[0]])
    
    assert nfib in [24,26], 'ERORR: FIBPARMS structure does not contain exactly 24 or exactly 26 fibres!!!'
    if nfib == 24:
        multipliers = np.r_[np.repeat(1, 2), 0.5, np.repeat(1, 18), 0.5, 1]
    if nfib == 26:
        multipliers = np.r_[np.repeat(1, 3), 0.5, np.repeat(1, 18), 0.5, np.repeat(1, 2)]
    
    for o in sorted(fibparms.keys()):
        mu = np.zeros((nfib, nx))
        for i, fib in enumerate(sorted(fibparms[o].keys())):
            # mu[i,:] = fibparms[o][fib]['mu_fit'](xx)
            mu[i, :] = fibparms[o][fib]['mu_fit']
            # get differences (ie fibre separations) and multiply some values with 0.5 because of the gaps between stellar and sky fibres
            diff = np.abs(np.diff(mu, axis=0)) * np.vstack([multipliers] * nx).T
        meansep[o] = np.average(diff, axis=0)

    return meansep



def make_chipmask(date, pathdict=None, savefile=False, combined_fibparms=True, timit=False, path=None):

    if pathdict is None and path is not None:
        archive_path = path
        outpath = path
    else:
        archive_path = pathdict['fp']
        outpath = pathdict['cm']

#    assert pathdict is not None, 'ERROR: path dictionary not provided!!!'

    print('Creating chipmask for ' + date + '...')

    if timit:
        start_time = time.time()

    object_list = ['stellar', 'sky2', 'sky3', 'thxe', 'lfc', 'bg']

    chipmask = {}

#    archive_path = pathdict['fp'] + 'archive/'
#    outpath = pathdict['cm']

    if combined_fibparms:
        fibparms = np.load(archive_path + 'combined_fibre_profile_fits_' + date + '.npy').item()
    else:
        fibparms = np.load(archive_path + 'fibre_profile_fits_' + date + '.npy').item()

    meansep = get_mean_fibre_separation(fibparms)

    for object in object_list:
        chipmask[object] = make_single_chipmask(fibparms, meansep, masktype=object)

    # stellar_chipmask = make_single_chipmask(fibparms, meansep, masktype='stellar')
    # sky2_chipmask = make_single_chipmask(fibparms, meansep, masktype='sky2')
    # sky3_chipmask = make_single_chipmask(fibparms, meansep, masktype='sky3')
    # thxe_chipmask = make_single_chipmask(fibparms, meansep, masktype='thxe')
    # lfc_chipmask = make_single_chipmask(fibparms, meansep, masktype='lfc')
    # bg_chipmask = make_single_chipmask(fibparms, meansep, masktype='background', exclude_top_and_bottom=False)
    # bg_chipmask_excl = make_single_chipmask(fibparms, meansep, masktype='background', exclude_top_and_bottom=True)

    if savefile:
        np.save(outpath + 'chipmask_' + date + '.npy', chipmask)

    if timit:
        print('Time elapsed: ' + str(np.round(time.time() - start_time, 1)) + ' seconds')

    return chipmask
