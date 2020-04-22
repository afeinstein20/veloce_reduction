'''
Created on 7 May 2018

@author: christoph
'''

import os
import glob
import astropy.io.fits as pyfits
import numpy as np

from veloce_reduction.veloce_reduction.helper_functions import laser_on, thxe_on, find_nearest
from veloce_reduction.veloce_reduction.calibration import correct_for_bias_and_dark_from_filename





def get_obstype_lists(pathdict, pattern=None, weeding=True, quick=False, raw_goodonly=True, savefiles=True):
    """
    This routine performs the "INGEST" step, ie for all files in a given night it identifies the type of observation and sorts the files into lists.
    For simcalib exposures it also determines which lamps were actually firing, no matter what the header says, as that can often be wrong (LC / SimTh / LC+SimTh).

    INPUT:
    "pathdict"      : dictionary containing all directories relevant to the reduction
    "pattern"       : if provided, only files containing a certain string pattern will be included
    "weeding"       : boolean - do you want to weed out binned observations?
    "quick"         : boolean - if TRUE, simcalib status in determined from headers alone (not from 2-dim images)
    "raw_goodonly"  : boolean - if TRUE, expect 8-digit date (YYYYMMDD) - if FALSE expect 6-digit date (YYMMDD)
    "savefiles"     : boolean - do you want to save the lists into output files

    OUTPUT:
    lists containing the filenames (incl. directory) of the respective observations of a certain type

    MODHIST:
    20200421 - CMB removed domeflat and skyflat lists (not used with Veloce)
    """

    path = pathdict['raw']
    chipmask_path = pathdict['cm']

    if raw_goodonly:
        date = path[-9:-1]
    else:
        date = '20' + path[-13:-7]

    if pattern is None:
        file_list = glob.glob(path + date[-2:] + "*.fits")
    else:
        file_list = glob.glob(path + '*' + pattern + '*.fits')
    
    
    # first weed out binned observations
    if weeding:
        unbinned = []
        binned = []
        for file in file_list:
            xdim = pyfits.getval(file, 'NAXIS2')
            if xdim == 4112:
                unbinned.append(file)
            else:
                binned.append(file)
    else:
        unbinned = file_list

    # prepare output lists
    if weeding:
        acq_list = binned[:]
    else:
        acq_list = []
    bias_list = []
    dark_list = []
    flat_list = []
    # skyflat_list = []
    # domeflat_list = []
    arc_list = []
    thxe_list = []
    laser_list = []
    laser_and_thxe_list = []
    stellar_list = []
    unknown_list = []

    for file in unbinned:
        obj_type = pyfits.getval(file, 'OBJECT')

        if obj_type.lower() == 'acquire':
            if not weeding:
                acq_list.append(file)
        elif obj_type.lower().startswith('bias'):
            bias_list.append(file)
        elif obj_type.lower().startswith('dark'):
            dark_list.append(file)
        elif obj_type.lower().startswith('flat'):
            flat_list.append(file)
        # elif obj_type.lower().startswith('skyflat'):
        #     skyflat_list.append(file)
        # elif obj_type.lower().startswith('domeflat'):
        #     domeflat_list.append(file)
        elif obj_type.lower().startswith('arc'):
            arc_list.append(file)
        elif obj_type.lower() in ["thxe", "thxe-only", "simth"]:
            thxe_list.append(file)
        elif obj_type.lower() in ["lc", "lc-only", "lfc", "lfc-only", "simlc"]:
            laser_list.append(file)
        elif obj_type.lower() in ["thxe+lfc", "lfc+thxe", "lc+simthxe", "lc+thxe"]:
            laser_and_thxe_list.append(file)
        elif obj_type.lower().startswith(("wasp","proxima","kelt","toi","tic","hd","hr","hip","gj","gl","ast","alpha","beta","gamma",
                                          "delta","tau","ksi","ach","zeta","ek",'1', '2', '3', '4', '5', '6', '7', '8', '9', 'mercury',
                                          'bd', 'bps', 'cd', 'he', 'g', 'cs', 'bkt', 'meingast', 'spangap', 'sarah', 'rm', 'fp', 'vel')):
            stellar_list.append(file)
        else:
            unknown_list.append(file)

    
    # sort out which calibration lamps were actually on for the exposures tagged as either "SimLC" or "SimTh"
    laser_only_list = []
    simth_only_list = []
    laser_and_simth_list = []
    calib_list = laser_list + thxe_list + laser_and_thxe_list
    calib_list.sort()
    
    if quick:
        checkdate = date[:]
    else:
        checkdate = '1' + date[1:]
    
    if int(checkdate) < 20190503:
        # check if chipmask for that night already exists (if not revert to the closest one in time (preferably earlier in time))
        if os.path.isfile(chipmask_path + 'chipmask_' + date + '.npy'):
            chipmask = np.load(chipmask_path + 'chipmask_' + date + '.npy').item()
        else:
            cm_list = glob.glob(chipmask_path + 'chipmask*.npy')
            cm_datelist = [int(cm.split('.')[-2][-8:]) for cm in cm_list]
            cm_datelist.sort()   # need to make sure it is sorted, so that find_nearest finds the earlier one in time if two dates are found that have the same delta_t to date
            cm_dates = np.array(cm_datelist)
            alt_date = find_nearest(cm_dates, int(date))
            chipmask = np.load(chipmask_path + 'chipmask_' + str(alt_date) + '.npy').item()
            
        # look at the actual 2D image (using chipmasks for LFC and simThXe) to determine which calibration lamps fired
        for file in calib_list:
            img = correct_for_bias_and_dark_from_filename(file, np.zeros((4096,4112)), np.zeros((4096,4112)), gain=[1., 1.095, 1.125, 1.], scalable=False, savefile=False, pathdict=pathdict)
            lc = laser_on(img, chipmask)
            thxe = thxe_on(img, chipmask)
            if (not lc) and (not thxe):
                unknown_list.append(file)
            elif (lc) and (thxe):
                laser_and_simth_list.append(file)
            else:
                if lc:
                    laser_only_list.append(file)
                elif thxe:
                    simth_only_list.append(file)
    else:
        # since May 2019 the header keywords are (mostly) correct, so could check for LFC / ThXe in header, as that is MUCH faster    
        for file in calib_list:
            lc = 0
            thxe = 0
            h = pyfits.getheader(file)
            if 'LCNEXP' in h.keys():   # this indicates the latest version of the FITS headers (from May 2019 onwards)
                if ('LCEXP' in h.keys()) or ('LCMNEXP' in h.keys()):   # this indicates the LFC actually was actually exposed (either automatically or manually)
                    lc = 1
            else:   # if not, just go with the OBJECT field
                if file in laser_list + laser_and_thxe_list:
                    lc = 1
            if (h['SIMCALTT'] > 0) and (h['SIMCALN'] > 0) and (h['SIMCALSE'] > 0):
                thxe = 1
            if lc+thxe == 1:
                if lc == 1:
                    laser_only_list.append(file)
                else:
                    simth_only_list.append(file)
            elif lc+thxe == 2:
                laser_and_simth_list.append(file)
            else:
                unknown_list.append(file)
        
    # sort all lists
    acq_list.sort()
    bias_list.sort()
    dark_list.sort()
    flat_list.sort()
    arc_list.sort()
    simth_only_list.sort()
    laser_only_list.sort()
    laser_and_simth_list.sort()
    stellar_list.sort()
    unknown_list.sort()

    if savefiles:
        shortfn_acq_list = [fn.split('/')[-1] for fn in acq_list]
        np.savetxt(path + date + '_acquire_list.txt', shortfn_acq_list, fmt='%s')
        shortfn_bias_list = [fn.split('/')[-1] for fn in bias_list]
        np.savetxt(path + date + '_bias_list.txt', shortfn_bias_list, fmt='%s')
        shortfn_dark_list = [fn.split('/')[-1] for fn in dark_list]
        np.savetxt(path + date + '_dark_list.txt', shortfn_dark_list, fmt='%s')
        shortfn_flat_list = [fn.split('/')[-1] for fn in flat_list]
        np.savetxt(path + date + '_flat_list.txt', shortfn_flat_list, fmt='%s')
        shortfn_arc_list = [fn.split('/')[-1] for fn in arc_list]
        np.savetxt(path + date + '_arc_list.txt', shortfn_arc_list, fmt='%s')
        shortfn_simth_only_list = [fn.split('/')[-1] for fn in simth_only_list]
        np.savetxt(path + date + '_simth_only_list.txt', shortfn_simth_only_list, fmt='%s')
        shortfn_laser_only_list = [fn.split('/')[-1] for fn in laser_only_list]
        np.savetxt(path + date + '_lfc_only_list.txt', shortfn_laser_only_list, fmt='%s')
        shortfn_laser_and_simth_list = [fn.split('/')[-1] for fn in laser_and_simth_list]
        np.savetxt(path + date + '_lfc_and_simth_list.txt', shortfn_laser_and_simth_list, fmt='%s')
        shortfn_stellar_list = [fn.split('/')[-1] for fn in stellar_list]
        np.savetxt(path + date + '_stellar_list.txt', shortfn_stellar_list, fmt='%s')
        shortfn_unknown_list = [fn.split('/')[-1] for fn in unknown_list]
        np.savetxt(path + date + '_unknown_list.txt', shortfn_unknown_list, fmt='%s')

    # return acq_list, bias_list, dark_list, flat_list, skyflat_list, domeflat_list, arc_list, simth_only_list, laser_only_list, laser_and_simth_list, stellar_list, unknown_list
    return acq_list, bias_list, dark_list, flat_list, arc_list, simth_only_list, laser_only_list, laser_and_simth_list, stellar_list, unknown_list





def identify_obstypes(path):
    """
    DUMMY ROUTINE: NOT CURRENTLY USED
    Identify the type of observation from the card in the FITS header, and create lists of filename for the different observation types.
    """

    file_list = glob.glob(path + "*.fits")

    bias_list = []
    dark_list = []
    sky_list = []
    # lflat_list = []
    skyflat_list = []
    # arc_list = []
    fibre_flat_list = []
    unknown_list = []
    # white_list = []
    thar_list = []
    thxe_list = []
    laser_list = []
    stellar_list = []

    for file in file_list:
        h = pyfits.getheader(file)
        # type = h['exptype']
        obstype = h['NDFCLASS']

        if obstype.upper() == 'BIAS':
            bias_list.append(file)
        elif obstype.upper() == 'DARK':
            dark_list.append(file)
        elif obstype.upper() == 'MFSKY':
            sky_list.append(file)
        # elif obstype.upper() == 'LFLAT':
        #     lflat_list.append(file)
        elif obstype.upper() == 'SFLAT':
            skyflat_list.append(file)
        # elif obstype.upper() == 'MFARC':
        #     arc_list.append(file)
        elif obstype.upper() == 'MFFFF':
            fibre_flat_list.append(file)
        elif obstype.upper() == 'MFOBJECT':
            if h['OBJECT'].lower() == 'thar':
                thar_list.append(file)
            elif h['OBJECT'].lower() == 'thxe':
                thxe_list.append(file)
            elif h['OBJECT'].lower() == 'laser':
                laser_list.append(file)
            else:
                stellar_list.append(file)
        # elif obstype.upper() in ('FLAT', 'WHITE'):
        #     white_list.append(file)
        # elif obstype.upper() == 'THAR':
        #     thar_list.append(file)
        # elif obstype.upper() == 'THXE':
        #     thxe_list.append(file)
        # elif obstype.upper() == 'LASER':
        #     laser_list.append(file)
        # elif obstype.upper() == 'STELLAR':
        #     stellar_list.append(file)
        else:
            print('WARNING: unknown exposure type encountered for', file)
            unknown_list.append(file)

    return bias_list, dark_list, sky_list, skyflat_list, fibre_flat_list, thar_list, thxe_list, laser_list, stellar_list, unknown_list





def get_obstype_lists_temp(path, pattern=None, weeding=True):
    """DUMMY ROUTINE: NOT CURRENTLY USED"""

    if pattern is None:
        file_list = glob.glob(path + "*.fits")
    else:
        file_list = glob.glob(path + '*' + pattern + '*.fits')
    
    
    # first weed out binned observations
    if weeding:
        unbinned = []
        binned = []
        for file in file_list:
            xdim = pyfits.getval(file, 'NAXIS2')
            if xdim == 4112:
                unbinned.append(file)
            else:
                binned.append(file)
    else:
        unbinned = file_list

    # prepare output lists
    if weeding:
        acq_list = binned[:]
    else:
        acq_list = []
    bias_list = []
    dark_list = []
    flat_list = []
    skyflat_list = []
    domeflat_list = []
    arc_list = []
    thxe_list = []
    laser_list = []
    laser_and_thxe_list = []
    stellar_list = []
    unknown_list = []

    for file in unbinned:
        obj_type = pyfits.getval(file, 'OBJECT')

        if obj_type.lower() == 'acquire':
            if not weeding:
                acq_list.append(file)
        elif obj_type.lower().startswith('bias'):
            bias_list.append(file)
        elif obj_type.lower().startswith('dark'):
            dark_list.append(file)
        elif obj_type.lower().startswith('flat'):
            flat_list.append(file)
        elif obj_type.lower().startswith('skyflat'):
            skyflat_list.append(file)
        elif obj_type.lower().startswith('domeflat'):
            domeflat_list.append(file)
        elif obj_type.lower().startswith('arc'):
            arc_list.append(file)
        elif obj_type.lower() in ["thxe","thxe-only", "simth"]:
            thxe_list.append(file)
        elif obj_type.lower() in ["lc","lc-only","lfc","lfc-only", "simlc"]:
            laser_list.append(file)
        elif obj_type.lower() in ["thxe+lfc","lfc+thxe","lc+simthxe","lc+thxe"]:
            laser_and_thxe_list.append(file)
        elif obj_type.lower().startswith(("wasp","proxima","kelt","toi","tic","hd","hr","hip","gj","gl","ast","alpha","beta","gamma",
                                          "delta","tau","ksi","ach","zeta","ek",'1', '2', '3', '4', '5', '6', '7', '8', '9')):
            stellar_list.append(file)
        else:
            unknown_list.append(file)

    return acq_list, bias_list, dark_list, flat_list, skyflat_list, domeflat_list, arc_list, thxe_list, laser_list, laser_and_thxe_list, stellar_list, unknown_list





def get_obs_coords_from_header(fn):
    """DUMMY ROUTINE: NOT CURRENTLY USED"""
    h = pyfits.getheader(fn)
    lat = h['LAT_OBS']
    long = h['LONG_OBS']
    alt = h['ALT_OBS']
    return lat,long,alt
    




