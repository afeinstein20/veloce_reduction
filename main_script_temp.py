import glob
import astropy.io.fits as pyfits
import numpy as np
import datetime
import copy
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from veloce_reduction.veloce_reduction.get_info_from_headers import get_obstype_lists
from veloce_reduction.veloce_reduction.helper_functions import short_filenames, laser_on, thxe_on, correct_orientation
from veloce_reduction.veloce_reduction.calibration import correct_for_bias_and_dark_from_filename, get_bias_and_readnoise_from_bias_frames, make_master_calib, \
make_ronmask, make_master_dark, make_master_darks, crop_overscan_region
from veloce_reduction.veloce_reduction.order_tracing import find_stripes, make_P_id, make_mask_dict, extract_stripes, make_order_traces_from_fibparms
# from veloce_reduction.veloce_reduction.spatial_profiles import fit_profiles, fit_profiles_from_indices
from veloce_reduction.veloce_reduction.flat_fielding import onedim_pixtopix_variations_spline
from veloce_reduction.veloce_reduction.profile_tests import fit_multiple_profiles_from_indices
from veloce_reduction.veloce_reduction.get_profile_parameters import make_real_fibparms_by_ord, combine_fibparms
from veloce_reduction.veloce_reduction.chipmasks import make_chipmask
from veloce_reduction.veloce_reduction.extraction import extract_spectrum_from_indices
from veloce_reduction.veloce_reduction.wavelength_solution import make_master_fibth, make_arc_dispsols
from process_scripts import process_whites, process_science_images


# THIS IS THE ONLY FIELD YOU NEED TO CHANGE ON A NIGHT BY NIGHT BASIS
date = '20191023'

# THIS BLOCK NEEDS TO BE SET UP JUST ONCE
pathdict = {}
pathdict['raw'] = '/Volumes/BERGRAID/data/veloce/raw_goodonly/' + date + '/'
pathdict['red'] = '/Volumes/BERGRAID/data/veloce/reduced/' + date + '/'
# pathdict['raw'] = '/Users/christoph/data/raw_goodonly/' + date + '/'
# pathdict['red'] = '/Users/christoph/data/reduced/' + date + '/'
pathdict['cm'] = '/Users/christoph/OneDrive - UNSW/chipmasks/archive/'
pathdict['fp'] = '/Users/christoph/OneDrive - UNSW/fibre_profiles/'
pathdict['lfc'] = '/Volumes/BERGRAID/data/veloce/lfc_peaks/'
pathdict['dispsol'] = '/Users/christoph/OneDrive - UNSW/dispsol/'
pathdict['root'] = '/Users/christoph/OneDrive - UNSW/'


### (0) GET INFO FROM FITS HEADERS ##################################################################################################################
# do a stock take of ALL FITS files in that folder
all_raw_files = glob.glob(pathdict['raw'] + date[-2:] + "*.fits")
all_raw_files.sort()
print('Identifying ' + str(len(all_raw_files)) + ' raw FITS files...')

# divide into lists according to exposure type
acq_list, bias_list, dark_list, flat_list, arc_list, thxe_list, laser_list, laser_and_thxe_list, stellar_list, unknown_list = get_obstype_lists(pathdict)
assert len(unknown_list) == 0, "WARNING: unknown files encountered!!!"

# obsnames = short_filenames(bias_list)
dumimg = crop_overscan_region(correct_orientation(pyfits.getdata(bias_list[0])))
ny,nx = dumimg.shape
del dumimg
#####################################################################################################################################################

# check white light exposures      
        
for file in flat_list:
#     fimg = crop_overscan_region(correct_orientation(pyfits.getdata(file)))
    fimg = correct_for_bias_and_dark_from_filename(file, np.zeros((4096,4112)), np.zeros((4096,4112)), gain=[1., 1.095, 1.125, 1.], scalable=False, savefile=False, path=pathdict['raw'])
#     plt.plot(fimg[165:230,3327])
#     plt.plot(fimg[1735:1802,2000])   # Q1
#     plt.plot(fimg[1650:1717,2709])   # Q2
    plt.plot(fimg[2305:2372,2709])   # Q3
#     plt.plot(fimg[2278:2345,2000])   # Q4


### (1) BAD PIXEL MASK ##############################################################################################################################
# bpm_list = glob.glob(pathdict['raw'] + '*bad_pixel_mask*')
# # read most recent bad pixel mask
# bpm_dates = [x[-12:-4] for x in bpm_list]
# most_recent_datestring = sorted(bpm_dates)[-1]
# bad_pixel_mask = np.load(pathdict['raw'] + 'bad_pixel_mask_' + most_recent_datestring + '.npy')
# 
# # update the pixel mask
# #blablabla
# 
# # save updated bad pixel mask
# now = datetime.datetime.now()
# dumstring = str(now)[:10].split('-')
# datestring = ''.join(dumstring)
# np.save(pathdict['raw']+'bad_pixel_mask_'+datestring+'.npy', bad_pixel_mask)
# ###################################################################################################################################################



### (2) CALIBRATIONS ################################################################################################################################
# gain = [0.88, 0.93, 0.99, 0.93]   # from "VELOCE_DETECTOR_REPORT_V1.PDF"
# gain = [1., 1., 1., 1.]
gain = [1., 1.095, 1.125, 1.]   # eye-balled from extracted flat fields

# (i) BIAS and READ NOISE
# check if MEDIAN BIAS already exists
choice = 'r'
if os.path.isfile(pathdict['raw'] + date + '_median_bias.fits'):
    choice = raw_input("MEDIAN BIAS image for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
if choice.lower() == 's':
    print('Loading MASTER BIAS for ' + date + '...')
    medbias = pyfits.getdata(pathdict['raw'] + date + '_median_bias.fits')
else:
    # get offsets and read-out noise from bias frames (units: [offsets] = ADUs; [RON] = e-)
    if len(bias_list) > 9:
        bias_list = sorted(bias_list)[0:9]
    else:
        bias_list = sorted(bias_list)
    medbias,coeffs,offsets,rons = get_bias_and_readnoise_from_bias_frames(bias_list, degpol=5, clip=5., gain=gain, save_medimg=True, debug_level=1, timit=True)

# check if read noise mask already exists
choice = 'r'
if os.path.isfile(pathdict['raw'] + date + '_read_noise_mask.fits'):
    choice = raw_input("READ NOISE FRAME for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
if choice.lower() == 's':
    print('Loading READ NOISE FRAME for ' + date + '...')
    ronmask = pyfits.getdata(pathdict['raw'] + date + '_read_noise_mask.fits')
else:
    ronmask = make_ronmask(rons, nx, ny, gain=gain, savefile=True, path=pathdict['raw'], timit=True)



# (ii) DARKS
# create (bias-subtracted) MASTER DARK frames (units = electrons)
# use make_master_darks() (ie plural) from now on!!!!!
# make_master_darks(dark_list, MB=medbias, gain=gain)
MDS = np.zeros(medbias.shape)



# (iii) WHITES 
# create (bias- & dark-subtracted) MASTER WHITE frame and corresponding error array (units = electrons)
choice_mw = 'r'
if os.path.isfile(pathdict['raw'] + date + '_master_white.fits'):
    choice_mw = raw_input("MASTER WHITE image for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
if choice_mw.lower() == 's':
    print('Loading MASTER WHITE for ' + date + '...')
    MW = pyfits.getdata(pathdict['raw'] + date + '_master_white.fits', 0)
    err_MW = pyfits.getdata(pathdict['raw'] + date + '_master_white.fits', 1)
else:
    # this is a first iteration without background removal - just so we can do the tracing; then we come back and do it properly later
    MW,err_MW = process_whites(flat_list, MB=medbias, ronmask=ronmask, MD=MDS, gain=gain, scalable=True, fancy=False, P_id=None,
                               clip=5., savefile=False, saveall=False, diffimg=False, remove_bg=False, path=pathdict['raw'], debug_level=1, timit=False)
    
#####################################################################################################################################################



### (3) INITIAL ORDER TRACING #######################################################################################################################
choice = 'r'
if os.path.isfile(pathdict['raw'] + date + '_P_id.npy') and os.path.isfile(pathdict['raw'] + date + '_mask.npy'):
    choice = raw_input("INITIAL ORDER TRACING has already been done for " + date + " ! Do you want to skip this step or recreate it? ['s' / 'r']")
if choice.lower() == 's':
    print('Loading initial order traces for ' + date + '...')
    P_id = np.load(pathdict['raw'] + date + '_P_id.npy').item()
    mask = np.load(pathdict['raw'] + date + '_mask.npy').item()
else:
    # find rough order locations
    #P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, gauss_filter_sigma=3., simu=False)
    P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, gauss_filter_sigma=10., simu=False, maskthresh = 400)
    # if the bad pixel column is found as an order:
    # del P[5]
    # tempmask = tempmask[np.r_[0:5, 6:40],:]
    # assign physical diffraction order numbers (this is only a dummy function for now) to order-fit polynomials and bad-region masks
    P_id_dum = make_P_id(P)
    assert len(P_id_dum) == 39, 'ERROR: not exactly 39 orders found!!!'
    mask = make_mask_dict(tempmask)
    P_id = copy.deepcopy(P_id_dum)
    # for the dates where fibre 8 had a ~20-30% drop in throughput, do NOT subtract 2
    if int(date) == 20190414:
        for o in P_id.keys():
            P_id[o][0] -= 1.
    elif (int(date) <= 20190203) or (int(date) >= 20190619):
        for o in P_id.keys():
            P_id[o][0] -= 2.
    np.save(pathdict['raw'] + date + '_P_id.npy', P_id)
    np.save(pathdict['raw'] + date + '_mask.npy', mask)
    

# now redo the master white properly, incl background removal, and save to file
if choice_mw.lower() == 'r':
    MW,err_MW = process_whites(flat_list, MB=medbias, ronmask=ronmask, MD=MDS, gain=gain, scalable=True, fancy=False, P_id=P_id,
                               clip=5., savefile=True, saveall=False, diffimg=False, remove_bg=True, path=pathdict['raw'], debug_level=1, timit=False)

#####################################################################################################################################################


### (4) CREATE FIBRE PROFILES #######################################################################################################################
slit_height = 30
MW_stripes,MW_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=slit_height)
del MW_stripes     # save memory
# err_MW_stripes = extract_stripes(err_MW, P_id, return_indices=False, slit_height=slit_height)

choice_fp = 'r'
if os.path.isfile(pathdict['fp'] + 'archive/' + 'combined_fibre_profile_fits_' + date + '.npy'):
    choice_fp = raw_input("FIBRE PROFILES for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
if choice_fp.lower() == 's':
    print('Loading fibre profiles for ' + date + '...')
    fibparms = np.load(pathdict['fp'] + 'archive/' + 'combined_fibre_profile_fits_' + date + '.npy').item()
else:
    sure = 'x'
    while sure.lower() not in ['y','n']:
        sure = raw_input("Are you sure you want to create fibre profiles for " + date + "??? This may take several hours!!! ['y' / 'n']")
    if sure == 'y':
        fp_in = fit_multiple_profiles_from_indices(P_id, MW, err_MW, MW_indices, slit_height=slit_height, timit=True, debug_level=1)
        np.save(pathdict['fp'] + 'archive/' + 'individual_fibre_profiles_' + date + '.npy', fp_in)
        stellar_fibparms = make_real_fibparms_by_ord(fp_in, date=date, path=pathdict['fp']+'archive/')
        fibparms = combine_fibparms(date=date, path=pathdict['fp'])
#####################################################################################################################################################


### (5) CREATE CHIPMASKS, FINAL ORDER TRACES, AND DETERMINE SLIT HEIGHTS FOR OPTIMAL EXTRACTION #####################################################
choice = 'r'
if os.path.isfile(pathdict['cm'] + 'chipmask_' + date + '.npy'):
    choice = raw_input("CHIPMASK for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
if choice.lower() == 's':
    print('Loading chipmask for ' + date + '...')
    chipmask = np.load(pathdict['cm'] + 'chipmask_' + date + '.npy').item()
else:
    chipmask = make_chipmask(date, pathdict=pathdict, combined_fibparms=False, savefile=True, timit=True)   # use combined_fibparms=True if you trust the simThXe and especially LFC traces (not for now!!!)

# make proper order traces from fibre profiles (this can now be used instead of "P_id", as it is exactly the same format, ie no need to change any subsequent routines!!!)
choice = 'r'
if os.path.isfile(pathdict['raw'] + date + '_traces.npy'):
    choice = raw_input("FINAL ORDER TRACING has already been done for " + date + " ! Do you want to skip this step or recreate it? ['s' / 'r']")
if choice.lower() == 's':
    print('Loading final order traces for ' + date + '...')
    traces = np.load(pathdict['raw'] + date + '_traces.npy').item()
else:
    traces = make_order_traces_from_fibparms(fibparms)
    np.save(pathdict['raw'] + date + '_traces.npy', traces)

# determine slit_heights from fibre profiles
slit_heights = []
for ord in sorted(traces['allfib'].keys()):
    # calculate median slit height between outermost sky fibres and add 5 pixels to make sure we're at least ~3-4 pixels above/beloce calib fibres
    slit_heights.append(np.ceil(np.median(np.abs(0.5 * (fibparms[ord]['fibre_02']['mu_fit'] - fibparms[ord]['fibre_27']['mu_fit'])))) + 5)
slit_height = np.max(slit_heights[1:]).astype(int)  # exclude order_01, as can be quite dodgy
#####################################################################################################################################################


### (6) PROCESS SIM. CALIBRATION LAMP MASTER FRAMES #################################################################################################
# (6a) define different stripes / strip_indices / slit_heights
# NOTE: these indices are now used for ALL subsequent extractions, so no need to run extract_stripes again
# stellar slit_height (want to exclude sim calibs)
stsh = 23
# sim calib slit_height (want just one fibre)
calsh = 4
stripes, indices = extract_stripes(MW, traces['allfib'], return_indices=True, slit_height=slit_height)
st_stripes, st_indices = extract_stripes(MW, traces['stellar'], return_indices=True, slit_height=stsh)
lfc_stripes, lfc_indices = extract_stripes(MW, traces['lfc'], return_indices=True, slit_height=calsh)
simth_stripes, simth_indices = extract_stripes(MW, traces['simth'], return_indices=True, slit_height=calsh)

# (6b) extract Master White
pix_q,flux_q,err_q = extract_spectrum_from_indices(MW, err_MW, indices, method='quick', slit_height=slit_height, ronmask=ronmask, savefile=True,
                                                   date=date, filetype='fits', obsname=date+'_master_white', pathdict=pathdict, timit=True)
pix,flux,err = extract_spectrum_from_indices(MW, err_MW, indices, method='optimal', slit_height=slit_height, fibs='all', slope=True, offset=True, date=date,
                                             individual_fibres=True, ronmask=ronmask, savefile=True, filetype='fits', obsname=date+'_master_white', pathdict=pathdict, timit=True)

# # get a smoothed Master White and a pixel-to-pixel sensitivity map (no, need to do that to 1-dim spectrum)
# smoothed_flat, pix_sens = onedim_pixtopix_variations_spline(MW, knots=9, savefits=True, path=pathdict['raw'], debug_level=1)


# (6c) MAKE MASTER FRAMES FOR EACH OF THE SIMULTANEOUS CALIBRATION SOURCES AND EXTRACT THEM
# TODO: use different traces and smaller slit_height for LFC only and lfc only???
if len(thxe_list) > 0:
    choice = 'r'
    if os.path.isfile(pathdict['raw'] + date + '_master_simth.fits'):
        choice = raw_input("Master SimThXe frame for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
    if choice.lower() == 's':
        print('Loading MASTER SimThXe frame for ' + date + '...')
        master_simth = pyfits.getdata(pathdict['raw'] + date + '_master_simth.fits', 0)
        err_master_simth = pyfits.getdata(pathdict['raw'] + date + '_master_simth.fits', 1)
    else:
        master_simth, err_master_simth = make_master_calib(thxe_list, lamptype='simth', MB=medbias, ronmask=ronmask, MD=MDS, gain=gain, chipmask=chipmask, remove_bg=True, savefile=True, pathdict=pathdict)
    # now do the extraction    
    pix_q,flux_q,err_q = extract_spectrum_from_indices(master_simth, err_master_simth, simth_indices, method='quick', slit_height=calsh, ronmask=ronmask, savefile=True,
                                                       date=date, filetype='fits', obsname=date+'_master_simth', pathdict=pathdict, timit=True)
    pix,flux,err = extract_spectrum_from_indices(master_simth, err_master_simth, indices, method='optimal', slit_height=slit_height, fibs='simth', slope=True, offset=True, date=date,
                                                 individual_fibres=True, ronmask=ronmask, savefile=True, filetype='fits', obsname=date+'_master_simth', pathdict=pathdict, timit=True)
    
if len(laser_list) > 0:
    choice = 'r'
    if os.path.isfile(pathdict['raw'] + date + '_master_lfc.fits'):
        choice = raw_input("Master LFC frame for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
    if choice.lower() == 's':
        print('Loading MASTER LFC frame for ' + date + '...')
        master_lfc = pyfits.getdata(pathdict['raw'] + date + '_master_lfc.fits', 0)
        err_master_lfc = pyfits.getdata(pathdict['raw'] + date + '_master_lfc.fits', 1)
    else:
        master_lfc, err_master_lfc = make_master_calib(laser_list, lamptype='lfc', MB=medbias, ronmask=ronmask, MD=MDS, gain=gain, chipmask=chipmask, remove_bg=True, savefile=True, pathdict=pathdict)
    # now do the extraction
    pix_q,flux_q,err_q = extract_spectrum_from_indices(master_lfc, err_master_lfc, lfc_indices, method='quick', slit_height=calsh, ronmask=ronmask, savefile=True,
                                                       date=date, filetype='fits', obsname=date+'_master_lfc', pathdict=pathdict, timit=True)
    pix,flux,err = extract_spectrum_from_indices(master_lfc, err_master_lfc, indices, method='optimal', slit_height=slit_height, fibs='lfc', slope=True, offset=True, date=date,
                                                 individual_fibres=True, ronmask=ronmask, savefile=True, filetype='fits', obsname=date+'_master_lfc', pathdict=pathdict, timit=True)
    
if len(laser_and_thxe_list) > 0:
    choice = 'r'
    if os.path.isfile(pathdict['raw'] + date + '_master_lfc_plus_simth.fits'):
        choice = raw_input("Master LFC_PLUS_SIMTH frame for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
    if choice.lower() == 's':
        print('Loading MASTER LFC + SimThXe frame for ' + date + '...')
        master_both = pyfits.getdata(pathdict['raw'] + date + '_master_lfc_plus_simth.fits', 0)
        err_master_both = pyfits.getdata(pathdict['raw'] + date + '_master_lfc_plus_simth.fits', 1)
    else:
        master_both, err_master_both = make_master_calib(laser_and_thxe_list, lamptype='both', MB=medbias, ronmask=ronmask, MD=MDS, gain=gain, chipmask=chipmask, remove_bg=True, savefile=True, pathdict=pathdict)
    # now do the extraction
    pix_q,flux_q,err_q = extract_spectrum_from_indices(master_both, err_master_both, indices, method='quick', slit_height=slit_height, ronmask=ronmask, savefile=True,
                                                       date=date, filetype='fits', obsname=date+'_master_lfc_plus_simth', pathdict=pathdict, timit=True)
    pix,flux,err = extract_spectrum_from_indices(master_both, err_master_both, indices, method='optimal', slit_height=slit_height, fibs='calibs', slope=True, offset=True, date=date,
                                                 individual_fibres=True, ronmask=ronmask, savefile=True, filetype='fits', obsname=date+'_master_lfc_plus_simth', pathdict=pathdict, timit=True)
#####################################################################################################################################################    


### (7) PROCESS and EXTRACT ARC IMAGES #######################################################################################################3######
# first, figure out the configuration of the calibration lamps for the ARC exposures
print('Processing ARC (fibre Thorium) images...')
arc_sublists = {'lfc':[], 'thxe':[], 'both':[], 'neither':[]}

# nasty temp fix to make sure we are always looking at the 2D images until the header keywords are reliable
checkdate = '1' + date[1:]

if int(checkdate) < 20190503:
    # look at the actual 2D image (using chipmasks for LFC and simThXe) to determine which calibration lamps fired
    for file in arc_list:
        img = correct_for_bias_and_dark_from_filename(file, medbias, MDS, gain=gain, scalable=False, savefile=False, path=pathdict['raw'])
        lc = laser_on(img, chipmask)
        thxe = thxe_on(img, chipmask)
        if (not lc) and (not thxe):
            arc_sublists['neither'].append(file)
        elif (lc) and (thxe):
            arc_sublists['both'].append(file)
        else:
            if lc:
                arc_sublists['lfc'].append(file)
            elif thxe:
                arc_sublists['thxe'].append(file)
else:
    # since May 2019 the header keywords are correct, so check for LFC / ThXe in header, as that is MUCH faster
    for file in arc_list:
        lc = 0
        thxe = 0
        h = pyfits.getheader(file)
        if 'LCNEXP' in h.keys():  # this indicates the latest version of the FITS headers (from May 2019 onwards)
            if ('LCEXP' in h.keys()) or ('LCMNEXP' in h.keys()):  # this indicates the LFC actually was actually exposed (either automatically or manually)
                lc = 1
        else:  # if not, just go with the OBJECT field
            if ('LC' in pyfits.getval(file, 'OBJECT').split('+')) or ('LFC' in pyfits.getval(file, 'OBJECT').split('+')):
                lc = 1
        if (h['SIMCALTT'] > 0) and (h['SIMCALN'] > 0) and (h['SIMCALSE'] > 0):
            thxe = 1
        assert lc+thxe in [0,1,2], 'ERROR: could not establish status of LFC and simultaneous ThXe for the exposures in this list!!!'    
        if lc+thxe == 0:
            arc_sublists['neither'].append(file)
        if lc+thxe == 1:
            if lc == 1:
                arc_sublists['lfc'].append(file)
            else:
                arc_sublists['thxe'].append(file)
        elif lc+thxe == 2:
            arc_sublists['both'].append(file)

for subl in arc_sublists.keys():
    if len(arc_sublists[subl]) > 0:
        dum = process_science_images(arc_sublists[subl], traces['allfib'], chipmask, mask=mask, stripe_indices=indices, quick_indices=indices, sampling_size=25,
                                     slit_height=slit_height, qsh=slit_height, gain=gain, MB=medbias, ronmask=ronmask, MD=MDS, scalable=True, saveall=False,
                                     pathdict=pathdict, ext_method='optimal', fibs='all', offset=True, slope=True, date=date, from_indices=True, timit=True)


# now make master ARC from individual extracted ARC frames, and create a rough wl-solution from that
if len(arc_list) > 0:
    choice = 'r'
    if os.path.isfile(pathdict['raw'] + date + '_master_ARC.fits'):
        choice = raw_input("Master ARC (extracted) for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
    if choice != 's':
        s,e = make_master_fibth(path=pathdict['raw'], date=date, savefile=True, overwrite=True)

    if date == '20180917':
        lamptype = 'thar'
    else:
        lamptype = 'thxe'
    choice = 'r'
    if os.path.isfile(pathdict['raw'] + date + '_' + lamptype + '_dispsol.fits'):
        choice = raw_input("Master ARC dispsol for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
    if choice != 's':
        dum = make_arc_dispsols(date=date, pathdict=pathdict, overwrite=True)
#####################################################################################################################################################


### (8) PROCESS INDIVIDUAL SIM. CALIBRATION FRAMES ##################################################################################################

if len(thxe_list) > 0:
    print('Processing sim-ThXe images...')
    dum = process_science_images(thxe_list, traces['simth'], chipmask, mask=mask, stripe_indices=indices, quick_indices=simth_indices,
                                 sampling_size=25, slit_height=slit_height, qsh=calsh, gain=gain, MB=medbias,
                                 ronmask=ronmask, MD=MDS, scalable=True, saveall=False, pathdict=pathdict, ext_method='optimal',
                                 offset=True, slope=True, fibs='simth', date=date, from_indices=True, timit=True)
if len(laser_list) > 0:
    print('Processing LFC images...')
    dum = process_science_images(laser_list, traces['lfc'], chipmask, mask=mask, stripe_indices=indices, quick_indices=lfc_indices,
                                 sampling_size=25, slit_height=slit_height, qsh=calsh, gain=gain, MB=medbias,
                                 ronmask=ronmask, MD=MDS, scalable=True, saveall=False, pathdict=pathdict, ext_method='optimal',
                                 offset=True, slope=True, fibs='lfc', date=date, from_indices=True, timit=True)
if len(laser_and_thxe_list) > 0:
    print('Processing LFC+sim-ThXe images...')
    dum = process_science_images(laser_and_thxe_list, traces['allfib'], chipmask, mask=mask, stripe_indices=indices, quick_indices=indices,
                                 sampling_size=25, slit_height=slit_height, qsh=slit_height, gain=gain, MB=medbias,
                                 ronmask=ronmask, MD=MDS, scalable=True, saveall=False, pathdict=pathdict, ext_method='optimal',
                                 offset=True, slope=True, fibs='calibs', date=date, from_indices=True, timit=True)
#####################################################################################################################################################


### (9) PROCESS STELLAR IMAGES ######################################################################################################################
if len(stellar_list) > 0:
    print('Processing stellar images...')
    dum = process_science_images(stellar_list, traces['stellar'], chipmask, mask=mask, stripe_indices=indices, quick_indices=st_indices,
                                 sampling_size=25, slit_height=slit_height, qsh=stsh, gain=gain, MB=medbias,
                                 ronmask=ronmask, MD=MDS, scalable=True, saveall=False, pathdict=pathdict, ext_method='optimal',
                                 offset=True, slope=True, fibs='all', date=date, from_indices=True, timit=True)
#####################################################################################################################################################


### (10) CALCULATE RADIAL VELOCITIES ################################################################################################################
# RV = ?
#
# b.c. will be already in the header
#
# TODO: also save CCF(s) as separate HDU to reduced spectrum file!!!
#####################################################################################################################################################




### (11) TIDY UP AND MOVE EXTRACTED SPECTRA TO REDUCED FOLDER #######################################################################################
# delete temporary background files
tempfiles = glob.glob(pathdict['raw'] + 'temp*')
for file in tempfiles:
    os.remove(file)

# find all reduced spectra, master frames, and auxiliary files and move them to the reduced folder
red_files = glob.glob(pathdict['raw'] + date + '*')
if not os.path.exists(pathdict['red']):
    os.system("mkdir " + pathdict['red'])
for file in red_files:
    short_fn = file.split('/')[-1]
    os.rename(file, pathdict['red'] + short_fn)
#####################################################################################################################################################





