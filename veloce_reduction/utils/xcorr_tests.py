import glob
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from readcol import readcol

from veloce_reduction.veloce_reduction.wavelength_solution import get_dispsol_for_all_fibs, get_dispsol_for_all_fibs_2
from veloce_reduction.veloce_reduction.get_radial_velocity import get_RV_from_xcorr, get_RV_from_xcorr_2, make_ccfs
from veloce_reduction.veloce_reduction.helper_functions import get_mean_snr
from veloce_reduction.veloce_reduction.flat_fielding import onedim_pixtopix_variations, deblaze_orders
from veloce_reduction.veloce_reduction.barycentric_correction import get_barycentric_correction


########################################################################################################################
# HOUSEKEEPING
path = '/Users/christoph/data/reduced/tauceti/tauceti_with_LFC/'
# path = '/Volumes/BERGRAID/data/veloce/reduced/tauceti/tauceti_with_LFC/'

files_sep = glob.glob(path + 'sep2018/' + '*10700*')
files_nov = glob.glob(path + 'nov2018/' + '*10700*')
files_jan = glob.glob(path + 'jan2019/' + '*10700*')

files = []
# sort list of files
for tempfiles in [files_sep, files_nov, files_jan]:
    all_shortnames = []
    for i,filename in enumerate(tempfiles):
        dum = filename.split('/')
        dum2 = dum[-1].split('.')
        dum3 = dum2[0]
        dum4 = dum3.split('_')
        shortname = dum4[1]
        all_shortnames.append(shortname)
    sortix = np.argsort(all_shortnames)
    tempfiles = np.array(tempfiles)
    tempfiles = tempfiles[sortix]
    all_obsnames = np.array(all_shortnames)
    all_obsnames = all_obsnames[sortix]
    files = files + list(tempfiles)


########################################################################################################################
########################################################################################################################
########################################################################################################################

# calculating BARYCORR

# all_jd = readcol(path + 'tauceti_all_jds.dat')
# all_bc = readcol(path + 'tauceti_all_bcs.dat')

all_jd = []
all_bc = []
# outfn = path + 'tauceti_all_info.dat'
# outfile = open(outfn, 'w')
# outfn_jd = path + 'tauceti_all_jds.dat'
# outfile_jd = open(outfn_jd, 'w')
# outfn_bc = path + 'tauceti_all_bcs.dat'
# outfile_bc = open(outfn_bc, 'w')
# outfn_names = path + 'tauceti_all_obsnames.dat'
# outfile_names = open(outfn_names, 'w')

for i,filename in enumerate(files):
    print('Processing BCs for tau Ceti observation '+str(i+1)+'/'+str(len(files)))
    # do some housekeeping with filenames
    dum = filename.split('/')
    dum2 = dum[-1].split('.')
    dum3 = dum2[0]
    dum4 = dum3.split('_')
    shortname = dum4[1]
    bc = get_barycentric_correction(filename, rvabs=-16.68)
    pyfits.setval(filename, 'BARYCORR', value=bc[0], comment='barycentric velocity correction [m/s]')
    all_bc.append(bc[0])
    jd = pyfits.getval(filename, 'UTMJD') + 2.4e6 + 0.5
    all_jd.append(jd)
    outfile_names.write(shortname + ' \n')
    outfile_jd.write('%14.6f \n' % (jd))
    outfile_bc.write('%14.6f \n' % (bc))
    outfile.write(shortname + '     %14.6f     %14.6f \n' % (jd, bc))

# outfile.close()
# outfile_jd.close()
# outfile_bc.close()
# outfile_names.close()

########################################################################################################################
########################################################################################################################
########################################################################################################################

# all_snr = readcol(path + 'tauceti_all_snr.dat')

# get mean SNR per collapsed pixel
all_snr = []
for i,file in enumerate(files):
    print('Estimating mean SNR for tau Ceti observation ' + str(i+1) + '/' + str(len(files)))
    flux = pyfits.getdata(file, 0)
    err = pyfits.getdata(file, 1)
    all_snr.append(get_mean_snr(flux, err))

########################################################################################################################
########################################################################################################################
########################################################################################################################

# calculate wl-solution for all fibres, including the LFC shifts and slopes; also append to reduced spectrum FITS file
signflip_shift = True
signflip_slope = True
signflip_secord = True
fudge = 1.
maxdiff= []
for i,filename in enumerate(files):
    print('Processing wl-solution for tau Ceti observation ' + str(i+1) + '/' + str(len(files)))
    dum = filename.split('/')
    dum2 = dum[-1].split('.')
    dum3 = dum2[0].split('_')
    obsname = dum3[1]
    # old_wldict, old_wl = get_dispsol_for_all_fibs(obsname, fudge=fudge, signflip_shift=signflip_shift, signflip_slope=signflip_slope, refit=True)
    # new_wldict, new_wl = get_dispsol_for_all_fibs(obsname, fudge=fudge, signflip_shift=signflip_shift, signflip_slope=signflip_slope, refit=False)
    new_wldict, new_wl = get_dispsol_for_all_fibs_2(obsname, fudge=fudge)
    # new_wl[0, :, :] = 1.
    # new_wl[-1, :, :] = 1.
    # old_wl[0, :, :] = 1.
    # old_wl[-1, :, :] = 1.
    wl[0, :, :] = 1.
    wl[-1, :, :] = 1.
    maxdiff.append(np.max(3.e8 * (old_wl.flatten() - new_wl.flatten()) / new_wl.flatten()))
    # pyfits.append(filename, wl, clobber=True)

########################################################################################################################
########################################################################################################################
########################################################################################################################

# calculating the CCFs for one order / 11 orders
# (either with or without the LFC shifts applied, comment out the 'wl' and 'wl0' you don't want)

all_xc = []
all_rv = np.zeros((len(files), 11, 19))
all_sumrv = np.zeros(len(files))
xcsums = np.zeros((len(files), 301))

# TEMPLATE:
f0 = pyfits.getdata(files[6], 0)   # one of the higher SNR obs but closest to FibThars used to define fibtofib wl shifts
# f0 = pyfits.getdata(files[69], 0)   # that's the highest SNR observation for Sep 18
# f0 = pyfits.getdata(files[2], 0)   # that's the highest SNR observation for Nov 18
# f0 = pyfits.getdata(files[35], 0)   # that's the 2nd highest SNR observation for Nov 18
# err0 = pyfits.getdata(files[69], 1)
# wl0 = pyfits.getdata(files[69], 2)
# wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')
obsname_0 = '20sep30087'     # one of the higher SNR obs but closest to FibThars used to define fibtofib wl shifts
date_0 = '20180920'
# obsname_0 = '24sep30078'     # that's the highest SNR observation for Sep 18
# obsname_0 = '16nov30128'     # that's the highest SNR observation for Nov 18
# obsname_0 = '25nov30084'     # that's the 2nd highest SNR observation for Nov 18
# wldict0,wl0 = get_dispsol_for_all_fibs(obsname_0, date=date_0, fudge=fudge, signflip_shift=signflip_shift,
#                                        signflip_slope=signflip_slope, signflip_secord=signflip_secord)
wldict0,wl0 = get_dispsol_for_all_fibs(obsname_0, date=date_0, fibs='stellar', refit=False, fibtofib=True, nightly_coeffs=True)
# wldict0,wl0 = get_dispsol_for_all_fibs_2(obsname_0)

# use a synthetic template?
# wl0, f0 = readcol('/Users/christoph/OneDrive - UNSW/synthetic_templates/' + 'synth_teff5250_logg45.txt', twod=False)
wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/synthetic_templates/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
f0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/synthetic_templates/phoenix/lte05400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')

for i,filename in enumerate(files):
    print('Processing RV for tau Ceti observation ' + str(i+1) + '/' + str(len(files)))
    f = pyfits.getdata(filename, 0)
    err = pyfits.getdata(file, 1)
    f_clean = f.copy()
    for o in range(f.shape[0]):
        f_clean[o,:],ncos = onedim_medfilt_cosmic_ray_removal(f[o,fib,:], err[o,fib,:], w=31, thresh=5., low_thresh=3.)
    f_dblz, err_dblz = deblaze_orders(f_clean, wl, smoothed_flat, maskdict, err=err, degpol=2, gauss_filter_sigma=3., maxfilter_size=100)    
    wl = pyfits.getdata(filename, 2)
    # wl = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')
    all_xc.append(make_ccfs(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[6], mask=None, smoothed_flat=None, delta_log_wl=1e-6, relgrid=False,
                            flipped=False, individual_fibres=False, debug_level=1, timit=False))
    rv,rverr,xcsum = get_RV_from_xcorr_2(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[6], individual_fibres=True, individual_orders=True, debug_level=1)
    sumrv,sumrverr,xcsum = get_RV_from_xcorr_2(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[6], individual_fibres=False, individual_orders=False, debug_level=1)
    all_rv[i,:,:] = rv
    all_sumrv[i] = sumrv
    xcsums[i,:] = xcsum
xcsums = np.array(xcsums)



########################################################################################################################
########################################################################################################################
########################################################################################################################

# overplot all tau ceti spectra with the barycorr removed so that in theory they should all perfectly overlap
# colour-coding by runs
o=26
for i,filename in enumerate(files):
    print('Processing RV for tau Ceti observation ' + str(i + 1) + '/' + str(len(files)))
    f = pyfits.getdata(filename, 0)
    dum = filename.split('/')
    dum2 = dum[-1].split('.')
    dum3 = dum2[0].split('_')
    obsname = dum3[1]
    wldict, wl = get_dispsol_for_all_fibs(obsname, fudge=fudge, signflip_shift=signflip_shift, signflip_slope=signflip_slope,
                                          signflip_secord=signflip_secord, nightly_coeffs=False)
    wl_bcc = (1 + all_bc[i] / c) * wl
    wl0_bcc = (1 + all_bc[6] / c) * wl0
    logwl = np.log(wl_bcc[o, :, :])
    logwl0 = np.log(wl0_bcc[o, :, :])
    min_wl = min_wl_arr[o]
    max_wl = max_wl_arr[o]
    logwlgrid = np.arange(np.log(min_wl), np.log(max_wl), delta_log_wl)
    logwl_sorted = logwl[:, ::-1].copy()
    logwl0_sorted = logwl0[:, ::-1].copy()
    ord_f0_sorted = f0[o, :, ::-1].copy()
    ord_f_sorted = f[o, :, ::-1].copy()
    nfib = ord_f0_sorted.shape[0]
    rebinned_f0 = np.zeros((nfib, len(logwlgrid)))
    rebinned_f = np.zeros((nfib, len(logwlgrid)))
    for ii in range(nfib):
        # spl_ref_f0 = interp.InterpolatedUnivariateSpline(logwl0_sorted[i,ordmask_sorted], ord_f0_sorted[i,ordmask_sorted], k=3)  # slightly slower than linear, but best performance for cubic spline
        # rebinned_f0[i,:] = spl_ref_f0(logwlgrid)
        # spl_ref_f = interp.InterpolatedUnivariateSpline(logwl_sorted[i,ordmask_sorted], ord_f_sorted[i,ordmask_sorted], k=3)  # slightly slower than linear, but best performance for cubic spline
        # rebinned_f[i,:] = spl_ref_f(logwlgrid)
        spl_ref_f0 = interp.InterpolatedUnivariateSpline(logwl0_sorted[ii, :], ord_f0_sorted[ii, :],
                                                         k=3)  # slightly slower than linear, but best performance for cubic spline
        rebinned_f0[ii, :] = spl_ref_f0(logwlgrid)
        spl_ref_f = interp.InterpolatedUnivariateSpline(logwl_sorted[ii, :], ord_f_sorted[ii, :],
                                                        k=3)  # slightly slower than linear, but best performance for cubic spline
        rebinned_f[ii, :] = spl_ref_f(logwlgrid)
    if i < 86:
        color='r'
    elif i < 138:
        color='b'
    else:
        color='k'
    plt.plot(logwlgrid, np.sum(rebinned_f, axis=0)/np.nanmedian(np.sum(rebinned_f, axis=0)), color=color)


########################################################################################################################
########################################################################################################################
########################################################################################################################



# WITH PRE-NORMALIZING FLATS

# calculating the CCFs for one order / 11 orders
# (either with or without the LFC shifts applied, comment out the 'wl' and 'wl0' you don't want)

all_xc = []
all_rv = np.zeros((len(files), 11, 19))
all_sumrv = np.zeros(len(files))
xcsums = np.zeros((len(files), 81))

# TEMPLATE:
# f0 = pyfits.getdata(files[69], 0)   # that's the highest SNR observation for Sep 18
# f0 = pyfits.getdata(files[2], 0)   # that's the highest SNR observation for Nov 18
f0 = pyfits.getdata(files[35], 0)   # that's the 2nd highest SNR observation for Nov 18
# err0 = pyfits.getdata(files[69], 1)
# wl0 = pyfits.getdata(files[69], 2)
# wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')
# obsname_0 = '24sep30078'     # that's the highest SNR observation for Sep 18
# obsname_0 = '16nov30128'     # that's the highest SNR observation for Nov 18
obsname_0 = '25nov30084'     # that's the highest SNR observation for Nov 18
# wldict0,wl0 = get_dispsol_for_all_fibs(obsname_0, fudge=fudge, signflip_shift=signflip_shift, signflip_slope=signflip_slope)
wldict0,wl0 = get_dispsol_for_all_fibs_2(obsname_0)

smoothed_flat_0, pix_sens_0 = onedim_pixtopix_variations(flat_0, filt='g', filter_width=25)
f0_dblz = deblaze_orders(f0, wl0, smoothed_flat_0, mask_0, err=None)

for i,filename in enumerate(files):
    print('Processing RV for tau Ceti observation ' + str(i+1) + '/' + str(len(files)))
    f = pyfits.getdata(filename, 0)
    # err = pyfits.getdata(file, 1)
    wl = pyfits.getdata(filename, 2)
    # wl = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')
    
    smoothed_flat, pix_sens = onedim_pixtopix_variations(flat, filt='g', filter_width=25)
    f_dblz = deblaze_orders(f, wl, smoothed_flat, mask, err=None)
    
    all_xc.append(make_ccfs(f, wl, f0, wl0, mask=None, smoothed_flat=None, delta_log_wl=1e-6, relgrid=False,
                            flipped=False, individual_fibres=False, debug_level=1, timit=False))
    rv,rverr,xcsum = get_RV_from_xcorr_2(f, wl, f0, wl0, individual_fibres=True, individual_orders=True, debug_level=1)
    sumrv,sumrverr,xcsum = get_RV_from_xcorr_2(f, wl, f0, wl0, individual_fibres=False, individual_orders=False, debug_level=1)
    all_rv[i,:,:] = rv
    all_sumrv[i] = sumrv
    xcsums[i,:] = xcsum
xcsums = np.array(xcsums) 

########################################################################################################################
########################################################################################################################
########################################################################################################################

# tests to determine the best fudge factor...
rvs_fudge_test = {}
# TEMPLATE:
f0 = pyfits.getdata(files[69], 0)   #that's the highest SNR observation for Sep 18
# f0 = pyfits.getdata(files[35], 0)   # that's the 2nd highest SNR observation for Nov 18
# err0 = pyfits.getdata(files[69], 1)
# wl0 = pyfits.getdata(files[69], 2)
# wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')
obsname_0 = '24sep30078'
# obsname_0 = '25nov30084'     # that's the highest SNR observation for Nov 18


for fudge in np.arange(0.5,1.26,0.05):
    print('FUDGE = ', fudge)
    signflip_shift = True
    signflip_slope = True

    wldict0, wl0 = get_dispsol_for_all_fibs(obsname_0, fudge=fudge, signflip_shift=signflip_shift,
                                            signflip_slope=signflip_slope, refit=True)

    all_sumrv = []

    for i, filename in enumerate(files):
        print('Processing tau Ceti observation ' + str(i + 1) + '/' + str(len(files)))
        dum = filename.split('/')
        dum2 = dum[-1].split('.')
        dum3 = dum2[0].split('_')
        obsname = dum3[1]
        wldict, wl = get_dispsol_for_all_fibs(obsname, fudge=fudge, signflip_shift=signflip_shift,
                                              signflip_slope=signflip_slope, refit=True)
        f = pyfits.getdata(filename, 0)
        # all_xc.append(get_RV_from_xcorr_2(f, wl, f0, wl0, individual_fibres=False, individual_orders=False))
        sumrv, sumrverr, xcsum = get_RV_from_xcorr_2(f, wl, f0, wl0,  individual_fibres=False, individual_orders=False, debug_level=1, fitrange=6)
        all_sumrv.append(sumrv)

    rvs_fudge_test['fudge_' + str(fudge)[:5]] = all_sumrv - np.array(all_bc)

########################################################################################################################
########################################################################################################################
########################################################################################################################


# plotting the CCFs on RV axis with and without BC applied

# speed of light in m/s
c = 2.99792458e8
delta_log_wl = 1e-6

# WHEN THERE IS ONLY ONE CCF PER OBSERVATION

plt.figure()
plt.title('OLD ; CCF flipped = False')
plt.xlim(-2e4,2e4)
plt.ylim(0.9980,1.0005)
for i in range(len(files)):
    plt.plot(c * (np.arange(len(old_all_xc[i])) - (len(old_all_xc[i]) // 2)) * delta_log_wl,
             old_all_xc[i] / np.max(old_all_xc[i]), 'k.-')
    plt.plot(c * (np.arange(len(old_all_xc[i])) - (len(old_all_xc[i]) // 2)) * delta_log_wl + all_bc[i],
             old_all_xc[i] / np.max(old_all_xc[i]), 'r.-')
    plt.plot(c * (np.arange(len(old_all_xc[i])) - (len(old_all_xc[i]) // 2)) * delta_log_wl - all_bc[i],
             old_all_xc[i] / np.max(old_all_xc[i]), 'b.-')


plt.figure()
plt.title('NEW ; CCF flipped = False')
plt.xlim(-2e4,2e4)
plt.ylim(0.9980,1.0005)
for i in range(len(files)):
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl,
             all_xc[i] / np.max(all_xc[i]), 'k.-')
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl + all_bc[i],
             all_xc[i] / np.max(all_xc[i]), 'r.-')
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl - all_bc[i],
             all_xc[i] / np.max(all_xc[i]), 'b.-')


plt.figure()
plt.title('NEW ; CCF flipped = True')
plt.xlim(-2e4,2e4)
plt.ylim(0.9980,1.0005)
for i in range(len(files)):
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl,
             all_xc_flipped[i] / np.max(all_xc_flipped[i]), 'k.-')
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl + all_bc[i],
             all_xc_flipped[i] / np.max(all_xc_flipped[i]), 'r.-')
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl - all_bc[i],
             all_xc_flipped[i] / np.max(all_xc_flipped[i]), 'b.-')


# WHEN THERE ARE MULTIPLE CCFs (for different orders) PER OBSERVATION
addrange = 40
# index = [4,5,6,25,26,33,34,35]
index = [5,6,17,25,26,27,31,34,35,36,37]

# # this was for all obs at once
# xcarr = np.zeros((len(all_xc), len(all_xc[0]), 2*addrange+1))
# for i in range(xcarr.shape[0]):
#     for j in range(xcarr.shape[1]):
#         dum = np.array(all_xc[i][j])
#         xcarr[i,j,:] = dum[len(dum)//2 - addrange : len(dum)//2 + addrange +1]
# xcsums = np.sum(xcarr,axis=1)


plt.figure()
plt.title('tau Ceti - CCFs for 11 orders added up')
plt.xlim(-2e4,2e4)
plt.ylim(0.9980,1.0005)
plt.xlabel('delta RV [m/s]')
for i in range(len(files)):
    plt.plot(c * (np.arange(len(xcsums[i,:])) - (len(xcsums[i,:]) // 2)) * delta_log_wl,
             xcsums[i,:] / np.max(xcsums[i,:]), 'k.-')
    plt.plot(c * (np.arange(len(xcsums[i,:])) - (len(xcsums[i,:]) // 2)) * delta_log_wl + all_bc[i],
             xcsums[i,:] / np.max(xcsums[i,:]), 'r.-')
    plt.plot(c * (np.arange(len(xcsums[i,:])) - (len(xcsums[i,:]) // 2)) * delta_log_wl - all_bc[i],
             xcsums[i,:] / np.max(xcsums[i,:]), 'b.-')
i=0
plt.plot(c * (np.arange(len(xcsums[i,:])) - (len(xcsums[i,:]) // 2)) * delta_log_wl,
         xcsums[i,:] / np.max(xcsums[i,:]), 'k.-', label='no BC')
plt.plot(c * (np.arange(len(xcsums[i,:])) - (len(xcsums[i,:]) // 2)) * delta_log_wl + all_bc[i],
         xcsums[i,:] / np.max(xcsums[i,:]), 'r.-', label='BC added')
plt.plot(c * (np.arange(len(xcsums[i,:])) - (len(xcsums[i,:]) // 2)) * delta_log_wl - all_bc[i],
         xcsums[i,:] / np.max(xcsums[i,:]), 'b.-', label='BC subtracted')
plt.legend()

########################################################################################################################

