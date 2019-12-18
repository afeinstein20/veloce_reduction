import glob
import os
import time
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from readcol import readcol
import scipy.interpolate as interp
import scipy.optimize as op

from veloce_reduction.veloce_reduction.wavelength_solution import get_dispsol_for_all_fibs, get_dispsol_for_all_fibs_2, get_dispsol_for_all_fibs_3
from veloce_reduction.veloce_reduction.get_radial_velocity import *
from veloce_reduction.veloce_reduction.helper_functions import get_snr, short_filenames, wm_and_wsv
from veloce_reduction.veloce_reduction.flat_fielding import onedim_pixtopix_variations, deblaze_orders
from veloce_reduction.veloce_reduction.barycentric_correction import get_barycentric_correction
from veloce_reduction.veloce_reduction.cosmic_ray_removal import onedim_medfilt_cosmic_ray_removal


########################################################################################################################

starname = 'delpav'   # 'tauceti', 'delpav', 'GJXXX', TOIXXX', 'HDXXX'

# HOUSEKEEPING
if starname.lower() == 'tauceti':
    path = '/Volumes/WORKING/data/veloce/reduced/tauceti/tauceti_with_LFC/'
else:
    path = '/Volumes/WORKING/data/veloce/reduced/' + starname + '/with_lfc/'


file_list = glob.glob(path + '*optimal3a_extracted.fits')
file_list.sort()  # this is not a proper sort, but see two lines below
wl_list = glob.glob(path + '*vac_wl.fits*')
wl_list.sort()   # this is not a proper sort, but consistent with file_list; proper sort is done below
lfc_wl_list = glob.glob(path + '*vac_wl_lfc.fits*')
lfc_wl_list.sort()   # this is not a proper sort, but consistent with file_list; proper sort is done below
assert len(file_list) == len(wl_list), 'ERROR: number of wl-solution files does not match the number of reduced spectra!!!'

obsname_list = [fn.split('_')[-3] for fn in file_list]
object_list = [pyfits.getval(fn, 'OBJECT').split('+')[0] for fn in file_list]
bc_list = [pyfits.getval(fn, 'BARYCORR') for fn in file_list]
texp_list = [pyfits.getval(fn, 'ELAPSED') for fn in file_list]
utmjd_start = np.array([pyfits.getval(fn, 'UTMJD') for fn in file_list]) + 2.4e6 + 0.5   # the fits header has 2,400,000.5 subtracted!!!!!
utmjd = utmjd_start + (np.array(texp_list)/2.)/86400.

sortix = np.argsort(utmjd)
all_obsnames = np.array(obsname_list)[sortix]
files = np.array(file_list)[sortix]
wls = np.array(wl_list)[sortix]
lfc_wls = np.array(lfc_wl_list)[sortix]
all_bc = np.array(bc_list)[sortix]
all_jd = utmjd[sortix]

all_dates = [pyfits.getval(file, 'UTDATE')[:4] + pyfits.getval(file, 'UTDATE')[5:7] + pyfits.getval(file, 'UTDATE')[8:] for file in files]
unique_dates = set(all_dates)


########################################################################################################################
########################################################################################################################
########################################################################################################################

all_snr = readcol(path + starname + '_all_snr.dat', twod=False)[0]

# # get mean SNR per collapsed pixel
# all_snr = []
# for i, file in enumerate(files):
#     print('Estimating mean SNR for observation ' + str(i + 1) + '/' + str(len(files)))
#     flux = pyfits.getdata(file, 0)
#     err = pyfits.getdata(file, 1)
#     all_snr.append(get_snr(flux, err))
# np.savetxt(path + 'tauceti_all_snr.dat', np.array(all_snr))

########################################################################################################################
########################################################################################################################
########################################################################################################################

# # make all new wl-solutions and save to file (only once)
# lfc_path = '/Volumes/BERGRAID/data/veloce/lfc_peaks/'
# for i,filename in enumerate(file_list):
#     dum = filename.split('/')
#     obsname = dum[-1].split('_')[1]
#     print(obsname, os.path.isfile(lfc_path + 'all/' + '2019' + '/' + obsname + 'olc.nst'))
#     if os.path.isfile(lfc_path + 'all/' + '2019' + '/' + obsname + 'olc.nst'):
#         utdate = pyfits.getval(filename, 'UTDATE')
#         date = utdate[:4] + utdate[5:7] + utdate[8:]
#         wldict, wl = get_dispsol_for_all_fibs_3(obsname, date=date)
#         pyfits.writeto(path + pyfits.getval(filename, 'OBJECT').split('+')[0] + '_' + obsname + '_vac_wl.fits', wl, clobber=True)

########################################################################################################################
########################################################################################################################
########################################################################################################################

relints = np.load(path + 'relints.npy')   # max = 1
norm_relints = np.load(path + 'norm_relints.npy')   # sum = 1

# # make quick and dirty relints (only once) - only for stellar fibres
# relints = []
# norm_relints = []
# for i,filename in enumerate(files):
#     print('Processing RV for observation ' + str(i + 1) + '/' + str(len(files)))
#     obsname = filename.split('_')[-3]
#     utdate = pyfits.getval(file_list[0], 'UTDATE')
#     date = utdate[:4] + utdate[5:7] + utdate[8:]
#     f = pyfits.getdata(filename, 0)
# #     f = pyfits.getdata(path + '190248_' + obsname + '_optimal3a_extracted_cleaned.fits')
#     err = pyfits.getdata(filename, 1)
#     obs_relints = np.zeros(19)
#     for j in range(3,22):
#         # obs_relints[j] = np.nanmedian(f[:,j+1,:]/f[:,12,:])   # relative to central fibre
#         obs_relints[j-3] = np.nanmedian(f[:,j,:])   # relative to stellar fibre that carries most flux
#         obs_relints = obs_relints / np.max(obs_relints)
#     relints.append(obs_relints)
#     norm_relints.append(obs_relints / np.max(obs_relints))

######################################################################################################################## np.max(param)
########################################################################################################################
########################################################################################################################

#################################################################################################################    
### OK, now let's Cross-Correlation RVs on real observations
#################################################################################################################

# calculating the CCFs for one order / 11 orders
# (either with or without the LFC shifts applied, comment out the 'wl' and 'wl0' you don't want)

# vers = 'v2d'

all_xc = []
all_rv = np.zeros((len(files), 19))
# all_rv = np.zeros((len(files), 10, 19))
# all_rv = np.zeros((len(files), 11, 19))
all_sumrv = []
# all_sumrv = np.zeros(len(files))
# all_sumrv_2 = np.zeros(len(files))
xcsums = np.zeros((len(files), 301))
# xcsums = np.zeros((len(files), 19, 301))
# xcsums = np.zeros((len(files), 38, 301))

# TEMPLATE:
# # tau Ceti
# ix0 = 7   # tau Ceti
# date_0 = '20180919'
# del Pav
ix0 = 36
date_0 = '20190518'

f0 = pyfits.getdata(files[ix0], 0)
err0 = pyfits.getdata(files[ix0], 1)
obsname_0 = all_obsnames[ix0]
# wl0 = pyfits.getdata(wls[ix0])
wl0 = pyfits.getdata(lfc_wls[ix0])
bc0 = pyfits.getval(files[ix0], 'BARYCORR')

maskdict = np.load('/Volumes/BERGRAID/data/veloce/reduced/' + date_0 + '/' + date_0 + '_mask.npy').item()

# wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')

mw_flux = pyfits.getdata('/Volumes/BERGRAID/data/veloce/reduced/' + date_0 + '/' + date_0 + '_master_white_optimal3a_extracted.fits')
smoothed_flat, pix_sens = onedim_pixtopix_variations(mw_flux, filt='gaussian', filter_width=25)

f0_clean = pyfits.getdata(path + '190248_' + obsname_0 + '_optimal3a_extracted_cleaned.fits')

# f0_clean = f0.copy()
# loop over orders
# for o in range(f0.shape[0]):
#     print('Order ', o + 1)
#     # loop over fibres (but exclude the simThXe and LFC fibres for obvious reasons!!!)
#     for fib in range(1,f0.shape[1]-1):
#         if (o == 0) and (fib == 1):
#             start_time = time.time()
#         f0_clean[o, fib, :], ncos = onedim_medfilt_cosmic_ray_removal(f0[o, fib, :], err0[o, fib, :], w=31, thresh=5., low_thresh=3.)
#         if (o == 38) and (fib == 24):
#             print('time elapsed ', time.time() - start_time, ' seconds')


# NO!!! I think we want to divide by the flat, not the smoothed flat otherwise we're not taking out the pix-to-pix sensitivity variations...
f0_dblz, err0_dblz = deblaze_orders(f0_clean, mw_flux, mask=maskdict, err=err0, combine_fibres=True,
                                    degpol=5, gauss_filter_sigma=3., maxfilter_size=100)
# f0_dblz, err0_dblz = deblaze_orders(f0_clean[:,3:22,:], smoothed_flat[:,3:22,:], mask=maskdict, err=err0[:,3:22,:], combine_fibres=True,
#                                     degpol=2, gauss_filter_sigma=3., maxfilter_size=100)
# f0_dblz, err0_dblz = deblaze_orders(f0_clean, smoothed_flat, mask=maskdict, err=err0, combine_fibres=True,
#                                     degpol=2, gauss_filter_sigma=3., maxfilter_size=100)



# use a synthetic template?
# wl0, f0 = readcol('/Users/christoph/OneDrive - UNSW/synthetic_templates/' + 'synth_teff5250_logg45.txt', twod=False)
# wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/synthetic_templates/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
# f0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/synthetic_templates/phoenix/lte05400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')


# loop over all observations
for i,filename in enumerate(files):
    print('Processing RV for observation ' + str(i + 1) + '/' + str(len(files)))
    # get obsname and date
    obsname = filename.split('_')[-3]
    utdate = pyfits.getval(filename, 'UTDATE')
    date = utdate[:4] + utdate[5:7] + utdate[8:]

    # read in spectrum
#     f = pyfits.getdata(filename, 0)
    err = pyfits.getdata(filename, 1)
#     wl = pyfits.getdata(wls[i])
    wl = pyfits.getdata(lfc_wls[i])
    bc = pyfits.getval(filename, 'BARYCORR')
    f_clean = pyfits.getdata(path + '190248_' + obsname + '_optimal3a_extracted_cleaned.fits')
#     f_clean = f.copy()
#     for o in range(f.shape[0]):
#         for fib in range(f.shape[1]):
#             f_clean[o,fib,:],ncos = onedim_medfilt_cosmic_ray_removal(f[o,fib,:], err[o,fib,:], w=31, thresh=5., low_thresh=3.)
#     pyfits.writeto(path + '190248_' + obsname + '_optimal3a_extracted_cleaned.fits', f_clean)
#     pyfits.append(path + '190248_' + obsname + '_optimal3a_extracted_cleaned.fits', err)
    #     wl = pyfits.getdata(filename, 2)
    #     wl = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')
    #     wldict,wl = get_dispsol_for_all_fibs(obsname, date=date, fibs='stellar', refit=False, fibtofib=True, nightly_coeffs=True)
#     wldict, wl = get_dispsol_for_all_fibs_2(obsname, refit=True, eps=2)
#     f_dblz, err_dblz = deblaze_orders(f_clean, mw_flux, mask=maskdict, err=err, combine_fibres=True, degpol=5, gauss_filter_sigma=3., maxfilter_size=100)
    #     all_xc.append(old_make_ccfs(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[6], mask=None, smoothed_flat=None, delta_log_wl=1e-6, relgrid=False,
    #                             flipped=False, individual_fibres=False, debug_level=1, timit=False))
    #     rv,rverr,xcsum = get_RV_from_xcorr_2(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[6], individual_fibres=True, individual_orders=True, old_ccf=True, debug_level=1)
    sumrv, sumrverr, xcsum = get_RV_from_xcorr_2(f_clean, wl, f0_clean, wl0, bc=bc, bc0=bc0, smoothed_flat=smoothed_flat, fitrange=35, individual_fibres=True,
                                                 individual_orders=False, deg_interp=3, norm_cont=True, fit_slope=False, old_ccf=False, debug_level=1)
    #     sumrv,sumrverr,xcsum = get_RV_from_xcorr_2(f_dblz, wl, f0_dblz, wl0, bc=all_bc[i], bc0=all_bc[6], individual_fibres=False, individual_orders=False, old_ccf=True, debug_level=1)
    #     all_rv[i,:,:] = rv
    all_sumrv.append(sumrv)
#     all_rv[i,:] = sumrv[0,:]
    #     all_sumrv[i] = sumrv
    xcsums[i, :] = xcsum
xcsums = np.array(xcsums)

np.savetxt('/Users/christoph/OneDrive - UNSW/tauceti/rvtest/sep_2019/rvs_' + vers + '.txt', all_sumrv)

# PLOT
plt.plot(all_sumrv, 'x')
plt.xlabel('# obs')
plt.ylabel('dRV [m/s]')
plt.title('Tau Ceti (N=147)   --   ' + vers)
plt.text(100, 30, 'RMS = ' + str(np.round(np.std(all_sumrv), 1)) + ' m/s', size='large')
plt.savefig('/Users/christoph/OneDrive - UNSW/tauceti/rvtest/sep_2019/rvs_' + vers + '.eps')

########################################################################################################################
########################################################################################################################
########################################################################################################################

rvs = np.squeeze(np.array(all_sumrv[10:]))
snr = np.squeeze(np.array(all_snr[10:]))
jds = np.squeeze(np.array(all_jd[10:]))
dates = np.array(all_dates[10:])
udates = set(dates)
wm_rvs = []
wm_rverrs = []
wm_jds = []
nightly_rms = []
nobs = []

for date in sorted(udates):
#     nightix = np.argwhere(dates == date)
    nightix = (dates == date)
    wm_rv, wm_rv_err = wm_and_wsv(rvs[nightix], snr[nightix])
    wm_rvs.append(wm_rv)
    wm_rverrs.append(wm_rv_err)
    wm_jd, wm_jd_err = wm_and_wsv(jds[nightix], snr[nightix])
    wm_jds.append(wm_jd)
    nightly_rms.append(np.std(rvs[nightix]))
    nobs.append(np.sum(nightix))
    print(date, np.sum(nightix), np.std(rvs[nightix]))


plt.scatter(jds - 2.458e6, rvs, marker='.', color='gray')
plt.errorbar(np.array(wm_jds)-2.458e6, wm_rvs, wm_rverrs, color='b', ecolor='b', fmt='o')
plt.ylim(-30,20)
plt.title('del Pav May/June 2019')
plt.ylabel('dRV [m/s]')
plt.xlabel('JD - 2458000.0')



########################################################################################################################
########################################################################################################################
########################################################################################################################

# CROSS-CORRELATION TESTS NOVEMBER 2019


# testing if I can recover correct RV shifts using real data and different CCFs (should also try different SNRs)

path = '/Volumes/WORKING/data/veloce/reduced/delpav/with_lfc/'
date = '20190518'

f = pyfits.getdata(path + '190248_18may30144_optimal3a_extracted_cleaned.fits')
err = pyfits.getdata(path + '190248_18may30144_optimal3a_extracted_cleaned.fits', 1)
wl = pyfits.getdata(path + '190248_18may30144_vac_wl.fits')
wl_lfc = pyfits.getdata(path + '190248_18may30144_vac_lfc_wl.fits')
bc = pyfits.getval(path + '190248_18may30144_optimal3a_extracted.fits', 'BARYCORR')

maskdict = np.load('/Volumes/BERGRAID/data/veloce/reduced/' + date + '/mask.npy').item()
mw_flux = pyfits.getdata('/Volumes/BERGRAID/data/veloce/reduced/' + date + '/master_white_optimal3a_extracted.fits')
smoothed_flat, pix_sens = onedim_pixtopix_variations(mw_flux, filt='gaussian', filter_width=25)

# shifts = [0, 10, 100, 1000, 10000]   # positive = REDshift
shifts = np.linspace(-30e3,30e3,720)
shifts_fine = np.arange(300)
shifts = np.arange(-300,301,10)
shifts = np.sort(np.r_[np.linspace(-30e3,30e3,720), np.arange(-300,301,10)])

all_xc = []
all_xc2 = []
all_sumrv = []
all_sumrv2 = []
all_sumrv3 = []
all_sumrv4 = []
xcsums = np.zeros((len(shifts), 301))
xcsums2 = np.zeros((len(shifts), 301))
xcsums3 = np.zeros((len(shifts), 301))
xcsums4 = np.zeros((len(shifts), 301))
# xcsums = np.zeros((len(shifts), 19, 301))
# xcsums2 = np.zeros((len(shifts), 19, 301))

# speed of light in m/s
c = 2.99792458e8

for i,shift in enumerate(shifts):
    print('Running test observation ' + str(i+1) + '/' + str(len(shifts)) + '...')
#     wl_shifted = wl * np.sqrt((1 + shift / c) / (1 - shift / c)) 
    wl_shifted = wl * np.sqrt((1 - shift / c) / (1 + shift / c)) 
#     all_xc.append(make_ccfs(f, wl, f, wl_shifted, bc=shift, bc0=0, smoothed_flat=smoothed_flat, deg_interp=3, individual_fibres=True, norm_cont=True, debug_level=1))
    sumrv, sumrverr, xcsum = get_RV_from_xcorr_2(f, wl_shifted, f, wl, bc=0, bc0=0, smoothed_flat=smoothed_flat, fitrange=35, individual_fibres=True,
                                                 individual_orders=False, scrunch=False, deg_interp=3, norm_cont=True, fit_slope=False, old_ccf=False, debug_level=1)
    all_sumrv.append(sumrv)
    xcsums[i, :] = xcsum
#     wl_lfc_shifted = wl_lfc * np.sqrt((1 + shift / c) / (1 - shift / c)) 
    wl_lfc_shifted = wl_lfc * np.sqrt((1 - shift / c) / (1 + shift / c)) 
    sumrv2, sumrverr2, xcsum2 = get_RV_from_xcorr_2(f, wl_lfc_shifted, f, wl_lfc, bc=0, bc0=0, smoothed_flat=smoothed_flat, fitrange=35, individual_fibres=True,
                                                    individual_orders=False, scrunch=False, deg_interp=3, norm_cont=True, fit_slope=False, old_ccf=False, debug_level=1)
    all_sumrv2.append(sumrv2)
    xcsums2[i, :] = xcsum2
#     all_xc2.append(make_ccfs(f, wl, f, wl_shifted, bc=shift, bc0=0, smoothed_flat=smoothed_flat, deg_interp=3, individual_fibres=True, norm_cont=False, debug_level=1))
#     sumrv3, sumrverr3, xcsum3 = get_RV_from_xcorr_2(f, wl, f, wl_shifted, bc=shift, bc0=0, smoothed_flat=smoothed_flat, fitrange=35, individual_fibres=False,
#                                                  individual_orders=False, deg_interp=3, norm_cont=False, fit_slope=True, old_ccf=False, debug_level=1)
#     all_sumrv3.append(sumrv3)
#     xcsums3[i, :] = xcsum3
#     sumrv4, sumrverr4, xcsum4 = get_RV_from_xcorr_2(f, wl, f, wl_shifted, bc=shift, bc0=0, smoothed_flat=smoothed_flat, fitrange=35, individual_fibres=False,
#                                                     individual_orders=False, deg_interp=3, norm_cont=False, fit_slope=False, old_ccf=False, debug_level=1)
#     all_sumrv4.append(sumrv4)
#     xcsums4[i, :] = xcsum4

plt.plot(shifts, np.squeeze(all_sumrv) - shifts, 'x')
plt.xlabel('input RV [m/s]')
plt.ylabel('output RV - input RV [m/s]')
plt.title('v4b')
plt.axvline(0, color='gray', linestyle=':')
plt.savefig(path + 'xc_tests/v4b_order_06.eps')

plt.xlim(-350,350)
plt.ylim(-2.,2.)
plt.savefig(path + 'xc_tests/v4b_order_06_zoom.eps')

# diffs = np.squeeze(all_sumrv) - shifts
# diffs2 = np.squeeze(all_sumrv2) - shifts
# diffs3 = np.squeeze(all_sumrv3) - shifts_fine
# 
# print(np.std(diffs), diffs[0])
# print(np.std(diffs2), diffs2[0])
# print(np.std(diffs3), diffs3[0])
#     
# plt.plot(diffs, 'x-')
# plt.plot(diffs2, 'x-')
# plt.plot(diffs3, 'x-')
    
    

# going back one level and testing things on one 1-D spectrum only 
# (either Veloce, or synthetic spectrum (which is thus known to have perfectly flat continuum), or even single (perfect-Gaussian) absorption line     

#################################################################################################################    
### using cross-correlation
#################################################################################################################
    
# make CCfs by shifting the continuum-normalized and rebinned_f0, to subsequently check if I can recover RVs for that
# speed of light in m/s
c = 2.99792458e8

# conversion factor between FWHM and sigma for pure Gaussian (FWHM = 2.355 * sigma)
fac = 2.*np.sqrt(2*np.log(2))

all_xc = []

logwlgrid = np.load('/Volumes/WORKING/data/veloce/reduced/delpav/with_lfc/xc_tests/logwlgrid_o17.npy')
# taper_func = np.load('/Volumes/WORKING/data/veloce/reduced/delpav/with_lfc/xc_tests/taper_func.npy')
# the continuum-normalized and rebinned_f0
### (1) using real Veloce data
# rebinned_f0 = np.load('/Volumes/WORKING/data/veloce/reduced/delpav/with_lfc/xc_tests/rebinned_f0.npy')
### (2) using a synthetic spectrum
# wl0, f0 = readcol('/Users/christoph/OneDrive - UNSW/synthetic_templates/' + 'synth_teff5250_logg45.txt', twod=False)
# # wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/synthetic_templates/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
# # f0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/synthetic_templates/phoenix/lte05400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
# logwlgrid = np.load('/Volumes/WORKING/data/veloce/reduced/delpav/with_lfc/xc_tests/logwlgrid_o36.npy')
# dum_spline_rep = interp.InterpolatedUnivariateSpline(np.log(wl0), f0, k=deg_interp)
# rebinned_f0 = dum_spline_rep(logwlgrid)
### (3) using a single absorption line
# test_f0 = np.ones(len(logwlgrid)) - CMB_pure_gaussian(np.arange(len(logwlgrid)), len(logwlgrid)/2., 10, 0.5)
test_f0 = np.ones(len(logwlgrid))
line_offsets = np.arange(-9000,9001,1000)
# line_offsets = np.array([-500,0, 1500])
line_pos = len(logwlgrid)/2. + line_offsets
# line_depths = [0.5,0.5,0.5]
line_depths = np.ones(len(line_offsets)) * 0.5
for pos,dep in zip(line_pos, line_depths):
    test_f0 -= CMB_pure_gaussian(np.arange(len(logwlgrid)), pos, 30./fac, dep)
rebinned_f0 = test_f0.copy()

taper_width = 0.05

taper_range = int(np.ceil(len(rebinned_f0) * taper_width))
taper_start = np.linspace(-np.pi, 0, taper_range)
taper_end = np.linspace(0, np.pi, taper_range)
taper_func = np.ones(len(rebinned_f0))
taper_func[:taper_range] = (np.cos(taper_start) / 2.) + 0.5
taper_func[-taper_range:] = (np.cos(taper_end) / 2.) + 0.5

deg_interp = 3

for i,shift in enumerate(shifts):
    print('Running test observation ' + str(i+1) + '/' + str(len(shifts)) + '...')
#     logwlgrid_shifted = logwlgrid + np.log(np.sqrt((1 + shift / c) / (1 - shift / c)))
# flipped should be set to TRUE, but can't be bothered to fix that now
    padded_logwlgrid = np.r_[np.arange(np.min(logwlgrid) - 151*delta_log_wl, np.min(logwlgrid) - 0.1*delta_log_wl, delta_log_wl), logwlgrid, 
                             np.arange(np.max(logwlgrid) + delta_log_wl, np.max(logwlgrid) + 151*delta_log_wl, delta_log_wl)]
    padded_rebinned_f0 = np.r_[np.ones(151), rebinned_f0, np.ones(151)]
#     logwlgrid_shifted = logwlgrid - np.log(np.sqrt((1 + shift / c) / (1 - shift / c)))
#     spline_rep = interp.InterpolatedUnivariateSpline(logwlgrid_shifted, rebinned_f0, k=deg_interp)
    logwlgrid_shifted = padded_logwlgrid - np.log(np.sqrt((1 + shift / c) / (1 - shift / c)))
    spline_rep = interp.InterpolatedUnivariateSpline(logwlgrid_shifted, padded_rebinned_f0, k=deg_interp)
    rebreb_f0 = spline_rep(logwlgrid)
#     rebreb_f0 = cmb_scrunch(logwlgrid, logwlgrid_shifted, padded_rebinned_f0)
    f0_in = (rebinned_f0 - 1.)*taper_func
    f_in = (rebreb_f0 - 1.)*taper_func
    f0_in -= np.mean(f0_in)
    f_in -= np.mean(f_in)
#     all_xc.append(xcorr(f0_in, f_in, scale='unbiased'))
    all_xc.append(xcorr(f0_in, f_in, scale='none'))
#     all_xc.append(xcorr(f0_in, f_in, scale='biased'))
#     all_xc.append(xcorr(f0_in, f_in, scale='unbiased'))
#     all_xc.append(xcorr(f0_in, f_in, scale='coeff'))
    
    
    
# speed of light in m/s
c = 2.99792458e8
delta_log_wl = 1e-6
fit_slope = False
addrange = 300
fitrange = 35
    
rv = np.zeros(len(shifts))
rverr = np.zeros(len(shifts))
fitparms = []

# for i in range(61):
for i in range(len(shifts)):
    
#     xc = xcsums[i,:]
#     xc = cen_xcarr[i,:]
    xc = all_xc[i]
    xc_cen = xc[len(xc) // 2 - addrange: len(xc) // 2 + addrange + 1]
    
    # want to fit a symmetric region around the peak, not around the "centre" of the xc
    
    # find peaks (the highest of which we assume is the real one we want) in case the delta-rvabs is non-zero
    peaks = np.r_[True, xc_cen[1:] > xc_cen[:-1]] & np.r_[xc_cen[:-1] > xc_cen[1:], True]
    # filter out maxima too close to the edges to avoid problems
    peaks[:5] = False
    peaks[-5:] = False
#     guessloc = np.argmax(xc*peaks)
    guessloc = np.argmax(xc_cen*peaks) + len(xc) // 2 - addrange
    if guessloc >= len(xc)//2:
        xrange = np.arange(np.minimum(len(xc) - 2*fitrange-1, guessloc - fitrange), np.minimum(guessloc + fitrange + 1, len(xc)), 1)
    else:
        xrange = np.arange(np.maximum(0, guessloc - fitrange), np.maximum(guessloc + fitrange + 1, 2*fitrange+1), 1)
    #             xrange = np.arange(guessloc - fitrange, guessloc + fitrange + 1, 1)
    #           xrange = np.arange(np.argmax(xc) - fitrange, np.argmax(xc) + fitrange + 1, 1)
    
    # make sure we have a dynamic range
    if debug_level >= 3:
        print(xrange)
    xc -= np.min(xc[xrange])
#   or???  xc -= np.min(xc) # doesn't make a difference at all if we are fitting an offset anyway
    # "normalize" it
    xc /= np.max(xc)
    xc *= 0.9
    xc += 0.1
    
#     plt.plot(xc)
    
#     # parameters: mu, sigma, amp, beta
#     guess = np.array([guessloc, fitrange//3, 0.9, 2.])
#     print('slope = FALSE')
#     print('offset = FALSE')
#     try:
#         # subtract the minimum of the fitrange so as to have a "dynamic range"
#         popt, pcov = op.curve_fit(fibmodel_with_amp, xrange, xc[xrange], p0=guess, maxfev=1000000)
#         mu = popt[0]
#         mu_err = np.sqrt(pcov[0, 0])
#         if debug_level >= 1:
#             print('Fit successful...')
#     except:
#         popt, pcov = (np.nan, np.nan)
#         mu = np.nan
#         mu_err = np.nan
    
    if fit_slope:
        print('slope = TRUE')
        print('offset = TRUE')
        # parameters: mu, sigma, amp, beta, offset, slope
        guess = np.array([guessloc, fitrange//3, 0.9, 2., np.min(xc[xrange]), 0.])
         
        try:
            # subtract the minimum of the fitrange so as to have a "dynamic range"
            popt, pcov = op.curve_fit(gausslike_with_amp_and_offset_and_slope, xrange, xc[xrange], p0=guess, maxfev=1000000)
            mu = popt[0]
            mu_err = np.sqrt(pcov[0, 0])
            if debug_level >= 1:
                print('Fit successful...')
        except:
            popt, pcov = (np.nan, np.nan)
            mu = np.nan
            mu_err = np.nan
    else:
        print('slope = FALSE')
        print('offset = TRUE')
        # parameters: mu, sigma, amp, beta, offset
        guess = np.array([guessloc, fitrange//3, 0.9, 2., np.min(xc[xrange])])
     
        try:
            # subtract the minimum of the fitrange so as to have a "dynamic range"
            popt, pcov = op.curve_fit(gausslike_with_amp_and_offset, xrange, xc[xrange], p0=guess, maxfev=1000000)
            mu = popt[0]
            mu_err = np.sqrt(pcov[0, 0])
            if debug_level >= 1:
                print('Fit successful...')
        except:
            popt, pcov = (np.nan, np.nan)
            mu = np.nan
            mu_err = np.nan
    
    # convert to RV in m/s
    rv[i] = c * (mu - (len(xc) // 2)) * delta_log_wl
    rverr[i] = c * mu_err * delta_log_wl
    fitparms.append(popt)
    
    
# mu_list = [fp[0] for fp in fitparms]
# phi = [mu - np.floor(mu) for mu in mu_list]
# plt.plot(phi, rv-shifts,'rx', label='linear interp.')
# plt.plot(phi, rv-shifts,'b+', label='cubic spline interp.')

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

#################################################################################################################    
### using template matching / shift-and_fit
#################################################################################################################

shifts = np.sort(np.r_[np.linspace(-30e3,30e3,720), np.arange(-300,301,10)])

# make CCfs by shifting the continuum-normalized and rebinned_f0, to subsequently check if I can recover RVs for that
# speed of light in m/s
c = 2.99792458e8

# conversion factor between FWHM and sigma for pure Gaussian (FWHM = 2.355 * sigma)
fac = 2.*np.sqrt(2*np.log(2))

all_xc = []

logwlgrid = np.load('/Volumes/WORKING/data/veloce/reduced/delpav/with_lfc/xc_tests/logwlgrid_o17.npy')
wl0 = np.exp(logwlgrid)

### (1) using real Veloce data
f0 = np.load('/Volumes/WORKING/data/veloce/reduced/delpav/with_lfc/xc_tests/rebinned_f0.npy')
### (2) using a synthetic spectrum
# wl0, f0 = readcol('/Users/christoph/OneDrive - UNSW/synthetic_templates/' + 'synth_teff5250_logg45.txt', twod=False)
### (3) using a single absorption line
# # test_f0 = np.ones(len(logwlgrid)) - CMB_pure_gaussian(np.arange(len(logwlgrid)), len(logwlgrid)/2., 10, 0.5)
# test_f0 = np.ones(len(logwlgrid))
# line_offsets = np.arange(-9000,9001,1000)
# # line_offsets = np.array([-500,0, 1500])
# line_pos = len(logwlgrid)/2. + line_offsets
# # line_depths = [0.5,0.5,0.5]
# line_depths = np.ones(len(line_offsets)) * 0.5
# for pos,dep in zip(line_pos, line_depths):
#     test_f0 -= CMB_pure_gaussian(np.arange(len(logwlgrid)), pos, 30./fac, dep)
# f0 = test_f0.copy()

f = f0.copy()
err = np.sqrt(f)
bad_threshold = 10

# initialise output arrays
# npix = len(f0)
# rv = np.zeros(len(shifts))
# rverr = np.zeros(len(shifts))
(nord, nfib, npix) = f.shape
rvs = np.zeros((nord,nfib))
rv_sigs = np.zeros((nord,nfib))
redchi2_arr = np.zeros((nord,nfib))
fitparms = []

for i,shift in enumerate(shifts):
    print('Running test observation ' + str(i+1) + '/' + str(len(shifts)) + '...')
    
    wl = wl0 * np.sqrt((1 + shift / c) / (1 - shift / c))
    
    # initial fit parameters
    initp = np.zeros(4)
    initp[0] = np.random.normal(shift, 1000)
#     initp[3] = 0.5
#     print('initp = ', initp)

    print("Order ")
    use_orders = np.arange(nord)
#     use_orders = [5, 6, 17, 25, 27, 31, 36]
#     use_orders = [17]
    
    # loop over all orders
    for o in use_orders:         # at the moment 17 and 34 give the lowest scatter

        print(str(o+1)),
        
        ord_wl = wl[o,:,:]
        ord_f = f[o,:,:]
        ord_err = err[o,:,:]
        ord_wl0 = wl0[o,:,:]
        ord_f0 = f0[o,:,:]
        
        nbad = 0
        
        # wavelength array must be increasing for "InterpolatedUnivariateSpline" to work --> turn arrays around if necessary!!!
        if (np.diff(ord_wl) < 0).any():
            ord_wl_sorted = ord_wl[:,::-1].copy()
            ord_f_sorted = ord_f[:,::-1].copy()
            ord_err_sorted = ord_err[:,::-1].copy()
            ord_wl0_sorted = ord_wl0[:,::-1].copy()
            ord_f0_sorted = ord_f0[:,::-1].copy()
        else:
            ord_wl_sorted = ord_wl.copy()
            ord_f_sorted = ord_f.copy()
            ord_err_sorted = ord_err.copy()
            ord_wl0_sorted = ord_wl0.copy()
            ord_f0_sorted = ord_f0.copy()
        
        
        # loop over all fibres
#         for fib in range(nfib):
        for fib in range(3,22):
            
            spl_ref = interp.InterpolatedUnivariateSpline(ord_wl0_sorted[fib,:], ord_f0_sorted[fib,:], k=3)
            args = (ord_wl_sorted[fib,:], ord_f_sorted[fib,:], ord_err_sorted[fib,:], spl_ref)
        
            # Remove edge effects in a slightly dodgy way by making their error bars infinity and hence giving them zero weights (20 pixels is about 30km/s) 
            args[2][:50] = np.inf
            args[2][-50:] = np.inf
            
#             model_init = rv_shift_resid(initp, *args, return_spect=True)
    
            # calculate the model and residuals starting with initial parameters
            # I don't have the Jacobian yet when I am using the proper relativistic Doppler shift, so using Dfun=None for now (still seems to work fine)
#             the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True, xtol=1e-6)
            the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=None, full_output=True, epsfcn=1e-3)   
#             the_fit[0] are the best-fit parms
#             the_fit[1] is the covariance matrix
#             the_fit[2] is auxiliary information on the fit (in form of a dictionary)
#             the_fit[3] is a string message giving information about the cause of failure.
#             the_fit[4] is an integer flag 
            # (if it is equal to 1, 2, 3 or 4, the solution was found. Otherwise, the solution was not found -- see op.leastsq documentation for more info)
#             model_before = rv_shift_resid(the_fit[0], *args, return_spect=True)
            resid_before = rv_shift_resid(the_fit[0], *args)
            wbad = np.where(np.abs(resid_before) > bad_threshold)[0]   # I should probably return the abs residuals and then divide them by the error outside "rv_shift_resid"
            nbad += len(wbad)
            
            chi2 = rv_shift_chi2(the_fit[0], *args)
            redchi2 = chi2 / (npix - len(initp))
    
#             fitted_spec = rv_shift_resid(the_fit[0], *args, return_spect=True)
            
            #################################################################
            # now repeat after removing outliers
            #################################################################
                        
            # make the errors for the "bad" pixels infinity (so as to make the weights zero)
            args[2][np.where(np.abs(resid_before) > bad_threshold)] = np.inf
        
#             the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True, xtol=1e-6)
            the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=None, full_output=True, epsfcn=1e-3)
#             residAfter = rv_shift_resid( the_fit[0], *args)
            chi2 = rv_shift_chi2(the_fit[0], *args)
            redchi2 = chi2 / (npix - len(initp))
            redchi2_arr[o,fib] = redchi2
            
            # Save the fit and the uncertainty (the_fit[0][0] is the RV shift)
#             fitparms.append(the_fit[0])
#             rv[i] = the_fit[0][0]
            rvs[o,fib] = the_fit[0][0]
#             rverr[i] = np.sqrt(the_fit[1][0,0])
            try:
                rv_sigs[o,fib] = np.sqrt(the_fit[1][0,0])
#                 rv_sigs[o,fib] = np.sqrt(redchi2 * the_fit[1][0,0])
            except:
                rv_sigs[o,fib] = np.nan


    
    
plt.plot(shifts, rv-shifts, 'x')
    

#################################################################################################################    
### OK, now let's try using template matching / shift-and-fit on real observations
#################################################################################################################

# TEMPLATE:
# # tau Ceti
# ix0 = 7   # tau Ceti
# date_0 = '20180919'
# del Pav
ix0 = 36
date_0 = '20190518'

f0 = pyfits.getdata(files[ix0], 0)
err0 = pyfits.getdata(files[ix0], 1)
obsname_0 = all_obsnames[ix0]
wl0 = pyfits.getdata(wls[ix0])
bc0 = pyfits.getval(files[ix0], 'BARYCORR')

maskdict = np.load('/Volumes/BERGRAID/data/veloce/reduced/' + date_0 + '/mask.npy').item()

# wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')

mw_flux = pyfits.getdata('/Volumes/BERGRAID/data/veloce/reduced/' + date_0 + '/master_white_optimal3a_extracted.fits')
smoothed_flat, pix_sens = onedim_pixtopix_variations(mw_flux, filt='gaussian', filter_width=25)

f0_clean = pyfits.getdata(path + '190248_' + obsname_0 + '_optimal3a_extracted_cleaned.fits')


all_rv = []
all_rverr = []
all_redchi2 = []

# loop over all observations
for i,filename in enumerate(files[:9]):
    print('Processing RV for observation ' + str(i + 1) + '/' + str(len(files)))
    # get obsname and date
    obsname = filename.split('_')[-3]
    utdate = pyfits.getval(filename, 'UTDATE')
    date = utdate[:4] + utdate[5:7] + utdate[8:]

    # read in spectrum
#     f = pyfits.getdata(filename, 0)
    err = pyfits.getdata(filename, 1)
    wl = pyfits.getdata(wls[i])
    bc = pyfits.getval(filename, 'BARYCORR')
    f_clean = pyfits.getdata(path + '190248_' + obsname + '_optimal3a_extracted_cleaned.fits')
    
    # call shift-and-fit RV routine (output format of rvs is (n_ord, n_fib) = (38,19)
    use_orders = [5,6,7,17,26,27,28,30,31,34,35,36]   # these give the best RVs (ie least affected by tellurics) when doing shit-and-fit RVs
    rvs, rv_errs, redchi2 = calculate_rv_shift(f0_clean[:,3:22,:], wl0[:,3:22,:], f_clean[:,3:22,:], wl[:,3:22,:], err[:,3:22,:], bc=bc, bc0=bc0, edge=50)
#     rvs, rv_errs, redchi2 = calculate_rv_shift(f0_clean[17:18,3:22,:], wl0[17:18,3:22,:], f_clean[17:18,3:22,:], wl[17:18,3:22,:], err[17:18,3:22,:], bc=bc, bc0=bc0, edge=50)
    
    all_rv.append(rvs)
    all_rverr.append(rv_errs)
    all_redchi2.append(redchi2)
    


















