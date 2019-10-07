import glob
import time
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from readcol import readcol

from veloce_reduction.veloce_reduction.wavelength_solution import get_dispsol_for_all_fibs, get_dispsol_for_all_fibs_2
from veloce_reduction.veloce_reduction.get_radial_velocity import get_RV_from_xcorr, get_RV_from_xcorr_2, make_ccfs, \
    old_make_ccfs
from veloce_reduction.veloce_reduction.helper_functions import get_snr, short_filenames, wm_and_wsv
from veloce_reduction.veloce_reduction.flat_fielding import onedim_pixtopix_variations, deblaze_orders
from veloce_reduction.veloce_reduction.barycentric_correction import get_barycentric_correction
from veloce_reduction.veloce_reduction.cosmic_ray_removal import onedim_medfilt_cosmic_ray_removal

########################################################################################################################
# HOUSEKEEPING
# path = '/Users/christoph/data/reduced/tauceti/tauceti_with_LFC/'
path = '/Volumes/BERGRAID/data/veloce/reduced/tauceti/tauceti_with_LFC/'
# path = '/Volumes/BERGRAID/data/veloce/reduced/TOI375/with_lfc/'
# path = '/Volumes/BERGRAID/data/veloce/reduced/delpav/with_lfc/'
# path = '/Volumes/BERGRAID/data/veloce/reduced/GJ674/with_lfc/'

file_list = glob.glob(path + '*optimal3a_extracted.fits')
wl_list = glob.glob(path + '*wl*')
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
all_bc = np.array(bc_list)[sortix]
all_jd = utmjd[sortix]

all_dates = [pyfits.getval(file, 'UTDATE')[:4] + pyfits.getval(file, 'UTDATE')[5:7] + pyfits.getval(file, 'UTDATE')[8:] for file in files]
unique_dates = set(all_dates)


########################################################################################################################
########################################################################################################################
########################################################################################################################

all_snr = readcol(path + 'tauceti_all_snr.dat', twod=False)[0]

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


# calculating the CCFs for one order / 11 orders
# (either with or without the LFC shifts applied, comment out the 'wl' and 'wl0' you don't want)

vers = 'v2d'

all_xc = []
all_rv = np.zeros((len(files), 19))
# all_rv = np.zeros((len(files), 10, 19))
# all_rv = np.zeros((len(files), 11, 19))
all_sumrv = []
# all_sumrv = np.zeros(len(files))
# all_sumrv_2 = np.zeros(len(files))
# xcsums = np.zeros((len(files), 301))
xcsums = np.zeros((len(files), 19, 301))

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

# f0_clean = f0.copy()
# # loop over orders
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


# loopo over all observations
for i,filename in enumerate(files):
    print('Processing RV for observation ' + str(i + 1) + '/' + str(len(files)))
#     # get obsname and date
#     obsname = all_obsnames[i]
    obsname = filename.split('_')[-3]
    utdate = pyfits.getval(filename, 'UTDATE')
    date = utdate[:4] + utdate[5:7] + utdate[8:]
#     dum = filename.split('/')
#     dum2 = dum[-1].split('.')
#     dum3 = dum2[0].split('_')
#     obsname = dum3[1]
#     day = obsname[:2]
#     mon = obsname[2:5]
#     if mon == 'jan':
#         year = '2019'
#         mondig = '01'
#     elif mon == 'sep':
#         year = '2018'
#         mondig = '09'
#     elif mon == 'nov':
#         year = '2018'
#         mondig = '11'
#     date = year + mondig + day
    # read in spectrum
#     f = pyfits.getdata(filename, 0)
    err = pyfits.getdata(filename, 1)
    wl = pyfits.getdata(wls[i])
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
    f_dblz, err_dblz = deblaze_orders(f_clean, mw_flux, mask=maskdict, err=err, combine_fibres=True, degpol=5, gauss_filter_sigma=3., maxfilter_size=100)
    #     all_xc.append(old_make_ccfs(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[6], mask=None, smoothed_flat=None, delta_log_wl=1e-6, relgrid=False,
    #                             flipped=False, individual_fibres=False, debug_level=1, timit=False))
    #     rv,rverr,xcsum = get_RV_from_xcorr_2(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[6], individual_fibres=True, individual_orders=True, old_ccf=True, debug_level=1)
    sumrv, sumrverr, xcsum = get_RV_from_xcorr_2(f_dblz, wl, f0_dblz, wl0, bc=bc, bc0=bc0, smoothed_flat=None, fitrange=35, individual_fibres=False,
                                                 individual_orders=False, deg_interp=1, norm_cont=True, fit_slope=True, old_ccf=False, debug_level=1)
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











