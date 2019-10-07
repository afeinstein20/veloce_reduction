import glob
import time
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from readcol import readcol

from veloce_reduction.veloce_reduction.wavelength_solution import get_dispsol_for_all_fibs, get_dispsol_for_all_fibs_2
from veloce_reduction.veloce_reduction.get_radial_velocity import get_RV_from_xcorr, get_RV_from_xcorr_2, make_ccfs, old_make_ccfs
from veloce_reduction.veloce_reduction.helper_functions import get_snr
from veloce_reduction.veloce_reduction.flat_fielding import onedim_pixtopix_variations, deblaze_orders
from veloce_reduction.veloce_reduction.barycentric_correction import get_barycentric_correction
from veloce_reduction.veloce_reduction.cosmic_ray_removal import onedim_medfilt_cosmic_ray_removal


########################################################################################################################
# HOUSEKEEPING
# path = '/Users/christoph/data/reduced/tauceti/tauceti_with_LFC/'
path = '/Volumes/BERGRAID/data/veloce/reduced_archive/v5_pre_July2019/tauceti/tauceti_with_LFC/'

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
    # get UT obs start time
    utmjd = pyfits.getval(filename, 'UTMJD') + 2.4e6 + 0.5  # the fits header has 2,400,000.5 subtracted!!!!!
    # add half the exposure time in days
    texp = pyfits.getval(filename, 'ELAPSED')
    utmjd = utmjd + (texp / 2.) / 86400.
    all_jd.append(utmjd)
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

# calculating the CCFs for one order / 11 orders
# (either with or without the LFC shifts applied, comment out the 'wl' and 'wl0' you don't want)

vers = 'v1c'
maskdict = np.load('/Volumes/BERGRAID/data/veloce/reduced/20190128/mask.npy').item()
# mw_flux = pyfits.getdata('/Volumes/BERGRAID/data/veloce/reduced/20190128/master_white_optimal3a_extracted.fits')
# smoothed_flat, pix_sens = onedim_pixtopix_variations(mw_flux, filt='gaussian', filter_width=25)

all_xc = []
all_rv = np.zeros((len(files), 11, 19))
all_sumrv = []
# all_sumrv = np.zeros(len(files))
# all_sumrv_2 = np.zeros(len(files))
xcsums = np.zeros((len(files), 301))

# TEMPLATE:
ix0 = 6
f0 = pyfits.getdata(files[ix0], 0)   # one of the higher SNR obs but closest to FibThars used to define fibtofib wl shifts
# f0 = pyfits.getdata(files[69], 0)   # that's the highest SNR observation for Sep 18
# f0 = pyfits.getdata(files[2], 0)   # that's the highest SNR observation for Nov 18
# f0 = pyfits.getdata(files[35], 0)   # that's the 2nd highest SNR observation for Nov 18
err0 = pyfits.getdata(files[ix0], 1)
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
# wldict0,wl0 = get_dispsol_for_all_fibs_2(obsname_0, refit=True, eps=2)
mw_flux = pyfits.getdata('/Volumes/BERGRAID/data/veloce/reduced/' + date_0 + '/master_white_optimal3a_extracted.fits')
smoothed_flat, pix_sens = onedim_pixtopix_variations(mw_flux, filt='gaussian', filter_width=25)

f0_clean = f0.copy()
  
for o in range(f0.shape[0]):
    for fib in range(f0.shape[1]):
        print('Order ',o+1)
        if (o==0) and (fib==0):
            start_time = time.time()
        f0_clean[o,fib,:],ncos = onedim_medfilt_cosmic_ray_removal(f0[o,fib,:], err0[o,fib,:], w=31, thresh=5., low_thresh=3.)
        if (o==38) and (fib==18):
            print('time elapsed ',time.time() - start_time, ' seconds')
          
f0_dblz, err0_dblz = deblaze_orders(f0_clean, smoothed_flat[:,2:21,:], maskdict, err=err0, combine_fibres=True, degpol=2, gauss_filter_sigma=3., maxfilter_size=100)    
                                          

# use a synthetic template?
# wl0, f0 = readcol('/Users/christoph/OneDrive - UNSW/synthetic_templates/' + 'synth_teff5250_logg45.txt', twod=False)
# wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/synthetic_templates/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
# f0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/synthetic_templates/phoenix/lte05400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')


# for i,filename in enumerate(files[73:83]):
for i,filename in enumerate(files):
#     i += 73
    print('Processing RV for tau Ceti observation ' + str(i+1) + '/' + str(len(files)))
    # get obsname and date
    dum = filename.split('/')
    dum2 = dum[-1].split('.')
    dum3 = dum2[0].split('_')
    obsname = dum3[1]
    day = obsname[:2]
    mon = obsname[2:5]
    if mon == 'jan':
        year = '2019'
        mondig = '01'
    elif mon == 'sep':
        year = '2018'
        mondig = '09'
    elif mon == 'nov':
        year = '2018'
        mondig = '11'
    date = year + mondig + day
    # read in spectrum
    f = pyfits.getdata(filename, 0)
    err = pyfits.getdata(filename, 1)
    # f_clean = f.copy()
    # for o in range(f.shape[0]):
    #     for fib in range(f.shape[1]):
    #         f_clean[o,fib,:],ncos = onedim_medfilt_cosmic_ray_removal(f[o,fib,:], err[o,fib,:], w=31, thresh=5., low_thresh=3.)
#     wl = pyfits.getdata(filename, 2)
#     wl = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')
#     wldict,wl = get_dispsol_for_all_fibs(obsname, date=date, fibs='stellar', refit=False, fibtofib=True, nightly_coeffs=True)
    wl = pyfits.getdata(path + 'wl_' + obsname + '_surface.fits')
#     wldict, wl = get_dispsol_for_all_fibs_2(obsname, refit=True, eps=2)
#     f_dblz, err_dblz = deblaze_orders(f_clean, smoothed_flat[:,2:21,:], maskdict, err=err, combine_fibres=True, skip_first_order=True, degpol=2, gauss_filter_sigma=3., maxfilter_size=100)   
#     all_xc.append(old_make_ccfs(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[6], mask=None, smoothed_flat=None, delta_log_wl=1e-6, relgrid=False,
#                             flipped=False, individual_fibres=False, debug_level=1, timit=False))
#     rv,rverr,xcsum = get_RV_from_xcorr_2(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[6], individual_fibres=True, individual_orders=True, old_ccf=True, debug_level=1)
    sumrv,sumrverr,xcsum = get_RV_from_xcorr_2(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[ix0], smoothed_flat=None, fitrange=35, individual_fibres=False, individual_orders=False, 
                                               fit_slope=True, old_ccf=False, debug_level=1)
#     sumrv,sumrverr,xcsum = get_RV_from_xcorr_2(f_dblz, wl, f0_dblz, wl0, bc=all_bc[i], bc0=all_bc[6], individual_fibres=False, individual_orders=False, old_ccf=True, debug_level=1)
#     all_rv[i,:,:] = rv
    all_sumrv.append(sumrv)
#     all_sumrv[i] = sumrv
    xcsums[i,:] = xcsum
xcsums = np.array(xcsums)

np.savetxt('/Users/christoph/OneDrive - UNSW/tauceti/rvtest/jul_2019/rvs_' + vers + '.txt', all_sumrv)

# PLOT
plt.plot(all_sumrv, 'x')
plt.xlabel('# obs')
plt.ylabel('dRV [m/s]')
plt.title('Tau Ceti (N=151)   --   ' + vers)
plt.text(100, 60, 'RMS = ' + str(np.round(np.std(all_sumrv),1)) + ' m/s', size='large')
plt.savefig('/Users/christoph/OneDrive - UNSW/tauceti/rvtest/jul_2019/rvs_' + vers + '.eps')

########################################################################################################################
########################################################################################################################
########################################################################################################################

