'''
Created on 21 Nov. 2017

@author: christoph
'''

import glob
import numpy as np
import astropy.io.fits as pyfits
import datetime
from scipy.signal import savgol_filter
from scipy import interpolate


from veloce_reduction.veloce_reduction.order_tracing import find_stripes, make_P_id, extract_stripes, flatten_single_stripe
from veloce_reduction.veloce_reduction.chipmasks import get_mean_fibre_separation




# fp_in = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/individual_fibre_profiles_20180924.npy').item()

def make_real_fibparms_by_ord(fp_in, degpol=7, savefile=True, date=None, simthxe=False, lfc=False, nx=4112,
                              path = '/Users/christoph/OneDrive - UNSW/fibre_profiles/archive/'):
    
    '''
    PURPOSE:
    Take the fitted values at the 25 pixel intervals and smooth them, and save them in the format that 
    "make_norm_profiles_5" can digest when calculating the normalized fibre profiles during optimal extraction.
    
    INPUT:
    'fp_in'    : dictionary containing the fibre profiles and traces from "fit_multiple_profiles(_from_indices)"
    'degpol'   : degree of the polynomial for fitting the fibre traces
    'savefile  : boolean - do you want to save the resulting output dictionary to file? 
    'date'     : date for the output filename - should be in standard string format 'YYYYMMDD'
    'simthxe'  : boolean - set to TRUE if creating the fibre profiles for the sim ThXe fibre
    'lfc'      : boolean - set to TRUE if creating the fibre profiles for the sim LFC fibre
    'nx'       : number of pixels in dispersion direction

    OUTPUT:
    'fibparms' : dictionary containing the smoothed fibre profiles and traces in the right format for "make_norm_profiles_5"
    '''
    
    xx = np.arange(nx)

    fibparms = {}

    for ord in sorted(fp_in.keys()):

        print('OK, processing ', ord)

        fibparms[ord] = {}

        #sanity check the dimensions are right
        nfib = fp_in[ord]['mu'].shape[1]
        if simthxe or lfc:
            assert nfib == 1, 'ERROR: input dictionary does NOT contain data for exactly 1 fibre!!!'
        else:
            assert nfib == 24, 'ERROR: input dictionary does NOT contain data for all 24 fibres!!!'


        # fibre numbers here increase from red to blue, ie from ThXe side to LFC side, as in:
        # pseudo-slit layout:   S5 S2 X 7 18 17 6 16 15 5 14 13  1 12 11  4 10  9  3  8 19  2 X S4 S3 S1
        # array indices     :    0  1   2  3  4 5  6  7 8  9 10 .................................. 22 23

        pix = np.array(fp_in[ord]['pix'])
        snr = np.array(fp_in[ord]['SNR'])

        #these are the "names" of the stellar and sky fibres (01=LFC, 05=blank, 25=blank, 28=ThXe)
        if simthxe:
            allfibs = ['fibre_28']
        elif lfc:
            allfibs = ['fibre_01']
        else:
            allfibs = ['fibre_02', 'fibre_03', 'fibre_04', 'fibre_06', 'fibre_07', 'fibre_08', 'fibre_09', 'fibre_10',
                       'fibre_11', 'fibre_12', 'fibre_13', 'fibre_14', 'fibre_15', 'fibre_16', 'fibre_17', 'fibre_18',
                       'fibre_19', 'fibre_20', 'fibre_21', 'fibre_22', 'fibre_23', 'fibre_24', 'fibre_26', 'fibre_27']

        # loop over 24 fibres (19 stellar + 5 sky) - NOTE the flipping - has to be that way!!!
        for i,fib in enumerate(allfibs[::-1]):

            fibparms[ord][fib] = {}

            mu = np.array(fp_in[ord]['mu'][:,i])
            sigma = np.array(fp_in[ord]['sigma'][:,i])
            beta = np.array(fp_in[ord]['beta'][:,i])

            goodmu = mu > 0
            goodsigma = sigma > 0
            goodbeta = beta > 0

            if (goodmu == goodsigma).all() and (goodmu == goodbeta).all():
                good = goodmu.copy()
                del goodmu
                del goodsigma
                del goodbeta
            else:
                print('ERROR: "mu", "sigma" and "beta" do not have the same dimensions for ', ord)
                return

            # define weights for the fitting based on the SNR
            w = np.array(fp_in[ord]['SNR']) ** 2

            # now fit polynomial to the fit parameters of profile measurements across the order
            mu_fit = np.poly1d(np.polyfit(pix[good], mu[good], degpol, w=w[good]))
#             sigma_fit = np.poly1d(np.polyfit(pix[good], sigma[good], degpol, w=w[good]))
#             beta_fit = np.poly1d(np.polyfit(pix[good], beta[good], degpol, w=w[good]))

            # instead of fitting, do a smoothing and linear extrapolation instead for sigma and beta
            # (otherwise we get bad oscillations of the polynomials, ie Runge's phenomenon !!!)
            xgrid = np.arange(np.min(pix[good]), np.max(pix[good]) + 1, 1)
            f_sigma = interpolate.interp1d(pix[good], sigma[good], fill_value='extrapolate')
            f_beta = interpolate.interp1d(pix[good], beta[good], fill_value='extrapolate')
            sigma_eqspace = f_sigma(xgrid)
            beta_eqspace = f_beta(xgrid)
            filtered_sigma = savgol_filter(sigma_eqspace, np.minimum(len(xgrid), 2001), 3)   # window size and order were just eye-balled to make it sensible
            filtered_beta = savgol_filter(beta_eqspace, np.minimum(len(xgrid), 2001), 3)     # window size and order were just eye-balled to make it sensible
            sigma_fit = interpolate.interp1d(xgrid, filtered_sigma, fill_value='extrapolate')
            beta_fit = interpolate.interp1d(xgrid, filtered_beta, fill_value='extrapolate')

            # TODO: add nice plots if debug_level>2 or sth

            # save fit parameters to dictionary - they will be used by "make_norm_profiles_5" to create the
            # normalized profiles during optimal extraction
            fibparms[ord][fib]['mu_fit'] = mu_fit(xx)
            fibparms[ord][fib]['sigma_fit'] = sigma_fit(xx)
            fibparms[ord][fib]['beta_fit'] = beta_fit(xx)
            # fibparms[fib][ord]['offset_fit'] = offset_fit
            # fibparms[fib][ord]['onchip'] = onchip


    if savefile:
        if simthxe:
            issim = 'sim_ThXe_'
        elif lfc:
            issim = 'LFC_'
        else:
            issim = ''
        if date is None:
            now = datetime.datetime.now()
            date = str(now)[:10].replace('-','')
        np.save(path + issim + 'fibre_profile_fits_' + date + '.npy', fibparms)

    return fibparms



def old_make_real_fibparms_by_ord(fp_in, savefile=True, degpol=6):

    path = '/Users/christoph/OneDrive - UNSW/fibre_profiles/'

    fibparms = {}

    for ord in sorted(fp_in.keys()):

        print('OK, processing ',ord)

        fibparms[ord] = {}

        #sanity check the dimensions are right
        nfib = fp_in['order_01']['mu'].shape[1]
        if nfib != 24:
            print('ERROR: input dictionary does NOT contain data for all 24 fibres!!!')
            return

        # fibre numbers here increase from red to blue, ie from ThXe side to LFC side, as in:
        # pseudo-slit layout:   S5 S2 X 7 18 17 6 16 15 5 14 13  1 12 11  4 10  9  3  8 19  2 X S4 S3 S1
        # array indices     :    0  1   2  3  4 5  6  7 8  9 10 .................................. 22 23

        pix = np.array(fp_in[ord]['pix'])
        snr = np.array(fp_in[ord]['SNR'])

        #these are the "names" of the stellar and sky fibres (01=LFC, 05=blank, 25=blank, 28=ThXe)
        allfibs = ['fibre_02', 'fibre_03', 'fibre_04', 'fibre_06', 'fibre_07', 'fibre_08', 'fibre_09', 'fibre_10',
                   'fibre_11', 'fibre_12', 'fibre_13', 'fibre_14', 'fibre_15', 'fibre_16', 'fibre_17', 'fibre_18',
                   'fibre_19', 'fibre_20', 'fibre_21', 'fibre_22', 'fibre_23', 'fibre_24', 'fibre_26', 'fibre_27']

        # loop over 24 fibres (19 stellar + 5 sky)
        # for i in range(24):
        for i,fib in enumerate(allfibs[::-1]):

            fibparms[ord][fib] = {}

            mu = np.array(fp_in[ord]['mu'][:,i])
            sigma = np.array(fp_in[ord]['sigma'][:,i])
            beta = np.array(fp_in[ord]['beta'][:,i])

            goodmu = mu > 0
            goodsigma = sigma > 0
            goodbeta = beta > 0

            if (goodmu == goodsigma).all() and (goodmu == goodbeta).all():
                good = goodmu.copy()
                del goodmu
                del goodsigma
                del goodbeta
            else:
                print('ERROR: "mu", "sigma" and "beta" do not have the same dimensions for ', ord)
                return

            # define weights for the fitting based on the SNR
            w = np.array(fp_in[ord]['SNR']) ** 2

            # now fit polynomial to the fit parameters of profile measurements across the order
            mu_fit = np.poly1d(np.polyfit(pix[good], mu[good], degpol, w=w[good]))
            sigma_fit = np.poly1d(np.polyfit(pix[good], sigma[good], degpol, w=w[good]))
            beta_fit = np.poly1d(np.polyfit(pix[good], beta[good], degpol, w=w[good]))

            # TODO: add nice plots if debug_level>2 or sth

            # save fit parameters to dictionary - they will be used by "make_norm_profiles_3" to create the
            # normalized profiles during optimal extraction
            fibparms[ord][fib]['mu_fit'] = mu_fit
            fibparms[ord][fib]['sigma_fit'] = sigma_fit
            fibparms[ord][fib]['beta_fit'] = beta_fit
            # fibparms[fib][ord]['offset_fit'] = offset_fit
            # fibparms[fib][ord]['onchip'] = onchip


    if savefile:
        now = datetime.datetime.now()
        np.save(path + 'fibre_profile_fits_'+str(now)[:10].replace('-','')+'.npy', fibparms)

    return fibparms



def make_fibparms_by_fib(savefile=True):

    path = '/Users/christoph/OneDrive - UNSW/fibre_profiles/'
    fp_files = glob.glob(path+"sim/"+"fibre_profiles*.npy")
    #mask_files = glob.glob(path+"masks/"+"mask*.npy")
    
    fibparms = {}
    
    for file in fp_files:
        fibnum = file[-6:-4]
        fib = 'fibre_'+fibnum
        fp = np.load(file).item()
        mask = np.load(path+"masks/"+"mask_"+fibnum+".npy").item()
        flatname = '/Volumes/BERGRAID/data/simu/veloce_flat_t70000_single_fib'+fibnum+'.fit'
        flat = pyfits.getdata(flatname)
        img = flat + 1.
    
        #we need the flux later only to weight the fits
        P,tempmask = find_stripes(flat, deg_polynomial=2)
        P_id = make_P_id(P)
        #mask = make_mask_dict(tempmask)
        stripes = extract_stripes(img, P_id, slit_height=10)
    
        fibparms[fib] = {}
        for ord in sorted(fp.iterkeys()):
            
            fibparms[fib][ord] = {}
            
            mu = np.array(fp[ord]['mu'])
            #amp = np.array(fp[ord]['amp'])
            sigma = np.array(fp[ord]['sigma'])
            beta = np.array(fp[ord]['beta'])
            #offset = np.array(fp[ord]['offset'])
            #slope = np.array(fp[ord]['slope'])
            
            onchip = mu>=0
            good = np.logical_and(mu>=0, mask[ord])
            
            stripe = stripes[ord]
            sc,sr = flatten_single_stripe(stripe,slit_height=10,timit=False)
            fluxsum = np.sum(sc,axis=0)
            w = np.sqrt(fluxsum)   #use relative error so that the order centres receive the largest weight-contribution
            
            xx = np.arange(len(mu))
            mu_fit = np.poly1d(np.polyfit(xx[good], mu[good], 5, w=w[good]))
            # xarr = xx*4*np.pi/np.max(xx) - 2*np.pi
            # amp_popt,amp_pcov = curve_fit(blaze,xarr,amp,sigma=w,p0=(0.2,np.max(amp),0.))
            sigma_fit = np.poly1d(np.polyfit(xx[good], sigma[good], 2, w=w[good]))
            beta_fit = np.poly1d(np.polyfit(xx[good], beta[good], 2, w=w[good]))
            #offset_fit = np.poly1d(np.polyfit(xx[good], offset[good], 0, w=w[good]))
            #offset_fit = np.average(offset, weights=w)
        
            fibparms[fib][ord]['mu_fit'] = mu_fit
            fibparms[fib][ord]['sigma_fit'] = sigma_fit
            fibparms[fib][ord]['beta_fit'] = beta_fit
            #fibparms[fib][ord]['offset_fit'] = offset_fit
            fibparms[fib][ord]['onchip'] = onchip
    
    if savefile:
        np.save('/Users/christoph/OneDrive - UNSW/fibre_profiles/sim/fibparms_by_fib.npy', fibparms) 
    
    return fibparms



def make_fibparms_by_ord(by_fib, savefile=True):
    
    by_orders = {}
    for fibkey in by_fib.keys():
        for ord in by_fib[fibkey].keys():
            by_orders[ord] = {}
    
    for ord in by_orders.keys():
        for fibkey in by_fib.keys():
                    
            dum = by_fib[fibkey][ord]
            by_orders[ord][fibkey] = dum
    
    if savefile:
        np.save('/Users/christoph/OneDrive - UNSW/fibre_profiles/sim/fibparms_by_ord.npy', by_orders) 
    
    return by_orders
    
    

def get_lfc_offset(date='20190503', return_median=False, norm=True):
    
    # read in fibparms for stellar & sky fibres
    stellar_fibparms = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/archive/fibre_profile_fits_' + date + '.npy').item()
    
    # read in fibparms for LFC fibre
    lfc_fibparms = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/laser/lfc_fibre_profile_fits_20190503.npy').item()
    
    # prepare output array
    offsets = np.zeros((39,24,4112))    
    
    for o,ord in enumerate(sorted(stellar_fibparms.keys())):
        for f,fib in enumerate(sorted(stellar_fibparms[ord].keys())):
            offsets[o,f,:] = stellar_fibparms[ord][fib]['mu_fit'] - lfc_fibparms[ord]['fibre_01']['mu_fit']
            if norm:
                # now "normalize" to one step
                offsets[o,f,:] /= int(fib[-2:]) - 1
    
    if return_median:
        # return the median values across fibres
        return np.median(offsets, axis=1)
    else:
        return offsets


    
def get_simthxe_offset(date='20190503', return_median=False, norm=True):
    
    # read in fibparms for stellar & sky fibres
    stellar_fibparms = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/archive/fibre_profile_fits_' + date + '.npy').item()
    
    # read in fibparms for simThXe fibre
    simthxe_fibparms = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/simthxe/sim_ThXe_fibre_profile_fits_20190503_nomask.npy').item()
    
    # prepare output array
    offsets = np.zeros((39,24,4112))    
    
    for o,ord in enumerate(sorted(stellar_fibparms.keys())):
        for f,fib in enumerate(sorted(stellar_fibparms[ord].keys())):
            offsets[o,f,:] = stellar_fibparms[ord][fib]['mu_fit'] - simthxe_fibparms[ord]['fibre_28']['mu_fit']
            if norm:
                # now "normalize" to one step
                offsets[o,f,:] /= 28 - int(fib[-2:])
    
    if return_median:
        # return the median values across fibres
        return np.median(offsets, axis=1)
    else:
        return offsets
    
    
    
def combine_fibparms(date, use_lfc=False, savefile=True):
    
    archive_path = '/Users/christoph/OneDrive - UNSW/fibre_profiles/archive/'
    simthxe_path = '/Users/christoph/OneDrive - UNSW/fibre_profiles/simthxe/'
    lfc_path = '/Users/christoph/OneDrive - UNSW/fibre_profiles/laser/'
    
    # read in fibparms for stellar & sky fibres
    stellar_fibparms = np.load(archive_path + 'fibre_profile_fits_' + date + '.npy').item()
    
    # read in fibparms for simThXe fibre and median offsets from stellar&sky fibres
    simthxe_fibparms = np.load(simthxe_path + 'sim_ThXe_fibre_profile_fits_20190503_nomask.npy').item()
#     med_simthxe_offsets = get_simthxe_offset(return_median=True)
    simthxe_offsets = get_simthxe_offset(date=date)
    
    # read in fibparms for LFC fibre and median offsets from stellar&sky fibres
    lfc_fibparms = np.load(lfc_path + 'lfc_fibre_profile_fits_20190503.npy').item()
#     med_lfc_offsets = get_lfc_offset(return_median=True)
    lfc_offsets = get_lfc_offset(date=date)
    
    # calculate a new simThXe trace based on the measured positions of the other fibres that night and the median offset as measured on the reference night (20190503)
    new_simthxe_traces  = np.zeros((39,24,4112))    
    for o,ord in enumerate(sorted(stellar_fibparms.keys())):
        for f,fib in enumerate(sorted(stellar_fibparms[ord].keys())):
            # new_simthxe_traces[o,f,:] = stellar_fibparms[ord][fib]['mu_fit'] - (28 - int(fib[-2:])) * med_simthxe_offsets[o,:]
            new_simthxe_traces[o,f,:] = stellar_fibparms[ord][fib]['mu_fit'] - (28 - int(fib[-2:])) * simthxe_offsets[o,f,:]
    med_new_simthxe_trace = np.median(new_simthxe_traces, axis=1)
    
    if use_lfc:
        # calculate a new LFC trace based on the measured positions of the other fibres that night and the median offset as measured on the reference night (20190503)
        new_lfc_traces  = np.zeros((39,24,4112))    
        for o,ord in enumerate(sorted(stellar_fibparms.keys())):
            for f,fib in enumerate(sorted(stellar_fibparms[ord].keys())):
                # new_lfc_traces[o,f,:] = stellar_fibparms[ord][fib]['mu_fit'] - (int(fib[-2:]) - 1) * med_lfc_offsets[o,:]
                new_lfc_traces[o,f,:] = stellar_fibparms[ord][fib]['mu_fit'] - (int(fib[-2:]) - 1) * lfc_offsets[o,f,:]
        med_new_lfc_trace = np.median(new_lfc_traces, axis=1)
    else:
        meansep = get_mean_fibre_separation(stellar_fibparms)
        
    
    # create and fill the combined fibparms structure
    combined_fibparms = {}
    for o,ord in enumerate(sorted(stellar_fibparms.keys())):
        combined_fibparms[ord] = {}
        for fib in stellar_fibparms[ord].keys():
            combined_fibparms[ord][fib] = stellar_fibparms[ord][fib]
        for fib in simthxe_fibparms[ord].keys():    # that's just 'fibre_28'
            combined_fibparms[ord][fib] = simthxe_fibparms[ord][fib]
            combined_fibparms[ord][fib]['mu_fit'] = med_new_simthxe_trace[o,:]
        if use_lfc:
            for fib in lfc_fibparms[ord].keys():    # that's just 'fibre_01'
                combined_fibparms[ord][fib] = lfc_fibparms[ord][fib]
                combined_fibparms[ord][fib]['mu_fit'] = med_new_lfc_trace[o,:]
        else:
            for fib in lfc_fibparms[ord].keys():    # that's just 'fibre_01'
                combined_fibparms[ord][fib] = lfc_fibparms[ord][fib]
                combined_fibparms[ord][fib]['mu_fit'] = stellar_fibparms[ord]['fibre_02']['mu_fit'] + meansep[ord]


    if savefile:
        if use_lfc:
            np.save(archive_path + 'combined_fibre_profile_fits_using_lfc_' + date + '.npy', combined_fibparms)
        else:
            np.save(archive_path + 'combined_fibre_profile_fits_' + date + '.npy', combined_fibparms)
        
    return combined_fibparms
    
    
    
    
    
    
    
    
    
    
    
    