'''
Created on 11 May 2018

@author: christoph
'''


import numpy as np
import astropy.io.fits as pyfits
import time
import os
import matplotlib.pyplot as plt

from veloce_reduction.veloce_reduction.helper_functions import get_iterable, find_nearest, get_datestring

#from matplotlib.cm import cmap_d
from plotting_helpers import circles

# C = []
# I = []
# O = []
# tot = []
# seeing = np.arange(.5,5.1,.1)
# # for fwhm in seeing:
# #     fluxes = flux_ratios_from_seeing(fwhm, return_values=True)
# #     C.append(fluxes[1])
# #     I.append(fluxes[2])
# #     O.append(fluxes[3])
# #     tot.append(fluxes[0])



#path = '/Volumes/BERGRAID/data/simu/'



def flux_ratios_from_seeing(seeing, verbose=False):
    '''
    This routine calculates, as a function of seeing (which is approximated by a 2D Gaussian with FWHM = seeing in both directions),
    the flux ratios that the Veloce IFU captures in total, as well as the ratios of the flux captured by the central fibre / inner-ring fibres / outer-ring fibres
    to the total flux captured by the IFU (NOT the TOTAL total flux!!!)
    
    INPUT:
    'seeing'        :  can be scalar or array/list; the FWHM of the seeing disk (approximated as a 2-dim Gaussian)
    
    KEYWORDS:
    'verbose'       : boolean - do you want verbose output printed to screen?
    
    OUTPUT:
    'flux_ratios'   : a python dictionary containing the results
    
    MODHIST:
    Dec 2017    - CMB create
    11/05/2018  - CMB modified output format and included possibility to iterate over 1-element seeing-array
    '''
    
    # initialise output dictionary
    flux_ratios = {}    
    
    # inner-radius r of hexagonal fibre is 0.26", therefore outer-radius R is (2/sqrt(3))*0.26" = 0.30"
    # what we really want is the "effective" radius though, for a regular hexagon that comes from A = 3*r*R = 2*sqrt(3)*r = pi * r_eff**2
    # ie r_eff = sqrt( (2*sqrt(3)) / pi )
#    fac = np.sqrt((2.*np.sqrt(3.)) / np.pi)
    rc = 0.26
    Rc = rc * 2. / np.sqrt(3.) 
#     ri = 0.78
#     ro = 1.30
#     reff = rc * fac
    di = 0.52
    do1 = 1.04
    xo2 = di + rc
    yo2 = 1.5 * Rc
#    do2 = 3. * (2./np.sqrt(3.)) * 0.26
    
    x = np.arange(-10,10.01,.01)
    y = np.arange(-10,10.01,.01)
    xx, yy = np.meshgrid(x, y)
    # define constant (FWHM vs sigma, because FWHM = sigma * 2*sqrt(2*log(2))
    cons = 2*np.sqrt(np.log(2)) 
    
    
    m1 = 1./np.sqrt(3.)   # slope1
    m2 = -m1              # slope2
    central = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx) <= rc, yy <= m1*xx + Rc), yy >= m1*xx - Rc), yy <= m2*xx + Rc), yy >= m2*xx - Rc)
    inner = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx-di) <= rc, yy <= m1*(xx-di) + Rc), yy >= m1*(xx-di) - Rc), yy <= m2*(xx-di) + Rc), yy >= m2*(xx-di) - Rc), ~central)
    outer1 = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx-do1) <= rc, yy <= m1*(xx-do1) + Rc), yy >= m1*(xx-do1) - Rc), yy <= m2*(xx-do1) + Rc), yy >= m2*(xx-do1) - Rc), ~inner)
    outer2 = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx-xo2) <= rc, yy <= m1*(xx-xo2) + Rc - yo2), yy >= m1*(xx-xo2) - Rc - yo2), yy <= m2*(xx-xo2) + Rc - yo2), yy >= m2*(xx-xo2) - Rc - yo2), ~inner), ~outer1)

#########################################################
#     this was the circular approximation:
#     central = (np.sqrt(xx*xx + yy*yy) <= rc)
#     inner = np.logical_and(np.sqrt(xx*xx + yy*yy) <= ri, ~central)
#     outer = np.logical_and(np.sqrt(xx*xx + yy*yy) <= ro, np.logical_and(~central, ~inner))
#     ifu = (np.sqrt(xx*xx + yy*yy) <= ro)
#     central = (np.sqrt(xx*xx + yy*yy) <= reff)
#     inner = np.sqrt((xx-di)*(xx-di) + yy*yy) <= reff
#     outer1 = np.sqrt((xx-do1)*(xx-do1) + yy*yy) <= reff
#     outer2 = np.sqrt((xx-do2)*(xx-do2) + yy*yy) <= reff
#     ifu = (np.sqrt(xx*xx + yy*yy) <= ro*fac)
#########################################################
    
#     if len(np.atleast_1d(seeing)) > 1:
#         
#         frac_ifu = []
#         renorm_frac_c = []
#         renorm_frac_i = []
#         renorm_frac_o = []
#         
#         for fwhm in seeing:
#         
#             if verbose:
#                 print('Simulating '+str(fwhm)+'" seeing...')
#             
#             #calculate 2-dim Gaussian flux distribution as function of input FWHM
#             fx = np.exp(-(np.absolute(xx) * cons / fwhm) ** 2.0)
#             fy = np.exp(-(np.absolute(yy) * cons / fwhm) ** 2.0)
#             f = fx * fy * 1e6
#             
#             frac_c = np.sum(f[central]) / np.sum(f)
#             frac_i = 6. * np.sum(f[inner]) / np.sum(f)
#             frac_o = 6. * ( np.sum(f[outer1]) + np.sum(f[outer2]) ) / np.sum(f)
#             ifu_frac = frac_c + frac_i + frac_o     #this is slightly overestimated (<1%) because they are not actually circular fibres
#             frac_ifu.append(ifu_frac)
#             
#             rfc = frac_c / ifu_frac
#             rfi = frac_i / ifu_frac
#             rfo = frac_o / ifu_frac
#             
#             renorm_frac_c.append(rfc)
#             renorm_frac_i.append(rfi)
#             renorm_frac_o.append(rfo)
#             
#             if verbose:
#                 print('Total fraction of flux captured by IFU: '+str(np.round(ifu_frac * 100,1))+'%')
#                 print('----------------------------------------------')
#                 print('Contribution from central fibre: '+str(np.round(rfc * 100,1))+'%')
#                 print('Contribution from inner-ring fibres: '+str(np.round(rfi * 100,1))+'%')
#                 print('Contribution from outer-ring fibres: '+str(np.round(rfo * 100,1))+'%')
#                 print
#     
#     else:
#         
#         fwhm = seeing
#         if verbose:
#                 print('Simulating '+str(fwhm)+'" seeing...')
#         #calculate 2-dim Gaussian flux distribution as function of input FWHM
#         fx = np.exp(-(np.absolute(xx) * cons / fwhm) ** 2.0)
#         fy = np.exp(-(np.absolute(yy) * cons / fwhm) ** 2.0)
#         f = fx * fy * 1e6
#         
#         frac_c = np.sum(f[central]) / np.sum(f)
#         frac_i = 6. * np.sum(f[inner]) / np.sum(f)
#         frac_o = 6. * ( np.sum(f[outer1]) + np.sum(f[outer2]) ) / np.sum(f)
#         frac_ifu = frac_c + frac_i + frac_o     #this is slightly overestimated (<1%) because they are not actually circular fibres
#         
#         renorm_frac_c = frac_c / frac_ifu
#         renorm_frac_i = frac_i / frac_ifu
#         renorm_frac_o = frac_o / frac_ifu
#         
#         if verbose:
#                 print('Total fraction of flux captured by IFU: '+str(np.round(frac_ifu * 100,1))+'%')
#                 print('----------------------------------------------')
#                 print('Contribution from central fibre: '+str(np.round(renorm_frac_c * 100,1))+'%')
#                 print('Contribution from inner-ring fibres: '+str(np.round(renorm_frac_i * 100,1))+'%')
#                 print('Contribution from outer-ring fibres: '+str(np.round(renorm_frac_o * 100,1))+'%')
#                 print
    
    
    flux_ratios['seeing'] = []
    flux_ratios['frac_ifu'] = []
    flux_ratios['central'] = []
    flux_ratios['inner'] = []
    flux_ratios['outer'] = []
    flux_ratios['outer1'] = []
    flux_ratios['outer2'] = []
    
    #stupid python!!!
    for fwhm in get_iterable(seeing):
    
        if verbose:
            print('Simulating '+str(fwhm)+'" seeing...')
        
        #calculate 2-dim Gaussian flux distribution as function of input FWHM
        fx = np.exp(-(np.absolute(xx) * cons / fwhm) ** 2.0)
        fy = np.exp(-(np.absolute(yy) * cons / fwhm) ** 2.0)
        #f = fx * fy
        f = fx * fy * 1e6
        
        frac_c = np.sum(f[central]) / np.sum(f)
        frac_i = 6. * np.sum(f[inner]) / np.sum(f)
        frac_o1 = 6. * np.sum(f[outer1]) / np.sum(f)
        frac_o2 = 6. * np.sum(f[outer2]) / np.sum(f)
        frac_o = 6. * ( np.sum(f[outer1]) + np.sum(f[outer2]) ) / np.sum(f)
        ifu_frac = frac_c + frac_i + frac_o     #this is slightly overestimated (<1%) because they are not actually circular fibres
        
        #renormalized fractions (renormalized to the total flux captured by the IFU, not the TOTAL total flux)
        rfc = frac_c / ifu_frac
        rfi = frac_i / ifu_frac
        rfo = frac_o / ifu_frac
        rfo1 = frac_o1 / ifu_frac
        rfo2 = frac_o2 / ifu_frac
        
        #fill the output dictionary  
        flux_ratios['seeing'].append(fwhm)
        flux_ratios['frac_ifu'].append(ifu_frac)
        flux_ratios['central'].append(rfc)
        flux_ratios['inner'].append(rfi)
        flux_ratios['outer'].append(rfo)
        flux_ratios['outer1'].append(rfo1)
        flux_ratios['outer2'].append(rfo2)
        
        if verbose:
            print('Total fraction of flux captured by IFU: '+str(np.round(ifu_frac * 100,1))+'%')
            print('----------------------------------------------')
            print('Contribution from central fibre: '+str(np.round(rfc * 100,1))+'%')
            print('Contribution from inner-ring fibres: '+str(np.round(rfi * 100,1))+'%')
            print('Contribution from outer-ring fibres: '+str(np.round(rfo * 100,1))+'%')
            print
    
    return flux_ratios





def get_pseudo_slitamps(central, inner, outer1, outer2):
    """
    INPUT:
    flux_ratios bzw. rel. intensities of the central / inner / outer(1/2) fibres (can be scalars or arrays)
    
    OUTPUT:
    'slit'   : 19-element array containing the rel. intensities arranged like shown in 
               "/Users/christoph/UNSW/IFU_layout/IFU_to_pseudoslit_fibre_mapping_20180222.png", ie [L S1 S3 S4 X 2 19 8 3 9 10 4 11 12 1 13 14 5 15 16 6 17 18 7 X S2 S5 ThXe]
               fibres 8,10,12,14,16,18 are "outer1"
               fibres 9,11,13,15,17,19 are "outer2"
               
    MODHIST:
    15/05/2018 - CMB create
    """
    
    slitamps = np.array([inner,outer2,outer1,inner,outer2,outer1,inner,outer2,outer1,central*6.,outer2,outer1,inner,outer2,outer1,inner,outer2,outer1,inner]) / 6.
     
    return slitamps





def make_pseudoslits(sigma=67., nfib=19):
    """
    This routine creates random-normal-sampled offsets for the stellar fibres that make up the Veloce pseudoslit. The user has the option to supply the desired level of scatter 
    in the fibre offsets, which is used to draw from a random-normal distribution the offsets (in steps of 10 m/s). Default value is 67 m/s, based on the considerations 
    in "WORD DOCUMENT HERE". Maximum offset is +/- 200 m/s, which corresponds to ~3 default sigmas, or a probability of ~0.27% that the offset more than that (in which case 
    the maximum offset is still used).
    NOTE: Alignment inaccuracies in the spatial directions are NOT treated here, as they should not influence the RV measurements.
    
    INPUT:
    'sigma'   : RMS of the fibre offsets (in m/s)
    'nfib'    : number of fibres (default for Veloce is 19 stellar fibres)
    
    OUTPUT:
    'offsets' : an nfib-element array of the individual fibre offsets (in m/s)
    
    MODHIST:
    15/05/2018 - CMB create
    """

    #For random samples from N(mu, sigma^2), use:   "sigma * np.random.randn(dimensions) + mu"
    pos = sigma * np.random.randn(nfib)
    
    #divide into discreet bins
    bins = np.arange(-200,201,10)
    
    #which bin is the right one?
    offsets = np.array([])
    for x in pos:
        offsets = np.append(offsets, find_nearest(bins,x))
        
    return offsets





def plot_pseudoslit(offsets, saveplot=False):
    """
    This routine plots the pseudo-slit corresponding to the input array 'offsets'. Fibres are color-coded blue to red.
    """
    
    ycen = np.arange(19) * 2.15
    xcen = 0.1 * offsets / 200.
    
    fig, ax = plt.subplots()
    
    circles(xcen, ycen, s=1, c=xcen, ec='black',cmap=plt.cm.get_cmap('bwr'),autoscale=True)
    ax.scatter(xcen,ycen,color='black',marker='o',s=.5)
    ax.axvline(x=0, color='grey', ls='--')
    ax.axvline(x=1, color='grey', ls='--')
    ax.axvline(x=-1, color='grey', ls='--')
    
#     plt.xlim(-1.15,1.15)
#     plt.xlim(-25,25)
#     plt.ylim(-5,45)
    
    #ax.set_aspect(aspect=.2)
    
    #plt.axis('scaled')
    plt.xlabel('dispersion direction pixels')
    plt.ylabel('spatial direction pixels')
    
    if saveplot:
        print("NIGHTMARE: I haven't done that yet...")
    return





def make_pseudo_obs(seeing=None, simpath='/Volumes/BERGRAID/data/simu/', nfib=19, outpath='/Volumes/BERGRAID/data/simu/composite/', fixed_offsets=False, fixoff=0., 
                    fixed_slitamps=False, slitamps=1., norm_slitamps=False, template=False, add_laser=False, savefile=True, verbose=False, debug_level=0, timit=False):    
    """
    This routine creates simulated observations by 
    (1) calculating the flux-ratios / relative intensities in the individual fibres,
    (2) simulating a pseudo-slit with a random normal distribution of relative offsets of the fibres relative to the centre (ie simulating alignment imperfections),
    (3) adding together the individual-fibre spectra from my EchelleSimulator library using the appropriate relative intensities.
    
    INPUT:
    'seeing'           : FWHM values of desired simulated seeing condition(s) - can be scalar or array
    'simpath'          : path to my EchelleSimulator spectrum library
    'nfib'             : the number of (stellar) fibres to use
    'outpath'          : root-directory for the output files (sub-directories will be created every time this routine is run because of the random-normal pseudo-slit alignments)
    'fixed_offsets'    : boolean - do you want to simulate fibre-offsets in dispersion direction for the pseudoslit? (set to TRUE if you don't want to simulate offsets, but rather use a constant offset 'fixoff' for all fibres)
    'fixoff'           : constant offset for all fibres (only used if 'fixed_offsets' is set to TRUE)
    'fixed_slitamps'   : boolean - do you want to simulate the effect of seeing or provide fixed relative intensities? (if TRUE, the relative intensities are taken from 'slitamps'; if set to FALSE
                         the relative intensities are calculated using the provided seeing values)     
    'slitamps'         : constant slitamps for all fibres (=relative intensities) (only used if 'fixed_slitamps' is set to TRUE)
    'norm_slitamps'    : boolean - do you want to 'normalize' the slitamps so they add up to 1? (only used if 'fixed_slitamps' is set to TRUE)     
    'template'         : boolean - is this going to be a template? in that case use the 'maxsnr' spectra...
    'add_laser'        : boolean - do you want to add the LFC in fibre 1?
    'savefile'         : boolean - do you want to save the simulated spectra as FITS files?
    'verbose'          : boolean - for debugging...
    'debug_level'      : boolean - for debugging...
    'timit'            : boolean - do you want to clock the run-time?
    
    OUTPUT:
    'master_dict'      : dictionary containing the simulated spectra for all seeing values (only if the 'savefile' keyword is not set, otherwise results are saved to FITS file(s))
    
    TODO: 
    add SNR   -   for now this is for a given (fixed) observing time, i.e. the total flux captured by the IFU drops as the seeing increases
    
    MODHIST:
    16/05/2018 - CMB create
    08/06/2018 - CMB added 'add_laser' keyword; also changed the addition of 1 to the first image only
    13/06/2018 - CMB added 'template' keyword (uses the 'maxsnr' simulated spectra)
    14/06/2018 - CMB added "slitamps.txt" output file; also added 'fixed_slitamps', 'slitamps', and 'norm_slitamps' keywords
    """
    
    if timit:
        start_time = time.time()
    
    ##### (1) #####
    if not fixed_slitamps:
        if seeing is not None:
            #first, calculate flux_ratios (ie relative intensities) for the different fibres from given seeing
            fr = flux_ratios_from_seeing(seeing, verbose=verbose)
            #then get an array of the relative intensities in the pseudo-slit
            slitamps = get_pseudo_slitamps(fr['central'], fr['inner'], fr['outer1'], fr['outer2'])
        else:
            print('ERROR: "fixed_slitamps" = False, but seeing not provided!!!')
            return
    else:
        if len(get_iterable(slitamps))==1:
            slitamps = np.repeat(slitamps, nfib).astype(float)
#         #reshape to accommodate the possibility of having multiple seeing conditions below
#         slitamps = np.reshape(slitamps, (19,1))
        if norm_slitamps:
            slitamps = slitamps.astype(float) / np.sum(slitamps)
    
    ##### (2) #####
    #simulate fibre-offsets by using RV offsets
    if not fixed_offsets:
        offsets = make_pseudoslits()
    else:
        if len(get_iterable(fixoff))==1:
            offsets = np.repeat(fixoff, nfib)
        elif len(get_iterable(fixoff))==nfib:
            offsets = np.array(fixoff)
        else:
            print('ERROR! input values for fibre offsets do not have correct length')
            return
    
    if debug_level >= 1:
        plot_pseudoslit(offsets)
    
    ##### (3) #####
    #some string manipulations for filenames in for loop below
    strshifts = np.abs(offsets).astype(int).astype(str)
    redblue = np.empty(nfib).astype(str)
    redblue[offsets > 0] = 'red'
    redblue[offsets < 0] = 'blue'
    redblue[offsets == 0] = ''
    if add_laser:
        laserstr = '_laser'
    else:
        laserstr = ''
    if template:
        tstring = '_template'
        maxsnr_string = '_maxsnr'
    else:
        tstring = ''
        maxsnr_string = ''
    
    if savefile:
        #create new sub-folder with info files containing info on the fibre offsets and the slitamps (=relative intensities)
        datestring = get_datestring()
        dum = 1
        newpath = outpath + 'tests_' + datestring
        dumpath = outpath + 'tests_' + datestring
        while os.path.exists(dumpath):
            dum += 1
            dumpath = newpath + '_' + str(dum)
        if dum > 1:
            newpath = newpath + '_' + str(dum)
        #create new folder
        os.makedirs(newpath)
        #write OFFSETS file
        outfn = newpath + '/' + 'offsets.txt' 
        outfile = open(outfn, 'w')
        outfile.writelines(["%s\n" % item for item in offsets.astype(str)])
        outfile.close()
        #write SLITAMPS file
        outfn = newpath + '/' + 'slitamps.txt' 
        outfile = open(outfn, 'w')
        outfile.writelines(["%s\n" % item for item in slitamps.astype(str)])
        outfile.close()
        
    else:
        master_dict = {}
    
    if seeing is not None:
        for i,fwhm in enumerate(get_iterable(seeing)):
            if verbose:
                print('SEEING-LOOP: ',fwhm)
            #use fibre-slots 6 to 24
            for n in range(nfib):
                fibslot = str(n+6).zfill(2)
                img = pyfits.getdata(simpath+'fib'+fibslot+'_'+redblue[n]+strshifts[n]+'ms'+maxsnr_string+'.fit') 
                if n==0:
                    master = (img.copy().astype(float) * slitamps[n,i]) + 1.
                    h = pyfits.getheader(simpath+'fib'+fibslot+'_'+redblue[n]+strshifts[n]+'ms'+maxsnr_string+'.fit')
                else:
                    master += img * slitamps[n,i]
            if add_laser:
                laser_img = pyfits.getdata(simpath + 'veloce_laser_comb.fit')
                master += laser_img/5.
            if savefile:
                #save to file
                pyfits.writeto(newpath+'/seeing'+fwhm.astype(str)+laserstr+tstring+'.fit', master, h, clobber=True)
            else:
                master_dict['seeing'+fwhm.astype(str)] = master
    else:
        for n in range(nfib):
            fibslot = str(n+6).zfill(2)
            img = pyfits.getdata(simpath+'fib'+fibslot+'_'+redblue[n]+strshifts[n]+'ms'+maxsnr_string+'.fit') 
            if n==0:
                master = (img.copy().astype(float) * slitamps[n]) + 1.
                h = pyfits.getheader(simpath+'fib'+fibslot+'_'+redblue[n]+strshifts[n]+'ms'+maxsnr_string+'.fit')
            else:
                master += img * slitamps[n]
        if add_laser:
            laser_img = pyfits.getdata(simpath + 'veloce_laser_comb.fit')
            master += laser_img/5.
        if savefile:
            #save to file
            pyfits.writeto(newpath+'/syntobs'+laserstr+tstring+'.fit', master, h, clobber=True)
        else:
            master_dict['syntobs'+tstring] = master
    
    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')
    
    if savefile:
        #print('Offsets: ',offsets)
        return
    else:
        return master_dict





def make_pseudo_obs_with_noise(seeing, snrs=None, RON=0., simpath='/Volumes/BERGRAID/data/simu/', nfib=19, outpath='/Volumes/BERGRAID/data/simu/composite/', fixed_offsets=False, fixoff=0.,
                               fixed_slitamps=False, slitamps=1., norm_slitamps=False, template=False, add_laser=False, savefile=True, verbose=False, debug_level=0, timit=False):    
    """
    This routine creates simulated observations by 
    (1) calculating the flux-ratios / relative intensities in the individual fibres,
    (2) simulating a pseudo-slit with a random normal distribution of relative offsets of the fibres relative to the centre (ie simulating alignment imperfections),
    (3) adding together the individual-fibre spectra from my EchelleSimulator library using the appropriate relative intensities.
    
    INPUT:
    'seeing'           : FWHM values of desired simulated seeing condition(s) - can be scalar or array
    'snrs'             : desired mean signal-to-noise ratio per extracted 1-dim pixel (WARNING: the seeing parameter does not renormalize to the flux captured by the IFU!!!)
                         floats can be provided for 'snrs', but it will be rounded to integer
    'RON'              : read-out noise
    'simpath'          : path to my EchelleSimulator spectrum library
    'nfib'             : the number of (stellar) fibres to use
    'outpath'          : root-directory for the output files (sub-directories will be created every time this routine is run because of the random-normal pseudo-slit alignments)
    'fixed_offsets'    : boolean - do you want to simulate fibre-offsets in dispersion direction for the pseudoslit? (set to TRUE if you don't want to simulate offsets, but rather use a constant offset 'fixoff' for all fibres)
    'fixoff'           : constant offset for all fibres (only used if 'fixed_offsets' is set to TRUE)
    'fixed_slitamps'   : boolean - do you want to simulate the effect of seeing or provide fixed relative intensities? (if TRUE, the relative intensities are taken from 'slitamps'; if set to FALSE
                         the relative intensities are calculated using the provided seeing values)     
    'slitamps'         : constant slitamps for all fibres (=relative intensities) (only used if 'fixed_slitamps' is set to TRUE)
    'norm_slitamps'    : boolean - do you want to 'normalize' the slitamps so they add up to 1? (only used if 'fixed_slitamps' is set to TRUE)     
    'template'         : boolean - is that going to be a template? in that case no noise is added...
    'add_laser'        : boolean - do you want to add the LFC in fibre 1?
    'savefile'         : boolean - do you want to save the simulated spectra as FITS files?
    'verbose'          : boolean - for debugging...
    'debug_level'      : boolean - for debugging...
    'timit'            : boolean - do you want to time the execution time?
    
    OUTPUT:
    'master_dict'      : dictionary containing the simulated spectra for all seeing values (only if the 'savefile' keyword is not set, otherwise results are saved to FITS file(s))
    
    TODO: add SNR   -   for now this is for a given (fixed) observing time, i.e. the total flux captured by the IFU drops as the seeing increases
    
    MODHIST:
    07/06/2018 - CMB create (clone of "make_pseudo_obs")
    08/06/2018 - CMB added 'add_laser' keyword; also changed the addition of 1 to the first image only
    14/06/2018 - CMB added "slitamps.txt" output file; also added 'fixed_slitamps', 'slitamps', and 'norm_slitamps' keywords
    """
    
    if timit:
        start_time = time.time()
    
    ##### (1) #####
    if not fixed_slitamps:
        if seeing is not None:
            #first, calculate flux_ratios (ie relative intensities) for the different fibres from given seeing
            fr = flux_ratios_from_seeing(seeing, verbose=verbose)
            #then get an array of the relative intensities in the pseudo-slit
            slitamps = get_pseudo_slitamps(fr['central'], fr['inner'], fr['outer1'], fr['outer2'])
        else:
            print('ERROR: "fixed_slitamps" = False, but seeing not provided!!!')
            return
    else:
        if len(get_iterable(slitamps))==1:
            slitamps = np.repeat(slitamps, nfib)
        #reshape to accommodate the possibility of having multiple seeing condittions below
        slitamps = np.reshape(slitamps, (19,1))
        if norm_slitamps:
            slitamps = slitamps / np.sum(slitamps)
    
    ##### (2) #####
    #simulate fibre-offsets by using RV offsets
    if not fixed_offsets:
        #offsets = make_pseudoslits()
        print('ERROR: offsets in pixel space have not been implemented yet!!!')
        return
    else:
        if len(get_iterable(fixoff))==1:
            offsets = np.repeat(fixoff, nfib)
        elif len(get_iterable(fixoff))==nfib:
            offsets = np.array(fixoff)
        else:
            print('ERROR! input values for fibre offsets do not have correct length')
            return
    if not (offsets == 0).all():
        print('ERROR: offsets in pixel space have not been implemented yet!!!')
        return
    
    if debug_level >= 1:
        plot_pseudoslit(offsets)
    
    ##### (3) #####
    #some string manipulations for filenames in for loop below
    strshifts = np.abs(offsets).astype(int).astype(str)
    redblue = np.empty(nfib).astype(str)
    redblue[offsets > 0] = 'red'
    redblue[offsets < 0] = 'blue'
    redblue[offsets == 0] = ''
    if add_laser:
        laserstr = '_laser'
    else:
        laserstr = ''
    
    if savefile:
        #create new sub-folder with README file containing info on the fibre offsets
        datestring = get_datestring()
        dum = 1
        newpath = outpath + 'tests_' + datestring
        dumpath = outpath + 'tests_' + datestring
        while os.path.exists(dumpath):
            dum += 1
            dumpath = newpath + '_' + str(dum)
        if dum > 1:
            newpath = newpath + '_' + str(dum)
        #create new folder
        os.makedirs(newpath)
        #write README file
        outfn = newpath + '/' + 'offsets.txt' 
        outfile = open(outfn, 'w')
        outfile.writelines(["%s\n" % item for item in offsets.astype(str)])
        outfile.close()
        #write SLITAMPS file
        outfn = newpath + '/' + 'slitamps.txt' 
        outfile = open(outfn, 'w')
        outfile.writelines(["%s\n" % item for item in slitamps.astype(str)])
        outfile.close()
        
    else:
        master_dict = {}
    
    for i,(fwhm,snr) in enumerate(zip(get_iterable(seeing), get_iterable(snrs))):
        snr = int(np.round(snr))
        if verbose:
            print('SEEING: ',fwhm)
            print('SNR: ',snr)
        #use fibre-slots 6 to 24
        for n in range(nfib):
            fibslot = str(n+6).zfill(2)
            img = pyfits.getdata(simpath+'fib'+fibslot+'_'+redblue[n]+strshifts[n]+'ms_maxsnr.fit') + 1.
            if n==0:
                master = img.copy().astype(float) * slitamps[n,i]
                h = pyfits.getheader(simpath+'fib'+fibslot+'_'+redblue[n]+strshifts[n]+'ms_maxsnr.fit')
            else:
                master += img * slitamps[n,i]
        
        #now add noise (a "maxsnr" spectrum has a mean SNR of ~555)
        if not template:
            fudge = (snr/555.)**2.
            scaled_master = master * fudge
            err_amps = np.sqrt(scaled_master + RON*RON)
            #add noise to the image
            scaled_noise = make_scaled_white_noise(err_amps)
            noisy_img = scaled_master + scaled_noise
        else:
            #don't want to add noise for the templates
            noisy_img = master.copy()
        
        #remove any negative pixels    
        noisy_img[noisy_img < 0] = 0.
        
        if add_laser:
            laser_img = pyfits.getdata(simpath + 'veloce_laser_comb.fit')
            noisy_img += laser_img/5.
        
        if savefile:
            #save to file
            pyfits.writeto(newpath+'/seeing'+fwhm.astype(str)+'_snr_'+str(snr)+laserstr+'.fit', noisy_img, h, clobber=True)
        else:
            master_dict['seeing'+fwhm.astype(str)] = master
    
    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')
    
    if savefile:
        #print('Offsets: ',offsets)
        return
    else:
        return master_dict





def make_mult_pseudo_obs(offsets=0., slitamps=1., nfib=19, simpath='/Volumes/BERGRAID/data/simu/',  outpath='/Volumes/BERGRAID/data/simu/composite/', norm_slitamps=False, 
                         template=False, add_laser=False, savefile=True, verbose=False, debug_level=0, timit=False):    
    """
    This routine creates simulated observations by adding together the individual-fibre spectra from my EchelleSimulator library using the appropriate relative intensities.
    It does essentially the same as "make_pseudo_obs", but takes fibre-offsets and slitamps as user-defined inputs (ie as if 'fixed_offsets' and 'fixed_slitamps' were both
    set to TRUE in "make_pseudo_obs", and no 'seeing' array was provided).
    
    INPUT:
    'offsets'          : constant offset for all fibres (can be scalar (all fibres have same offset), or array of size (nobs,nfib))
    'slitamps'         : constant slitamps for all fibres (=relative intensities) (can be scalar (all fibres have same offset), or array of size (nobs,nfib))
    'nfib'             : the number of (stellar) fibres to use
    'simpath'          : path to my EchelleSimulator spectrum library
    'outpath'          : root-directory for the output files (sub-directories will be created every time this routine is run because of the random-normal pseudo-slit alignments)
    'norm_slitamps'    : boolean - do you want to 'normalize' the slitamps so they add up to 1? (only used if 'fixed_slitamps' is set to TRUE)     
    'template'         : boolean - is this going to be a template? in that case use the 'maxsnr' spectra...
    'add_laser'        : boolean - do you want to add the LFC in fibre 1?
    'savefile'         : boolean - do you want to save the simulated spectra as FITS files?
    'verbose'          : boolean - for debugging...
    'debug_level'      : boolean - for debugging...
    'timit'            : boolean - do you want to clock the run-time?
    
    OUTPUT:
    'master_dict'      : dictionary containing the simulated spectra for all seeing values (only if the 'savefile' keyword is not set, otherwise results are saved to FITS file(s))
    
    TODO: 
    add SNR   -   for now this is for a given (fixed) observing time, i.e. the total flux captured by the IFU drops as the seeing increases
    
    MODHIST:
    02/07/2018 - CMB create (clone of "make_pseudo_obs")
    """
    
    if timit:
        start_time = time.time()
    
    ##### (1) #####
    # fibre-offsets formatting
    # (i) if a scalar is provided; nobs=1, use for all fibres
    if len(get_iterable(offsets)) == 1:
        nobs_from_offsets = 1
        offsets = np.repeat(offsets, nfib)
        #how can you do this for multiple obs?
        if debug_level >= 1:
            plot_pseudoslit(offsets)
    # (ii) if a nfib-element array is provided; nobs=1
    elif np.array(offsets).ndim == 1:
        if offsets.shape[0] == nfib:
            nobs_from_offsets = 1
            offsets = np.array(offsets)
        else:
            print('ERROR: "offsets" has the wrong shape!!!')
            return
        #how can you do this for multiple obs?
        if debug_level >= 1:
            plot_pseudoslit(offsets)
    # (iii) a (nx,nfib)-element array is provided; nobs=nx
    else:
        if offsets.shape[1] != nfib:
            print('ERROR: "offsets" has the wrong shape!!!')
            return
        else:
            nobs_from_offsets = offsets.shape[0]        
        
        
    ##### (2) #####
    # slitamps formatting
    # (i) if a scalar is provided; nobs=1, use for all fibres
    if len(get_iterable(slitamps)) == 1:
        nobs_from_slitamps = 1
        slitamps = np.repeat(slitamps, nfib).astype(float)
    # (ii) if a nfib-element array is provided; nobs=1
    elif np.array(slitamps).ndim == 1:
        if slitamps.shape[0] == nfib:
            nobs_from_slitamps = 1
        else:
            print('ERROR: "slitamps" has the wrong shape!!!')
            return
    # (iii) a (nx,nfib)-element array is provided; nobs=nx
    else:
        if slitamps.shape[1] != nfib:
            print('ERROR: "slitamps" has the wrong shape!!!')
            return
        else:
            nobs_from_slitamps = slitamps.shape[0]
            
            
    #make sure we have consistent dimensions between "offsets" and "slitamps"
    if nobs_from_offsets > nobs_from_slitamps:
        if nobs_from_slitamps == 1:
            print('WARNING: number of fibres inferred from "offsets" is LARGER than from "slitamps" !!!')
            print('Using same slitamps for all observations...')
            nobs = nobs_from_offsets
            slitamps_1dim = slitamps.copy()
            while nobs_from_slitamps < nobs_from_offsets: 
                slitamps = np.vstack((slitamps,slitamps))
                nobs_from_slitamps += 1
        else:
            print('ERROR: dimensions of "offsets" and "slitamps" do not agree!!!')
            return
    elif nobs_from_offsets < nobs_from_slitamps:
        if nobs_from_offsets == 1:
            print('WARNING: number of fibres inferred from "offsets" is SMALLER than from "slitamps" !!!')
            print('Using same offsets for all observations...')
            nobs = nobs_from_slitamps
            offsets_1dim = offsets.copy()
            while nobs_from_offsets < nobs_from_slitamps: 
                offsets = np.vstack((offsets,offsets_1dim))
                nobs_from_offsets += 1
        else:
            print('ERROR: dimensions of "offsets" and "slitamps" do not agree!!!')
            return
    else:
        nobs = nobs_from_offsets
                
        
    ## reshape to accommodate the possibility of having multiple seeing conditions below
    # slitamps = np.reshape(slitamps, (19,1))

    if norm_slitamps:
        slitsums = slitamps.sum(axis=1, keepdims=True)
        slitamps = slitamps.astype(float) / slitsums
    
    
    if savefile:
        #create new sub-folder with info files containing info on the fibre offsets and the slitamps (=relative intensities)
        datestring = get_datestring()
        dum = 1
        newpath = outpath + 'tests_' + datestring
        dumpath = outpath + 'tests_' + datestring
        while os.path.exists(dumpath):
            dum += 1
            dumpath = newpath + '_' + str(dum)
        if dum > 1:
            newpath = newpath + '_' + str(dum)
        #create new folder
        os.makedirs(newpath)
    else:
        master_dict = {}
    
    
    #loop over all observations
    for i in np.arange(nobs):
        if verbose:
            print('Simulating observation '+str(i+1)+'/'+str(nobs))
            
        #some string manipulations for filenames in for loop below
        strshifts = np.abs(offsets[i,:]).astype(int).astype(str)
        redblue = np.empty(nfib).astype(str)
        redblue[offsets[i,:] > 0] = 'red'
        redblue[offsets[i,:] < 0] = 'blue'
        redblue[offsets[i,:] == 0] = ''
        if add_laser:
            laserstr = '_laser'
        else:
            laserstr = ''
        if template:
            tstring = '_template'
            maxsnr_string = '_maxsnr'
        else:
            tstring = ''
            maxsnr_string = ''    
        
        
        #write OFFSETS file
        outfn = newpath + '/' + 'offsets_'+str(i+1).zfill(len(str(nobs)))+'.txt' 
        outfile = open(outfn, 'w')
        outfile.writelines(["%s\n" % item for item in offsets[i,:].astype(str)])
        outfile.close()
        #write SLITAMPS file
        outfn = newpath + '/' + 'slitamps_'+str(i+1).zfill(len(str(nobs)))+'.txt' 
        outfile = open(outfn, 'w')
        outfile.writelines(["%s\n" % item for item in slitamps[i,:].astype(str)])
        outfile.close()
        
            
        #use fibre-slots 6 to 24
        for n in range(nfib):
            fibslot = str(n+6).zfill(2)
            img = pyfits.getdata(simpath+'fib'+fibslot+'_'+redblue[n]+strshifts[n]+'ms'+maxsnr_string+'.fit') 
            if n==0:
                master = (img.copy().astype(float) * slitamps[i,n]) + 1.
                h = pyfits.getheader(simpath+'fib'+fibslot+'_'+redblue[n]+strshifts[n]+'ms'+maxsnr_string+'.fit')
            else:
                master += img * slitamps[i,n]
        if add_laser:
            laser_img = pyfits.getdata(simpath + 'veloce_laser_comb.fit')
            master += laser_img/5.
        if savefile:
            #save to file
            pyfits.writeto(newpath+'/syntobs_'+str(i+1).zfill(len(str(nobs)))+laserstr+tstring+'.fit', master, h, clobber=True)
        else:
            master_dict['syntobs_'+str(i+1)] = master

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')
    
    if savefile:
        #print('Offsets: ',offsets)
        return
    else:
        return master_dict





def make_scaled_white_noise(err_amps):
    ny,nx = err_amps.shape
    #noise with sigma=1
    noise = np.resize(np.random.normal(0, 1, nx*ny),(ny,nx))
    #scale noise with "error-amplitudes"
    scaled_noise = noise * err_amps
    return scaled_noise





def add_rel_scatter_to_slitamps(slitamps, relerr):    
    ny,nx = slitamps.shape
    #noise with sigma=1
    noise = np.resize(np.random.normal(0, 1, nx*ny),(ny,nx))
    #scale noise with "error-amplitudes"
    scaled_noise = noise * relerr * slitamps
    noisy_slitamps = slitamps + scaled_noise
    return noisy_slitamps
    




def slitamps_scatter_test(seeing=1.5, ntest=10, relerr=0.1):
    
    #use typical seeing condition
    fr = flux_ratios_from_seeing(seeing, verbose=False)
    #then get an array of the relative intensities in the pseudo-slit
    theo_slitamps = get_pseudo_slitamps(fr['central'], fr['inner'], fr['outer1'], fr['outer2'])

    #weird formatting thing
    n=1
    nf_slitamps = theo_slitamps.T
    while n < ntest:
        nf_slitamps = np.vstack((nf_slitamps,theo_slitamps.T))
        n += 1
    
    ##this is the array that we will pass to "make_mult_pseudo_obs"
    #allslitamps = []

    for re in relerr:
        #add some scatter to the slitamps
        re_slitamps = add_rel_scatter_to_slitamps(nf_slitamps, re)
        try:
            allslitamps = np.vstack((allslitamps, re_slitamps))
        except:
            allslitamps = re_slitamps

    #make on-the-fly templates
    dum = make_mult_pseudo_obs(offsets=0., slitamps=allslitamps, nfib=19, norm_slitamps=True, template=True, verbose=True)

    return










# def shift_simuspec(shift, fibnum, simpath='/Volumes/BERGRAID/data/simu/', fibnum=15, outpath='/Volumes/BERGRAID/data/simu/composite/', savefile=True, verbose=False, debug_level=0, timit=False):    
#     """WORK IN PROGRESS"""
#     
#     redblue = ''
#     #positive (ie right) shift in pixels is towards smaller wavelengths for simulated spectra, ie blue-shift
#     if shift < 0:
#         redblue = 'red'
#     if shift > 0:
#         redblue = 'blue'
#     
#     
#     if savefile:
#         #create new sub-folder with README file containing info on the simulated observations
#         datestring = get_datestring()
#         dum = 1
#         newpath = outpath + 'tests_' + datestring
#         dumpath = outpath + 'tests_' + datestring
#         while os.path.exists(dumpath):
#             dum += 1
#             dumpath = newpath + '_' + str(dum)
#         if dum > 1:
#             newpath = newpath + '_' + str(dum)
#         #create new folder
#         os.makedirs(newpath)
#         #write README file
#         outfn = newpath + '/' + 'README.txt' 
#         outfile = open(outfn, 'w')
#         outfile.write('Fibre '+str(fibnum).zfill(2)+'\n')
#         outfile.write(redblue+'-shifted by '+str(shift)+' pixels\n')
#         #outfile.writelines(["%s\n" % item for item in offsets.astype(str)])
#         outfile.close()
#     
#     #read unshifted image
#     fibslot = str(fibnum).zfill(2)
#     img = pyfits.getdata(simpath+'fib'+fibslot+'_0ms_maxsnr.fit') + 1.
#     
#     ny,nx = img.shape
#     xx = np.arange(nx)
#     #shift x-axis
#     xx_shifted = xx + shift
#     #then interpolate back onto original x-axis
#     spl = interp.InterpolatedUnivariateSpline(xx_shifted, spec, k=3)    #slightly slower than linear, but best performance for cubic spline
#     shifted_spec = spl(xx)
# 
#     return



